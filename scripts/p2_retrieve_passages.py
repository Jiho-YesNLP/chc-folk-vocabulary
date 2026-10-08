"""Per-ability seed retrieval over BGE-M3 FAISS + sparse indices (Stage 2).

For each narrow ability, encodes its seed sentences and retrieves the top-K
candidate passages from every available index using hybrid (dense + sparse)
scoring. Results are saved to:

  data/processed/retrieval/track_a/{ability_code}_raw.jsonl

One record per (passage, seed) pair:
  {passage_id, source, text, source_id, seed_id, seed_type,
   dense_score, sparse_score, hybrid_score, ability_code}

Seeds come from the ``seeds`` list of each narrow ability in the CHC inventory
(seed_type: "inventory_seed"). An external file at
data/raw/seeds/track_a/{ability_code}.jsonl overrides them when present. There is
no fallback: an ability without seeds is an error (the inventory loader refuses it).

Because every descriptor seed shares a "person-narrative" (she/he ...) framing, that
common-mode direction dominates the embeddings and inflates similarity to any
person-narrative passage. When ``background_subtraction`` / ``remove_top_pcs``
are set, an All-but-the-Top transform (Mu & Viswanath 2018) is fit on the pooled
seeds across all abilities and projected out of each query before search.

Usage:
  uv run python scripts/p2_retrieve_passages.py configs/p2_retrieval.yaml --ability I
  uv run python scripts/p2_retrieve_passages.py configs/p2_retrieval.yaml --all
  uv run python scripts/p2_retrieve_passages.py configs/p2_retrieval.yaml --all --debug
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import faiss
import numpy as np
import yaml
from tqdm import tqdm

from src.models.encoder import BGEM3Encoder
from src.data.inventory import read_inventory

# ── seed loading ──────────────────────────────────────────────────────────────

def _load_inventory(inventory_path: pathlib.Path) -> dict[str, dict]:
    """Load narrow abilities keyed by globally-unique ``uid`` (``{broad}-{code}``).

    Narrow ``code``s are unique only *within* a broad stratum: a few (RS, WS) are
    cross-listed across strata, and two pairs (WA/Wa, PC/Pc) differ only by case.
    Keying by the composite uid makes every ability distinct and filesystem-safe,
    so cross-listed/case-twin abilities never overwrite one another.
    """
    data = read_inventory(inventory_path)
    abilities: dict[str, dict] = {}
    for broad in data["broad_abilities"]:
        for narrow in broad.get("narrow_abilities", []):
            code = narrow["code"]
            uid = narrow.get("uid") or f"{broad['code']}-{code}"
            if uid in abilities:
                raise ValueError(f"Duplicate ability uid '{uid}' in inventory")
            abilities[uid] = {
                "uid": uid,
                "code": code,
                "name": narrow["name"],
                "broad_stratum": broad["code"],
                "definition": narrow["definition"],
                "seeds": narrow.get("seeds", []),
            }
    return abilities


def load_seeds(
    ability_code: str,
    seed_dir: pathlib.Path,
    inventory: dict[str, dict],
) -> list[dict]:
    """Return seed dicts for the ability whose uid is *ability_code*.

    ``ability_code`` is the composite uid (``{broad}-{code}``); it keys the
    inventory and names the seed/output files.

    Each dict has: ability_code (uid), code (bare mnemonic), ability_name,
    broad_stratum, seed_id, text, seed_type.

    Resolution order:
      1. An external seed JSONL file (data/raw/seeds/track_a/{uid}.jsonl), if present.
      2. The ``seeds`` list on the ability in the project annotations (the normal
         path, seed_type: inventory_seed).

    There is no further fallback: an ability with no seeds raises.
    """
    seed_file = seed_dir / f"{ability_code}.jsonl"
    if seed_file.exists():
        seeds = []
        for line in seed_file.read_text().splitlines():
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
        if seeds:
            return seeds

    if ability_code not in inventory:
        raise ValueError(f"Ability code '{ability_code}' not found in inventory")
    ab = inventory[ability_code]
    base = {
        "ability_code": ability_code,   # composite uid, e.g. "Grw-RS"
        "code": ab["code"],             # bare mnemonic, e.g. "RS"
        "ability_name": ab["name"],
        "broad_stratum": ab["broad_stratum"],
    }

    inv_seeds = ab.get("seeds") or []
    if not inv_seeds:
        raise ValueError(f"Ability '{ability_code}' has no seeds")
    return [
        {**base, "seed_id": f"{ability_code}_seed_{i}", "text": text, "seed_type": "inventory_seed"}
        for i, text in enumerate(inv_seeds)
    ]


# ── sparse scoring ────────────────────────────────────────────────────────────

def sparse_dot(query_weights: dict, passage_weights: dict) -> float:
    """Dot product of two sparse lexical weight dicts (shared keys only)."""
    score = 0.0
    # iterate the smaller dict
    if len(query_weights) > len(passage_weights):
        query_weights, passage_weights = passage_weights, query_weights
    for tok, w in query_weights.items():
        if tok in passage_weights:
            score += w * passage_weights[tok]
    return float(score)


# ── query debiasing (anisotropy / common-mode removal) ─────────────────────────

def fit_debias_model(seed_dense: np.ndarray, remove_top_pcs: int) -> dict:
    """Fit a debiasing transform on the pooled seed embeddings of *all* abilities.

    The structure shared by every ability's seeds — the "person-narrative" she/he
    framing — is high-variance and common-mode, so it surfaces as the embedding
    mean plus the top few principal components. We capture both so they can be
    projected out of each query, leaving the lower-variance, ability-specific
    signal to drive retrieval.

    Returns {"mean": (d,), "components": (k, d) orthonormal rows or None}.
    Reference: Mu & Viswanath (2018), "All-but-the-Top: Simple and Effective
    Postprocessing for Word Representations", ICLR. https://arxiv.org/abs/1702.01417
    """
    mean = seed_dense.mean(axis=0).astype(np.float32)
    components = None
    if remove_top_pcs > 0:
        centered = seed_dense - mean
        # SVD of the centered sample matrix: rows of Vt are the principal axes,
        # ordered by descending singular value (=> descending variance). The
        # right singular vectors equal the eigenvectors of the covariance matrix.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        components = np.ascontiguousarray(vt[:remove_top_pcs]).astype(np.float32)
    return {"mean": mean, "components": components}


def apply_debias(vecs: np.ndarray, model: dict, subtract_mean: bool) -> np.ndarray:
    """Debias query vectors and re-normalize to unit length (for the IP index).

    The FAISS index stores raw L2-normalized passage vectors and scores by dot
    product, which is bilinear: zeroing a direction in the *query* nulls that
    direction's contribution to every score. So debiasing the query alone is
    sufficient to remove the common-mode axes from the ranking — no reindexing.
    """
    out = vecs.astype(np.float32).copy()
    if subtract_mean:
        out = out - model["mean"]
    comps = model["components"]
    if comps is not None:
        # subtract the projection onto each retained top principal component
        out = out - (out @ comps.T) @ comps
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (out / norms).astype(np.float32)


# ── index loading ─────────────────────────────────────────────────────────────

def load_available_indices(index_dir: pathlib.Path, debug: bool = False) -> list[dict]:
    """Return a list of loaded index dicts for every complete triplet found.

    Each dict: {name, faiss_index, sparse (list[dict] by passage_idx), meta (list[dict])}
    """
    faiss_paths = sorted(index_dir.glob("*_faiss.index"))
    if debug:
        faiss_paths = faiss_paths[:1]

    indices = []
    for fp in faiss_paths:
        name = fp.stem.replace("_faiss", "")
        sparse_path = index_dir / f"{name}_sparse.jsonl"
        meta_path = index_dir / f"{name}_meta.jsonl"
        if not sparse_path.exists() or not meta_path.exists():
            print(f"[skip] {name} — missing sparse or meta file")
            continue

        print(f"[load] {name}")
        fi = faiss.read_index(str(fp))

        sparse: list[dict] = []
        for line in sparse_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                sparse.append(rec["sparse"])

        meta: list[dict] = []
        for line in meta_path.read_text().splitlines():
            if line.strip():
                meta.append(json.loads(line))

        if len(sparse) != len(meta) or fi.ntotal != len(meta):
            print(f"  [warn] {name} — size mismatch (faiss={fi.ntotal}, sparse={len(sparse)}, meta={len(meta)}); skipping")
            continue

        indices.append({"name": name, "faiss": fi, "sparse": sparse, "meta": meta})
        print(f"  → {fi.ntotal:,} passages")

    return indices


# ── retrieval ────────────────────────────────────────────────────────────────

def retrieve_for_ability(
    ability_code: str,
    seeds: list[dict],
    seed_dense: np.ndarray,
    seed_sparse: list[dict],
    indices: list[dict],
    cfg: dict,
    debug: bool = False,
) -> list[dict]:
    """Retrieve top-K hybrid-scored passages across all indices for one ability.

    Seed embeddings are supplied pre-encoded (and already debiased, if enabled)
    so the debias transform can be fit once on the full seed pool. ``seed_dense``
    is (n_seeds, 1024); ``seed_sparse`` is the matching list of weight dicts.

    Returns a list of result dicts, one per (passage, seed) hit. Duplicates
    (same passage_id from different seeds) are kept — dedup happens downstream.
    """
    top_k: int = cfg["top_k"]
    threshold: float = cfg["similarity_threshold"]
    alpha: float = cfg["alpha"]

    results: list[dict] = []
    cand_scores: list[float] = []  # all pre-threshold dense scores, for calibration

    for idx in indices:
        fi: faiss.Index = idx["faiss"]
        sparse_store: list[dict] = idx["sparse"]
        meta_store: list[dict] = idx["meta"]
        source_name: str = idx["name"]

        # Dense search: top_k candidates per seed (no threshold yet)
        scores_mat, ids_mat = fi.search(seed_dense.astype(np.float32), top_k)

        for seed_i, seed in enumerate(seeds):
            q_sparse = seed_sparse[seed_i]
            for rank, (dense_score, passage_idx) in enumerate(
                zip(scores_mat[seed_i], ids_mat[seed_i])
            ):
                if passage_idx < 0:
                    continue
                cand_scores.append(float(dense_score))
                if float(dense_score) < threshold:
                    continue

                p_sparse = sparse_store[passage_idx]
                sp_score = sparse_dot(q_sparse, p_sparse)
                hybrid = (1 - alpha) * float(dense_score) + alpha * sp_score

                meta = meta_store[passage_idx]
                passage_id = f"{source_name}_{passage_idx}"
                results.append(
                    {
                        "passage_id": passage_id,
                        "source": source_name,
                        "text": meta.get("text", ""),
                        "source_id": meta.get("source_id", ""),
                        "seed_id": seed["seed_id"],
                        "seed_type": seed.get("seed_type", ""),
                        "dense_score": round(float(dense_score), 6),
                        "sparse_score": round(float(sp_score), 6),
                        "hybrid_score": round(float(hybrid), 6),
                        "ability_code": ability_code,                # composite uid
                        "code": seed.get("code", ""),               # bare mnemonic
                        "broad_stratum": seed.get("broad_stratum", ""),
                    }
                )

        if debug:
            break  # one index in debug mode

    # ── cap the per-ability pool ────────────────────────────────────────────────
    # top_k per seed × ~10 seeds × ~54 indices => ~54k (passage, seed) hits per
    # ability, dominated by a long low-score tail that the downstream dedup +
    # top-100 selection never reaches. Sort all hits by hybrid score and keep the
    # top ``max_per_ability`` so the raw files stay small without dropping any
    # selection-relevant candidate. Absent/null => keep everything (unsorted).
    max_per_ability = cfg.get("max_per_ability")
    if max_per_ability:
        results.sort(key=lambda r: r["hybrid_score"], reverse=True)
        if len(results) > max_per_ability:
            results = results[:max_per_ability]

    # Dense-score distribution of the candidate pool, *before* the threshold.
    # Debiasing shifts the cosine scale, so this is what to recalibrate
    # `similarity_threshold` against (it is printed regardless of the cutoff).
    if cand_scores:
        arr = np.array(cand_scores)
        pct = np.percentile(arr, [50, 90, 99])
        cap_note = f" (capped at {max_per_ability:,})" if max_per_ability and arr.size > len(results) else ""
        print(f"  dense candidates: n={arr.size:,}  "
              f"min={arr.min():.3f}  p50={pct[0]:.3f}  p90={pct[1]:.3f}  "
              f"p99={pct[2]:.3f}  max={arr.max():.3f}  | kept≥{threshold}: {len(results):,}{cap_note}")

    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main(config_path: str, ability_args: list[str] | None, run_all: bool, debug: bool) -> None:
    cfg = yaml.safe_load(pathlib.Path(config_path).read_text())
    required = {
        "index_dir", "seed_dir", "inventory_path", "output_dir",
        "encoder_model", "top_k", "similarity_threshold", "alpha", "batch_size",
    }
    missing = required - cfg.keys()
    if missing:
        sys.exit(f"Config missing required fields: {missing}")

    index_dir = pathlib.Path(cfg["index_dir"])
    seed_dir = pathlib.Path(cfg["seed_dir"])
    inventory_path = pathlib.Path(cfg["inventory_path"])
    output_dir = pathlib.Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if not inventory_path.exists():
        sys.exit(f"Inventory not found: {inventory_path}")

    inventory = _load_inventory(inventory_path)

    if run_all:
        ability_codes = list(inventory.keys())
    elif ability_args:
        # Accept composite uids ("Grw-RS") or bare codes ("RS"/"FI"). A bare
        # code is resolved only if it is unambiguous across strata.
        ability_codes = []
        for ability_arg in ability_args:
            if ability_arg in inventory:
                ability_codes.append(ability_arg)
                continue
            matches = [uid for uid, ab in inventory.items() if ab["code"] == ability_arg]
            if len(matches) == 1:
                ability_codes.extend(matches)
            elif len(matches) > 1:
                sys.exit(f"Ambiguous code '{ability_arg}'; specify a uid: {sorted(matches)}")
            else:
                sys.exit(f"Unknown ability '{ability_arg}'")
    else:
        sys.exit("Specify --ability UID|CODE or --all")

    print(f"Loading indices from {index_dir} …")
    indices = load_available_indices(index_dir, debug=debug)
    if not indices:
        sys.exit(f"No complete index triplets found in {index_dir}")

    print(f"\nLoading encoder: {cfg['encoder_model']}")
    enc = BGEM3Encoder(model_name=cfg["encoder_model"], device=cfg.get("device", "cpu"))

    seed_dir.mkdir(parents=True, exist_ok=True)

    # ── encode every ability's seeds ──────────────────────────────────────────
    # The debias transform is fit on the seed pool across ALL abilities, so the
    # common she/he "person-narrative" direction can be estimated and removed —
    # independent of which abilities this particular run retrieves for.
    all_codes = list(inventory.keys())
    seeds_by_code = {c: load_seeds(c, seed_dir, inventory) for c in all_codes}
    flat_texts = [s["text"] for c in all_codes for s in seeds_by_code[c]]
    print(f"\nEncoding {len(flat_texts)} seed(s) across {len(all_codes)} abilities …")
    enc_out = enc.encode(flat_texts, batch_size=cfg["batch_size"])
    flat_dense: np.ndarray = enc_out["dense"]
    flat_sparse: list[dict] = enc_out["sparse"]

    # split the flat encodings back into per-ability blocks
    dense_by_code: dict[str, np.ndarray] = {}
    sparse_by_code: dict[str, list[dict]] = {}
    pos = 0
    for c in all_codes:
        n = len(seeds_by_code[c])
        dense_by_code[c] = flat_dense[pos:pos + n]
        sparse_by_code[c] = flat_sparse[pos:pos + n]
        pos += n

    # ── fit & apply query debiasing (anisotropy / common-mode removal) ──────────
    subtract_mean = bool(cfg.get("background_subtraction", False))
    remove_top_pcs = int(cfg.get("remove_top_pcs", 0))
    if subtract_mean or remove_top_pcs > 0:
        model = fit_debias_model(flat_dense, remove_top_pcs)
        print(f"Debiasing queries: background_subtraction={subtract_mean}, "
              f"remove_top_pcs={remove_top_pcs}")
        for c in all_codes:
            dense_by_code[c] = apply_debias(dense_by_code[c], model, subtract_mean)
    else:
        print("Query debiasing disabled (raw seed embeddings).")

    # ── retrieve per requested ability ──────────────────────────────────────────
    for ability_code in tqdm(ability_codes, desc="abilities"):
        out_path = output_dir / f"{ability_code}_raw.jsonl"
        seeds = seeds_by_code[ability_code]
        seed_source = seeds[0].get("seed_type", "")
        print(f"\n[{ability_code}] {len(seeds)} seed(s)  ({seed_source})")

        results = retrieve_for_ability(
            ability_code,
            seeds,
            dense_by_code[ability_code],
            sparse_by_code[ability_code],
            indices,
            cfg,
            debug=debug,
        )

        with out_path.open("w") as fh:
            for rec in results:
                fh.write(json.dumps(rec) + "\n")

        print(f"  → {len(results):,} hits  saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve passages for CHC narrow abilities")
    parser.add_argument("config", help="Path to YAML config file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ability", metavar="CODE", action="append",
                       help="Ability uid or code, e.g. Gf-I; repeat to run several in one pass")
    group.add_argument("--all", action="store_true", help="Run all abilities in inventory")
    parser.add_argument("--debug", action="store_true",
                        help="Use first index only; fast smoke-test")
    args = parser.parse_args()
    main(args.config, args.ability, args.all, args.debug)
