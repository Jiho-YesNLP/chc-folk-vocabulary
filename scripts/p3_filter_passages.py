"""Stage 3 — apply the seed-conditioned ability-descriptor classifier to the corpus.

For each CHC narrow ability, score its full pre-filtered candidate pool with the
trained cross-encoder (``models/cad_classifier``) and retain passages the model
calls a *folk_description* of that ability. The classifier is *seed-conditioned*:
each candidate is paired with an anchor built from the ability name + a sample of
that ability's descriptor seeds — the exact pairing used at training time
(see scripts/p3_train_classifier.py ``PairDataset._anchor`` and
src/data/classifier_dataset.py ``build_anchor``).

Pipeline (per ability uid, e.g. ``Gr-FA``):
  {uid}_raw.jsonl
    -> select_candidates (p3_annotate.py): max-hybrid per passage_id -> text dedup
       -> technical-label lexicon pre-filter -> rank by hybrid -> top `pool_top`
    -> pair each candidate with the ability anchor; cross-encoder -> prob_cad
       (averaged over `seed_subsets` random seed draws, all deterministic)
    -> retain where prob_cad >= `threshold`

Outputs (under `output_dir`):
  {uid}_scored.jsonl    every scored candidate + prob_cad + retained  (re-thresholdable, no GPU)
  {uid}_filtered.jsonl  retained passages only: retrieval fields + classifier_label,
                        classifier_confidence, classifier_version   (the Stage 3 deliverable)
  coverage_report.json  per-ability pool/retained/retention + source breakdown + retention tier
  filter_run_meta.json  run configuration

Requires the project GPU env (torch cu124); run on the server, not macOS.

Usage:
  uv run python scripts/p3_filter_passages.py configs/p3_filter.yaml [--overwrite] [--debug] [--ability Gr-FA]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import random

import yaml

import p3_annotate as A  # same scripts/ dir; reuse the proven selection + inventory logic
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.inventory import read_inventory  # noqa: E402  taxonomy + project annotations, merged

REQUIRED_KEYS = (
    "output_dir", "raw_dir", "inventory_path", "model_path", "classifier_version",
    "pool_top", "n_seeds", "seed_subsets", "seed", "threshold",
    "max_len", "batch_size", "device",
)

CAD_LABEL = "folk_description"


# ── seeds (mirrors src/data/classifier_dataset.py so anchors match training) ──────
def load_seeds(inventory_path: pathlib.Path, seed_dir: pathlib.Path | None) -> dict[str, list[str]]:
    """{uid: [seed, ...]} from the inventory `seeds`, overridden by external files."""
    inv = read_inventory(inventory_path)
    seeds: dict[str, list[str]] = {}
    for ba in inv["broad_abilities"]:
        for na in ba.get("narrow_abilities", []):
            uid = na.get("uid") or f"{ba['code']}-{na['code']}"
            seeds[uid] = list(na["seeds"])
    if seed_dir and seed_dir.exists():
        for uid in seeds:
            f = seed_dir / f"{uid}.jsonl"
            if not f.exists():
                continue
            texts = []
            for line in f.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                texts.append(rec["text"] if isinstance(rec, dict) else str(rec))
            if texts:
                seeds[uid] = texts
    return seeds


def build_anchor(name: str, seeds: list[str], sep: str) -> str:
    """Ability name + seeds joined by the tokenizer's own separator (backbone-agnostic)."""
    return f" {sep} ".join([name, *seeds])


def sample_seeds(pool: list[str], n: int, rng: random.Random) -> list[str]:
    return rng.sample(pool, min(n, len(pool))) if pool else []


# ── classifier ───────────────────────────────────────────────────────────────────
class CADClassifier:
    def __init__(self, model_path: str, device: str, max_len: int, batch_size: int,
                 seed_token_budget: int = 0):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(device).eval()
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed_token_budget = seed_token_budget  # 0 = truncate the longer side
        self.sep_token = self.tok.sep_token or "[SEP]"

    @property
    def cad_index(self) -> int:
        """Logit index of the descriptor class — read from the model's label2id so it
        works for both binary (cad=1) and three-way (folk_description=2) checkpoints;
        falls back to the top class (3-way) or index 1 (binary).

        Several spellings are matched because checkpoints name the class differently in
        their label2id (``folk_description``, ``folk`` or ``cad``)."""
        l2i = getattr(self.model.config, "label2id", None) or {}
        for k, v in l2i.items():
            if str(k).lower() in ("folk_description", "folk", "cad"):
                return int(v)
        n = int(getattr(self.model.config, "num_labels", 2) or 2)
        return n - 1 if n > 2 else 1

    def prob_cad(self, anchors: list[str], targets: list[str]) -> list[float]:
        """Softmax descriptor probability for each (anchor, target) pair."""
        torch = self.torch
        out: list[float] = []
        with torch.no_grad():
            for i in range(0, len(anchors), self.batch_size):
                a = anchors[i:i + self.batch_size]
                t = targets[i:i + self.batch_size]
                if self.seed_token_budget:
                    # same encoding as training: anchor capped at the budget, target cut by max_len
                    feats = [self.tok.prepare_for_model(
                                 self.tok(ai, add_special_tokens=False)["input_ids"][:self.seed_token_budget],
                                 self.tok(ti, add_special_tokens=False)["input_ids"],
                                 truncation="only_second", max_length=self.max_len)
                             for ai, ti in zip(a, t)]
                    enc = self.tok.pad(feats, return_tensors="pt").to(self.device)
                else:
                    enc = self.tok(
                        a, t, truncation=True, max_length=self.max_len,
                        padding=True, return_tensors="pt",
                    ).to(self.device)
                logits = self.model(**enc).logits
                probs = torch.softmax(logits, dim=-1)[:, self.cad_index]
                out.extend(probs.detach().cpu().tolist())
        return out


class CADEnsemble:
    """Several checkpoints scored on identical anchors; prob_cad is the mean of the members'
    descriptor probabilities. Same interface as CADClassifier, so score_ability is unchanged."""

    def __init__(self, model_paths: list[str], device: str, max_len: int, batch_size: int,
                 seed_token_budget: int = 0):
        self.members = [CADClassifier(p, device, max_len, batch_size, seed_token_budget) for p in model_paths]
        self.sep_token = self.members[0].sep_token
        self.tok = self.members[0].tok

    def prob_cad(self, anchors: list[str], targets: list[str]) -> list[float]:
        per = [m.prob_cad(anchors, targets) for m in self.members]
        return [sum(v) / len(v) for v in zip(*per)]


def load_classifier(cfg: dict):
    """`model_path` may be one checkpoint or a list of checkpoints (averaged)."""
    paths = cfg["model_path"] if isinstance(cfg["model_path"], list) else [cfg["model_path"]]
    args = (cfg["device"], int(cfg["max_len"]), int(cfg["batch_size"]), int(cfg.get("seed_token_budget") or 0))
    return CADClassifier(paths[0], *args) if len(paths) == 1 else CADEnsemble(paths, *args)


# ── per-ability scoring ───────────────────────────────────────────────────────────
def score_ability(
    pool: list[dict], ability_name: str, seed_pool: list[str], clf: CADClassifier,
    cfg: dict, base_rng_seed: int, uid: str,
) -> list[float]:
    """Mean descriptor probability per candidate, averaged over `seed_subsets` anchor draws."""
    targets = [c["text"] for c in pool]
    n_subsets = max(1, int(cfg["seed_subsets"]))
    acc = [0.0] * len(pool)
    for k in range(n_subsets):
        rng = random.Random(f"{base_rng_seed}-{uid}-{k}")
        seeds = sample_seeds(seed_pool, cfg["n_seeds"], rng)
        anchor = build_anchor(ability_name, seeds, clf.sep_token)
        probs = clf.prob_cad([anchor] * len(targets), targets)
        for i, p in enumerate(probs):
            acc[i] += p
    return [a / n_subsets for a in acc]


def retention_tier(retention_rate: float) -> str:
    """Retention tier from the *normalized* retention rate (retained / pool_size).

    This describes how much the classifier keeps, not the manuscript's coverage tier: that
    comes from the random-sample table (p3_annotation_stats.py -> coverage.csv), the single
    source of coverage tiers, which p4_build_fingerprint.py copies into each fingerprint.

    Normalizing decouples coverage from pool size: an ability whose pool is
    smaller than `pool_top` (fewer unique candidates) is judged on the fraction
    of its pool that is a descriptor, not an absolute count it could never reach. The
    cutoffs preserve the original coverage calibration — its raw 100/10 thresholds were
    implicitly "out of a ~1000 pool", i.e. 0.10 / 0.01 retention.
    """
    if retention_rate >= 0.10:
        return "high"
    if retention_rate >= 0.01:
        return "sparse"
    return "none"


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 — classifier relevance filter")
    ap.add_argument("config")
    ap.add_argument("--overwrite", action="store_true", help="allow writing into a populated output_dir")
    ap.add_argument("--debug", action="store_true", help="small slice: pool_top=20, first 3 abilities")
    ap.add_argument("--ability", default=None, help="restrict to one uid (e.g. Gr-FA)")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise SystemExit(f"Config missing required keys: {missing}")

    out_dir = pathlib.Path(cfg["output_dir"])
    raw_dir = pathlib.Path(cfg["raw_dir"])
    inventory_path = pathlib.Path(cfg["inventory_path"])
    seed_dir = pathlib.Path(cfg["seed_dir"]) if cfg.get("seed_dir") else None
    threshold = float(cfg["threshold"])

    if out_dir.exists() and any(out_dir.glob("*_filtered.jsonl")) and not args.overwrite:
        raise SystemExit(f"{out_dir} already has filtered output; pass --overwrite to replace.")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.debug:
        cfg["pool_top"] = 20

    # inventory + selection scaffolding (shared with p3_annotate.py)
    inv = A.load_inventory(inventory_path)
    label_re = A.build_label_matcher(inv["labels"])
    seeds_by_uid = load_seeds(inventory_path, seed_dir)

    # which abilities
    if args.ability:
        uids = [args.ability]
    else:
        uids = sorted(p.name[:-len("_raw.jsonl")] for p in raw_dir.glob("*_raw.jsonl"))
        if args.debug:
            uids = uids[:3]

    clf = load_classifier(cfg)
    print(f"model={cfg['model_path']}  device={cfg['device']}  threshold={threshold}  "
          f"pool_top={cfg['pool_top']}  n_seeds={cfg['n_seeds']}x{cfg['seed_subsets']}")

    coverage: dict[str, dict] = {}
    for uid in uids:
        if uid not in inv["abilities"]:
            print(f"[skip] unknown uid: {uid}")
            continue
        raw_path = raw_dir / f"{uid}_raw.jsonl"
        if not raw_path.exists():
            print(f"[skip] no raw file: {raw_path}")
            continue
        seed_pool = seeds_by_uid.get(uid, [])
        if not seed_pool:
            raise SystemExit(f"{uid}: no seeds in inventory/{seed_dir} — anchor cannot match training.")

        name = inv["abilities"][uid]["name"]
        pool, stats = A.select_candidates(raw_path, label_re, int(cfg["pool_top"]))
        if not pool:
            print(f"[{uid}] empty pool (prefiltered={stats['prefiltered_label_hits']}) — skipped")
            coverage[uid] = {"name": name, "pool_size": 0, "n_retained": 0,
                             "retention_rate": 0.0, "retention_tier": "none"}
            continue

        probs = score_ability(pool, name, seed_pool, clf, cfg, int(cfg["seed"]), uid)

        scored_path = out_dir / f"{uid}_scored.jsonl"
        filtered_path = out_dir / f"{uid}_filtered.jsonl"
        n_retained = 0
        src_counts: dict[str, int] = {}
        sum_prob_ret = 0.0
        sum_hybrid_ret = 0.0
        with scored_path.open("w") as sf, filtered_path.open("w") as ff:
            for c, p in zip(pool, probs):
                retained = p >= threshold
                p = round(float(p), 4)
                sf.write(json.dumps({**c, "prob_cad": p, "retained": retained}) + "\n")
                if retained:
                    n_retained += 1
                    src = c.get("source", "")
                    src_counts[src] = src_counts.get(src, 0) + 1
                    sum_prob_ret += p
                    sum_hybrid_ret += float(c.get("hybrid_score") or 0.0)
                    ff.write(json.dumps({
                        **c,
                        "classifier_label": CAD_LABEL,
                        "classifier_confidence": p,
                        "classifier_version": cfg["classifier_version"],
                    }) + "\n")

        n_pool = len(pool)
        retention_rate = n_retained / n_pool if n_pool else 0.0
        cov = {
            "name": name,
            "broad_stratum": inv["abilities"][uid]["broad_stratum"],
            "pool_size": n_pool,
            "n_retained": n_retained,
            "retention_rate": round(retention_rate, 4),
            "mean_prob_retained": round(sum_prob_ret / n_retained, 4) if n_retained else 0.0,
            "mean_hybrid_retained": round(sum_hybrid_ret / n_retained, 4) if n_retained else 0.0,
            "source_breakdown": dict(sorted(src_counts.items())),
            "retention_tier": retention_tier(retention_rate),
            "prefiltered_label_hits": stats["prefiltered_label_hits"],
        }
        coverage[uid] = cov
        print(f"[{uid}] pool={n_pool} retained={n_retained} "
              f"({cov['retention_rate']*100:.1f}%) retention_tier={cov['retention_tier']}")

    (out_dir / "coverage_report.json").write_text(json.dumps(coverage, indent=2))
    (out_dir / "filter_run_meta.json").write_text(json.dumps({
        "model_path": cfg["model_path"],
        "classifier_version": cfg["classifier_version"],
        "threshold": threshold,
        "pool_top": cfg["pool_top"],
        "n_seeds": cfg["n_seeds"],
        "seed_subsets": cfg["seed_subsets"],
        "seed": cfg["seed"],
        "n_abilities": len(coverage),
        "debug": args.debug,
    }, indent=2))

    tiers = {"high": 0, "sparse": 0, "none": 0}
    for c in coverage.values():
        tiers[c["retention_tier"]] += 1
    print(f"\ndone: {len(coverage)} abilities  high={tiers['high']} "
          f"sparse={tiers['sparse']} none={tiers['none']}")
    print(f"coverage -> {out_dir / 'coverage_report.json'}")


if __name__ == "__main__":
    main()
