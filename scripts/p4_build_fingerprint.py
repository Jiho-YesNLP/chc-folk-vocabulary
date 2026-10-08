"""Stage 4 · Step B — deterministic per-ability fingerprint assembly.

Consumes the Stage-3 filtered corpus (passage text) and the Step-A unified
extraction (spans, conceptual schema, construal, register, evaluative) and
assembles the four-component ability-descriptor fingerprint + the attribute/
process register complement. No LLM — only the BGE-M3 encoder (GPU) for the
synonym-merge of spans and schema phrases. Re-runnable; threshold/weight tuning
needs no re-extraction.

Per ability (uid):
  1. lexical            — content-word over-representation vs. the pooled corpus (PMI lift)
  2. conceptual_frame   — dominant clusters of the conceptual_schema phrases,
                          ranked by freq x distinctiveness x support (same score as spans)
  3. construal_pattern  — distribution over dispositional/agentive/spontaneous/output_based
  4. evaluative_framing — rarity / tone distribution + salient social contexts
  + register            — attribute vs. process proportion (store-vs-operation complement)
  + canonical_descriptions — synonym-merged spans ranked by freq x distinctiveness x support

Only ``on_target`` extraction records feed the fingerprint (off-target = classifier
false positives). Outputs to ``output_dir``:
  {uid}.json                 per-ability fingerprint
  {uid}_canonical_review.json full ranked canonical clusters (human label review)
  fingerprints.json          all abilities combined
  fingerprint_report.md      human-readable summary + coverage tiers + register crosstab
  fingerprint_vectors.npz    BGE-M3 vectors for clustering/viz (no extra GPU pass):
                             ability prototypes (fp_*) + per-span members (member_*).
                             Full-build only; skipped on --ability runs.

Needs a CUDA GPU (BGE-M3). Usage:
  uv run python scripts/p4_build_fingerprint.py configs/p4_fingerprint.yaml [--ability Gr-FA] [--overwrite]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import p3_annotate as A  # noqa: E402  inventory loader
from src.data import lexical as LX  # noqa: E402
from src.data import cluster as CL  # noqa: E402

REQUIRED_KEYS = ("output_dir", "corpus_dir", "descriptions_dir", "inventory_path", "coverage_path",
                 "encoder_model", "device", "merge_threshold", "schema_merge_threshold",
                 "lexical_top_k", "lexical_min_count", "canonical_top_k", "batch_size")

# CHC store-vs-operation grouping for the attribute/process complement
STORE_STRATA = {"Gc", "Gkn", "Gl"}
OPERATION_STRATA = {"Gf", "Gs", "Gps"}


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def load_coverage(path: pathlib.Path) -> dict[str, dict]:
    """{uid: {"tier", "prev"}} from the random-sample coverage table (p3_annotation_stats.py),
    the single source of coverage tiers. The fingerprint does not define its own tier: the
    extraction cap (100 passages) and backfill floor make any count-based tier incomparable."""
    with path.open() as f:
        return {r["uid"]: {"tier": r["tier"], "prev": float(r["prev"])} for r in csv.DictReader(f)}


def coverage_fields(uid: str, cov: dict, corpus_report: dict, n_on: int, n_extracted: int) -> dict:
    """Coverage fields of a fingerprint record. `coverage_tier`/`prevalence` come from the
    random-sample table, so they equal the manuscript's numbers by construction;
    `corpus_prevalence_est` (share of the top-N pool the classifier retains x share of the
    extracted passages judged on-target) is the corpus-side estimate on the same scale."""
    c = cov.get(uid, {})
    rep = corpus_report.get(uid, {})
    rate = n_on / n_extracted if n_extracted else 0.0
    pool = rep.get("pool_size") or 0
    est = (rep.get("n_retained", 0) / pool) * rate if pool else None
    return {"coverage_tier": c.get("tier"), "prevalence": c.get("prev"),
            "corpus_prevalence_est": round(est, 4) if est is not None else None}


def proportions(counter: Counter) -> dict:
    total = sum(counter.values()) or 1
    return {k: round(v / total, 4) for k, v in counter.most_common()}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 Step B — fingerprint assembly")
    ap.add_argument("config")
    ap.add_argument("--ability", default=None, help="single uid (e.g. Gr-FA)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--update-coverage-only", action="store_true",
                    help="rewrite only the coverage fields of existing fingerprints (no encoder, "
                         "no reclustering), e.g. after the coverage table is regenerated")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")

    out_dir = pathlib.Path(cfg["output_dir"])
    corpus_dir = pathlib.Path(cfg["corpus_dir"])
    desc_dir = pathlib.Path(cfg["descriptions_dir"])
    if (out_dir.exists() and any(out_dir.glob("*.json")) and not args.overwrite and not args.ability
            and not args.update_coverage_only):
        sys.exit(f"{out_dir} already populated; pass --overwrite.")
    out_dir.mkdir(parents=True, exist_ok=True)

    cov = load_coverage(pathlib.Path(cfg["coverage_path"]))
    rep_path = corpus_dir / "coverage_report.json"
    corpus_report = json.loads(rep_path.read_text()) if rep_path.exists() else {}

    if args.update_coverage_only:
        fps = json.loads((out_dir / "fingerprints.json").read_text())
        for uid, f in fps.items():
            f.update(coverage_fields(uid, cov, corpus_report, f["n_on_target"], f["n_extracted"]))
            (out_dir / f"{uid}.json").write_text(json.dumps(f, indent=2))
        (out_dir / "fingerprints.json").write_text(json.dumps(fps, indent=2))
        write_report(out_dir, fps)
        print(f"updated coverage fields of {len(fps)} fingerprints -> {out_dir}")
        return

    inv = A.load_inventory(pathlib.Path(cfg["inventory_path"]))

    if args.ability:
        uids = [args.ability]
    else:
        uids = sorted(p.name[:-len("_extracted.jsonl")] for p in desc_dir.glob("*_extracted.jsonl"))

    # ── pass 1: load on-target records + passage text per ability ────────────────
    ab_records: dict[str, list[dict]] = {}      # uid -> on_target extraction records (with text)
    ab_tokens: dict[str, Counter] = {}          # uid -> content-token counts (lexical)
    ab_token_set: dict[str, set] = {}           # uid -> content-token set (IDF)
    ab_n_extracted: dict[str, int] = {}         # uid -> total extracted (on_target_rate denominator)
    ab_n_backfill: dict[str, int] = {}          # uid -> of those, how many were sub-threshold backfill
    background = Counter()
    cap_counts: Counter = Counter()             # proper-noun filter: capitalized occurrences
    low_counts: Counter = Counter()             # ... and lowercase ones (pooled across abilities)
    for uid in uids:
        if uid not in inv["abilities"]:
            print(f"[skip] unknown uid: {uid}")
            continue
        ext_path = desc_dir / f"{uid}_extracted.jsonl"
        filt_path = corpus_dir / f"{uid}_filtered.jsonl"
        scored_path = corpus_dir / f"{uid}_scored.jsonl"
        if not ext_path.exists():
            print(f"[skip] no extraction: {ext_path}")
            continue
        # scored.jsonl is the superset (every candidate, retained or not) so it also
        # covers p4_extract_descriptions.py's backfill_subthreshold passages, which
        # filtered.jsonl alone does not contain.
        text_by_id = {r["passage_id"]: r["text"] for r in read_jsonl(scored_path)} if scored_path.exists() \
            else ({r["passage_id"]: r["text"] for r in read_jsonl(filt_path)} if filt_path.exists() else {})
        extracted = read_jsonl(ext_path)
        ab_n_extracted[uid] = len(extracted)
        ab_n_backfill[uid] = sum(1 for r in extracted if r.get("extraction_source") == "backfill_subthreshold")
        recs = [r for r in extracted if r.get("on_target")]
        for r in recs:
            r["text"] = text_by_id.get(r["passage_id"], "")
        ab_records[uid] = recs
        toks = Counter()
        for r in recs:
            toks.update(LX.tokens(r["text"]))
            # casing evidence comes from the raw text, before tokens() lowercases it.
            # Pooled across abilities on purpose: a name is a name corpus-wide, and one
            # ability's slice is far too small to judge a lemma on its own.
            LX.case_counts(r["text"], cap_counts, low_counts)
        ab_tokens[uid] = toks
        ab_token_set[uid] = set(toks)
        background.update(toks)
    idf = LX.ability_idf(ab_token_set)

    # ── proper-noun filter for the lexical component ─────────────────────────────
    # Personal/platform names ("sharleen", "reddit") are corpus artifacts, not folk
    # vocabulary for an ability. Scoped to the lexical component only: spans and
    # conceptual schemas are annotator-written and already free of them, and excluding
    # terms from the IDF would silently reweight canonical-description salience.
    if bool(cfg.get("proper_noun_filter", True)):
        excluded = LX.proper_nouns(cap_counts, low_counts,
                                   min_obs=int(cfg.get("proper_noun_min_obs", 4)),
                                   cap_ratio=float(cfg.get("proper_noun_cap_ratio", 0.85)))
    else:
        excluded = set()
    if excluded:
        shown = ", ".join(sorted(excluded)[:20])
        print(f"[proper-noun filter] excluding {len(excluded)} lemma(s) from lexical markers: "
              f"{shown}{' ...' if len(excluded) > 20 else ''}")

    # ── embed all unique spans + schema phrases once (BGE-M3, GPU) ────────────────
    span_strings: set[str] = set()
    schema_strings: set[str] = set()
    for recs in ab_records.values():
        for r in recs:
            for s in r.get("spans", []):
                if _norm(s["text"]):
                    span_strings.add(_norm(s["text"]))
            if r.get("conceptual_schema"):
                schema_strings.add(_norm(r["conceptual_schema"]))
    to_embed = sorted(span_strings | schema_strings)
    vec_of: dict[str, np.ndarray] = {}
    if to_embed:
        from src.models.encoder import BGEM3Encoder
        enc = BGEM3Encoder(model_name=cfg["encoder_model"], device=cfg["device"])
        dense = enc.encode(to_embed, batch_size=int(cfg["batch_size"]))["dense"]
        vec_of = {s: dense[i] for i, s in enumerate(to_embed)}
        print(f"embedded {len(to_embed)} unique strings ({len(span_strings)} spans, {len(schema_strings)} schemas)")

    # ── per-ability assembly ─────────────────────────────────────────────────────
    fingerprints: dict[str, dict] = {}
    # embedding artifact accumulators (reuse the already-loaded vec_of; no extra GPU pass)
    fp_rows: list[np.ndarray] = []          # (A, dim) salience-weighted ability prototypes
    fp_meta: list[tuple[str, str]] = []     # (uid, broad_stratum) per prototype row
    mem_rows: list[np.ndarray] = []         # (M, dim) per-span member vectors (dedup within ability)
    mem_meta: list[tuple] = []              # (uid, broad_stratum, norm_text, count, mean_conf) per member
    for uid in uids:
        if uid not in ab_records:
            continue
        ab = inv["abilities"][uid]
        recs = ab_records[uid]
        n_on = len(recs)
        # on_target_rate is over what was actually SENT to extraction (retained +
        # backfill_subthreshold), not n_retained -- with backfill, those can differ a lot
        # (e.g. 3 retained but 100 extracted), and n_retained alone would make the rate
        # nonsensical (e.g. > 1).
        n_extracted = ab_n_extracted.get(uid, n_on)
        n_backfill = ab_n_backfill.get(uid, 0)

        # 1) lexical
        lexical = LX.lexical_markers(ab_tokens[uid], background,
                                     min_count=int(cfg["lexical_min_count"]),
                                     top_k=int(cfg["lexical_top_k"]),
                                     exclude=excluded)

        # canonical descriptions (synonym-merged spans)
        span_freq: Counter = Counter()
        span_support: dict[str, list[float]] = defaultdict(list)
        span_raw: dict[str, Counter] = defaultdict(Counter)
        for r in recs:
            conf = r.get("classifier_confidence") or 0.0
            for s in r.get("spans", []):
                key = _norm(s["text"])
                if not key:
                    continue
                span_freq[key] += 1
                span_support[key].append(float(conf))
                span_raw[key][s["text"].strip()] += 1
        canonical = build_clusters(
            span_freq, span_support, span_raw, vec_of, idf,
            float(cfg["merge_threshold"]), total=sum(span_freq.values()),
        )

        # 2) conceptual frame (clusters of schema phrases)
        schema_freq: Counter = Counter()
        schema_support: dict[str, list[float]] = defaultdict(list)
        schema_raw: dict[str, Counter] = defaultdict(Counter)
        for r in recs:
            sc = r.get("conceptual_schema")
            if sc:
                k = _norm(sc)
                schema_freq[k] += 1
                schema_support[k].append(float(r.get("classifier_confidence") or 0.0))
                schema_raw[k][sc.strip()] += 1
        frames = build_clusters(schema_freq, schema_support, schema_raw, vec_of, idf,
                                float(cfg["schema_merge_threshold"]),
                                total=sum(schema_freq.values()))

        # 3) construal pattern, 4) evaluative, register
        con = proportions(Counter(r["construal"] for r in recs))
        rarity = proportions(Counter(r["evaluative"]["rarity"] for r in recs))
        tone = proportions(Counter(r["evaluative"]["tone"] for r in recs))
        social = Counter(r["evaluative"]["social_context"] for r in recs if r["evaluative"].get("social_context"))
        register = proportions(Counter(r["register"] for r in recs))

        filt_path = corpus_dir / f"{uid}_filtered.jsonl"
        n_filt = len(read_jsonl(filt_path)) if filt_path.exists() else None  # n_retained, informational

        fingerprints[uid] = {
            "uid": uid,
            "name": ab["name"],
            "broad_stratum": ab["broad_stratum"],
            "broad_name": ab["broad_name"],
            "n_filtered": n_filt,
            "n_extracted": n_extracted,
            "n_backfill_subthreshold": n_backfill,
            "n_on_target": n_on,
            "on_target_rate": round(n_on / n_extracted, 4) if n_extracted else None,
            **coverage_fields(uid, cov, corpus_report, n_on, n_extracted),
            "lexical": lexical,
            "conceptual_frame": frames[: int(cfg["canonical_top_k"])],
            "construal_pattern": con,
            "evaluative_framing": {"rarity": rarity, "tone": tone,
                                   "social_contexts": [t for t, _ in social.most_common(8)]},
            "register": register,
            "canonical_descriptions": canonical[: int(cfg["canonical_top_k"])],
            "prompt_version": recs[0].get("prompt_version") if recs else None,
        }
        # embedding artifact: ability prototype (from the stored canonical clusters) +
        # per-span member vectors with dedup count and mean classifier support.
        fp_vec = fingerprint_vector(canonical[: int(cfg["canonical_top_k"])], vec_of)
        if fp_vec is not None:
            fp_rows.append(fp_vec)
            fp_meta.append((uid, ab["broad_stratum"]))
        for key, cnt in span_freq.items():
            v = vec_of.get(key)
            if v is None:
                continue
            sup = span_support.get(key) or []
            mem_rows.append(v)
            mem_meta.append((uid, ab["broad_stratum"], key, int(cnt),
                             round(float(np.mean(sup)) if sup else 0.0, 6)))

        # per-ability fingerprint + full ranked clusters for manual label review
        (out_dir / f"{uid}.json").write_text(json.dumps(fingerprints[uid], indent=2))
        (out_dir / f"{uid}_canonical_review.json").write_text(json.dumps(canonical, indent=2))
        print(f"[{uid}] {ab['name']}: on_target={n_on} canon={len(canonical)} "
              f"frames={len(frames)} tier={fingerprints[uid]['coverage_tier']}")

    # ── combined outputs + report ────────────────────────────────────────────────
    # Audit trail: what the proper-noun filter removed, written next to the fingerprints
    # so a reviewer can check it without re-running the build.
    (out_dir / "excluded_proper_nouns.json").write_text(json.dumps({
        "enabled": bool(cfg.get("proper_noun_filter", True)),
        "min_obs": int(cfg.get("proper_noun_min_obs", 4)),
        "cap_ratio": float(cfg.get("proper_noun_cap_ratio", 0.85)),
        "n_excluded": len(excluded),
        "excluded": sorted(excluded),
    }, indent=2))
    (out_dir / "fingerprints.json").write_text(json.dumps(fingerprints, indent=2))
    write_report(out_dir, fingerprints)
    print(f"\n{len(fingerprints)} fingerprints -> {out_dir}")

    # ── embedding artifact (single matrix file; downstream clustering/viz, no GPU) ──
    # Skipped on single-ability runs so a `--ability` rerun cannot clobber the full matrix.
    if args.ability:
        print("[vectors] single-ability run — skipped fingerprint_vectors.npz "
              "(re-run full build to refresh the matrix)")
    elif fp_rows or mem_rows:
        dim = (fp_rows or mem_rows)[0].shape[0]
        out_path = out_dir / "fingerprint_vectors.npz"
        np.savez_compressed(
            out_path,
            dim=np.int64(dim),
            encoder_model=str(cfg["encoder_model"]),
            # ability prototypes — salience-weighted mean of canonical-description clusters, L2-normalized
            fp_uid=np.array([m[0] for m in fp_meta]),
            fp_broad=np.array([m[1] for m in fp_meta]),
            fp_matrix=np.stack(fp_rows).astype(np.float32) if fp_rows else np.empty((0, dim), np.float32),
            # member span vectors (BGE-M3 dense, already L2-normalized), deduped within ability
            member_uid=np.array([m[0] for m in mem_meta]),
            member_broad=np.array([m[1] for m in mem_meta]),
            member_text=np.array([m[2] for m in mem_meta]),
            member_count=np.array([m[3] for m in mem_meta], dtype=np.int64),
            member_conf=np.array([m[4] for m in mem_meta], dtype=np.float32),
            member_matrix=np.stack(mem_rows).astype(np.float32) if mem_rows else np.empty((0, dim), np.float32),
        )
        print(f"[vectors] {len(fp_rows)} prototypes + {len(mem_rows)} member spans "
              f"(dim={dim}) -> {out_path}")
    else:
        print("[vectors] no embeddings available (vec_of empty) — skipped fingerprint_vectors.npz")


def build_clusters(freq, support, raw_map, vec_of, idf, threshold, total):
    """Synonym-merge keyed strings into ranked clusters with freq x distinctiveness x support."""
    keys = [k for k in freq if k in vec_of]
    if not keys:
        return []
    vectors = np.stack([vec_of[k] for k in keys])
    priority = np.array([freq[k] for k in keys], dtype=float)
    labels, reps = CL.cluster_by_cosine(vectors, priority, threshold)
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(labels):
        clusters[c].append(i)
    out = []
    for cid, members in clusters.items():
        m_keys = sorted(members, key=lambda i: -freq[keys[i]])
        cfreq = sum(freq[keys[i]] for i in members)
        # canonical label = most frequent raw surface form across the cluster
        raw = Counter()
        for i in members:
            raw.update(raw_map[keys[i]])
        label = raw.most_common(1)[0][0] if raw else keys[m_keys[0]]
        content = LX.tokens(label)
        distinct = float(np.mean([idf.get(t, 1.0) for t in content])) if content else 1.0
        sup_vals = [v for i in members for v in support.get(keys[i], [])]
        sup = float(np.mean(sup_vals)) if sup_vals else 0.0
        freq_norm = cfreq / total if total else 0.0
        salience = freq_norm * distinct * sup
        out.append({
            "canonical": label,
            "freq": cfreq,
            "share": round(freq_norm, 4),
            "distinctiveness": round(distinct, 4),
            "support": round(sup, 4),
            "salience": round(salience, 6),
            "members": [keys[i] for i in m_keys][:12],
        })
    out.sort(key=lambda d: -d["salience"])
    return out


def fingerprint_vector(canonical_clusters: list[dict], vec_of: dict) -> np.ndarray | None:
    """Assemble one ability prototype: salience-weighted mean of its canonical-description
    clusters, then L2-normalize. Each cluster contributes the mean of its member-span
    vectors (those present in ``vec_of``). Returns None if no member has an embedding."""
    rows, weights = [], []
    for c in canonical_clusters:
        mem = [vec_of[m] for m in c.get("members", []) if m in vec_of]
        if not mem:
            continue
        rows.append(np.mean(np.stack(mem), axis=0))
        weights.append(float(c.get("salience", 0.0)))
    if not rows:
        return None
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        w = np.ones_like(w)
    v = (np.stack(rows) * w[:, None]).sum(axis=0)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n else None


def write_report(out_dir: pathlib.Path, fps: dict) -> None:
    lines = ["# Stage 4 — Ability-Descriptor Fingerprints (Track A)\n"]
    tiers = Counter(f["coverage_tier"] for f in fps.values())
    lines.append(f"Abilities: {len(fps)}  |  high={tiers['high']} sparse={tiers['sparse']} none={tiers['none']}\n")

    # register store-vs-operation complement
    grp = {"store": [], "operation": [], "other": []}
    for f in fps.values():
        if f["n_on_target"] == 0:
            continue
        proc = f["register"].get("process", 0.0)
        g = "store" if f["broad_stratum"] in STORE_STRATA else \
            "operation" if f["broad_stratum"] in OPERATION_STRATA else "other"
        grp[g].append(proc)
    lines.append("\n## Attribute/process register (mean process-share by CHC group)\n")
    for g, vals in grp.items():
        if vals:
            lines.append(f"- **{g}**: {np.mean(vals):.2f} process  (n={len(vals)})")
    lines.append("")

    lines.append("\n## Per-ability summary\n")
    lines.append("| uid | ability | tier | n | backfill% | register | top lexical | top schema |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for uid, f in sorted(fps.items()):
        reg = "/".join(f"{k[:4]}{int(v*100)}" for k, v in f["register"].items()) or "—"
        lex = ", ".join(d["term"] for d in f["lexical"][:5]) or "—"
        sch = f["conceptual_frame"][0]["canonical"] if f["conceptual_frame"] else "—"
        n_ext = f.get("n_extracted") or 0
        bf_pct = f"{100 * f.get('n_backfill_subthreshold', 0) / n_ext:.0f}%" if n_ext else "—"
        lines.append(f"| {uid} | {f['name']} | {f['coverage_tier']} | {f['n_on_target']} | {bf_pct} | {reg} | {lex} | {sch} |")
    (out_dir / "fingerprint_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
