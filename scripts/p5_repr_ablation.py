"""Stage 5 — representation ablation: span vs. source vs. span[SEP]source.

Answers a question §3.G would otherwise have to assert: the span is a substring of
its source sentence ~96% of the time, so does concatenating them buy anything over
encoding either alone? Three conditions, identical metrics, same encoder:

    span          the distilled descriptor phrase only
    source        the one-sentence passage only
    span_source   "<span> [SEP] <source>"  -- what p5_embed_joint.py actually emits

One design: every on-target (span, source) pair, the same points in all three
conditions, so differences are tested by a paired bootstrap over passages. A point's
neighbors exclude itself and every other span from the same passage -- co-occurring spans
share their source sentence and would otherwise match each other trivially under `source`
and `span_source`. Results are also split by span count: in a passage that yields several
spans each span is a fragment, and that is where conditioning on the source should matter.

Reported raw and under a per-ability cap repeated over several seeds (mean +/- sd),
matching p5_cluster_viz.py. CIs are given only for the paired differences, which carry the
reliability claim; per-condition means are descriptive.

Encoding is cached to ``cache_path``; re-running with different k, seeds or bootstrap
settings needs no re-encode. Delete the cache after changing the corpus or the
extraction records.

Usage:
  uv run python scripts/p5_repr_ablation.py configs/p5_repr_ablation.yaml
  uv run python scripts/p5_repr_ablation.py <config> --no-cache
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

REQUIRED_KEYS = ("descriptions_dir", "corpus_dir", "output_dir", "cache_path",
                 "encoder_model", "device", "batch_size", "sep", "knn_k",
                 "cap_per_ability", "seeds", "bootstrap_n", "bootstrap_alpha")
CONDITIONS = ("span", "source", "span_source")


def read_jsonl(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def load_rows(desc_dir: pathlib.Path, corpus_dir: pathlib.Path, sep: str) -> list[dict]:
    """One row per (span, source) pair from on-target records, with its ability labels."""
    rows, skipped_no_source = [], 0
    for f in sorted(desc_dir.glob("*_extracted.jsonl")):
        uid = f.name[: -len("_extracted.jsonl")]
        broad = uid.split("-")[0]
        text: dict[str, str] = {}
        for cand in (corpus_dir / f"{uid}_filtered.jsonl", corpus_dir / f"{uid}_scored.jsonl"):
            for r in read_jsonl(cand):
                text.setdefault(r["passage_id"], r["text"])
        for r in read_jsonl(f):
            if not r.get("on_target"):
                continue
            src = text.get(r["passage_id"])
            if not src:
                skipped_no_source += 1
                continue
            src = " ".join(src.split())
            spans = [s["text"].strip() for s in r.get("spans", []) if s.get("text", "").strip()]
            for sp in spans:
                rows.append({"uid": uid, "broad": broad, "passage_id": r["passage_id"],
                             "span": sp, "source": src,
                             "span_source": f"{sp}{sep}{src}", "n_spans": len(spans)})
    if skipped_no_source:
        print(f"[warn] {skipped_no_source} on-target records had no source sentence; skipped")
    return rows


def encode_conditions(rows, cfg, cache_path: pathlib.Path, use_cache: bool) -> dict[str, np.ndarray]:
    """Encode every condition's full string pool once. Subsampling later indexes in."""
    if use_cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if int(z["n_rows"]) == len(rows):
            print(f"[cache] reusing {cache_path} ({len(rows)} rows)")
            return {c: z[c] for c in CONDITIONS}
        print(f"[cache] stale ({int(z['n_rows'])} rows cached, {len(rows)} now) — re-encoding")

    from src.models.encoder import BGEM3Encoder
    enc = BGEM3Encoder(model_name=cfg["encoder_model"], device=cfg["device"])
    out = {}
    for cond in CONDITIONS:
        # dedup before the encoder, expand after: source strings repeat heavily
        strings = [r[cond] for r in rows]
        uniq = sorted(set(strings))
        print(f"[encode] {cond}: {len(uniq)} unique of {len(strings)}", flush=True)
        dense = np.asarray(enc.encode(uniq, batch_size=int(cfg["batch_size"]))["dense"],
                           dtype=np.float32)
        pos = {s: i for i, s in enumerate(uniq)}
        out[cond] = dense[[pos[s] for s in strings]]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, n_rows=np.int64(len(rows)), **out)
    print(f"[cache] wrote {cache_path}")
    return out


def knn_purity_per_point(X: np.ndarray, labels: np.ndarray, pid: np.ndarray, k: int) -> np.ndarray:
    """Fraction of each point's k nearest cosine neighbors sharing its label, excluding the
    point itself and every other span from the same passage (co-occurring spans share their
    source sentence, so under `source` and `span_source` they would match trivially)."""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    sims = X @ X.T
    sims[pid[:, None] == pid[None, :]] = -np.inf
    k = min(k, X.shape[0] - 1)
    idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    return (labels[idx] == labels[:, None]).mean(axis=1)


def recovery_metrics(X: np.ndarray, uid: np.ndarray, broad: np.ndarray, seeds: int) -> dict:
    """Label-blind recovery, as in p5_cluster_viz.py: k-means at the true cluster count,
    ARI against the true abilities and strata, averaged over `seeds` restarts. X is
    L2-normalized, so Euclidean k-means ranks like cosine. (Cosine average-linkage
    agglomerative clustering chains these vectors into one giant cluster and forces
    ARI to ~0 -- do not use.)"""
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    out = {}
    for name, truth in (("ari_narrow", uid), ("ari_broad", broad)):
        nc = len(set(truth))
        out[name] = round(float(np.mean([
            adjusted_rand_score(truth, KMeans(nc, n_init=10, random_state=s).fit_predict(X))
            for s in range(seeds)])), 4)
    return out


def boot_ci(values: np.ndarray, groups: np.ndarray, n: int, alpha: float,
            rng) -> tuple[float, float, float]:
    """Percentile cluster bootstrap: resample passages (`groups`) with replacement and take
    the mean over every point in the drawn passages. `values` is a per-point paired
    difference. Spans from one passage share a source sentence, so their differences are
    not independent; resampling points would understate the variance."""
    g, inv = np.unique(groups, return_inverse=True)
    sums, cnts = np.bincount(inv, weights=values), np.bincount(inv).astype(float)
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=n).astype(float)
    stats = (w @ sums) / (w @ cnts)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def select(rows, cap: int | None, seed: int, exclude: frozenset = frozenset()) -> list[int]:
    """Row indices: every (span, source) pair, optionally capped per ability; broad strata
    with a single narrow ability dropped, as p5_cluster_viz.py does. The same indices serve all three
    conditions, so every comparison is over identical points."""
    keep = [i for i, r in enumerate(rows) if r["uid"] not in exclude]
    if cap:
        rng = np.random.default_rng(seed)
        by_uid = defaultdict(list)
        for i in keep:
            by_uid[rows[i]["uid"]].append(i)
        keep = []
        for idxs in by_uid.values():
            keep.extend(idxs if len(idxs) <= cap
                        else list(rng.choice(idxs, cap, replace=False)))
    # drop broad strata with a single narrow ability (e.g. Go), counted over ALL rows so
    # the cap cannot change which strata qualify
    abil = defaultdict(set)
    for r in rows:
        abil[r["broad"]].add(r["uid"])
    return sorted(i for i in keep if len(abil[rows[i]["broad"]]) > 1)


def evaluate(rows, vecs, idxs, cond, k, seeds):
    X = vecs[cond][idxs]
    uid = np.array([rows[i]["uid"] for i in idxs])
    broad = np.array([rows[i]["broad"] for i in idxs])
    pid = np.array([rows[i]["passage_id"] for i in idxs])
    pn = knn_purity_per_point(X, uid, pid, k)
    pb = knn_purity_per_point(X, broad, pid, k)
    m = {"n_points": len(idxs), "knn_narrow": round(float(pn.mean()), 4),
         "knn_broad": round(float(pb.mean()), 4)}
    m.update(recovery_metrics(X, uid, broad, seeds))
    return m, pn, pb


def chance_purity(rows, idxs, key) -> float:
    """Same-label neighbor rate under random labels, self excluded (as p5_cluster_viz.py)."""
    n = np.array(list(Counter(rows[i][key] for i in idxs).values()), float)
    N = n.sum()
    return round(float((n * (n - 1)).sum() / (N * (N - 1))), 4)


SUBSETS = {"all": lambda r: True,
           "single_span": lambda r: r["n_spans"] == 1,
           "multi_span": lambda r: r["n_spans"] > 1}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 — representation ablation")
    ap.add_argument("config")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")

    k = int(cfg["knn_k"])
    cap = int(cfg["cap_per_ability"])
    seeds = list(cfg["seeds"])
    km_seeds = int(cfg.get("kmeans_seeds", 3))
    exclude = frozenset(cfg.get("exclude_abilities", []))  # CHC cross-listed duplicates
    nb, alpha = int(cfg["bootstrap_n"]), float(cfg["bootstrap_alpha"])
    rng = np.random.default_rng(seeds[0])
    out_dir = pathlib.Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(pathlib.Path(cfg["descriptions_dir"]),
                     pathlib.Path(cfg["corpus_dir"]), cfg["sep"])
    if not rows:
        sys.exit("No (span, source) pairs found — check descriptions_dir / corpus_dir.")
    n_multi = sum(1 for r in rows if r["n_spans"] > 1)
    print(f"(span, source) pairs: {len(rows)}  |  from multi-span records: {n_multi} "
          f"({100 * n_multi / len(rows):.1f}%)")

    vecs = encode_conditions(rows, cfg, pathlib.Path(cfg["cache_path"]), not args.no_cache)
    report: dict = {"config": {kk: cfg[kk] for kk in
                               ("knn_k", "cap_per_ability", "seeds", "bootstrap_n",
                                "encoder_model", "sep", "corpus_dir", "descriptions_dir")},
                    "excluded_abilities": sorted(exclude),
                    "n_pairs": len(rows), "n_multi_span_pairs": n_multi}

    # ---- raw: every pair, one neighbor graph per condition over identical points ----
    idxs = select(rows, None, seeds[0], exclude)
    report["chance"] = {"narrow": chance_purity(rows, idxs, "uid"),
                        "broad": chance_purity(rows, idxs, "broad")}
    raw, pur = {}, {}
    for cond in CONDITIONS:
        m, pn, pb = evaluate(rows, vecs, idxs, cond, k, km_seeds)
        raw[cond], pur[cond] = m, {"knn_narrow": pn, "knn_broad": pb}
    report["raw"] = raw

    # ---- paired differences vs. span_source, overall and split by span count ----
    # Purity is always read off the full graph; the split only selects which points'
    # scores are averaged.
    pid = np.array([rows[i]["passage_id"] for i in idxs])
    pairs: dict = {}
    for sname, pred in SUBSETS.items():
        sel = np.array([pred(rows[i]) for i in idxs])
        block = {"n": int(sel.sum()),
                 "purity": {c: {lbl: round(float(pur[c][lbl][sel].mean()), 4)
                                for lbl in ("knn_narrow", "knn_broad")} for c in CONDITIONS}}
        for other in ("span", "source"):
            for lbl in ("knn_narrow", "knn_broad"):
                mean, lo, hi = boot_ci((pur["span_source"][lbl] - pur[other][lbl])[sel],
                                       pid[sel], nb, alpha, rng)
                block[f"span_source_minus_{other}.{lbl}"] = {
                    "mean": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "excludes_zero": bool(lo > 0 or hi < 0)}
        pairs[sname] = block
    report["paired"] = pairs

    # ---- balanced: per-ability cap, repeated over seeds ----
    per_seed = defaultdict(list)
    for s in seeds:
        sidx = select(rows, cap, s, exclude)
        for cond in CONDITIONS:
            per_seed[cond].append(evaluate(rows, vecs, sidx, cond, k, km_seeds)[0])
    report["balanced"] = {
        cond: {mk: {"mean": round(float(np.mean([m[mk] for m in ms])), 4),
                    "sd": round(float(np.std([m[mk] for m in ms])), 4)}
               for mk in ("n_points", "knn_narrow", "knn_broad", "ari_narrow", "ari_broad")}
        for cond, ms in per_seed.items()}

    (out_dir / "repr_ablation.json").write_text(json.dumps(report, indent=2))
    write_markdown(out_dir / "repr_ablation.md", report, k, cap, seeds)
    print(f"\n-> {out_dir}/repr_ablation.json  and  repr_ablation.md")


def write_markdown(path, report, k, cap, seeds) -> None:
    L = ["# Representation ablation — span vs. source vs. span[SEP]source\n",
         f"`knn_k={k}`  `cap_per_ability={cap}`  `seeds={seeds}`  "
         f"`n_pairs={report['n_pairs']}` (multi-span: {report['n_multi_span_pairs']})\n",
         "Same points in every condition; self and same-passage spans excluded as neighbors. "
         "ARI = label-blind k-means (k = true count) vs. the labels.\n",
         f"Chance kNN purity — narrow {report['chance']['narrow']}, "
         f"broad {report['chance']['broad']}\n",
         "\n## Raw (all points)\n",
         "| condition | n | kNN narrow | kNN broad | ARI narrow | ARI broad |",
         "|---|---|---|---|---|---|"]
    for c, m in report["raw"].items():
        L.append(f"| {c} | {m['n_points']} | {m['knn_narrow']} | "
                 f"{m['knn_broad']} | {m['ari_narrow']} | {m['ari_broad']} |")
    L += ["\n## Paired differences (span_source minus other), by span count\n",
          "95% percentile bootstrap CI, resampling passages; `ns` = CI includes zero.\n",
          "| subset | n | purity span / source / span_source (narrow) | vs span (narrow) | "
          "vs source (narrow) | vs span (broad) | vs source (broad) |",
          "|---|---|---|---|---|---|---|"]
    for sname, b in report["paired"].items():
        p = b["purity"]
        f = lambda key: (f"{b[key]['mean']:+.4f} [{b[key]['ci95'][0]:+.4f}, {b[key]['ci95'][1]:+.4f}]"
                         + ("" if b[key]["excludes_zero"] else " ns"))
        L.append(f"| {sname} | {b['n']} | {p['span']['knn_narrow']} / {p['source']['knn_narrow']} / "
                 f"{p['span_source']['knn_narrow']} | {f('span_source_minus_span.knn_narrow')} | "
                 f"{f('span_source_minus_source.knn_narrow')} | {f('span_source_minus_span.knn_broad')} | "
                 f"{f('span_source_minus_source.knn_broad')} |")
    L += [f"\n## Balanced (cap {cap}/ability, mean ± sd over {len(seeds)} seeds)\n",
          "| condition | n | kNN narrow | kNN broad | ARI narrow | ARI broad |",
          "|---|---|---|---|---|---|"]
    for c, m in report["balanced"].items():
        L.append(f"| {c} | {m['n_points']['mean']:.0f} | "
                 f"{m['knn_narrow']['mean']} ± {m['knn_narrow']['sd']} | "
                 f"{m['knn_broad']['mean']} ± {m['knn_broad']['sd']} | "
                 f"{m['ari_narrow']['mean']} ± {m['ari_narrow']['sd']} | "
                 f"{m['ari_broad']['mean']} ± {m['ari_broad']['sd']} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
