"""Held-out evaluation of the relevance classifier on the random annotation set.

Reproduces the classifier numbers the paper reports (Classifier Training): the averaged
classifier scores every random-set passage, both raters' labels are pooled, the operating
threshold is the F1-maximizing point of a 0.05-0.95 sweep, and precision / recall / F1 at
that threshold get 95% bootstrap CIs over passages.

Two steps:

1. **Score** (GPU). Each checkpoint in the filter config's `model_path` scores every
   random-set pool with `p3_filter_passages.score_ability`, using the config's `n_seeds`,
   `seed_subsets`, `seed`, `max_len` and `batch_size`, so the eval mirrors exactly how the
   classifier is applied to the corpus. Per-checkpoint probabilities are cached as
   `{scores_dir}/{parent}_{name}.jsonl` (`{uid, passage_id, prob}`, 5 decimals) and reused
   on later runs unless `--rescore` is given.
2. **Evaluate** (CPU, from the cache). The ensemble probability is the mean of the
   checkpoints' cached probabilities. A passage is positive for a rater when that rater
   labels it `folk_description` with `best_ability_code` equal to the pool's ability.
   "Pooled" scoring stacks the two raters' labels (each passage counted once per rater),
   which with soft targets equals scoring against the averaged label. Only passages both
   raters judged are used.

The random set is disjoint from the classifier's training data by construction
(`p3_build_annotation_sets.py` draws random first; `p3_build_soft_dataset.py` also drops
any training text that appears in it).

Writes `{out_dir}/report.json` and `{out_dir}/report.md`.

Usage:
  uv run python scripts/p3_eval_classifier.py configs/p3_filter.yaml              # GPU: score + evaluate
  uv run python scripts/p3_eval_classifier.py configs/p3_filter.yaml --device cpu  # slow, same numbers
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import yaml

import p3_annotate as A           # annotation-set layout + inventory
import p3_filter_passages as F    # CADClassifier, load_seeds, score_ability — identical scoring

CAD = "folk_description"
DEFAULT_ANNOTATORS = ["opus-5.5-in-conversation@v2", "llm-gpt-6-astra@v2"]


# ── step 1: per-checkpoint scores ────────────────────────────────────────────────
def scores_path(scores_dir: pathlib.Path, model_path: str) -> pathlib.Path:
    p = pathlib.Path(model_path)
    return scores_dir / f"{p.parent.name}_{p.name}.jsonl"


def score_checkpoint(model_path: str, cfg: dict, layout: A.SetLayout, inv: dict, seeds: dict,
                     out: pathlib.Path) -> None:
    clf = F.CADClassifier(model_path, cfg["device"], int(cfg["max_len"]), int(cfg["batch_size"]),
                          int(cfg.get("seed_token_budget") or 0))
    rows = []
    for uid in layout.uids():
        pool = [json.loads(l) for l in layout.pool(uid).read_text(encoding="utf-8").splitlines() if l.strip()]
        probs = F.score_ability(pool, inv["abilities"][uid]["name"], seeds[uid], clf, cfg,
                                int(cfg["seed"]), uid)
        rows += [{"uid": uid, "passage_id": p["passage_id"], "prob": round(float(q), 5)}
                 for p, q in zip(pool, probs)]
        print(f"  {uid} {len(pool)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(rows)} passages)")


# ── step 2: metrics ──────────────────────────────────────────────────────────────
def load_gold(layout: A.SetLayout, annotators: list[str]) -> tuple[list[tuple[str, str]], dict]:
    """Passage keys in pool order (judged by every annotator) and per-annotator 0/1 gold."""
    gold: dict[str, dict] = {a: {} for a in annotators}
    for a in annotators:
        for f in layout.judgments.glob(f"*/{a}.jsonl"):
            uid = f.parent.name
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    j = json.loads(l)
                    gold[a][(uid, j["passage_id"])] = int(j["label"] == CAD and j.get("best_ability_code") == uid)
    keys = []
    for uid in layout.uids():
        for l in layout.pool(uid).read_text(encoding="utf-8").splitlines():
            if l.strip():
                k = (uid, json.loads(l)["passage_id"])
                if all(k in gold[a] for a in annotators):
                    keys.append(k)
    return keys, {a: np.array([gold[a][k] for k in keys], bool) for a in annotators}


def load_probs(path: pathlib.Path, keys: list[tuple[str, str]]) -> np.ndarray:
    p = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            p[(r["uid"], r["passage_id"])] = r["prob"]
    missing = [k for k in keys if k not in p]
    if missing:
        raise SystemExit(f"{path}: {len(missing)} evaluated passages have no score (e.g. {missing[0]}); rerun with --rescore")
    return np.array([p[k] for k in keys])


def prf(g: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = int((g & pred).sum()); fp = int((~g & pred).sum()); fn = int((g & ~pred).sum())
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


def avg_precision(g: np.ndarray, s: np.ndarray) -> float:
    o = np.argsort(-s); g = g[o]
    prec = np.cumsum(g) / np.arange(1, len(g) + 1)
    return float((prec * g).sum() / g.sum())


def evaluate(prob: np.ndarray, gold: dict, grid: list[float], n_boot: int, boot_seed: int) -> dict:
    golds = list(gold.values())
    n = len(prob)

    def pooled(t: float, idx: np.ndarray | None = None) -> tuple[float, float, float]:
        idx = np.arange(n) if idx is None else idx
        return prf(np.concatenate([g[idx] for g in golds]), np.concatenate([prob[idx]] * len(golds)) >= t)

    sweep = {t: pooled(t)[2] for t in grid}
    thr = max(grid, key=lambda t: sweep[t])          # first F1 maximum in grid order
    P, R, F1 = pooled(thr)
    rng = np.random.default_rng(boot_seed)
    boots = np.array([pooled(thr, rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return {
        "n_passages": n,
        "positives": {a: int(g.sum()) for a, g in gold.items()},
        "ap_pooled": avg_precision(np.concatenate(golds), np.concatenate([prob] * len(golds))),
        "threshold": thr,
        "pooled": {"precision": P, "recall": R, "f1": F1,
                   "ci95": {"precision": [lo[0], hi[0]], "recall": [lo[1], hi[1]], "f1": [lo[2], hi[2]]}},
        "per_rater_f1": {a: prf(g, prob >= thr)[2] for a, g in gold.items()},
        "f1_by_threshold": sweep,
        "n_boot": n_boot,
        "boot_seed": boot_seed,
    }


def write_md(rep: dict, path: pathlib.Path) -> None:
    p, ci = rep["pooled"], rep["pooled"]["ci95"]
    fmt = lambda k: f"{p[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]"
    lines = [
        "# Relevance classifier: held-out evaluation (random set)",
        "",
        f"Checkpoints (averaged): {', '.join(rep['checkpoints'])}",
        f"Passages: {rep['n_passages']}; positives per rater: "
        + ", ".join(f"{a} {n}" for a, n in rep["positives"].items()),
        "",
        f"- Pooled AP: {rep['ap_pooled']:.3f}",
        f"- Operating threshold (F1-max over the sweep): {rep['threshold']}",
        f"- Precision {fmt('precision')}, recall {fmt('recall')}, F1 {fmt('f1')} "
        f"(95% CI, {rep['n_boot']} bootstrap resamples of passages, seed {rep['boot_seed']})",
        "- F1 per rater: " + ", ".join(f"{a} {f:.3f}" for a, f in rep["per_rater_f1"].items()),
        "",
        "| threshold | pooled F1 |",
        "|---|---|",
        *[f"| {t} | {f:.3f} |" for t, f in rep["f1_by_threshold"].items()],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="filter config (configs/p3_filter.yaml): checkpoints + scoring settings")
    ap.add_argument("--root", default=A.ANNOTATION_ROOT)
    ap.add_argument("--set", dest="set_name", default="random", choices=sorted(A.SETS),
                    help="evaluate on the held-out random set; ranked is the training set")
    ap.add_argument("--annotators", nargs="+", default=DEFAULT_ANNOTATORS,
                    help="judgment file stems ({annotator}@{prompt_version}) to pool")
    ap.add_argument("--scores-dir", default="results/classifier_eval/scores")
    ap.add_argument("--out-dir", default="results/classifier_eval")
    ap.add_argument("--rescore", action="store_true", help="re-score checkpoints even if cached")
    ap.add_argument("--device", help="override the config's device (e.g. cpu)")
    ap.add_argument("--grid-step", type=float, default=0.05, help="threshold sweep step over [0.05, 0.95]")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if args.device:
        cfg["device"] = args.device
    models = cfg["model_path"] if isinstance(cfg["model_path"], list) else [cfg["model_path"]]
    layout = A.SetLayout(args.root, args.set_name)
    scores_dir = pathlib.Path(args.scores_dir)

    todo = [m for m in models if args.rescore or not scores_path(scores_dir, m).exists()]
    if todo:
        inv_path = pathlib.Path(cfg["inventory_path"])
        inv = A.load_inventory(inv_path)
        seeds = F.load_seeds(inv_path, pathlib.Path(cfg["seed_dir"]) if cfg.get("seed_dir") else None)
    for m in models:
        out = scores_path(scores_dir, m)
        if m in todo:
            print(f"scoring {m} -> {out}")
            score_checkpoint(m, cfg, layout, inv, seeds, out)
        else:
            print(f"using cached scores {out}")

    keys, gold = load_gold(layout, args.annotators)
    if not keys:
        raise SystemExit(f"no passage under {layout.judgments} is judged by all of {args.annotators}; "
                         "evaluation needs both raters' judgment files")
    prob = np.mean([load_probs(scores_path(scores_dir, m), keys) for m in models], axis=0)
    steps = int(round(0.90 / args.grid_step))
    grid = [round(0.05 + i * args.grid_step, 2) for i in range(steps + 1)]
    rep = {"checkpoints": models, "annotators": args.annotators, "set": args.set_name,
           **evaluate(prob, gold, grid, args.n_boot, args.boot_seed)}

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    write_md(rep, out_dir / "report.md")
    print((out_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
