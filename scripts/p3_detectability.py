"""Stage 3 — detectability of individual differences vs. folk coverage (exploratory, RQ1).

CPU/local. Reads the frozen detectability ratings (one JSONL per rater, produced blind from
the frozen v3 rubric; see data/raw/detectability/README.md) and the per-ability coverage from
p3_annotation_stats.py, and writes out_dir/{report.json, report.md}.

For each rater pair: three-way kappa and the kappa of the INSTRUMENTED-vs-rest contrast (the
only contrast any claim rests on). On the abilities the pair agrees on, the INSTRUMENTED class
is compared with the rest on descriptor prevalence by a one-sided Mann-Whitney U test (no
cutoff); the share at zero prevalence is reported alongside with Fisher's exact test.

The rubric was developed while its author could see the coverage results, so the test is
exploratory; the report says so.

Usage:
  uv run python scripts/p3_detectability.py configs/p3_detectability.yaml
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import statistics

import yaml
from scipy.stats import fisher_exact, mannwhitneyu
from sklearn.metrics import cohen_kappa_score

POS = "INSTRUMENTED"


def load_ratings(path: pathlib.Path, key: dict[str, str]) -> dict[str, str]:
    """{uid: category}. Records carry either an opaque blind `id` or a `uid`."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["uid"] if "uid" in r else key[r["id"]]] = r["category"]
    if len(out) != len(key) or "ERROR" in out.values():
        raise SystemExit(f"{path}: expected {len(key)} valid ratings, got {collections.Counter(out.values())}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = yaml.safe_load(pathlib.Path(ap.parse_args().config).read_text())
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    key = json.loads(pathlib.Path(cfg["blind_key_path"]).read_text())
    rdir = pathlib.Path(cfg["ratings_dir"])
    ratings = {r: load_ratings(rdir / f"{r}.jsonl", key) for pair in cfg["rater_pairs"] for r in pair}
    with open(cfg["coverage_path"]) as f:
        prev = {row["uid"]: float(row["prev"]) for row in csv.DictReader(f)}
    # Abilities rated but no longer in the inventory drop out; inventory abilities added after rating are listed.
    uids = sorted(u for u in key.values() if u in prev)
    unrated = sorted(set(prev) - set(key.values()))

    by_votes = collections.Counter(u for r in ratings.values() for u, c in r.items() if c == POS and u in prev)
    rep = {"coverage_path": cfg["coverage_path"], "n_rated_in_coverage": len(uids),
           "unrated": unrated, "pairs": [],
           "instrumented_by_votes": {u: n for u, n in sorted(by_votes.items(), key=lambda x: (-x[1], x[0]))}}
    for a, b in cfg["rater_pairs"]:
        A, B = ratings[a], ratings[b]
        ca, cb = [A[u] for u in uids], [B[u] for u in uids]
        ba, bb = [c == POS for c in ca], [c == POS for c in cb]
        agreed = [u for u in uids if (A[u] == POS) == (B[u] == POS)]
        ins = [u for u in agreed if A[u] == POS]
        rest = [u for u in agreed if A[u] != POS]
        pi, pr = [prev[u] for u in ins], [prev[u] for u in rest]
        mw = mannwhitneyu(pi, pr, alternative="less")
        table = [[sum(p == 0 for p in pi), sum(p > 0 for p in pi)], [sum(p == 0 for p in pr), sum(p > 0 for p in pr)]]
        odds, fp = fisher_exact(table, alternative="greater")
        rep["pairs"].append({
            "raters": [a, b],
            "kappa_3way": float(cohen_kappa_score(ca, cb)),
            "binary_agreement": sum(x == y for x, y in zip(ba, bb)), "n": len(uids),
            "kappa_instrumented": float(cohen_kappa_score(ba, bb)),
            "instrumented": {u: prev[u] for u in ins},
            "disagreements": {u: [A[u], B[u]] for u in uids if u not in agreed},
            "n_rest": len(rest),
            "median_prev": {"instrumented": statistics.median(pi), "rest": statistics.median(pr)},
            "mean_prev": {"instrumented": statistics.mean(pi), "rest": statistics.mean(pr)},
            "mann_whitney": {"U": float(mw.statistic), "p_one_sided": float(mw.pvalue)},
            "zero_prevalence": {"table": table, "odds_ratio": float(odds), "p_one_sided": float(fp)},
        })
    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))

    L = ["# Detectability vs. folk coverage (exploratory)\n",
         f"Coverage: {cfg['coverage_path']} (column `prev`). The rubric was developed while its author could see "
         "coverage results; treat every result below as exploratory.\n",
         f"Rated abilities in coverage: {len(uids)}; unrated (added after rating): {unrated}\n",
         f"INSTRUMENTED votes (of {len(ratings)} raters): {rep['instrumented_by_votes']}\n"]
    for p in rep["pairs"]:
        a, b = p["raters"]
        ins = ", ".join(f"{u} {100 * v:.0f}%" for u, v in sorted(p["instrumented"].items(), key=lambda x: x[1]))
        L += [f"\n## {a} x {b}\n",
              f"- three-way kappa {p['kappa_3way']:.3f}; INSTRUMENTED-vs-rest agreement {p['binary_agreement']}/{p['n']}, kappa {p['kappa_instrumented']:.3f}",
              f"- INSTRUMENTED (both raters, n = {len(p['instrumented'])}): {ins}",
              f"- disagreements: {p['disagreements']}",
              f"- median prevalence {100 * p['median_prev']['instrumented']:.1f}% vs {100 * p['median_prev']['rest']:.1f}% "
              f"(rest, n = {p['n_rest']}); Mann-Whitney U = {p['mann_whitney']['U']:.0f}, one-sided p = {p['mann_whitney']['p_one_sided']:.1e}",
              f"- zero prevalence: {p['zero_prevalence']['table'][0][0]}/{len(p['instrumented'])} vs "
              f"{p['zero_prevalence']['table'][1][0]}/{p['n_rest']}; Fisher one-sided p = {p['zero_prevalence']['p_one_sided']:.2g}"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
