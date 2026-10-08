"""Stage 6 — manuscript tables from a validation run: pools, support split, error analysis.

CPU/local, no encoder. Reads the raw score matrices p6_validation_pilot.py saves
(out_dir/scores.npz) with the frozen materials and writes out_dir/report.json + report.md:

  pools      open (every support-pruned ability), within-stratum (abilities sharing the
             primary target's stratum), WJ IV-restricted (abilities any kept test lists as a
             target, within the open pool; answer-informed, diagnostic only). acc@1, acc@5,
             MRR, broad@1 (open only), each with its chance rate averaged over tests.
  support    open-pool acc@1 and MRR split by the primary target's descriptor-span count
             (>= cap: full member set; below cap), plus the unscorable tests (< min_support).
  errors     where each condition's top-1 misses land: a listed secondary target, the same
             stratum, or elsewhere.
  zero-prevalence check: tests whose primary target is a zero-prevalence ability (RQ1), i.e.
             tier "none" in coverage_path, the single source of coverage tiers.

Usage:
  uv run python scripts/p6_validation_report.py configs/p6_validation_wj4.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import numpy as np
import yaml

CONDITIONS = ("A_label", "B_definition", "C_seeds", "D_spans", "E_profile")


def zero_prevalence(coverage_path: str) -> set[str]:
    with open(coverage_path) as f:
        return {row["uid"] for row in csv.DictReader(f) if row["tier"] == "none"}


def pool_metrics(S, tests, pool, cands_for):
    ranks, acc1_ch, acc5_ch, mrr_ch = [], [], [], []
    for t, row in zip(tests, S):
        cand = cands_for(t)
        idx = [pool.index(a) for a in cand]
        tgt = pool.index(t["targets"][0])
        r = int((row[idx] > row[tgt]).sum()) + 1
        ranks.append(r)
        n = len(idx)
        acc1_ch.append(1 / n); acc5_ch.append(min(5, n) / n); mrr_ch.append(np.mean([1 / i for i in range(1, n + 1)]))
    ranks = np.array(ranks)
    return ({"acc1": float((ranks == 1).mean()), "acc5": float((ranks <= 5).mean()), "mrr": float((1 / ranks).mean())},
            {"acc1": float(np.mean(acc1_ch)), "acc5": float(np.mean(acc5_ch)), "mrr": float(np.mean(mrr_ch))}, ranks)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = yaml.safe_load(pathlib.Path(ap.parse_args().config).read_text())
    out_dir = pathlib.Path(cfg["out_dir"])
    z = np.load(out_dir / "scores.npz")
    pool = [str(u) for u in z["pool"]]
    support = dict(zip(pool, [int(s) for s in z["support"]]))
    mats = json.loads(pathlib.Path(cfg["materials_path"]).read_text())
    by_id = {t["id"]: t for t in mats["tests"]}
    tests = [by_id[str(i)] for i in z["test_ids"]]
    stratum = lambda u: u.split("-", 1)[0]
    wj_abilities = sorted({u for t in mats["tests"] for u in t["targets"] if u in pool})
    cap = int(cfg["cap_per_ability"])
    zero = zero_prevalence(cfg["coverage_path"])

    pools = {"open": lambda t: pool,
             "within_stratum": lambda t: [a for a in pool if stratum(a) == stratum(t["targets"][0])],
             "wj4_restricted": lambda t: wj_abilities}
    rep = {"n_scorable": len(tests), "pool_size": len(pool), "wj4_restricted_size": len(wj_abilities),
           "pools": {}, "chance": {}, "support": {}, "errors": {}, "zero_prevalence": {}}
    for pname, cf in pools.items():
        rep["pools"][pname] = {}
        for c in CONDITIONS:
            m, ch, ranks = pool_metrics(z[c], tests, pool, cf)
            if pname == "open":
                top = [pool[int(np.argmax(row))] for row in z[c]]
                m["broad1"] = float(np.mean([stratum(a) == stratum(t["targets"][0]) for a, t in zip(top, tests)]))
            rep["pools"][pname][c] = m
        rep["chance"][pname] = ch
    rep["chance"]["open"]["broad1"] = float(np.mean([sum(stratum(a) == stratum(t["targets"][0]) for a in pool) / len(pool) for t in tests]))

    full = [i for i, t in enumerate(tests) if support[t["targets"][0]] >= cap]
    part = [i for i, t in enumerate(tests) if support[t["targets"][0]] < cap]
    for c in CONDITIONS:
        _, _, ranks = pool_metrics(z[c], tests, pool, pools["open"])
        rep["support"][c] = {f">={cap}": {"n": len(full), "acc1": float((ranks[full] == 1).mean()), "mrr": float((1 / ranks[full]).mean())},
                             f"<{cap}": {"n": len(part), "acc1": float((ranks[part] == 1).mean()), "mrr": float((1 / ranks[part]).mean())}}
        top = [pool[int(np.argmax(row))] for row in z[c]]
        miss = [(a, t) for a, t in zip(top, tests) if a != t["targets"][0]]
        rep["errors"][c] = {"misses": len(miss),
                            "secondary_target": sum(a in t["targets"][1:] for a, t in miss),
                            "same_stratum_other": sum(a not in t["targets"] and stratum(a) == stratum(t["targets"][0]) for a, t in miss),
                            "other_stratum": sum(a not in t["targets"] and stratum(a) != stratum(t["targets"][0]) for a, t in miss),
                            "detail": [{"id": t["id"], "target": t["targets"][0], "top": a} for a, t in miss]}
    rep["support"]["unscorable"] = [{"id": t["id"], "target": t["targets"][0]} for t in mats["tests"] if t["targets"][0] not in pool]
    for t in mats["tests"]:
        if t["targets"][0] in zero:
            entry = {"target": t["targets"][0], "scorable": t["targets"][0] in pool}
            if entry["scorable"]:
                i = [x["id"] for x in tests].index(t["id"])
                for c in CONDITIONS:
                    _, _, ranks = pool_metrics(z[c][i:i + 1], [t], pool, pools["open"])
                    entry[c] = int(ranks[0])
            rep["zero_prevalence"][t["id"]] = entry

    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    L = [f"# Stage 6 report — {cfg['materials_path']}\n", f"scorable {len(tests)}, open pool {len(pool)}, WJ IV-restricted pool {len(wj_abilities)}\n"]
    for pname in pools:
        L += [f"\n## {pname}\n", "| cond | acc@1 | acc@5 | MRR | broad@1 |", "|---|---|---|---|---|"]
        for c in CONDITIONS:
            m = rep["pools"][pname][c]
            L.append(f"| {c} | {m['acc1']:.3f} | {m['acc5']:.3f} | {m['mrr']:.3f} | {m.get('broad1', float('nan')):.3f} |")
        ch = rep["chance"][pname]
        L.append(f"| chance | {ch['acc1']:.3f} | {ch['acc5']:.3f} | {ch['mrr']:.3f} | {ch.get('broad1', float('nan')):.3f} |")
    L += ["\n## support split (open pool)\n"] + [f"- {c}: " + json.dumps(rep["support"][c]) for c in CONDITIONS]
    L += ["\n## errors (top-1 misses)\n"] + [f"- {c}: " + json.dumps({k: v for k, v in rep["errors"][c].items() if k != 'detail'}) for c in CONDITIONS]
    L += ["\n## zero-prevalence targets\n", json.dumps(rep["zero_prevalence"])]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
