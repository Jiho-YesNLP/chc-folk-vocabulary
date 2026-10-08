"""Stage 3 — coverage (RQ1) and inter-annotator agreement from the random annotation set.

CPU/local, no model. Reads the frozen Layer-2 judgments of the RANDOM set
(``{judgments_dir}/{uid}/{rater}.jsonl``, 50 uniform-random passages per ability) and writes
out_dir/{coverage.csv, report.json, report.md}.

Coverage. For each ability and each coverage rater: descriptor prevalence (share of the 50
passages labeled folk_description *for the target ability*), incidental and off-topic rates,
and the neighbor count (folk_description attributed via best_ability_code to a different
ability). The ability's value is the mean over the coverage raters; its tier is high at
prevalence >= tier_high, sparse above zero, none at zero (so none means no rater found a
single target descriptor). Summaries by broad stratum and by tier.

Agreement. Cohen's kappa and raw agreement between the two agreement raters on the three-way
label, overall and per broad stratum; which label pairs the disagreements fall on; each
rater's folk_description rate; Spearman rho between the raters' per-ability rates, both for
folk_description of any ability and for target prevalence. The rejected third rater is
reported by its incidental-label rate and its kappa against each agreement rater.

Usage:
  uv run python scripts/p3_annotation_stats.py configs/p3_annotation_stats.yaml
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import sys

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.inventory import read_inventory  # noqa: E402  taxonomy + project annotations, merged

LABELS = ("folk_description", "incidental_mention", "off_topic")


def load_inventory(path: pathlib.Path) -> tuple[list[str], dict[str, str], list[str]]:
    """Ability uids in inventory order, uid -> broad code, and broad codes in order."""
    inv = read_inventory(path)
    uids, broad, strata = [], {}, []
    for ba in inv["broad_abilities"]:
        strata.append(ba["code"])
        for na in ba["narrow_abilities"]:
            uid = na.get("uid") or f"{ba['code']}-{na['code']}"
            uids.append(uid)
            broad[uid] = ba["code"]
    return uids, broad, strata


def load_rater(judgments_dir: pathlib.Path, uids: list[str], rater: str) -> dict[str, dict[str, dict]]:
    """{uid: {passage_id: judgment}}; fails if any ability lacks this rater's file."""
    out = {}
    for uid in uids:
        path = judgments_dir / uid / f"{rater}.jsonl"
        if not path.exists():
            raise SystemExit(f"missing judgments: {path}")
        out[uid] = {r["passage_id"]: r for r in map(json.loads, path.read_text().splitlines()) if r}
    return out


def ability_rates(rows: dict[str, dict], uid: str) -> dict[str, float]:
    n = len(rows)
    folk = [r for r in rows.values() if r["label"] == "folk_description"]
    target = sum(r.get("best_ability_code") in (None, uid) for r in folk)
    return {"n": n,
            "prev": target / n,
            "inc": sum(r["label"] == "incidental_mention" for r in rows.values()) / n,
            "off": sum(r["label"] == "off_topic" for r in rows.values()) / n,
            "nb": len(folk) - target,
            "folk_any": len(folk) / n}


def tier(prev: float, high: float) -> str:
    return "high" if prev >= high else "sparse" if prev > 0 else "none"


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = yaml.safe_load(pathlib.Path(ap.parse_args().config).read_text())
    jdir = pathlib.Path(cfg["judgments_dir"])
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    uids, broad, strata = load_inventory(pathlib.Path(cfg["inventory_path"]))
    high = float(cfg["tier_high"])

    # ── coverage ────────────────────────────────────────────────────────────────
    cov_raters = cfg["coverage_raters"]
    per_rater = {r: load_rater(jdir, uids, r) for r in cov_raters}
    cov = []
    for uid in uids:
        rates = {r: ability_rates(per_rater[r][uid], uid) for r in cov_raters}
        row = {"uid": uid, "broad": broad[uid]}
        for k in ("prev", "inc", "off", "nb"):
            row[k] = float(np.mean([rates[r][k] for r in cov_raters]))
            for r in cov_raters:
                row[f"{k}[{r}]"] = rates[r][k]
        row["tier"] = tier(row["prev"], high)
        cov.append(row)
    with open(out_dir / "coverage.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov[0]))
        w.writeheader()
        w.writerows(cov)

    by_stratum = {}
    for s in strata:
        rs = [r for r in cov if r["broad"] == s]
        if not rs:
            continue
        t = collections.Counter(r["tier"] for r in rs)
        by_stratum[s] = {"n": len(rs), "high": t["high"], "sparse": t["sparse"], "none": t["none"],
                         **{k: float(np.mean([r[k] for r in rs])) for k in ("prev", "inc", "off")}}
    by_tier = {}
    for t in ("high", "sparse", "none"):
        rs = [r for r in cov if r["tier"] == t]
        by_tier[t] = {"n": len(rs), **{k: float(np.mean([r[k] for r in rs])) if rs else None
                                        for k in ("prev", "inc", "off")}}
    coverage = {"raters": cov_raters, "tier_high": high,
                "tiers": dict(collections.Counter(r["tier"] for r in cov)),
                "mean": {k: float(np.mean([r[k] for r in cov])) for k in ("prev", "inc", "off")},
                "none": [r["uid"] for r in cov if r["tier"] == "none"],
                "by_stratum": by_stratum, "by_tier": by_tier}

    # ── agreement ───────────────────────────────────────────────────────────────
    ra, rb = cfg["agreement_raters"]
    A = per_rater.get(ra) or load_rater(jdir, uids, ra)
    B = per_rater.get(rb) or load_rater(jdir, uids, rb)
    keys = [(u, p) for u in uids for p in A[u] if p in B[u]]
    la = [A[u][p]["label"] for u, p in keys]
    lb = [B[u][p]["label"] for u, p in keys]

    def agree(x, y):
        return {"n": len(x), "raw": float(np.mean([i == j for i, j in zip(x, y)])),
                "kappa": float(cohen_kappa_score(x, y, labels=list(LABELS)))}

    strat_agree = {}
    for s in strata:
        idx = [i for i, (u, _) in enumerate(keys) if broad[u] == s]
        if idx:
            x, y = [la[i] for i in idx], [lb[i] for i in idx]
            strat_agree[s] = {**agree(x, y), f"folk[{ra}]": x.count("folk_description") / len(x),
                              f"folk[{rb}]": y.count("folk_description") / len(y)}
    disagree = collections.Counter(" vs ".join(sorted((x, y))) for x, y in zip(la, lb) if x != y)
    rates_a = {u: ability_rates(A[u], u) for u in uids}
    rates_b = {u: ability_rates(B[u], u) for u in uids}
    rho = {}
    for k in ("folk_any", "prev"):
        s = spearmanr([rates_a[u][k] for u in uids], [rates_b[u][k] for u in uids])
        rho[k] = {"rho": float(s.statistic), "p": float(s.pvalue)}

    third = cfg.get("third_rater")
    third_rep = None
    if third:
        C = load_rater(jdir, uids, third)
        tk = [(u, p) for u in uids for p in C[u]]
        lc = [C[u][p]["label"] for u, p in tk]
        third_rep = {"rater": third, "n": len(lc),
                     "incidental": lc.count("incidental_mention"),
                     "incidental_rate": lc.count("incidental_mention") / len(lc),
                     f"incidental_rate[{ra}]": la.count("incidental_mention") / len(la),
                     f"kappa_vs[{ra}]": agree([A[u][p]["label"] for u, p in tk if p in A[u]],
                                             [C[u][p]["label"] for u, p in tk if p in A[u]])["kappa"],
                     f"kappa_vs[{rb}]": agree([B[u][p]["label"] for u, p in tk if p in B[u]],
                                             [C[u][p]["label"] for u, p in tk if p in B[u]])["kappa"]}

    agreement = {"raters": [ra, rb], "overall": agree(la, lb), "by_stratum": strat_agree,
                 "disagreements": dict(disagree),
                 "folk_rate": {ra: la.count("folk_description") / len(la),
                               rb: lb.count("folk_description") / len(lb)},
                 "spearman": rho, "third_rater": third_rep}

    (out_dir / "report.json").write_text(json.dumps({"coverage": coverage, "agreement": agreement}, indent=2))

    # ── report.md ───────────────────────────────────────────────────────────────
    L = ["# Stage 3 annotation statistics — random set\n",
         f"Coverage raters (mean): {', '.join(cov_raters)}. Tier: high >= {pct(high)}%, sparse > 0, none = 0.\n",
         f"Tiers: {coverage['tiers']}. Mean prevalence {pct(coverage['mean']['prev'])}%, "
         f"incidental {pct(coverage['mean']['inc'])}%, off-topic {pct(coverage['mean']['off'])}%.\n",
         f"None tier: {', '.join(coverage['none']) or '—'}\n",
         "\n## Coverage by stratum\n", "| stratum | N | high | sparse | none | prev % | inc % | off % |",
         "|---|---|---|---|---|---|---|---|"]
    for s, v in by_stratum.items():
        L.append(f"| {s} | {v['n']} | {v['high']} | {v['sparse']} | {v['none']} | {pct(v['prev'])} | {pct(v['inc'])} | {pct(v['off'])} |")
    L += ["\n## Coverage by tier\n", "| tier | N | prev % | inc % | off % |", "|---|---|---|---|---|"]
    for t, v in by_tier.items():
        L.append(f"| {t} | {v['n']} | " + " | ".join(pct(v[k]) if v[k] is not None else "—" for k in ("prev", "inc", "off")) + " |")
    L += ["\n## Per ability (mean of coverage raters; nb = mean neighbor-attributed count of 50)\n",
          "| ability | prev % | inc % | off % | nb | tier |", "|---|---|---|---|---|---|"]
    for r in cov:
        L.append(f"| {r['uid']} | {pct(r['prev'])} | {pct(r['inc'])} | {pct(r['off'])} | {r['nb']:.1f} | {r['tier']} |")
    o = agreement["overall"]
    L += [f"\n## Agreement: {ra} x {rb}\n",
          f"n = {o['n']}, kappa = {o['kappa']:.3f}, raw = {o['raw']:.3f}. "
          f"folk_description rate: {ra} {pct(agreement['folk_rate'][ra])}%, {rb} {pct(agreement['folk_rate'][rb])}%.\n",
          f"Disagreements: {dict(disagree)}\n",
          f"Spearman across abilities: folk of any ability rho = {rho['folk_any']['rho']:.3f} (p = {rho['folk_any']['p']:.1e}); "
          f"target prevalence rho = {rho['prev']['rho']:.3f} (p = {rho['prev']['p']:.1e}).\n",
          "| stratum | N | raw | kappa | folk % A | folk % B |", "|---|---|---|---|---|---|"]
    for s, v in strat_agree.items():
        L.append(f"| {s} | {v['n']} | {v['raw']:.3f} | {v['kappa']:.3f} | {pct(v[f'folk[{ra}]'])} | {pct(v[f'folk[{rb}]'])} |")
    if third_rep:
        L += [f"\n## Third rater: {third}\n",
              f"incidental_mention {third_rep['incidental']} of {third_rep['n']} ({pct(third_rep['incidental_rate'])}%, "
              f"against {pct(third_rep[f'incidental_rate[{ra}]'])}% for {ra}); "
              f"kappa vs {ra} {third_rep[f'kappa_vs[{ra}]']:.3f}, vs {rb} {third_rep[f'kappa_vs[{rb}]']:.3f}.\n"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L[:4]))
    print(f"kappa {o['kappa']:.3f} raw {o['raw']:.3f}; wrote {out_dir}/coverage.csv, report.json, report.md")


if __name__ == "__main__":
    main()
