"""Stage 5 — linguistic discriminant validity: which ability pairs folk language fails to separate.

CPU/local. Reads the source-conditioned joint vectors (p5_embed_joint.py), the CHC inventory,
and the Stage-3 random-set judgments. Two lines of evidence per ability pair:

  geometric leak   share of the pair's kNN edges that cross between the two abilities,
                   (edges A->B + edges B->A) / (k * (n_A + n_B)), on the SAME neighbor graph
                   as p5_cluster_viz.py's kNN purity: k nearest cosine neighbors excluding the
                   point itself and every other span from the same passage, over the same point
                   set (single-ability strata and config `exclude_abilities` dropped).
  annotation       passages a rater labeled folk_description of one ability while assigning
                   best_ability_code to the other, pooled over the configured raters (raw count).

Collapse is a test, not a ranking: a pair collapses when its leak exceeds a label-permutation
null (labels shuffled over points, the neighbor graph held fixed) at Benjamini-Hochberg
q < fdr_alpha across all eligible pairs. A pair separates when its leak is at or below the
null mean. Eligible pairs have at least `min_n` descriptors per ability (default k + 1: below
that an ability cannot fill its own neighborhood, so its leak is inflated by arithmetic).

Relations: N = same broad stratum (given by the taxonomy); D = the inventory's `discriminator`
field ("vs <CODE> [(<Gx>)]"), project-authored and reported as such.

Outputs (to out_dir): pairs.csv (every eligible pair), summary.json, summary.md (cell counts
and the top pairs per cell).

Usage:
  uv run python scripts/p5_discriminant.py configs/p5_discriminant.yaml
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import pathlib
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from p5_cluster_viz import knn_excluding_passage  # noqa: E402  (one neighbor graph for both)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.inventory import read_inventory  # noqa: E402  taxonomy + project annotations, merged

REQUIRED_KEYS = ("joint_vectors_path", "inventory_path", "judgments_dir", "raters", "knn_k",
                 "exclude_abilities", "n_permutations", "fdr_alpha", "out_dir", "random_seed")


def discriminator_pairs(inv: dict) -> set[frozenset]:
    """Unordered narrow-ability pairs named in the inventory's discriminator glosses. A bare
    code resolves within the ability's own stratum unless qualified "(Gx)"; references to a
    broad stratum ("vs Gkn") are not narrow pairs and are skipped."""
    uids = {n["uid"] for b in inv["broad_abilities"] for n in b["narrow_abilities"]}
    broads = {b["code"] for b in inv["broad_abilities"]}
    by_code = collections.defaultdict(list)
    for u in uids:
        by_code[u.split("-", 1)[1]].append(u)
    pairs = set()
    for b in inv["broad_abilities"]:
        for n in b["narrow_abilities"]:
            for seg in (n.get("discriminator") or "").split(";"):
                mo = re.match(r"\s*vs\s+([A-Za-z0-9/]+)(?:\s*\((G[a-z]+)\))?", seg)
                if not mo:
                    continue
                for c in mo.group(1).split("/"):
                    if c in broads:
                        continue
                    tgt = f"{mo.group(2) or b['code']}-{c}"
                    if tgt not in uids:
                        if len(by_code[c]) != 1:
                            sys.exit(f"cannot resolve discriminator code {c!r} in {n['uid']}")
                        tgt = by_code[c][0]
                    if tgt != n["uid"]:
                        pairs.add(frozenset((n["uid"], tgt)))
    return pairs


def annotation_counts(judgments_dir: pathlib.Path, raters: list[str]) -> collections.Counter:
    cross = collections.Counter()
    for d in sorted(p for p in judgments_dir.iterdir() if p.is_dir()):
        for rater in raters:
            f = d / rater
            if not f.exists():
                sys.exit(f"missing judgments: {f}")
            for line in f.read_text().splitlines():
                j = json.loads(line)
                bac = j.get("best_ability_code")
                if j["label"] == "folk_description" and bac and bac != d.name:
                    cross[frozenset((d.name, bac))] += 1
    return cross


def cross_edge_matrix(labels: np.ndarray, idx: np.ndarray, L: int) -> np.ndarray:
    """C[a, b] = number of kNN edges from a point labeled a to a neighbor labeled b."""
    src = np.repeat(labels, idx.shape[1])
    dst = labels[idx.ravel()]
    return np.bincount(src * L + dst, minlength=L * L).reshape(L, L)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 — linguistic discriminant validity")
    ap.add_argument("config")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")
    k = int(cfg["knn_k"])
    min_n = int(cfg.get("min_n", k + 1))
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- the point set of p5_cluster_viz.py's cohesion metrics ----
    z = np.load(cfg["joint_vectors_path"])
    X = z["member_matrix"].astype(np.float64)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    uid = z["member_uid"].astype(str)
    broad = z["member_broad"].astype(str)
    pid = z["member_passage_id"].astype(str)
    abil_per_broad = collections.defaultdict(set)
    for u, b in zip(uid, broad):
        abil_per_broad[b].add(u)
    single = {b for b, a in abil_per_broad.items() if len(a) < 2}
    keep = ~np.isin(broad, list(single)) & ~np.isin(uid, cfg["exclude_abilities"])
    X, uid, broad, pid = X[keep], uid[keep], broad[keep], pid[keep]
    print(f"points {len(uid)}  abilities {len(set(uid))}  (dropped strata {sorted(single)}, "
          f"excluded {sorted(cfg['exclude_abilities'])})")

    idx = knn_excluding_passage(X, pid, k)
    names = sorted(set(uid))
    L = len(names)
    lab = np.array([names.index(u) for u in uid])
    n = np.bincount(lab, minlength=L)
    stratum = {u: b for u, b in zip(uid, broad)}

    obs = cross_edge_matrix(lab, idx, L)
    obs_sym = obs + obs.T
    rng = np.random.default_rng(int(cfg["random_seed"]))
    n_perm = int(cfg["n_permutations"])
    ge = np.zeros((L, L))
    null_sum = np.zeros((L, L))
    for _ in range(n_perm):
        c = cross_edge_matrix(rng.permutation(lab), idx, L)
        c = c + c.T
        ge += c >= obs_sym
        null_sum += c
    denom = k * (n[:, None] + n[None, :])

    D = discriminator_pairs(read_inventory(cfg["inventory_path"]))
    ann = annotation_counts(pathlib.Path(cfg["judgments_dir"]), list(cfg["raters"]))

    rows = []
    for i in range(L):
        for j in range(i + 1, L):
            if n[i] < min_n or n[j] < min_n:
                continue
            a, b = names[i], names[j]
            rows.append({
                "a": a, "b": b, "n_a": int(n[i]), "n_b": int(n[j]),
                "N": stratum[a] == stratum[b], "D": frozenset((a, b)) in D,
                "leak": obs_sym[i, j] / denom[i, j],
                "leak_null": null_sum[i, j] / n_perm / denom[i, j],
                "p": (1 + ge[i, j]) / (1 + n_perm),
                "ann": ann[frozenset((a, b))],
            })
    q = bh_qvalues(np.array([r["p"] for r in rows]))
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
        r["collapse"] = bool(qq < float(cfg["fdr_alpha"]))
        r["separate"] = bool(r["leak"] <= r["leak_null"])

    with (out_dir / "pairs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in sorted(rows, key=lambda r: -r["leak"]):
            w.writerow({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in r.items()})

    cells = {
        "N_collapse": [r for r in rows if r["N"] and r["collapse"]],
        "notN_collapse": [r for r in rows if not r["N"] and r["collapse"]],
        "D_collapse": [r for r in rows if r["D"] and r["collapse"]],
        "notD_collapse": [r for r in rows if not r["D"] and r["collapse"]],
        "D_separate": [r for r in rows if r["D"] and r["separate"]],
    }
    elig = {"all": len(rows), "N": sum(r["N"] for r in rows), "D": sum(r["D"] for r in rows)}
    summary = {"config": {kk: cfg[kk] for kk in REQUIRED_KEYS}, "min_n": min_n,
               "n_points": int(len(uid)), "n_abilities": L,
               "D_pairs_in_inventory": len(D), "eligible_pairs": elig,
               "cell_sizes": {c: len(v) for c, v in cells.items()},
               "top_annotation_pairs": [
                   {"pair": sorted(p), "count": c} for p, c in ann.most_common(10)]}
    top = lambda c: (sorted(cells[c], key=lambda r: r["leak"])[:5] if c == "D_separate"
                     else sorted(cells[c], key=lambda r: -r["leak"])[:5])
    summary["top"] = {c: [{kk: (round(v, 4) if isinstance(v, float) else v)
                           for kk, v in r.items()} for r in top(c)] for c in cells}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    L_ = ["# Linguistic discriminant validity\n",
          f"points={len(uid)}  abilities={L}  k={k}  min_n={min_n}  permutations={n_perm}  "
          f"FDR q<{cfg['fdr_alpha']}\n",
          f"Eligible pairs: {elig['all']} (N: {elig['N']}, D: {elig['D']} of {len(D)} in the inventory)\n",
          "Cell sizes: " + ", ".join(f"{c}={len(v)}" for c, v in cells.items()) + "\n"]
    for c in cells:
        L_ += [f"\n## {c}\n", "| A | B | leak | null | q | ann | n_a / n_b |", "|---|---|---|---|---|---|---|"]
        for r in top(c):
            L_.append(f"| {r['a']} | {r['b']} | {r['leak']:.3f} | {r['leak_null']:.4f} | "
                      f"{r['q']:.3g} | {r['ann']} | {r['n_a']} / {r['n_b']} |")
    L_ += ["\n## Largest annotation cross-attributions (any pair)\n"]
    L_ += [f"- {' / '.join(sorted(p))}: {c}" for p, c in ann.most_common(10)]
    (out_dir / "summary.md").write_text("\n".join(L_) + "\n")
    print(f"-> {out_dir}/pairs.csv, summary.json, summary.md")


if __name__ == "__main__":
    main()
