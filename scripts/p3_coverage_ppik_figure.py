"""Stage 3 — manuscript figure: coverage of the 71 abilities grouped by the definitions sheet's
broad-ability color codes (Ackerman's PPIK grouping): process (Gf, Gwm, Gl, Gv, Ga), knowledge
(Gc, Gkn, Grw, Gq) and speed/fluency (Gr, Gs, Gt, Gps).

Panels: (a) on-target folk-description prevalence and (b) incidental-mention rate per ability
(mean of the two raters, one dot per ability, bar at the group median); (c) coverage tiers.
Reads `{out_dir}/coverage.csv` written by p3_annotation_stats.py.

Usage:
  uv run python scripts/p3_coverage_ppik_figure.py configs/p3_annotation_stats.yaml \
      results/figures/coverage_by_ppik.png
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GROUP = {"Gf": "P", "Gwm": "P", "Gl": "P", "Gv": "P", "Ga": "P",
         "Gc": "K", "Gkn": "K", "Grw": "K", "Gq": "K",
         "Gr": "S", "Gs": "S", "Gt": "S", "Gps": "S"}
TICK = {"P": "Process\n(blue)", "K": "Knowledge\n(gray)", "S": "Speed/\nfluency\n(green)"}
BAR_LABEL = {"P": "Process (blue)", "K": "Knowledge (gray)", "S": "Speed/fluency (green)"}
COLOR = {"P": "#2a5caa", "K": "#9a9a9a", "S": "#1f6e33"}   # approximate the sheet's colors
INK, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#e6e6e6"
ORDER = ["P", "K", "S"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="configs/p3_annotation_stats.yaml (out_dir, tier_high)")
    ap.add_argument("out", help="output image path (.png at 300 dpi, or .pdf)")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    with open(pathlib.Path(cfg["out_dir"]) / "coverage.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups = collections.defaultdict(list)
    for r in rows:
        groups[GROUP[r["broad"]]].append(r)
    high_pct = 100 * float(cfg.get("tier_high", 0.10))

    plt.rcParams.update({"font.size": 7})
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(7.16, 2.55), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    rng = np.random.default_rng(3)   # horizontal jitter only
    for ax, key, title in ((a1, "prev", "(a) On-target folk descriptions (%)"),
                           (a2, "inc", "(b) Incidental mentions (%)")):
        for i, k in enumerate(ORDER):
            y = np.array([float(r[key]) * 100 for r in groups[k]])
            x = i + rng.uniform(-0.18, 0.18, len(y))
            ax.scatter(x, y, s=11, color=COLOR[k], edgecolor="white", linewidth=0.6, zorder=3)
            ax.hlines(np.median(y), i - 0.3, i + 0.3, color=INK, lw=1.4, zorder=4)
            ax.text(i + 0.33, np.median(y), f"{np.median(y):.1f}", va="center", fontsize=6, color=INK)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICK[k] for k in ORDER], fontsize=6.3, color=INK)
        ax.set_xlim(-0.5, 2.65)
        ax.set_title(title, fontsize=7, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=6)
    a1.axhline(high_pct, color=MUTED, lw=0.6, ls=(0, (3, 3)))
    a1.text(2.6, high_pct + 0.5, f"high tier ≥ {high_pct:g}%", fontsize=5.5, color=MUTED, ha="right")

    shades = {"high": 1.0, "sparse": 0.55, "none": 0.18}
    for i, k in enumerate(ORDER):
        n, left = len(groups[k]), 0.0
        for t in ("high", "sparse", "none"):
            c = sum(r["tier"] == t for r in groups[k])
            w = c / n
            a3.barh(i, w, left=left, color=COLOR[k], alpha=shades[t], edgecolor="white", linewidth=1.2, height=0.6)
            if c:
                a3.text(left + w / 2, i, f"{c}", ha="center", va="center", fontsize=6,
                        color="white" if shades[t] > 0.5 else INK)
            left += w
    a3.set_yticks(range(3))
    a3.set_yticklabels([BAR_LABEL[k] for k in ORDER], fontsize=6.3, color=INK)
    a3.invert_yaxis()
    a3.set_xlim(0, 1)
    a3.set_xticks([0, 0.5, 1])
    a3.set_xticklabels(["0", "50%", "100%"], fontsize=6, color=MUTED)
    a3.set_title("(c) Tier: high · sparse · none", fontsize=7, color=INK, loc="left")
    for s in ("top", "right", "left"):
        a3.spines[s].set_visible(False)
    a3.tick_params(left=False)

    fig.tight_layout(pad=0.4)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
