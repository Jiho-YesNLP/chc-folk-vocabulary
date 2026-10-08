"""Stage 3 — manuscript figure: share of zero-prevalence abilities by the two correlates of absence.

Rows: instrument-dependent (both detectability raters say INSTRUMENTED), the other abilities of
the speed/fluency broad abilities (Gr, Gs, Gt, Gps), and all other abilities. Counts come from
coverage.csv (tier "none") and the detectability ratings named in configs/p3_detectability.yaml.
Writes a vector PDF.

Usage:
  uv run python scripts/p3_absence_routes_figure.py configs/p3_detectability.yaml \
      results/figures/absence_routes.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPEED_FLUENCY = {"Gr", "Gs", "Gt", "Gps"}
BAR = "#2A5CAA"


def instrumented(cfg: dict) -> set[str]:
    key = json.loads(pathlib.Path(cfg["blind_key_path"]).read_text())
    rdir = pathlib.Path(cfg["ratings_dir"])
    sets = []
    for rater in cfg["rater_pairs"][0]:
        recs = [json.loads(l) for l in (rdir / f"{rater}.jsonl").read_text().splitlines() if l.strip()]
        sets.append({r["uid"] if "uid" in r else key[r["id"]] for r in recs if r["category"] == "INSTRUMENTED"})
    return set.intersection(*sets)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("out")
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    with open(cfg["coverage_path"]) as f:
        tier = {r["uid"]: r["tier"] for r in csv.DictReader(f)}
    inst = instrumented(cfg) & set(tier)
    rest = [u for u in tier if u not in inst]
    groups = [("Instrument-dependent", sorted(inst)),
              ("Speed/fluency, not\ninstrument-dependent", [u for u in rest if u.split("-")[0] in SPEED_FLUENCY]),
              ("All other abilities", [u for u in rest if u.split("-")[0] not in SPEED_FLUENCY])]
    rows = [(name, sum(tier[u] == "none" for u in us), len(us)) for name, us in groups]
    for name, z, n in rows:
        print(f"{name.replace(chr(10), ' ')}: {z} of {n}")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.6, 1.65))
    y = list(range(len(rows)))[::-1]
    ax.barh(y, [z / n for _, z, n in rows], height=0.55, color=BAR)
    for yi, (_, z, n) in zip(y, rows):
        ax.text(z / n + 0.02, yi, f"{z} of {n}", va="center", ha="left", color="#222222")
    ax.set_yticks(y, [name for name, _, _ in rows], color="#444444")
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25%", "50%", "75%", "100%"], color="#444444")
    ax.set_xlabel("Abilities with zero prevalence", color="#222222")
    ax.xaxis.grid(True, color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", color="#666666")
    fig.savefig(args.out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
