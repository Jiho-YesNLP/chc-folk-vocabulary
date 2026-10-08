"""Export browser-fetchable data for the interactive ability-descriptor cluster page.

Data-prep only (writes JSON; no HTML). Mirrors ``p5_cluster_viz.py``'s joint mode so the 2-D
layout matches its ``scatter.png``: each point is a source-conditioned span vector from
``p5_embed_joint.py`` (config ``joint_vectors_path``), with the config's ``exclude_abilities``
dropped, projected by UMAP(metric=cosine, random_state=seed, min_dist=..., n_neighbors=...).
A ``prototype``-mode config (additive blend toward the ability prototype, display only, not
used in the paper) is also accepted.

Writes into ``<web_dir>/data[/<dataset>]``:
  cluster_layout.json   — projected points, per-stratum hull ellipse, tab20 colors,
                          narrow-ability label centroids (the only artifact that needs
                          the offline embeddings / UMAP).
  inventory.json        — merged CHC taxonomy + project annotations (definitions, seeds, …).
  fingerprints.json     — copy of the per-ability ability-descriptor fingerprints.

The web page (index.html) loads ``data/joint/`` and joins the three client-side.

Usage:
  uv run python scripts/p5_export_cluster_web_data.py \
      --config configs/p5_cluster_viz.yaml \
      --fingerprints data/processed/fingerprints/track_a_opus55/fingerprints.json --dataset joint
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import yaml

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.inventory import read_inventory  # noqa: E402  taxonomy + project annotations, merged


def l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(n, 1e-12, None)


def project_2d(X, seed, min_dist, n_neighbors):
    import umap  # noqa: WPS433 — optional heavy dep, only needed here
    reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed,
                        min_dist=min_dist, n_neighbors=n_neighbors)
    return reducer.fit_transform(X)


def palette(strata) -> dict:
    """tab20 over the sorted unique broad codes — identical to cluster_viz.palette()."""
    cm = plt.get_cmap("tab20")
    return {b: mcolors.to_hex(cm(i % 20)) for i, b in enumerate(sorted(strata))}


def ellipse_params(pts: np.ndarray, std: float) -> dict | None:
    """Covariance-boundary ellipse (cx, cy, rx, ry, angle°) — matches scatter_plot()."""
    if len(pts) < 3:
        return None
    c = pts.mean(axis=0)
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = float(np.degrees(np.arctan2(*vecs[:, 0][::-1])))
    w, h = 2 * std * np.sqrt(np.maximum(vals, 1e-9))
    return {"cx": round(float(c[0]), 3), "cy": round(float(c[1]), 3),
            "rx": round(float(w) / 2, 3), "ry": round(float(h) / 2, 3),
            "angle": round(ang, 3)}


def load_blended(cfg: dict) -> dict:
    """Member coordinates + metadata for the configured mode (mirrors cluster_viz.load_points).

    Returns dict with: blended (M,dim) L2-normed points, mem_uid, mem_broad, mem_text,
    mem_count, fp_broad (unique broad codes), alpha (float|None), default_fp_json (path).
    """
    mode = cfg.get("mode", "prototype").lower()

    if mode == "joint":
        vec_path = pathlib.Path(cfg["joint_vectors_path"])
        if not vec_path.exists():
            sys.exit(f"{vec_path} not found — run p5_embed_joint.py (CUDA GPU) first.")
        z = np.load(vec_path)
        blended = l2norm(z["member_matrix"].astype(np.float64))
        mem_uid = z["member_uid"].astype(str)
        mem_broad = z["member_broad"].astype(str)
        # joint npz stores the in-focus descriptor phrase as `member_span` (no `member_text`)
        text_key = "member_span" if "member_span" in z else "member_text"
        mem_text = z[text_key].astype(str)
        mem_count = z["member_count"].astype(np.int64) if "member_count" in z \
            else np.ones(len(mem_uid), np.int64)
        fp_broad = mem_broad
        print(f"loaded {len(mem_uid)} joint members across "
              f"{len(set(mem_uid.tolist()))} abilities (mode=joint, no blend)")
        return {"mode": mode, "blended": blended, "mem_uid": mem_uid,
                "mem_broad": mem_broad, "mem_text": mem_text, "mem_count": mem_count,
                "fp_broad": fp_broad, "alpha": None,
                "default_fp_json": vec_path.with_name("fingerprints.json")}

    # ── prototype (additive blend) mode ───────────────────────────────────────────
    for k in ("vectors_path", "alpha"):
        if k not in cfg:
            sys.exit(f"mode=prototype requires config key '{k}'")
    vec_path = pathlib.Path(cfg["vectors_path"])
    if not vec_path.exists():
        sys.exit(f"{vec_path} not found — run p4_build_fingerprint.py (CUDA GPU) first.")
    z = np.load(vec_path)
    fp_mat = l2norm(z["fp_matrix"].astype(np.float64))
    fp_uid = z["fp_uid"].astype(str)
    fp_broad = z["fp_broad"].astype(str)
    mem_mat = z["member_matrix"].astype(np.float64)
    mem_uid = z["member_uid"].astype(str)
    mem_broad = z["member_broad"].astype(str)
    mem_text = z["member_text"].astype(str)
    mem_count = z["member_count"].astype(np.int64)
    print(f"loaded {len(fp_uid)} prototypes, {len(mem_uid)} member spans")

    alpha = float(cfg["alpha"])
    fp_lookup = {u: fp_mat[i] for i, u in enumerate(fp_uid)}
    blended = l2norm(np.stack([
        alpha * fp_lookup[u] + (1 - alpha) * mem_mat[i] if u in fp_lookup else mem_mat[i]
        for i, u in enumerate(mem_uid)
    ]))
    return {"mode": mode, "blended": blended, "mem_uid": mem_uid,
            "mem_broad": mem_broad, "mem_text": mem_text, "mem_count": mem_count,
            "fp_broad": fp_broad, "alpha": alpha,
            "default_fp_json": vec_path.with_name("fingerprints.json")}


def main() -> None:
    ap = argparse.ArgumentParser(description="Export web data for cluster page")
    ap.add_argument("--config", default="configs/p5_cluster_viz.yaml",
                    help="cluster_viz config; its mode + paths pick the source directory")
    ap.add_argument("--inventory", default="data/raw/chc_taxonomy.json")
    ap.add_argument("--fingerprints", default=None,
                    help="per-ability fingerprints.json to copy (default: next to the "
                         "vectors file, else data/processed/fingerprints/track_a_opus55/)")
    ap.add_argument("--web-dir", default="webapp")
    ap.add_argument("--dataset", default=None,
                    help="write into <web-dir>/data/<dataset>/ so methods coexist; "
                         "the page selects it via ?data=<dataset>. Default: flat data/.")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    V = load_blended(cfg)
    excluded = sorted(cfg.get("exclude_abilities", []))
    if excluded:   # same point set as p5_cluster_viz.py's scatter.png
        keep = ~np.isin(V["mem_uid"], excluded)
        for key in ("blended", "mem_uid", "mem_broad", "mem_text", "mem_count"):
            V[key] = V[key][keep]
        V["fp_broad"] = V["mem_broad"]
        print(f"excluded {len(excluded)} abilities: {excluded}")
    blended = V["blended"]
    mem_uid, mem_broad, mem_text, mem_count = (
        V["mem_uid"], V["mem_broad"], V["mem_text"], V["mem_count"])

    # fingerprints.json: explicit flag > next to vectors file > track_a fallback
    if args.fingerprints:
        fp_json = pathlib.Path(args.fingerprints)
    elif V["default_fp_json"].exists():
        fp_json = V["default_fp_json"]
    else:
        fp_json = pathlib.Path("data/processed/fingerprints/track_a_opus55/fingerprints.json")
    if not fp_json.exists():
        sys.exit(f"fingerprints.json not found at {fp_json} — pass --fingerprints")

    inv_path = pathlib.Path(args.inventory)
    data_dir = pathlib.Path(args.web_dir) / "data"
    if args.dataset:
        data_dir = data_dir / args.dataset
    data_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg["random_seed"])
    xy = np.asarray(project_2d(blended, seed,
                               float(cfg.get("umap_min_dist", 0.0)),
                               int(cfg.get("umap_n_neighbors", 15))), dtype=float)
    print(f"projected to 2-D via umap (seed={seed})")

    colors = palette(set(mem_broad.tolist()) | set(V["fp_broad"].tolist()))
    std = float(cfg.get("scatter_ellipse_std", 2.0))

    # points
    points = [
        {"x": round(float(xy[i, 0]), 3), "y": round(float(xy[i, 1]), 3),
         "u": mem_uid[i], "n": mem_uid[i].split("-", 1)[-1], "b": mem_broad[i],
         "c": int(mem_count[i]), "t": mem_text[i]}
        for i in range(len(mem_uid))
    ]

    # per-broad hull + count
    broads = []
    for b in sorted(set(mem_broad.tolist())):
        sel = mem_broad == b
        broads.append({"b": b, "n": int(sel.sum()),
                       "ellipse": ellipse_params(xy[sel], std)})

    # narrow-ability label centroids (median, matching scatter_plot)
    narrows = []
    for u in sorted(set(mem_uid.tolist())):
        sel = mem_uid == u
        c = np.median(xy[sel], axis=0)
        narrows.append({"u": u, "n": u.split("-", 1)[-1], "b": mem_broad[sel][0],
                        "x": round(float(c[0]), 3), "y": round(float(c[1]), 3)})

    layout = {"proj": "umap", "mode": V["mode"], "alpha": V["alpha"], "colors": colors,
              "broads": broads, "points": points, "narrows": narrows}
    (data_dir / "cluster_layout.json").write_text(json.dumps(layout))

    # copy the two source files the page joins against
    # The webapp reads one merged file: taxonomy + project annotations.
    (data_dir / "inventory.json").write_text(json.dumps(read_inventory(inv_path), indent=2, ensure_ascii=False) + "\n")
    shutil.copyfile(fp_json, data_dir / "fingerprints.json")

    print(f"wrote {data_dir}/cluster_layout.json (mode={V['mode']}, "
          f"{len(points)} points, {len(broads)} strata, {len(narrows)} narrows)")
    print(f"copied inventory.json + fingerprints.json -> {data_dir}")


if __name__ == "__main__":
    main()
