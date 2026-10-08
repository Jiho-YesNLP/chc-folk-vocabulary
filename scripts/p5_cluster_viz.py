"""Stage 5 — cohesion metrics & visualization of ability-descriptor vectors (RQ2).

CPU/local, no re-embedding. Two input modes (see ``load_points``): ``joint`` reads the
source-conditioned span vectors from ``p5_embed_joint.py`` and is the MEASURED path for
RQ2; ``prototype`` pulls each span toward its ability prototype and is display-only (not used in the paper).

Core question: does the geometry of folk descriptors recover CHC structure? Measured as
one kNN graph read at narrow (ability) and broad (stratum) granularity -- kNN purity with
chance baselines, overall and per stratum -- plus label-blind recovery: k-means ARI against
the true abilities and strata.

Outputs (to ``out_dir``):
  metrics.json            — cohesion metrics (raw + per-ability-balanced)
  metrics.md              — same, human-readable
  scatter.png             — 2-D projection, colored by broad stratum, per-stratum hull
  dendrogram.png          — Ward tree over the ability centroids (display only)
  similarity_heatmap.png  — prototype cosine sim, block-ordered by broad stratum

All metrics are computed on the full-dimensional (1024-d) vectors; the 2-D projection
is for display only.

Usage:
  uv run python scripts/p5_cluster_viz.py configs/p5_cluster_viz.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy.cluster.hierarchy import dendrogram, linkage  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

REQUIRED_KEYS = ("out_dir", "max_members_per_ability", "knn_k", "projection",
                 "drop_singleton_broad", "random_seed")


def l2norm(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=-1, keepdims=True)
    return m / np.clip(n, 1e-12, None)


def project_2d(X: np.ndarray, how: str, seed: int,
               min_dist: float = 0.0, n_neighbors: int = 15) -> tuple[np.ndarray, str]:
    """2-D embedding for display only, never for metrics. `how` selects the method
    deliberately -- umap (the reported default), tsne, or pca -- and a requested method
    that is unavailable raises rather than substituting another.
    Lower ``min_dist`` packs each cluster tighter (less whitespace)."""
    how = how.lower()
    if how == "umap":
        # No silent fallback: UMAP and t-SNE give visibly different pictures, so a
        # missing dependency must not quietly change what the figure shows. umap-learn
        # is a declared `viz` extra; install it, or set `projection` explicitly.
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise SystemExit(
                f"projection: umap requested but umap-learn is not installed ({e}). "
                "Install the viz extra (uv pip install umap-learn), or set "
                "`projection: tsne` in the config to choose t-SNE deliberately."
            ) from e
        reducer = umap.UMAP(n_components=2, metric="cosine", random_state=seed,
                            min_dist=min_dist, n_neighbors=n_neighbors)
        return reducer.fit_transform(X), "umap"
    if how == "tsne":
        from sklearn.manifold import TSNE
        perp = float(min(30, max(5, (len(X) - 1) / 3)))
        ts = TSNE(n_components=2, metric="cosine", init="random",
                  perplexity=perp, random_state=seed)
        return ts.fit_transform(X), "tsne"
    from sklearn.decomposition import PCA
    return PCA(n_components=2, random_state=seed).fit_transform(X), "pca"


def chance_purity(labels: np.ndarray) -> float:
    """Expected same-label neighbor rate if labels were assigned at random:
    sum_l n_l (n_l - 1) / (N (N - 1)), self excluded."""
    n = np.array(list(Counter(labels).values()), float)
    N = n.sum()
    return float((n * (n - 1)).sum() / (N * (N - 1)))


def kmeans_ari(pts: np.ndarray, truth: np.ndarray, n_clusters: int, seeds: int) -> dict:
    """Label-blind recovery: k-means at the true cluster count vs. the ground truth, over
    `seeds` restarts. pts are L2-normalized, so Euclidean k-means ranks like cosine.
    (Cosine average-linkage agglomerative clustering, used previously, chains these
    vectors into one giant cluster and forces ARI to ~0 whatever the data -- do not use.)"""
    nc = max(2, min(n_clusters, len(pts) - 1))
    aris = [adjusted_rand_score(truth, KMeans(nc, n_init=10, random_state=s).fit_predict(pts))
            for s in range(seeds)]
    return {"k": int(nc), "mean": round(float(np.mean(aris)), 4),
            "sd": round(float(np.std(aris)), 4), "seeds": int(seeds)}


def knn_excluding_passage(pts: np.ndarray, pid: np.ndarray | None, k: int) -> np.ndarray:
    """Indices of each point's k nearest cosine neighbors, excluding itself and -- when
    passage ids are given -- every other span extracted from the same passage. Co-occurring
    spans share their source sentence, so under span[SEP]source they sit next to each other
    and would count as same-ability neighbors trivially."""
    S = pts @ pts.T
    if pid is None:
        np.fill_diagonal(S, -np.inf)
    else:
        S[pid[:, None] == pid[None, :]] = -np.inf
    k = min(k, len(pts) - 1)
    return np.argpartition(-S, k - 1, axis=1)[:, :k]


def cohesion_metrics(pts: np.ndarray, broad: np.ndarray, uid: np.ndarray,
                     k: int, seeds: int, pid: np.ndarray | None = None) -> dict:
    """All on full-dim vectors. Cosine == dot since pts are L2-normalized.

    One kNN graph read at two granularities (narrow = uid, broad = stratum), overall and
    per stratum, with chance baselines; plus one label-blind recovery test (k-means ARI)
    at each granularity. The pairwise view of the same kNN graph (geometric leak) lives in
    the discriminant analysis."""
    out: dict = {"n_points": int(len(pts)), "n_broad": int(len(set(broad))),
                 "n_uid": int(len(set(uid)))}
    if len(pts) < 3 or out["n_broad"] < 2:
        out["note"] = "too few points/strata for metrics"
        return out

    # kNN purity: fraction of k nearest neighbors (excluding self and same-passage
    # siblings) sharing label.
    idx = knn_excluding_passage(pts, pid, k)
    out["same_passage_excluded"] = pid is not None
    out["knn_purity_uid"] = round(float(np.mean(uid[idx] == uid[:, None])), 4)
    out["knn_purity_broad"] = round(float(np.mean(broad[idx] == broad[:, None])), 4)
    out["chance_purity_uid"] = round(chance_purity(uid), 4)
    out["chance_purity_broad"] = round(chance_purity(broad), 4)

    # per stratum: narrow and broad purity of that stratum's points (Table: per-stratum cohesion)
    out["knn_purity_by_broad"] = {
        b: {"n": int((broad == b).sum()),
            "narrow": round(float(np.mean(uid[idx[broad == b]] == uid[broad == b][:, None])), 4),
            "broad": round(float(np.mean(broad[idx[broad == b]] == b)), 4)}
        for b in sorted(set(broad))}

    out["ari_broad"] = kmeans_ari(pts, broad, out["n_broad"], seeds)
    out["ari_uid"] = kmeans_ari(pts, uid, out["n_uid"], seeds)
    return out


def balanced_subsample(uid: np.ndarray, cap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    by = defaultdict(list)
    for i, u in enumerate(uid):
        by[u].append(i)
    for u, rows in by.items():
        rows = np.array(rows)
        if len(rows) > cap:
            rows = rng.choice(rows, size=cap, replace=False)
        keep.extend(rows.tolist())
    return np.array(sorted(keep))


# 15 mutually distinct colors that stay legible on white (mostly Kelly's max-contrast set).
DISTINCT = ["#0067A5", "#BE0032", "#008856", "#F38400", "#875692", "#882D17", "#E68FAC",
            "#848482", "#8DB600", "#604E97", "#F6A600", "#B3446C", "#2B3D26", "#00A3A3",
            "#1F1F7A"]


def palette(strata: list[str]) -> dict:
    """One distinct color per stratum. `strata` may repeat (callers concatenate member and
    prototype strata), so de-duplicate before indexing."""
    uniq = sorted(set(strata))
    if len(uniq) > len(DISTINCT):
        cmap = plt.get_cmap("tab20")
        return {b: cmap(i % 20) for i, b in enumerate(uniq)}
    return {b: DISTINCT[i] for i, b in enumerate(uniq)}


def scatter_plot(xy, broad, uid, count, colors, how, path, dpi=300, ellipse_std=2.0):
    from matplotlib.patches import Ellipse

    xy = xy.astype(float)
    fig, ax = plt.subplots(figsize=(14, 11))
    # covariance-ellipse boundary per broad stratum (the broad-ability grouping)
    for b in sorted(set(broad)):
        pts = xy[broad == b]
        if len(pts) < 3:
            continue
        c = pts.mean(axis=0)
        vals, vecs = np.linalg.eigh(np.cov(pts.T))
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        ang = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
        w, h = 2 * ellipse_std * np.sqrt(np.maximum(vals, 1e-9))
        ax.add_patch(Ellipse(c, w, h, angle=ang, facecolor=colors[b],
                             edgecolor=colors[b], lw=1.3, alpha=0.10, zorder=0))
    # points: thin white rim so overlaps stay legible
    sizes = 12 + 12 * np.log1p(count.astype(float))
    for b in sorted(set(broad)):
        sel = broad == b
        ax.scatter(xy[sel, 0], xy[sel, 1], s=sizes[sel], c=[colors[b]],
                   label=b, alpha=0.75, linewidths=0.25, edgecolors="white")
    # labels as Text objects so adjustText can repel them apart (with leader lines)
    texts = []
    for u in sorted(set(uid)):  # narrow-ability label at each ability's cluster centroid
        sel = uid == u
        c = np.median(xy[sel], axis=0)
        texts.append(ax.text(c[0], c[1], u.split("-", 1)[-1], fontsize=7, fontweight="bold",
                             ha="center", va="center", color="black", zorder=6,
                             bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                       ec=colors[broad[sel][0]], lw=0.8, alpha=0.9)))
    for b in sorted(set(broad)):  # bold broad-stratum label at each stratum's centroid
        c = np.median(xy[broad == b], axis=0)
        texts.append(ax.text(c[0], c[1], b, fontsize=12, fontweight="bold",
                             ha="center", va="center", color="black", zorder=7,
                             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[b],
                                       lw=1.6, alpha=0.92)))
    try:
        from adjustText import adjust_text
        arrow = dict(arrowstyle="-", color="0.5", lw=0.5)
        try:  # newer adjustText kwargs
            adjust_text(texts, ax=ax, x=list(xy[:, 0]), y=list(xy[:, 1]),
                        arrowprops=arrow, expand=(1.15, 1.3), force_text=(0.4, 0.6))
        except TypeError:  # older signature
            adjust_text(texts, ax=ax, arrowprops=arrow)
    except Exception as e:
        print(f"[labels] adjustText unavailable ({e}); labels left at centroids")
    ax.set_title(f"Ability-descriptor fingerprints ({how}) — color/ellipse = broad stratum, "
                 f"label = narrow ability")
    ax.set_xticks([]); ax.set_yticks([])
    ax.margins(0.02)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="broad",
              markerscale=2.0)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def dendrogram_plot(fp, fp_uid, fp_broad, colors, path, dpi=300):
    """Ward tree over the L2-normalized ability centroids, leaves along the bottom so it
    fits a full-width figure. Ward, not average linkage: on these vectors cosine average
    linkage chains small, diffuse abilities onto the root one by one (the same failure
    that made it unusable for the recovery test). Display only -- no metric uses it."""
    if len(fp) < 3:
        return
    Z = linkage(fp, method="ward")  # Euclidean on unit vectors ~ cosine
    # drawn at print size for a full-width IEEE figure (7.16 in), so font sizes are real
    fig, ax = plt.subplots(figsize=(7.16, 2.9))
    dendrogram(Z, labels=list(fp_uid), orientation="top", ax=ax, leaf_font_size=5,
               leaf_rotation=90, color_threshold=0, above_threshold_color="#888")
    b_of = dict(zip(fp_uid, fp_broad))
    for lab in ax.get_xmajorticklabels():
        lab.set_color(colors[b_of[lab.get_text()]])
        lab.set_fontweight("bold")
    handles = [plt.Line2D([], [], marker="s", ls="", color=colors[b], label=b)
               for b in sorted(set(fp_broad))]
    ax.legend(handles=handles, ncol=len(handles), fontsize=6, frameon=False, markerscale=0.8,
              loc="upper center", bbox_to_anchor=(0.5, 1.14), handletextpad=0.1,
              columnspacing=0.6)
    ax.set_ylabel("Ward linkage distance", fontsize=7)
    ax.tick_params(axis="y", labelsize=6)
    for ln in ax.collections:
        ln.set_linewidth(0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def heatmap_plot(fp, fp_uid, fp_broad, colors, path, dpi=300):
    if len(fp) < 2:
        return
    order = np.argsort([f"{b}:{u}" for b, u in zip(fp_broad, fp_uid)])
    S = (fp @ fp.T)[np.ix_(order, order)]
    labs = [fp_uid[i] for i in order]
    bro = [fp_broad[i] for i in order]
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(S, cmap="magma", vmin=float(S.min()), vmax=1.0)
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=90, fontsize=6)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=6)
    for i, lab in enumerate(ax.get_yticklabels()):
        lab.set_color(colors[bro[i]])
    for i, lab in enumerate(ax.get_xticklabels()):
        lab.set_color(colors[bro[i]])
    # broad-block boundary lines
    bounds = [i for i in range(1, len(bro)) if bro[i] != bro[i - 1]]
    for bnd in bounds:
        ax.axhline(bnd - 0.5, color="cyan", lw=0.6); ax.axvline(bnd - 0.5, color="cyan", lw=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="cosine similarity")
    ax.set_title("Ability-prototype cosine similarity (block-ordered by broad stratum)")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_md(path, cfg, raw, bal, proj_used, dropped, rep="prototype blend"):
    L = ["# Ability-Descriptor Fingerprint Clustering\n"]
    L.append(f"representation={rep}  member=span  projection={proj_used}  "
             f"knn_k={cfg['knn_k']}  cap/ability={cfg['max_members_per_ability']}\n")
    if dropped:
        L.append(f"Singleton broad strata excluded from cohesion metrics: {', '.join(dropped)}\n")

    def block(title, m):
        L.append(f"\n## {title}\n")
        if "note" in m:
            L.append(m["note"]); return
        L.append(f"points={m['n_points']}  abilities={m['n_uid']}  strata={m['n_broad']}\n")
        L.append("| granularity | kNN purity | chance | k-means ARI (mean ± sd) |")
        L.append("|---|---|---|---|")
        for g, lbl in (("uid", "narrow"), ("broad", "broad")):
            a = m[f"ari_{g}"]
            L.append(f"| {lbl} | {m[f'knn_purity_{g}']} | {m[f'chance_purity_{g}']} | "
                     f"{a['mean']} ± {a['sd']} (k={a['k']}, {a['seeds']} seeds) |")
        L.append("\n*Per-stratum kNN purity:*\n")
        L.append("| broad | n | narrow | broad | gap |")
        L.append("|---|---|---|---|---|")
        for b, v in sorted(m.get("knn_purity_by_broad", {}).items(), key=lambda kv: -kv[1]["broad"]):
            L.append(f"| {b} | {v['n']} | {v['narrow']} | {v['broad']} | "
                     f"{round(v['broad'] - v['narrow'], 4)} |")

    block("Raw (all member spans)", raw)
    block(f"Balanced (≤{cfg['max_members_per_ability']} spans/ability)", bal)
    L.append("\n*kNN purity = share of each point's k nearest neighbors carrying its label "
             "(self and spans from the same passage excluded as neighbors); "
             "ARI = agreement of label-blind k-means (k = true count) with the labels.*\n")
    path.write_text("\n".join(L) + "\n")


def exclude_abilities(V: dict, drop: set) -> dict:
    """Remove abilities from the members and the prototypes. Used for the CHC cross-listed
    duplicates (e.g. Reading Speed under both Grw and Gs): one construct under several
    labels, whose mutual neighbors would otherwise register as errors."""
    keep = ~np.isin(V["mem_uid"], list(drop))
    fkeep = ~np.isin(V["fp_uid"], list(drop))
    V = dict(V)
    for key in ("pts", "mem_uid", "mem_broad", "mem_count", "mem_pid"):
        if V.get(key) is not None:
            V[key] = V[key][keep]
    for key in ("fp_mat", "fp_uid", "fp_broad"):
        V[key] = V[key][fkeep]
    return V


def load_points(cfg: dict) -> dict:
    """Load member points + ability prototypes for the configured mode.

    mode == "joint" (Stage-5, method 2): each point is an independent source-conditioned
      span vector from ``p5_embed_joint.py`` — no prototype blend, so within-ability variance
      is genuine and the recovery test against CHC labels is not circular. Ability
      prototypes are *derived* as member centroids (display only; never fed back into
      the member coordinates).

    mode == "prototype" (display variant, not used in the paper): each member span is pulled toward its
      salience-weighted ability prototype, pt = L2norm(alpha*proto + (1-alpha)*span).
      alpha is a cohesion dial — it shrinks within-ability scatter by (1-alpha)^2 and
      bakes label identity into the coordinates, so use this for figures, not for the
      measured RQ2 / discriminant-validity claims.
    """
    mode = cfg.get("mode", "prototype").lower()

    if mode == "joint":
        vec_path = pathlib.Path(cfg["joint_vectors_path"])
        if not vec_path.exists():
            sys.exit(f"{vec_path} not found — run p5_embed_joint.py (CUDA GPU) first.")
        z = np.load(vec_path)
        pts = l2norm(z["member_matrix"].astype(np.float64))
        mem_uid = z["member_uid"].astype(str)
        mem_broad = z["member_broad"].astype(str)
        mem_count = z["member_count"].astype(np.int64) if "member_count" in z \
            else np.ones(len(mem_uid), np.int64)
        # ability prototype = L2-normalized centroid of its member points (display only)
        fp_uid = np.array(sorted(set(mem_uid)))
        fp_broad = np.array([mem_broad[mem_uid == u][0] for u in fp_uid])
        fp_mat = l2norm(np.stack([pts[mem_uid == u].mean(axis=0) for u in fp_uid]))
        print(f"loaded {len(mem_uid)} joint members across {len(fp_uid)} abilities "
              f"(dim={int(z['dim'])}); mode=joint (no prototype blend)")
        mem_pid = z["member_passage_id"].astype(str) if "member_passage_id" in z else None
        return {"mode": mode, "pts": pts, "mem_uid": mem_uid, "mem_broad": mem_broad,
                "mem_count": mem_count, "mem_pid": mem_pid, "fp_mat": fp_mat, "fp_uid": fp_uid,
                "fp_broad": fp_broad, "desc": "joint span[SEP]source, alpha=0"}

    # ── prototype-blend mode (display only) ──────────────────────────────────────────
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
    mem_count = z["member_count"].astype(np.int64)
    print(f"loaded {len(fp_uid)} prototypes, {len(mem_uid)} member spans (dim={int(z['dim'])})")

    alpha = float(cfg["alpha"])
    fp_lookup = {u: fp_mat[i] for i, u in enumerate(fp_uid)}
    missing_fp = sorted(set(mem_uid) - set(fp_uid))
    if missing_fp:
        print(f"[warn] {len(missing_fp)} abilities lack a prototype; their spans use alpha=0: {missing_fp}")
    blended = l2norm(np.stack([
        alpha * fp_lookup[u] + (1 - alpha) * mem_mat[i] if u in fp_lookup else mem_mat[i]
        for i, u in enumerate(mem_uid)
    ]))
    return {"mode": mode, "pts": blended, "mem_uid": mem_uid, "mem_broad": mem_broad,
            "mem_count": mem_count, "mem_pid": None, "fp_mat": fp_mat, "fp_uid": fp_uid,
            "fp_broad": fp_broad, "desc": f"prototype blend, alpha={alpha}"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4/5 — clustering & viz")
    ap.add_argument("config")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")

    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    V = load_points(cfg)
    excluded = sorted(cfg.get("exclude_abilities", []))
    if excluded:
        V = exclude_abilities(V, set(excluded))
        print(f"excluded {len(excluded)} abilities from all metrics and figures: {excluded}")
    blended = V["pts"]
    fp_mat, fp_uid, fp_broad = V["fp_mat"], V["fp_uid"], V["fp_broad"]
    mem_uid, mem_broad, mem_count = V["mem_uid"], V["mem_broad"], V["mem_count"]

    # optionally drop singleton broad strata (e.g. Gq) from cohesion metrics
    dropped: list[str] = []
    keep_mask = np.ones(len(mem_uid), bool)
    if cfg["drop_singleton_broad"]:
        abil_per_broad = defaultdict(set)
        for u, b in zip(fp_uid, fp_broad):
            abil_per_broad[b].add(u)
        dropped = sorted(b for b, a in abil_per_broad.items() if len(a) < 2)
        if dropped:
            keep_mask = ~np.isin(mem_broad, dropped)

    k = int(cfg["knn_k"])

    pts, br, ui = blended[keep_mask], mem_broad[keep_mask], mem_uid[keep_mask]
    seeds = int(cfg.get("kmeans_seeds", 10))
    pid = V["mem_pid"][keep_mask] if V["mem_pid"] is not None else None
    raw_m = cohesion_metrics(pts, br, ui, k, seeds, pid)

    bsel = balanced_subsample(ui, int(cfg["max_members_per_ability"]), int(cfg["random_seed"]))
    bal_m = cohesion_metrics(pts[bsel], br[bsel], ui[bsel], k, seeds,
                             pid[bsel] if pid is not None else None)

    # ── figures ──────────────────────────────────────────────────────────────────
    colors = palette(list(set(mem_broad)) + list(set(fp_broad)))
    dpi = int(cfg.get("figure_dpi", 300))
    xy, proj_used = project_2d(blended, cfg["projection"], int(cfg["random_seed"]),
                               min_dist=float(cfg.get("umap_min_dist", 0.0)),
                               n_neighbors=int(cfg.get("umap_n_neighbors", 15)))
    scatter_plot(xy, mem_broad, mem_uid, mem_count, colors, proj_used, out_dir / "scatter.png", dpi,
                 ellipse_std=float(cfg.get("scatter_ellipse_std", 2.0)))
    dendrogram_plot(fp_mat, fp_uid, fp_broad, colors, out_dir / "dendrogram.png", dpi)
    heatmap_plot(fp_mat, fp_uid, fp_broad, colors, out_dir / "similarity_heatmap.png", dpi)

    # ── metrics out ───────────────────────────────────────────────────────────────
    metrics = {"mode": V["mode"], "representation": V["desc"], "member_unit": "span",
               "projection": proj_used, "knn_k": k, "dropped_singleton_broad": dropped,
               "excluded_abilities": excluded,
               "raw": raw_m, "balanced": bal_m}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_md(out_dir / "metrics.md", cfg, raw_m, bal_m, proj_used, dropped, rep=V["desc"])

    print(f"\nraw: knn narrow={raw_m.get('knn_purity_uid')} broad={raw_m.get('knn_purity_broad')}  "
          f"ARI narrow={raw_m.get('ari_uid', {}).get('mean')} broad={raw_m.get('ari_broad', {}).get('mean')}")
    print(f"figures + metrics -> {out_dir}")


if __name__ == "__main__":
    main()
