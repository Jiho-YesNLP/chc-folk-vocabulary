"""Stage 5 · Step A — source-conditioned joint embedding of ability descriptors.

GPU/BGE-M3. The honest, label-free input for Stage-5 clustering, alignment, and the
discriminant analysis. Unlike ``p4_build_fingerprint.py``'s ``fingerprint_vectors.npz``
(which blends each member span toward a per-ability *prototype* — a label-derived
vector that mechanically shrinks within-ability variance and bakes CHC identity into
the coordinates), this emits **one independent vector per (span, source sentence)**:

    vec(i) = BGE-M3( "<span>  [SEP]  <source_sentence>" )   # L2-normalized dense

No prototype is injected, so within-ability spread is whatever the data gives and a
label-blind clustering of these points is a genuine recovery test of CHC structure
(RQ2) — and FA/FI overlap surfaces honestly for the discriminant analysis, instead of
being hidden by a pull toward each ability's own mean.

The span is the distilled descriptor phrase (in focus); the source sentence is the
one-sentence passage it was extracted from (disambiguating context — "sharp" in a
recall passage vs. a reasoning passage separates by frame). No surrounding sentences
are introduced, consistent with the pipeline-wide single-sentence passage unit.

Only ``on_target`` extraction records feed the output. Records whose source sentence
is missing from BOTH the filtered and scored corpus are skipped (cannot be conditioned).

Output ``joint_vectors.npz`` (member-level only — ability "prototypes" are derived
downstream as member centroids, never embedded separately):
  member_matrix      (M, dim) float32, L2-normalized joint vectors
  member_uid         (M,)     ability uid (e.g. Gr-FA)        — ground-truth label
  member_broad       (M,)     broad stratum (e.g. Gr)         — ground-truth label
  member_span        (M,)     normalized span text (dedup key / metadata)
  member_passage_id  (M,)     source passage id
  member_count       (M,)     # exact-duplicate joint strings collapsed into this point
  member_conf        (M,)     mean Stage-3 classifier confidence of the source passage(s)
  dim, encoder_model, template, n_skipped_no_source

Needs a CUDA GPU. Usage:
  uv run python scripts/p5_embed_joint.py configs/p5_joint_embed.yaml [--ability Gr-FA] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import p3_annotate as A  # noqa: E402  inventory loader

REQUIRED_KEYS = ("output_path", "corpus_dir", "descriptions_dir", "inventory_path",
                 "encoder_model", "device", "batch_size", "template")


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _norm(s: str) -> str:
    return " ".join((s or "").split()).lower()


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 Step A — source-conditioned joint embedding")
    ap.add_argument("config")
    ap.add_argument("--ability", default=None, help="single uid (e.g. Gr-FA); for spot checks")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")

    out_path = pathlib.Path(cfg["output_path"])
    if out_path.exists() and not args.overwrite:
        sys.exit(f"{out_path} exists; pass --overwrite.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_dir = pathlib.Path(cfg["corpus_dir"])
    desc_dir = pathlib.Path(cfg["descriptions_dir"])
    template = cfg["template"]
    min_span_chars = int(cfg.get("min_span_chars", 2))

    inv = A.load_inventory(pathlib.Path(cfg["inventory_path"]))

    if args.ability:
        uids = [args.ability]
    else:
        uids = sorted(p.name[:-len("_extracted.jsonl")] for p in desc_dir.glob("*_extracted.jsonl"))

    # ── gather (uid, joint_string) members, deduped on the exact joint string ────────
    # acc[(uid, joint_str)] = {"span","passage_id","broad","count","conf_sum"}
    acc: dict[tuple, dict] = {}
    order: list[tuple] = []          # preserve first-seen order for reproducibility
    n_skipped_no_source = 0
    n_on_target = 0
    for uid in uids:
        if uid not in inv["abilities"]:
            print(f"[skip] unknown uid: {uid}")
            continue
        ext_path = desc_dir / f"{uid}_extracted.jsonl"
        filt_path = corpus_dir / f"{uid}_filtered.jsonl"
        if not ext_path.exists():
            print(f"[skip] no extraction: {ext_path}")
            continue
        text_by_id = {r["passage_id"]: r["text"] for r in read_jsonl(filt_path)} if filt_path.exists() else {}
        # scored.jsonl is the superset (every candidate, retained or not), so it is also
        # the only place backfill_subthreshold passages appear. Fill gaps from it rather
        # than only substituting when `filtered` is wholly empty: an ability can have a
        # populated filtered file and *still* have backfilled passages missing from it,
        # in which case those spans were silently dropped (Gr-FX lost its only span this
        # way, disappearing from the matrix entirely). Matches p4_build_fingerprint.py,
        # which prefers scored.jsonl for the same reason.
        scored_path = corpus_dir / f"{uid}_scored.jsonl"
        if scored_path.exists():
            for r in read_jsonl(scored_path):
                text_by_id.setdefault(r["passage_id"], r["text"])
        broad = inv["abilities"][uid]["broad_stratum"]
        for r in read_jsonl(ext_path):
            if not r.get("on_target"):
                continue
            n_on_target += 1
            source = (text_by_id.get(r["passage_id"]) or "").strip()
            if not source:
                n_skipped_no_source += 1
                continue
            conf = float(r.get("classifier_confidence") or 0.0)
            for s in r.get("spans", []):
                span = (s.get("text") or "").strip()
                if len(span) < min_span_chars:
                    continue
                joint = template.format(span=span, source=source)
                key = (uid, joint)
                rec = acc.get(key)
                if rec is None:
                    acc[key] = {"span": _norm(span), "passage_id": r["passage_id"],
                                "broad": broad, "count": 1, "conf_sum": conf}
                    order.append(key)
                else:
                    rec["count"] += 1
                    rec["conf_sum"] += conf

    if not order:
        sys.exit("No (span, source) members gathered — nothing to embed.")
    joint_strings = [k[1] for k in order]
    print(f"on_target records seen: {n_on_target}  |  members (deduped joint strings): {len(order)}  "
          f"|  skipped (no source sentence): {n_skipped_no_source}")

    # ── single BGE-M3 pass over the joint strings ────────────────────────────────────
    from src.models.encoder import BGEM3Encoder
    enc = BGEM3Encoder(model_name=cfg["encoder_model"], device=cfg["device"])
    dense = enc.encode(joint_strings, batch_size=int(cfg["batch_size"]))["dense"]
    dense = np.asarray(dense, dtype=np.float32)
    # encoder returns L2-normalized dense; re-normalize defensively.
    dense /= np.clip(np.linalg.norm(dense, axis=1, keepdims=True), 1e-12, None)
    dim = dense.shape[1]

    member_uid = np.array([k[0] for k in order])
    member_broad = np.array([acc[k]["broad"] for k in order])
    member_span = np.array([acc[k]["span"] for k in order])
    member_pid = np.array([acc[k]["passage_id"] for k in order])
    member_count = np.array([acc[k]["count"] for k in order], dtype=np.int64)
    member_conf = np.array([acc[k]["conf_sum"] / acc[k]["count"] for k in order], dtype=np.float32)

    np.savez_compressed(
        out_path,
        dim=np.int64(dim),
        encoder_model=str(cfg["encoder_model"]),
        template=str(template),
        n_skipped_no_source=np.int64(n_skipped_no_source),
        member_matrix=dense,
        member_uid=member_uid,
        member_broad=member_broad,
        member_span=member_span,
        member_passage_id=member_pid,
        member_count=member_count,
        member_conf=member_conf,
    )
    n_uid = len(set(member_uid.tolist()))
    n_broad = len(set(member_broad.tolist()))
    print(f"[joint] {len(order)} members across {n_uid} abilities / {n_broad} broad strata "
          f"(dim={dim}) -> {out_path}")


if __name__ == "__main__":
    main()
