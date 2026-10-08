"""Merge in-conversation annotation labels with their frozen candidate pools.

Companion to `p3_annotate.py --freeze-only`. The annotator (a human or Claude working
in-conversation, not the OpenAI API) reads each frozen pool
({root}/{set}/pools/{uid}.jsonl) and writes a labels file;
this script writes the Layer-2 judgment file
(``{set_dir}/{uid}/{annotator_id}@{prompt_version}.jsonl``) and rebuilds the derived lean view from every
annotator's Layer-2 file for that uid, then rebuilds that set's summary CSV.

``--set`` picks which frozen pool is being labelled and where the output goes; it is the
same switch p3_annotate.py takes.

Labels file (JSON): one object, keyed by uid; each value maps the candidate index
(as a string) to either a label string, or a [label, best_ability_uid] pair for
folk_description. best_ability_uid must be a composite uid (e.g. "Gr-FI"); use the
target's own uid for on-target descriptors, a neighbour's uid for a neighbour false positive.

  {
    "Gr-FI": {
      "0": ["folk_description", "Gr-FI"],
      "1": "off_topic",
      "2": ["folk_description", "Gr-FA"],
      "3": "incidental_mention"
    }
  }

Every index 0..N-1 present in the pool must appear in the labels for that uid.

Usage:
  uv run python scripts/p3_apply_inconv_labels.py --set random --labels labels.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

import p3_annotate as A  # reuse the Layer-1/Layer-2 I/O + write_summary

VALID = {"folk_description", "incidental_mention", "off_topic"}


def parse_entry(entry) -> tuple[str, str | None]:
    """Normalize a labels entry to (label, best_ability_uid_or_None)."""
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, (list, tuple)) and entry:
        label = entry[0]
        bac = entry[1] if len(entry) > 1 else None
        return label, bac
    raise ValueError(f"bad labels entry: {entry!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply in-conversation labels to pools")
    ap.add_argument("--labels", required=True, help="JSON: {uid: {index: label|[label,uid]}}")
    ap.add_argument("--inventory", default="data/raw/chc_taxonomy.json")
    ap.add_argument("--root", default=A.ANNOTATION_ROOT,
                    help="annotation tree root; all set paths derive from it")
    ap.add_argument("--set", choices=sorted(A.SETS), required=True,
                    help="which annotation set these labels belong to")

    ap.add_argument("--annotator", default="claude:in-conversation")
    ap.add_argument("--prompt-version", default="v1", help="Layer-2 provenance")
    ap.add_argument("--gaps-only", action="store_true",
                    help="require labels only for pool indices no annotator has judged yet, "
                         "and MERGE with this annotator's existing judgments instead of "
                         "requiring the whole pool. Use to resume a partially-labelled set.")
    args = ap.parse_args()

    inv = A.load_inventory(pathlib.Path(args.inventory))
    code_set = set(inv["abilities"])
    layout = A.SetLayout(args.root, args.set)
    layout.mkdirs()

    labels = json.loads(pathlib.Path(args.labels).read_text())
    errors: list[str] = []
    written = 0

    for uid, idx_map in labels.items():
        if uid not in inv["abilities"]:
            errors.append(f"{uid}: unknown uid")
            continue
        ppath = layout.pool(uid)
        if not ppath.exists():
            errors.append(f"{uid}: no frozen pool {ppath.name}")
            continue
        annotator_id, annotator_type = A.normalize_annotator(args.annotator)
        pool = A.read_pool(ppath)
        cands = {c["index"]: c for c in pool}
        # which indices must this labels file cover?
        if args.gaps_only:
            judged = A.judged_pids(layout, uid)
            required = {c["index"] for c in pool if c["passage_id"] not in judged}
        else:
            required = set(cands)
        missing = sorted(required - {int(k) for k in idx_map})
        if missing:
            errors.append(f"{uid}: missing labels for indices {missing}")
            continue
        extra = sorted({int(k) for k in idx_map} - set(cands))
        if extra:
            errors.append(f"{uid}: labels for indices not in pool {extra}")
            continue

        now = dt.datetime.now(dt.timezone.utc).isoformat()
        recs = []
        for k, entry in idx_map.items():
            i = int(k)
            if i not in cands:
                errors.append(f"{uid}: label index {i} not in pool")
                continue
            label, bac = parse_entry(entry)
            if label not in VALID:
                errors.append(f"{uid}[{i}]: invalid label {label!r}")
                continue
            if label == "folk_description":
                if bac is None:
                    errors.append(f"{uid}[{i}]: folk_description needs a best_ability uid")
                    continue
                if bac not in code_set:
                    errors.append(f"{uid}[{i}]: best_ability {bac!r} not a known uid")
                    continue
            else:
                bac = None
            c = cands[i]
            recs.append({
                "pool_id": uid, "pool_version": c["pool_version"], "passage_id": c["passage_id"],
                "annotator_id": annotator_id, "annotator_type": annotator_type,
                "prompt_version": args.prompt_version,
                "label": label, "best_ability_code": bac, "confidence": None,
                "note": "", "timestamp": now,
            })

        if errors:
            continue  # fail closed; don't write a partial/contaminated file
        jpath = layout.judgment(uid, annotator_id, args.prompt_version)
        n_new = len(recs)
        n_kept = 0
        if jpath.exists():
            # Merge rather than overwrite so a resumed run keeps this annotator's earlier
            # judgments (e.g. carried-forward labels). New rows win a passage_id collision.
            fresh = {r["passage_id"] for r in recs}
            prior = [r for r in A.read_judgments(jpath) if r["passage_id"] not in fresh]
            n_kept = len(prior)
            recs = prior + recs
        A.write_judgments(jpath, recs)
        merged = A.materialize_lean(layout, uid)
        written += 1
        print(f"[{uid}] +{n_new} new, {n_kept} kept -> {jpath.relative_to(layout.base)} "
              f"-> view {len(merged)}")

    if errors:
        print("\nERRORS (no files written for affected uids):", file=sys.stderr)
        for e in errors:
            print("  " + e, file=sys.stderr)
        sys.exit(1)

    A.write_summary(layout)
    print(f"\nDone: {written} abilities annotated ({args.set} set).")


if __name__ == "__main__":
    main()
