#!/usr/bin/env python3
"""Disk-state survey for Stage 4 Step A in-conversation extraction.

No separate progress-tracking file -- the data files ARE the source of truth, so
status can never drift from what's actually on disk. For each inventory ability,
compares the existing {uid}_extracted.jsonl (if any) against what
p4_extract_descriptions.select_passages() would currently select (same cap/floor/seed,
configs/p4_extract.yaml), and against the expected in-conversation annotator.
An ability is "done" only if the line count matches AND every record's annotator
is the expected one (so a run by another model, or one extracted under different
cap/floor settings, shows up as "stale", not "done").

Defaults survey the reported run (data/processed/descriptions/track_a_opus55,
llm:claude-opus-5.5).

Usage:
  uv run python scripts/p4_extraction_status.py                  # summary + per-uid table
  uv run python scripts/p4_extraction_status.py --pending         # just the uids left to do
  uv run python scripts/p4_extraction_status.py --desc-dir DIR    # survey another run
  uv run python scripts/p4_extraction_status.py --annotator ID    # expect a different annotator
"""
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import p4_extract_descriptions as E  # noqa: E402
import p3_annotate as A  # noqa: E402

# The model that produced the reported in-conversation extraction. Override with
# --annotator / --desc-dir when surveying a run produced by a different model.
EXPECTED_ANNOTATOR = "llm:claude-opus-5.5"
DEFAULT_DESC_DIR = "data/processed/descriptions/track_a_opus55"


def main():
    pending_only = "--pending" in sys.argv
    expected_annotator = EXPECTED_ANNOTATOR
    if "--annotator" in sys.argv:
        expected_annotator = sys.argv[sys.argv.index("--annotator") + 1]
    desc_rel = DEFAULT_DESC_DIR
    if "--desc-dir" in sys.argv:
        desc_rel = sys.argv[sys.argv.index("--desc-dir") + 1]

    cfg = yaml.safe_load((ROOT / "configs/p4_extract.yaml").read_text())
    cap, floor, seed = (int(cfg.get("extract_cap", 100)), int(cfg.get("extract_floor", 10)),
                       int(cfg.get("extract_seed", 42)))

    inv = A.load_inventory(ROOT / "data/raw/chc_taxonomy.json")
    uids = sorted(inv["abilities"])

    corpus_dir = ROOT / cfg["corpus_dir"]
    desc_dir = ROOT / desc_rel

    rows = []
    for uid in uids:
        filtered_path = corpus_dir / f"{uid}_filtered.jsonl"
        scored_path = corpus_dir / f"{uid}_scored.jsonl"
        if not filtered_path.exists():
            rows.append((uid, None, None, "no-filtered-file"))
            continue
        expected = E.select_passages(filtered_path, scored_path, cap, floor, seed)
        expected_n = len(expected)

        out_path = desc_dir / f"{uid}_extracted.jsonl"
        if not out_path.exists():
            rows.append((uid, expected_n, 0, "missing"))
            continue
        existing = E.read_jsonl(out_path)
        actual_n = len(existing)
        annotators = {r.get("annotator") for r in existing}
        if actual_n == expected_n and annotators == {expected_annotator}:
            status = "done"
        elif actual_n == 0:
            status = "missing"
        else:
            status = "stale"
        rows.append((uid, expected_n, actual_n, status, annotators))

    if pending_only:
        for r in rows:
            if r[3] != "done":
                print(r[0])
        return

    counts = {}
    for r in rows:
        counts[r[3]] = counts.get(r[3], 0) + 1
    print(f"{len(uids)} abilities: " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print()
    print(f"{'uid':<10} {'expected':>8} {'actual':>8}  status")
    for r in rows:
        uid, expected_n, actual_n, status = r[0], r[1], r[2], r[3]
        ann = r[4] if len(r) > 4 and status != "done" else ""
        ann_str = f"  ({', '.join(sorted(a or 'None' for a in ann))})" if ann else ""
        print(f"{uid:<10} {str(expected_n):>8} {str(actual_n):>8}  {status}{ann_str}")


if __name__ == "__main__":
    main()
