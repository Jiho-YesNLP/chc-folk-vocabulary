"""Print a frozen Stage-3 annotation pool for in-conversation labeling.

The in-conversation flow (a human, or Claude reading in the conversation itself, rather
than the OpenAI API path in p3_annotate.py):

  1. this script prints the frozen pool + the target ability's definition
  2. the annotator emits a labels JSON:  {uid: {index: label | [label, best_ability_uid]}}
  3. p3_apply_inconv_labels.py --set {set} --labels labels.json  writes Layer 2 and rebuilds
     the derived view + summary

Note p3_apply_inconv_labels.py fails closed unless EVERY index in the pool is labelled, so a
batch should cover whole abilities. Indices already judged are marked `*` here — they can
still be relabelled (a second rater judging the same items is how cross-annotator kappa is
computed; Layer 2 keeps each rater's file separate, so nothing is overwritten).

Usage:
  uv run python scripts/p3_show_pool.py --set random --ability Gf-RQ
  uv run python scripts/p3_show_pool.py --set random --ability Gf-RQ --unjudged-only
  uv run python scripts/p3_show_pool.py --set random --batch 3 --start 0     # next 3 abilities
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import p3_annotate as A


def show(layout: A.SetLayout, inv: dict, uid: str, unjudged_only: bool) -> int:
    p = layout.pool(uid)
    if not p.exists():
        print(f"[skip] no frozen pool: {p}")
        return 0
    text, shown = A.render_pool(layout, inv, uid, unjudged_only)
    print(text)
    return len(shown)


def main() -> None:
    ap = argparse.ArgumentParser(description="Print a frozen annotation pool for labeling")
    ap.add_argument("--set", choices=sorted(A.SETS), required=True)
    ap.add_argument("--ability", action="append", default=None, help="repeatable uid")
    ap.add_argument("--batch", type=int, default=0,
                    help="print the next N abilities that still have unjudged passages")
    ap.add_argument("--start", type=int, default=0, help="offset into that ability list")
    ap.add_argument("--unjudged-only", action="store_true",
                    help="hide already-judged indices (NOTE: p3_apply_inconv_labels.py needs "
                         "every index, so use this only to gauge remaining work)")
    ap.add_argument("--inventory", default="data/raw/chc_taxonomy.json")
    ap.add_argument("--root", default=A.ANNOTATION_ROOT)
    args = ap.parse_args()

    layout = A.SetLayout(args.root, args.set)
    inv = A.load_inventory(pathlib.Path(args.inventory))

    if args.ability:
        uids = args.ability
    elif args.batch:
        pending = [u for u in layout.uids()
                   if len(A.judged_pids(layout, u)) < len(A.read_pool(layout.pool(u)))]
        uids = pending[args.start: args.start + args.batch]
        done = len(layout.uids()) - len(pending)
        print(f"# {len(pending)} abilities still incomplete ({done} done); "
              f"showing {len(uids)} from offset {args.start}")
    else:
        sys.exit("Specify --ability UID (repeatable) or --batch N.")

    total = sum(show(layout, inv, u, args.unjudged_only) for u in uids)
    print(f"\n# {total} passages shown across {len(uids)} abilities")
    print("# label -> {uid: {index: \"label\" | [\"folk_description\", \"<best_uid>\"]}}")
    print(f"# then: uv run python scripts/p3_apply_inconv_labels.py --set {args.set} "
          f"--labels labels.json --annotator <annotator-id> --prompt-version v2")


if __name__ == "__main__":
    main()
