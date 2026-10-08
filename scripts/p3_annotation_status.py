"""Report annotation progress per set, computed from disk (never a tracking file).

Completion is derived: an ability is done when every passage in its frozen pool has a
Layer-2 judgment. So this is always correct after any interruption — there is no state to
keep in sync. Use it to resume: it prints what is left and the exact next command.

Usage:
  uv run python scripts/p3_annotation_status.py                 # both sets, summary + remaining
  uv run python scripts/p3_annotation_status.py --set random    # one set
  uv run python scripts/p3_annotation_status.py --set random --verbose   # per-ability table
"""

from __future__ import annotations

import argparse

import p3_annotate as A


def rows_for(layout: A.SetLayout) -> list[tuple[str, int, int]]:
    out = []
    for uid in layout.uids():
        n = len(A.read_pool(layout.pool(uid)))
        j = len(A.judged_pids(layout, uid))
        out.append((uid, j, n))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotation progress from disk")
    ap.add_argument("--set", choices=sorted(A.SETS), default=None, help="default: both")
    ap.add_argument("--root", default=A.ANNOTATION_ROOT)
    ap.add_argument("--verbose", action="store_true", help="per-ability table")
    args = ap.parse_args()

    names = [args.set] if args.set else sorted(A.SETS)
    for name in names:
        L = A.SetLayout(args.root, name)
        rows = rows_for(L)
        done = [r for r in rows if r[1] >= r[2]]
        pending = [r for r in rows if r[1] < r[2]]
        judged = sum(r[1] for r in rows)
        total = sum(r[2] for r in rows)
        print(f"\n=== {name} ===")
        print(f"abilities: {len(done)}/{len(rows)} complete   "
              f"passages: {judged}/{total} labeled   gaps: {total - judged}")
        if args.verbose:
            for uid, j, n in rows:
                bar = "done" if j >= n else f"{n - j} left"
                print(f"  {uid:9s} {j:3d}/{n:<3d}  {bar}")
        if pending:
            nxt = ", ".join(u for u, _, _ in pending[:5])
            print(f"next: {nxt}{' …' if len(pending) > 5 else ''}")
            print(f"  uv run python scripts/p3_show_pool.py --set {name} --batch 3 --unjudged-only")


if __name__ == "__main__":
    main()
