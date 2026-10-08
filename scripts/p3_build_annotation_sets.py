"""Build both Stage-3 annotation sets per CHC ability, in one pass.

Stage 3 uses two annotation sets. Each has one job and one fixed depth applied to every
ability — there is no per-ability adaptivity, no stratification, and no branch where one
ability is annotated more deeply than another:

  RANDOM set  — a uniform sample of the top-`pool_top` retrieval pool. This is the
    *measurement* instrument: per-ability descriptor prevalence (the coverage analysis) and
    the held-out precision number for the relevance classifier.

  RANKED set  — the top-`depth` passages by hybrid score, skipping anything the random
    set drew. This is the *training* set for the relevance classifier. It is rank-biased
    by construction and is never read as a coverage estimate.

**Order is the design.** The random set is drawn FIRST, from an untouched pool, with no
exclusions of any kind; the ranked set then takes what is left. Because nothing is ever
withheld from the random draw, it is a uniform sample of the pool for every ability, the
rule is identical across all 84, and the number carrying the study's claims needs no
correction. Reversing the order would reintroduce exactly the depth-vs-outcome confound
this design exists to remove. Building both here, from one read of the pool, makes the
order impossible to get wrong.

Both sets derive from a single `annotate.select_candidates(..., top=pool_top)` call per
ability (max-hybrid per passage_id -> exact normalized-text dedup -> technical-label
lexicon pre-filter -> rank by hybrid score), so both are drawn from byte-identical
material to what retrieval and the filter saw. The random draw is salted per ability:
independent across abilities, exactly reproducible within one.

Pools are frozen on first write (Layer 1, never re-sampled) via annotate.load_or_freeze_pool,
so re-running is a no-op that reports what already exists.

Output (one JSONL per ability per set), under the annotation tree:
  {root}/random/pools/{uid}.jsonl   {root}/ranked/pools/{uid}.jsonl
  one row per passage: index, passage_id, text, source, seed_id, hybrid_score, pool_version.
  Ability metadata is NOT stored here; look it up by uid in the inventory.

Usage:
  uv run python scripts/p3_build_annotation_sets.py --all
  uv run python scripts/p3_build_annotation_sets.py --all --dry-run     # report sizes, write nothing
  uv run python scripts/p3_build_annotation_sets.py --ability Gv-PI
"""

from __future__ import annotations

import argparse
import pathlib
import random
import sys

import p3_annotate as A  # same scripts/ dir; reuse the proven selection + Layer-1 I/O


def build_sets(raw_path: pathlib.Path, label_re, n_random: int, n_ranked: int,
               seed: str, pool_top: int, uid: str) -> tuple[list[dict], list[dict], dict]:
    """Return (random_pool, ranked_pool, stats) from ONE read of the ability's raw file.

    Random first from the untouched pool, then ranked from what remains. The two pools are
    disjoint by construction, so no passage is ever annotated as part of both sets.
    """
    pool, stats = A.select_candidates(raw_path, label_re, pool_top)

    rng = random.Random(f"{seed}:{uid}")
    shuffled = list(pool)
    rng.shuffle(shuffled)
    rnd = shuffled[:n_random]

    taken = {c["passage_id"] for c in rnd}
    # `pool` is already sorted by hybrid score descending, so this is the top-N by score.
    ranked = [c for c in pool if c["passage_id"] not in taken][:n_ranked]

    stats.update({
        "n_random": len(rnd),
        "n_ranked": len(ranked),
        "displaced": sum(1 for c in pool[:n_ranked] if c["passage_id"] in taken),
    })
    return rnd, ranked, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze the random + ranked annotation pools")
    ap.add_argument("--ability", default=None, help="single uid, e.g. Gv-PI")
    ap.add_argument("--all", action="store_true", help="every ability with a *_raw.jsonl")
    ap.add_argument("--inventory", default="data/raw/chc_taxonomy.json")
    ap.add_argument("--raw-dir", default="data/processed/retrieval/track_a")
    ap.add_argument("--root", default=A.ANNOTATION_ROOT,
                    help="annotation tree root; pool paths derive from it")
    ap.add_argument("--n-random", type=int, default=50, help="uniform-random N per ability")
    ap.add_argument("--n-ranked", type=int, default=50, help="top-N by hybrid score per ability")
    ap.add_argument("--pool-top", type=int, default=1000,
                    help="both sets draw from the top-K by hybrid score (the pool the filter scores)")
    ap.add_argument("--seed", default="r2607", help="RNG seed for the random draw, salted per ability")
    ap.add_argument("--dry-run", action="store_true", help="report only; freeze nothing")
    args = ap.parse_args()

    inv = A.load_inventory(pathlib.Path(args.inventory))
    label_re = A.build_label_matcher(inv["labels"])
    raw_dir = pathlib.Path(args.raw_dir)
    lay = A.layouts(args.root)
    if not args.dry_run:
        for L in lay.values():
            L.mkdirs()

    if args.all:
        uids = sorted(p.name[: -len("_raw.jsonl")] for p in raw_dir.glob("*_raw.jsonl"))
    elif args.ability:
        uids = [args.ability]
    else:
        sys.exit("Specify --ability UID or --all.")

    froze = kept = short = 0
    tot_random = tot_ranked = tot_displaced = 0
    for uid in uids:
        if uid not in inv["abilities"]:
            print(f"[skip] unknown uid: {uid}")
            continue
        raw_path = raw_dir / f"{uid}_raw.jsonl"
        rpath = lay["random"].pool(uid)
        kpath = lay["ranked"].pool(uid)

        if rpath.exists() and kpath.exists():
            rnd, ranked = A.read_pool(rpath), A.read_pool(kpath)
            print(f"[{uid}] frozen already: random={len(rnd)} ranked={len(ranked)}")
            kept += 1
            tot_random += len(rnd)
            tot_ranked += len(ranked)
            continue
        if not raw_path.exists():
            print(f"[skip] no raw file: {raw_path}")
            continue

        rnd, ranked, stats = build_sets(raw_path, label_re, args.n_random, args.n_ranked,
                                        args.seed, args.pool_top, uid)
        print(f"[{uid}] pool={stats['pool_size']} (raw={stats['raw_records']:,} "
              f"prefiltered={stats['prefiltered_label_hits']:,}) "
              f"-> random={stats['n_random']} ranked={stats['n_ranked']} "
              f"(displaced {stats['displaced']} from the top-{args.n_ranked})")

        if len(rnd) < args.n_random or len(ranked) < args.n_ranked:
            print(f"  [warn] {uid}: pool too small for the requested depths")
            short += 1

        if not args.dry_run:
            # Freeze random first: if this run dies between the two writes, the next run
            # rebuilds an identical ranked pool from the same frozen random draw.
            A.load_or_freeze_pool(rpath, lambda: (rnd, stats))
            A.load_or_freeze_pool(kpath, lambda: (ranked, stats))
        froze += 1
        tot_random += len(rnd)
        tot_ranked += len(ranked)
        tot_displaced += stats["displaced"]

    verb = "would freeze" if args.dry_run else "froze"
    print(f"\n{verb} {froze} abilities ({kept} already frozen)"
          + (f", {short} short of the requested depths" if short else ""))
    print(f"random total={tot_random}  ranked total={tot_ranked}  "
          f"displaced from top-{args.n_ranked}={tot_displaced}")


if __name__ == "__main__":
    main()
