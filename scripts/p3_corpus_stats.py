"""Stages 2–3 — corpus funnel and per-ability retention (manuscript corpus statistics tables).

CPU/local. Writes out_dir/{report.json, report.md} with:

  funnel      subreddits (the configured list) and raw comments (index preflight manifests,
              merged), indexed passages and comments contributing >= 1 passage (index *_meta.jsonl), retrieval records and unique (ability, passage)
              candidates, frozen annotation-set passages, retained folk_description passages.
  retention   per-ability retained counts from the filter's coverage_report.json: abilities with
              >= 1 retained passage, median, min/max, abilities below the extraction floor.
  sources     distinct subreddits among retrieval candidates and among retained passages.

The index meta files are written by p2_build_index.py on the GPU machine; run there for the index rows. Rows whose inputs are
absent are reported as unavailable rather than guessed.

Usage:
  uv run python scripts/p3_corpus_stats.py configs/p3_corpus_stats.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import yaml


def count_lines(path: pathlib.Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    cfg = yaml.safe_load(pathlib.Path(ap.parse_args().config).read_text())
    out_dir = pathlib.Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rep: dict = {"funnel": {}, "retention": {}, "sources": {}, "raw": {}, "unavailable": []}
    F = rep["funnel"]

    # raw comments: the index preflight manifests (p2_build_index.py --inspect), merged by
    # subreddit; restricted to, and checked against, the configured subreddit list.
    subs = {x.lower() for x in yaml.safe_load(pathlib.Path(cfg["subreddits_config"]).read_text())["subreddits"]}
    index_dir = pathlib.Path(cfg["index_dir"])
    recs: dict[str, dict] = {}
    for mp in cfg["manifest_paths"]:
        mp = pathlib.Path(mp)
        if mp.exists():
            for name, rec in json.loads(mp.read_text())["files"].items():
                sub = name.split("_comments")[0].lower()
                if sub in subs and rec.get("status") == "ok":
                    recs.setdefault(sub, rec)
    F["subreddits"] = len(subs)
    missing = sorted(subs - set(recs))
    if missing:
        rep["unavailable"].append(f"comments_raw (no manifest record for {len(missing)}: {', '.join(missing)})")
    else:
        years = sorted({int(y) for r in recs.values() for y in r["year_counts"]})
        per = {s_: r["total_comments"] for s_, r in recs.items()}
        F["comments_raw"] = sum(per.values())
        rep["raw"] = {"years": [years[0], years[-1]],
                      "min": [min(per, key=per.get), min(per.values())],
                      "max": [max(per, key=per.get), max(per.values())]}
    metas = sorted(index_dir.glob("*_meta.jsonl")) if index_dir.exists() else []
    if metas:
        n, comments = 0, set()
        for m in metas:
            for line in open(m):
                r = json.loads(line)
                n += 1
                comments.add((m.name, r.get("source_id")))
        F["comments_indexed"] = len(comments)
        F["passages_indexed"] = n
        rep["sources"]["indexed"] = len(metas)
    else:
        rep["unavailable"] += ["comments_indexed", "passages_indexed"]

    # retrieval candidates
    records, pairs, cand_sources = 0, 0, set()
    for f in sorted(pathlib.Path(cfg["retrieval_dir"]).glob("*_raw.jsonl")):
        pids = set()
        for line in open(f):
            r = json.loads(line)
            records += 1
            pids.add(r["passage_id"])
            cand_sources.add(r["source"])
        pairs += len(pids)
    F["retrieval_records"] = records
    F["retrieval_candidates_unique"] = pairs

    # frozen annotation sets
    for s in cfg["annotation_sets"]:
        F[f"annotated_{s}"] = sum(count_lines(p) for p in pathlib.Path(cfg["annotation_root"], s, "pools").glob("*.jsonl"))

    # retention (filter output)
    cov = json.loads(pathlib.Path(cfg["coverage_report_path"]).read_text())
    n_ret = {u: v["n_retained"] for u, v in cov.items()}
    floor = int(cfg["extract_floor"])
    F["retained"] = sum(n_ret.values())
    vals = list(n_ret.values())
    rep["retention"] = {"n_abilities": len(vals),
                        "with_retained": sum(v > 0 for v in vals),
                        "median": statistics.median(vals), "min": min(vals), "max": max(vals),
                        "below_floor": {u: v for u, v in sorted(n_ret.items()) if v < floor},
                        "extract_floor": floor,
                        "min_ability": min(n_ret, key=n_ret.get), "max_ability": max(n_ret, key=n_ret.get)}
    ret_sources = set()
    for v in cov.values():
        ret_sources |= set(v["source_breakdown"])
    rep["sources"].update({"retrieval_candidates": len(cand_sources), "retained": len(ret_sources)})

    (out_dir / "report.json").write_text(json.dumps(rep, indent=2))
    R = rep["retention"]
    L = ["# Corpus statistics\n", "## Funnel\n", "| row | count |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in F.items()]
    if rep["raw"]:
        L += [f"\nRaw comments span {rep['raw']['years'][0]}–{rep['raw']['years'][1]}; per subreddit from "
              f"{rep['raw']['min'][1]:,} (r/{rep['raw']['min'][0]}) to {rep['raw']['max'][1]:,} (r/{rep['raw']['max'][0]}).\n"]
    L += [f"\nUnavailable here: {'; '.join(rep['unavailable'])}\n" if rep["unavailable"] else "",
          "## Retention per ability\n",
          f"- abilities with >= 1 retained passage: {R['with_retained']} / {R['n_abilities']}",
          f"- median retained: {R['median']}; min {R['min']} ({R['min_ability']}), max {R['max']} ({R['max_ability']})",
          f"- below the extraction floor (< {floor}): {len(R['below_floor'])} {R['below_floor']}",
          f"\n## Sources\n\n- distinct subreddits among retrieval candidates: {rep['sources']['retrieval_candidates']}",
          f"- distinct subreddits among retained passages: {rep['sources']['retained']}"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
