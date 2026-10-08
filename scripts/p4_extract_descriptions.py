"""Stage 4 · Step A — unified per-passage ability-descriptor extraction (LLM).

Reads each ability's classifier-retained passages
(``data/processed/corpus/track_a/{uid}_filtered.jsonl``) and, in one structured LLM
call per passage, extracts all four fingerprint signals + the attribute/process
register (see src/data/description_schema.py). Output is one lean record per passage:

  data/processed/descriptions/track_a/{uid}_extracted.jsonl

Per-ability sample is bounded by ``extract_cap`` (config; default 100) and
``extract_floor`` (default 10) -- deliberately different numbers, not one "target
count", because flattening every ability to the same extracted-passage count would
erase the relative descriptor-richness signal Stage 5's clustering/density view depends
on. Above `extract_cap`, random subsample down to it (so no single descriptor-rich
ability dominates p4_build_fingerprint.py's shared cross-ability lexical background).
Between the floor and the cap, extracted AS-IS -- no padding, no subsampling, the
true retained count is preserved exactly. Below `extract_floor`, backfilled UP TO
the floor (never to the cap) with the highest-scoring SUB-threshold candidates from
``{uid}_scored.jsonl`` -- a safety net against classifier false negatives and
against an ability with 0 retained passages silently vanishing from extraction
entirely, without inflating it to look as rich as a well-populated ability. The
extraction LLM still makes its own ``on_target`` call per passage, so a backfilled
passage that's genuinely off-topic is simply dropped there, not laundered through.
Each output record carries ``extraction_source``: ``"retained"`` or
``"backfill_subthreshold"``. ``n_retained``/``coverage_tier`` in
``coverage_report.json`` are computed upstream by p3_filter_passages.py and are
unaffected by either bound -- they remain the true coverage measurement.

Reuses p3_annotate.py's inventory loader and its batch-with-per-item-fallback shape; the
prompt/schema live in description_schema.py (versioned). Runs against the same
OpenAI-compatible gateway as p3_annotate.py (OPENAI_API_KEY + OPENAI_BASE_URL in env);
no GPU needed.

Usage:
  uv run python scripts/p4_extract_descriptions.py configs/p4_extract.yaml --pilot --limit 10
  uv run python scripts/p4_extract_descriptions.py configs/p4_extract.yaml --ability Gr-FA
  uv run python scripts/p4_extract_descriptions.py configs/p4_extract.yaml --all [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time

import yaml

# scripts/ is sys.path[0] (sibling imports); add the repo root so `src` resolves too.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import p3_annotate as A  # noqa: E402  sibling script: reuse the inventory loader
from src.data import description_schema as schema  # noqa: E402  versioned prompt + parser

REQUIRED_KEYS = ("output_dir", "corpus_dir", "inventory_path", "model",
                 "batch_size", "temperature", "prompt_version_pin", "sleep")

# pilot set from Step 7 of the Track-A instructions (one per diverse stratum/coverage level)
PILOT_UIDS = ["Gr-FA", "Gf-I", "Gv-Vz", "Gwm-AC", "Gc-VL"]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def select_passages(filtered_path: pathlib.Path, scored_path: pathlib.Path,
                    cap: int, floor: int, seed: int) -> list[dict]:
    """Select passages for extraction.

    `cap` and `floor` are deliberately different numbers, not one "target count" --
    flattening every ability to the same extracted-passage count would erase the
    relative descriptor-richness signal that Stage 5's clustering/density view depends on
    (an ability with 3 retained passages must still look sparser than one with 46).

    - More than `cap` retained: random subsample down to `cap`, so no single
      descriptor-rich ability dominates the cross-ability lexical background in
      p4_build_fingerprint.py.
    - Between `floor` and `cap` retained: extracted AS-IS, no padding, no
      subsampling -- the true retained count is preserved exactly.
    - Fewer than `floor` retained: backfilled UP TO `floor` (never to `cap`) with the
      highest-scoring SUB-threshold candidates from `{uid}_scored.jsonl` -- a safety
      net against classifier false negatives and the silent-drop bug (an ability with
      0 retained previously vanished from extraction entirely), without inflating a
      genuinely sparse ability to look as rich as a well-populated one. The
      per-passage extraction LLM still makes its own on_target call, so a backfilled
      passage that's genuinely off-topic is simply dropped there, not laundered through.

    Every row is tagged `extraction_source`: "retained" or "backfill_subthreshold".
    """
    retained = read_jsonl(filtered_path)
    for r in retained:
        r["extraction_source"] = "retained"
    if len(retained) > cap:
        rng = random.Random(seed)
        return rng.sample(retained, cap)
    if len(retained) >= floor:
        return retained
    have = {r["passage_id"] for r in retained}
    candidates = [r for r in read_jsonl(scored_path)
                 if not r.get("retained") and r["passage_id"] not in have]
    candidates.sort(key=lambda r: r.get("prob_cad", 0.0), reverse=True)
    backfill = []
    for c in candidates[: floor - len(retained)]:
        c = dict(c)
        c["classifier_confidence"] = c.get("prob_cad")
        c["extraction_source"] = "backfill_subthreshold"
        backfill.append(c)
    return retained + backfill


def call_batch(client, model, temperature, ability, batch):
    sys_msg = schema.build_system_prompt(ability) + "\n\n" + schema.SCHEMA_INSTRUCTION
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": schema.build_user_prompt(batch)},
        ],
    )
    return schema.parse_response(resp.choices[0].message.content, batch)


def call_with_fallback(client, model, temperature, ability, batch):
    """Whole batch; on failure retry each item alone so one bad sentence can't sink the batch."""
    try:
        return call_batch(client, model, temperature, ability, batch), None
    except Exception as e:
        out: dict[int, dict] = {}
        for item in batch:
            try:
                out.update(call_batch(client, model, temperature, ability, [item]))
            except Exception:
                pass
        return out, str(e)


def extract_ability(client, cfg, ability, passages, out_path):
    model, temperature, bs = cfg["model"], cfg["temperature"], int(cfg["batch_size"])
    annotator = f"llm:{model}"
    dist = {"on_target": 0, "off_target": 0, "error": 0, "spans": 0}
    with out_path.open("w") as jf:
        for start in range(0, len(passages), bs):
            chunk = passages[start:start + bs]
            batch = [(start + j, p["text"]) for j, p in enumerate(chunk)]
            ann, err = call_with_fallback(client, model, temperature, ability, batch)
            if err:
                print(f"  [warn] batch {start}-{start + len(chunk) - 1} fell back "
                      f"(recovered {len(ann)}/{len(chunk)}): {err[:80]}")
            for j, p in enumerate(chunk):
                a = ann.get(start + j)
                if a is None:
                    a = schema.normalize_annotation({})
                    dist["error"] += 1
                    err_flag = "no_annotation"
                else:
                    err_flag = None
                    dist["on_target" if a["on_target"] else "off_target"] += 1
                    dist["spans"] += len(a["spans"])
                rec = {
                    "passage_id": p["passage_id"],
                    "source": p.get("source", ""),
                    "hybrid_score": p.get("hybrid_score"),
                    "classifier_confidence": p.get("classifier_confidence"),
                    "extraction_source": p.get("extraction_source", "retained"),
                    **a,
                    "prompt_version": schema.PROMPT_VERSION,
                    "schema_version": schema.SCHEMA_VERSION,
                    "annotator": annotator,
                    **({"error": err_flag} if err_flag else {}),
                }
                jf.write(json.dumps(rec) + "\n")
            print(f"  [{min(start + len(chunk), len(passages))}/{len(passages)}] "
                  + "  ".join(f"{k}={v}" for k, v in dist.items() if v))
            time.sleep(cfg["sleep"])
    return dist


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 Step A — unified per-passage extraction")
    ap.add_argument("config")
    ap.add_argument("--ability", default=None, help="single uid (e.g. Gr-FA)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pilot", action="store_true", help=f"the 5-ability pilot set: {PILOT_UIDS}")
    ap.add_argument("--limit", type=int, default=0, help="cap passages/ability (smoke test)")
    ap.add_argument("--overwrite", action="store_true", help="re-extract abilities that already have output")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        sys.exit(f"Config missing required keys: {missing}")
    if cfg["prompt_version_pin"] != schema.PROMPT_VERSION:
        sys.exit(f"prompt_version_pin={cfg['prompt_version_pin']} != PROMPT_VERSION={schema.PROMPT_VERSION}; "
                 "bump the config pin deliberately after a prompt change.")

    out_dir = pathlib.Path(cfg["output_dir"])
    corpus_dir = pathlib.Path(cfg["corpus_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    inv = A.load_inventory(pathlib.Path(cfg["inventory_path"]))

    if args.pilot:
        uids = PILOT_UIDS
    elif args.ability:
        uids = [args.ability]
    elif args.all:
        uids = sorted(p.name[:-len("_filtered.jsonl")] for p in corpus_dir.glob("*_filtered.jsonl"))
    else:
        sys.exit("Specify --pilot, --ability UID, or --all.")

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set (gateway via OPENAI_BASE_URL, as in p3_annotate.py).")
    from openai import OpenAI
    client = OpenAI()

    for uid in uids:
        if uid not in inv["abilities"]:
            print(f"[skip] unknown uid: {uid}")
            continue
        fpath = corpus_dir / f"{uid}_filtered.jsonl"
        if not fpath.exists():
            print(f"[skip] no filtered file: {fpath}")
            continue
        out_path = out_dir / f"{uid}_extracted.jsonl"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {uid}: output exists (use --overwrite)")
            continue
        spath = corpus_dir / f"{uid}_scored.jsonl"
        passages = select_passages(fpath, spath, int(cfg.get("extract_cap", 100)),
                                   int(cfg.get("extract_floor", 10)), int(cfg.get("extract_seed", 42)))
        n_backfill = sum(1 for p in passages if p["extraction_source"] == "backfill_subthreshold")
        if args.limit:
            passages = passages[: args.limit]
        if not passages:
            print(f"[{uid}] 0 passages — skipped")
            continue
        ability = inv["abilities"][uid]
        print(f"[{uid}] {ability['name']}: extracting {len(passages)} passages"
              + (f" ({n_backfill} backfilled sub-threshold)" if n_backfill else "")
              + (f" [LIMIT {args.limit}]" if args.limit else ""))
        dist = extract_ability(client, cfg, ability, passages, out_path)
        print(f"[{uid}] done -> {out_path.name}  {dist}")


if __name__ == "__main__":
    main()
