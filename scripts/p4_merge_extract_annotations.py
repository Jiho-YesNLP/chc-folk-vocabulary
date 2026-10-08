#!/usr/bin/env python3
"""Merge hand-authored in-conversation annotations with carry-through fields from the
SAME cap/floor/seed-bounded candidate selection p4_extract_descriptions.py would have
sent to the LLM (select_passages(), config: configs/p4_extract.yaml) --
preserving its order. Used for Stage 4 Step A in-conversation extraction so the
annotator only emits the per-passage analysis (keyed by passage_id) and never has
to echo source/hybrid_score/classifier_confidence/extraction_source.

Usage:
  p4_merge_extract_annotations.py <uid> <annotations.json> --annotator <id> [--resume-from N]

--annotator : REQUIRED, records what produced these annotations, e.g.
  "llm:claude-opus-5.5" or "llm:gpt-4o". There is deliberately no default -- a wrong
  annotator stamp is unrecoverable once the source annotations are gone.

<annotations.json>: a JSON list of objects, each:
  {"passage_id": "...", "spans": [...], "conceptual_schema": ...,
   "construal": "...", "register": "...",
   "evaluative": {"rarity": "...", "tone": "...", "social_context": ...},
   "on_target": true}

--resume-from N : keep the first N existing output lines, append the rest. The
annotations file must then cover passages [N: ] only.

--out-dir DIR : write {uid}_extracted.jsonl into DIR (relative to the repo root)
instead of the default data/processed/descriptions/track_a. Used to keep a new
extraction run alongside an earlier one instead of overwriting it. --resume-from
reads its kept lines from the same DIR.

Validation is delegated to description_schema.normalize_annotation(), the same
function the API path uses, so both extraction paths clean and coerce identically.
Verifies final line count == input line count.
"""
import json, sys, os, pathlib

ROOT = str(pathlib.Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import p4_extract_descriptions as E  # noqa: E402
from src.data.description_schema import (  # noqa: E402
    PROMPT_VERSION, SCHEMA_VERSION, normalize_annotation)
import yaml

DEFAULT_OUT_DIR = "data/processed/descriptions/track_a"
OUT_T = os.path.join(ROOT, DEFAULT_OUT_DIR, "{uid}_extracted.jsonl")


def select_input(uid):
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/p4_extract.yaml")))
    corpus_dir = pathlib.Path(ROOT) / cfg["corpus_dir"]
    filtered_path = corpus_dir / f"{uid}_filtered.jsonl"
    scored_path = corpus_dir / f"{uid}_scored.jsonl"
    return E.select_passages(filtered_path, scored_path,
                             int(cfg.get("extract_cap", 100)), int(cfg.get("extract_floor", 10)),
                             int(cfg.get("extract_seed", 42)))

def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    uid = sys.argv[1]
    ann_path = sys.argv[2]
    if "--annotator" not in sys.argv:
        sys.exit("--annotator is required (e.g. --annotator llm:claude-opus-5.5); "
                 "it records what produced these annotations and has no safe default.")
    annotator = sys.argv[sys.argv.index("--annotator") + 1]
    resume_from = 0
    if "--resume-from" in sys.argv:
        resume_from = int(sys.argv[sys.argv.index("--resume-from") + 1])
    out_dir = DEFAULT_OUT_DIR
    if "--out-dir" in sys.argv:
        out_dir = sys.argv[sys.argv.index("--out-dir") + 1]

    inp = select_input(uid)
    anns = json.load(open(ann_path))
    by_id = {a["passage_id"]: a for a in anns}

    out_path = os.path.join(ROOT, out_dir, f"{uid}_extracted.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    kept = []
    if resume_from:
        with open(out_path) as f:
            kept = [l.rstrip("\n") for l in f if l.strip()][:resume_from]
        assert len(kept) == resume_from, f"{uid}: only {len(kept)} existing lines, need {resume_from}"

    records = list(kept)
    missing = []
    for row in inp[resume_from:]:
        pid = row["passage_id"]
        a = by_id.get(pid)
        if a is None:
            missing.append(pid)
            continue
        a = normalize_annotation(a)
        rec = {
            "passage_id": pid,
            "source": row.get("source"),
            "hybrid_score": row.get("hybrid_score"),
            "classifier_confidence": row.get("classifier_confidence"),
            "extraction_source": row.get("extraction_source", "retained"),
            "spans": a["spans"],
            "conceptual_schema": a["conceptual_schema"],
            "construal": a["construal"],
            "register": a["register"],
            "evaluative": a["evaluative"],
            "on_target": a["on_target"],
            "prompt_version": PROMPT_VERSION,
            "schema_version": SCHEMA_VERSION,
            "annotator": annotator,
        }
        records.append(json.dumps(rec, ensure_ascii=False))

    if missing:
        print(f"ERROR {uid}: {len(missing)} passages missing annotations:", missing[:10])
        sys.exit(1)

    if len(records) != len(inp):
        print(f"ERROR {uid}: output {len(records)} != input {len(inp)}")
        sys.exit(1)

    with open(out_path, "w") as f:
        f.write("\n".join(records) + "\n")
    print(f"OK {uid}: {len(records)} lines -> {out_path}")


if __name__ == "__main__":
    main()
