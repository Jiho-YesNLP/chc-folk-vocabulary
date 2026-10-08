# Stage 4 · Step A: in-conversation extraction protocol

These are the instructions given to the in-conversation extractor (Claude Opus 5.5 in the reported run), which performs per-passage descriptor extraction directly in a chat session rather than through an API. The extraction rubric is `docs/stage4-cad_extraction_rubric.md`; the prompt and schema it restates live in `src/data/description_schema.py`. Everything below the rule is the text the extractor receives in a fresh session. All commands run from the repository root.

---

You are performing **Stage 4, Step A** of the CHC folk-vocabulary pipeline: per-passage descriptor extraction, done by you directly in this conversation rather than by an API model.

This is the *extraction* step only. The fingerprint itself is assembled afterwards by `scripts/p4_build_fingerprint.py`, which is deterministic and needs a GPU; do not attempt it here.

## Before you start

Read `docs/stage4-cad_extraction_rubric.md` in full. It is the authoritative specification for what to extract, the controlled vocabularies, and the normalization rules. Do not work from memory or from any summary of it, including this prompt: where this prompt and the rubric disagree, the rubric wins. `src/data/description_schema.py` is the source of truth behind both.

## Run directory

Each extraction run writes to its own directory under `data/processed/descriptions/`, named for the model (the reported run is `track_a_opus55`, annotator `llm:claude-opus-5.5`). Every status and merge command below passes that directory explicitly as `<RUN_DIR>`. Never drop the flag: both scripts default to `data/processed/descriptions/track_a`, and the merge opens its output for writing, so running it without `--out-dir` overwrites whatever run is stored there.

A different model is a different run and needs its own directory.

## Scope of this run

Get the worklist:

```
uv run python scripts/p4_extraction_status.py --pending --desc-dir <RUN_DIR>
```

An ability counts as done only when its line count matches the current selection *and* every record carries the expected annotator, so abilities extracted against an older corpus show as `stale` and need redoing. Unless specific abilities are named, work through the pending list in the order it prints, one ability at a time to completion. Report progress after each.

## Per ability

**1. Load the ability's context** from the inventory (`p3_annotate.load_inventory`, which joins `data/raw/chc_taxonomy.json` with `data/raw/chc_project_annotations.json`): `name`, `code`, `broad_name`, `definition`, `discriminator`. Hold the definition in mind to orient yourself, but never echo its wording into your output: you are recording how people describe the ability in ordinary language, not restating the construct.

**2. Get the passages.** This selector is authoritative. Do not read the corpus files directly, because the cap, floor and seed selection determines which passages belong to the run:

```
uv run python -c "
import sys, json; sys.path.insert(0,'scripts'); sys.path.insert(0,'.')
import p4_merge_extract_annotations as M
for i, r in enumerate(M.select_input('<UID>')):
    print(json.dumps({'i': i, 'passage_id': r['passage_id'], 'text': r['text']}))
"
```

**3. Annotate every passage**, in the order given, following the rubric. Abilities run up to 100 passages; work in batches of roughly 20 and keep going until the ability is finished. Judge each passage on its own: do not let a run of similar passages pull you toward a default answer, and do not infer content the sentence does not contain.

**4. Write the annotations** to `data/interim/stage4_annotations/<UID>_annotations.json` (gitignored; create the directory once). A JSON list, one object per passage, in input order, containing only `passage_id` and the six extraction fields. No metadata, no version stamps, no annotator field; the merge step adds those.

**5. Merge**, substituting your own model identifier:

```
uv run python scripts/p4_merge_extract_annotations.py <UID> \
    data/interim/stage4_annotations/<UID>_annotations.json \
    --annotator llm:<your-model-id> \
    --out-dir <RUN_DIR>
```

The `--annotator` value must name the model that actually produced the annotations, in this session. It is the only record of provenance and cannot be recovered once the annotations file is gone. Do not copy an annotator string from the rubric or from existing records without checking that it is true of you.

**6. Verify** before moving on:

```
uv run python scripts/p4_extraction_status.py --desc-dir <RUN_DIR> | grep '<UID>'
```

It must read `done`. If it reads `stale`, the line count or annotator disagrees; investigate rather than re-running blindly.

## Hard constraints

- **Never call an API.** You are the annotator.
- **Never write `{uid}_extracted.jsonl` in any descriptions directory yourself.** Only the merge script writes it. Writing it directly bypasses validation, the metadata join, version stamping, and the line-count check, which is how out-of-vocabulary values get into the corpus.
- **Exactly one annotation per input passage, in input order.** Never drop, merge, reorder, or invent a passage. The merge step fails on a missing `passage_id`; that failure is the safety net.
- **Emit already-normalized values** per the rubric's normalization section: stripped span text, no empty-string fields (`null` instead), span objects rather than bare strings, schema defaults rather than invented or omitted enum values.
- If a passage is genuinely not a descriptor of the target ability, mark `on_target: false` rather than forcing a reading. Those records are dropped at assembly; that is the mechanism working.
- If anything is ambiguous (a missing file, a count mismatch, an ability whose passages look wrong), stop and ask rather than guessing. A silently wrong batch costs far more than a question.
