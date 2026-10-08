# Stage 4 · Step A — Ability-Descriptor Extraction Rubric (shared, `extract-v2`)

You are performing the unified per-passage extraction *in-conversation* (the same role the script `scripts/p4_extract_descriptions.py` would hand to an API model, but done by you directly). Do **not** call any API.

`src/data/description_schema.py` is the single source of truth for the prompt and schema (`PROMPT_VERSION = "extract-v2"`). This rubric restates it for in-conversation use; if the two ever disagree, the module wins and this file must be regenerated from it.

> **Why the details below matter.** The API path pipes every annotation through `description_schema.normalize_annotation()`, which strips whitespace, drops malformed spans, and turns empty strings into `null`. The in-conversation path only gets `p4_merge_extract_annotations.py`'s `coerce()`, which enforces the enums but does **not** clean span text, empty schemas, or empty social contexts. Emitting already-normalized output is what keeps the two paths comparable.

## What each passage is

Every input passage has **already cleared the relevance classifier** as an *ability descriptor* of one target CHC ability. Your job is to characterize **HOW** the ability is described in everyday language — never the technical construct. **Do not invent content beyond the sentence.** You may flag a passage as off-target (`on_target: false`) if it actually describes a different ability or nothing (a classifier false positive); the deterministic build step drops those.

## Per-ability context (look it up before extracting an ability)

For each ability uid (e.g. `Gr-FA`), load its metadata via `p3_annotate.load_inventory` (the taxonomy in `data/raw/chc_taxonomy.json` joined with the project annotations in `data/raw/chc_project_annotations.json`). You need: `name`, `code` (the uid), `broad_name`, `definition`, `discriminator`.

Frame your analysis as: *"This sentence is an ability descriptor of **{name}** ({code}), under **{broad_name}**. Definition (for understanding only — do NOT echo its wording): {definition}. How it differs from neighbors: {discriminator}."* Folk register only — characterize the behavior/experience, never name the faculty.

## Fields to extract per sentence

1. **`spans`** — short descriptor phrases/words that characterize the ability AS SHOWN in the sentence.
    - Each span: `{"text": "<verbatim or lightly trimmed substring>", "pos": "adjective" | "phrase"}`.
    - INCLUDE only language about the cognitive process/capacity/tendency itself (e.g. `"ideas spark off each other"`, `"links things that seem unrelated"`, `"quick-witted"`).
    - EXCLUDE: the ability's name or any technical/clinical label; generic praise with no process (`"amazing"`, `"so smart"`); topic words not about the ability. Use `[]` if none qualify.
2. **`conceptual_schema`** — a ≤5-word lay label for the underlying schema the description uses (e.g. `"branching/traversal"`, `"production volume"`, `"mental rotation"`, `"holding briefly"`). `null` if the sentence does not really characterize this ability.
3. **`construal`** — exactly one of:
    - `dispositional`: a standing trait of a person ("she's the kind who...", "he's quick to...")
    - `agentive`: a deliberate action the person worked through ("he worked out...", "she pieced together...")
    - `spontaneous`: something that happened to/for the person, unbidden ("the ideas just came", "it popped into my head")
    - `output_based`: framed through the product/result ("her answer was...", "the drawing turned out...")
    - `other`: none of the above fits
4. **`register`** — `attribute` (a durable property predicated of a person) or `process` (a specific occasion on which the ability was exercised).
5. **`evaluative`** — `{"rarity": "rare"|"common"|"unmarked", "tone": "positive"|"neutral"|"negative", "social_context": "<≤4-word phrase>"|null}`
    - `rarity`: is the ability framed as exceptional or ordinary?
    - `social_context`: the social setting, if salient; else `null`.
6. **`on_target`** — `true` if the sentence genuinely characterizes the target ability; `false` if it actually describes a different ability or nothing.

### Controlled vocabularies (enforce exactly)

- `construal` ∈ {dispositional, agentive, spontaneous, output_based, other}
- `register` ∈ {attribute, process}
- `rarity` ∈ {rare, common, unmarked}
- `tone` ∈ {positive, neutral, negative}
- span `pos` ∈ {adjective, phrase}

### Normalization (match `normalize_annotation()` exactly)

These are the rules the API path applies automatically. `coerce()` does **not** apply them, so you must:

- **Span text**: strip leading/trailing whitespace. Drop any span whose text is empty after stripping. Always emit span objects (`{"text": ..., "pos": ...}`), never bare strings.
- **`conceptual_schema`**: strip whitespace. Emit `null` — never `""` — when there is no schema.
- **`social_context`**: strip whitespace. Emit `null` — never `""` — when no setting is salient.
- **Unsure on an enum**: do not invent a value and do not omit the key. Use the schema default (`construal: "other"`, `register: "process"`, `rarity: "unmarked"`, `tone: "neutral"`, `pos: "phrase"`), which is what the API path would coerce it to.
- **Never omit a field.** Every record carries all six keys, even when a value is `null` or `[]`.

## Input

Passage selection is owned by `p4_merge_extract_annotations.py`'s `select_input(uid)`, which reads the same corpus that `select_passages()` sent to extraction (`configs/p4_extract.yaml`). Use these fields from each line: `passage_id`, `source`, `hybrid_score`, `classifier_confidence`, and `text` (the sentence to analyze).

## Output — emit annotations, then merge

Do **not** write `{uid}_extracted.jsonl` yourself. Writing it directly bypasses `coerce()`, the metadata join, the version stamping, and the line-count check. Instead, emit an annotations file and run the merge script.

**Step 1.** Write `<uid>_annotations.json`: a JSON **list**, one object per input passage, **in input order**, containing only the passage id and the six extraction fields:

```json
[{"passage_id": "<from input>", "spans": [{"text": "...", "pos": "phrase"}], "conceptual_schema": "branching/traversal", "construal": "dispositional", "register": "process", "evaluative": {"rarity": "unmarked", "tone": "positive", "social_context": null}, "on_target": true}]
```

Do not add `source`, `hybrid_score`, `classifier_confidence`, `prompt_version`, `schema_version`, or `annotator` — the merge step carries the metadata through and stamps the rest.

**Step 2.** Run the merge:

```
uv run python scripts/p4_merge_extract_annotations.py <uid> <uid>_annotations.json \
    --annotator llm:<your-model-id> --out-dir <run directory> [--resume-from N]
```

It joins the passage metadata, enforces the controlled vocabularies, stamps `prompt_version` and `schema_version` (read from `description_schema.py`, never hardcoded) plus the `--annotator` you pass, verifies the output line count equals the input line count, and writes `<run directory>/{uid}_extracted.jsonl`.

`--out-dir` names the current run's directory, which `docs/stage4-inconv_extraction_prompt.md` states. Never omit it: the default is `data/processed/descriptions/track_a`, which holds an earlier run, and the merge overwrites what it finds there.

`--annotator` is required and has no default: it is the only record of what produced each annotation, and a wrong stamp is unrecoverable once the source annotations are gone. Pass the identifier of the model that actually did the annotating — yours, in this session. Do not copy the stamp from this rubric, the prompt, or existing records without checking it is true of you.

- Every input passage must produce exactly one annotation. Do not drop or reorder; the merge step hard-fails on a missing `passage_id`.
- `--resume-from N` keeps the first N existing output lines and appends the rest; the annotations file must then cover passages `[N:]` only.
