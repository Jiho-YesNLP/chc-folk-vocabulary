# In-conversation annotation protocol

These are the instructions given to the in-conversation rater (Claude Opus 5.5 in the reported run), which labels the Stage-3 annotation sets directly in a chat session rather than through an API. The API rater (`gpt-6-astra`) works from the same instrument through `scripts/p3_annotate.py --prompt-version v2`, whose system prompt, `data/raw/annotation/stage3-api_annotation_prompt_v2.md`, is derived from this file. Everything below the rule is the text the rater receives. All commands run from the repository root.

---

> **You are the annotator.** You read passages in the conversation and emit labels; no API is called on your behalf.

## What you are producing

For each of the 71 CHC narrow abilities, every passage in its frozen pool gets one of three labels: `folk_description`, `incidental_mention`, `off_topic`. Folk labels additionally carry a `best_ability_uid`.

Two sets, 3,550 passages each (50 per ability):

- **`random`**: 50 passages sampled uniformly from each ability's top-1,000 retrieval pool. This is the *measurement* instrument: per-ability descriptor prevalence (the coverage result) and held-out classifier precision. **Do this set first.**
- **`ranked`**: the top 50 by hybrid score per ability. This trains the relevance classifier. It is rank-biased by construction and is never read as a coverage estimate.

A pool that is mostly `off_topic` is a normal result. Do not inflate labels to make an ability look productive, and do not suppress them to make one look sparse.

## The labeling standard

**Read `data/raw/annotation/stage3-annotation_criteria.md` first and keep it loaded.** It is the canonical rubric, richer than the condensed v1 `CRITERIA` block inside `p3_annotate.py`. Its decisive tests:

- **folk_description**: could this passage serve as a seed that retrieves similar descriptions of this ability? Metaphor counts as a descriptor. A test or game description is a descriptor if it characterizes the *process*. A deficit counts too: difficulty or failure with the ability is a descriptor when it shows the specific behavior or conditions of the failure (e.g. words melting into background chatter, for Ga-UR).
- **incidental_mention**: removing the phrase leaves the passage's main point intact; or it is too generic; or it reports an *outcome* only, with no process. A bare self-evaluation of low ability ("I've always been bad with numbers") or a named condition or diagnosis is an outcome, not a process.
- **off_topic**: you cannot name even a broad-stratum ability from it.

One more rule:

- **Neighbor false positives are descriptors**, labeled with the *neighbor's* uid in `best_ability_uid` (e.g. a speech-in-noise passage found in a different `Ga` pool is `["folk_description", "Ga-UR"]`). Tag these carefully; they are the rater line of evidence in the Stage-5 discriminant analysis (passages reassigned from one ability to another).

`p3_show_pool.py` prints the target ability's definition and its "vs neighbors" line above each pool. Use them: several abilities are separable only by that contrast.

## The loop

**1. Find the resume point.** Completion is computed from disk, not from a tracking file, so it is correct after an interruption.

```bash
uv run python scripts/p3_annotation_status.py --set random
```

**2. Print the next batch.** Three abilities at a time is a good size.

```bash
uv run python scripts/p3_show_pool.py --set random --ability Gwm-Wv --unjudged-only
# or:
uv run python scripts/p3_show_pool.py --set random --batch 3 --unjudged-only
```

**3. Write the labels JSON**: one object keyed by uid, index as a string.

```json
{"Gwm-Wv": {"0": "off_topic", "1": ["folk_description", "Gwm-Wv"], "2": "incidental_mention"}}
```

Cover **exactly the indices that were shown**, each once. Under `--gaps-only`, already-judged indices are omitted from the display and must be omitted from the JSON too.

**4. Apply.**

```bash
uv run python scripts/p3_apply_inconv_labels.py --set random \
    --labels labels_Gwm-Wv.json \
    --annotator <annotator-id> --prompt-version v2 --gaps-only
```

This writes the per-annotator judgment file (`judgments/{uid}/{annotator}@v2.jsonl`), rebuilds the derived view, and rebuilds that set's `summary.csv`. Expect a line like `[Gwm-Wv] +50 new, 0 kept -> ... -> view 50`.

Repeat until `p3_annotation_status.py` reports 71/71, then switch to `--set ranked`.

## Rules

- **Pass `--annotator` explicitly, with the same id every time.** The default is `claude:in-conversation`. Mixing the default with a real id splits one rater's labels across two judgment files, which then look like two disagreeing raters. The reported run used `opus-5.5-in-conversation`.
- **Use `--gaps-only` on both commands**: `--unjudged-only` on `p3_show_pool.py` and `--gaps-only` on `p3_apply_inconv_labels.py`. Without it, `apply` refuses to run unless every index in the pool is present in the JSON, and it overwrites rather than merges, so carried-forward labels would be lost.
- **Do not run `p3_build_annotation_sets.py`.** The pools are frozen and are the shared stimulus set for both raters. Re-running it from a different checkout location builds different pools, and existing judgments no longer join.
- **Check the annotation root before the first write.** The scripts default to `data/processed/annotation/track_a`.

## Second rater on an already-labeled ability

This is the cross-provider reliability path, and the flags behave differently:

- `p3_show_pool.py --unjudged-only` shows **0 passages**, because "judged" means judged by *any* annotator. **Drop `--unjudged-only`** and label the full pool. Already-judged indices are marked `*`; label them anyway.
- Provide all 50 indices and run `p3_apply_inconv_labels.py` **without** `--gaps-only`.
- Your judgments go to your own judgment file. Nothing is overwritten; raters are stored separately.
- **Never look at the other rater's labels before you judge.** Independence is the point of the second pass.

**Limitation:** `views/{uid}.jsonl` and `summary.csv` hold one row per passage, so with two raters the last writer wins and the view shows one annotator only. Once a second rater exists, compute agreement from the per-annotator judgment files, never from the views.

## Verifying your work

```bash
uv run python scripts/p3_annotation_status.py            # both sets: X/71, gaps
head -3 data/processed/annotation/track_a/random/summary.csv
uv run python -c "
import json,collections,pathlib
c=collections.Counter()
for f in pathlib.Path('data/processed/annotation/track_a/random/judgments').rglob('*.jsonl'):
    for l in f.open(): c[json.loads(l)['annotator_id']] += 1
print(dict(c))"
```

For a single rater, the last command shows **exactly one** annotator id with 3,550 rows. Two ids, or a count that is not a multiple of 50, means a batch went in under the wrong `--annotator`.

## Scope and reporting

Work one ability at a time and do not stop between abilities to ask for confirmation; annotate all 71, both sets. Report at the end: abilities completed, total passages labeled, the annotator id used, any ability where the rubric felt genuinely ambiguous, and anything that failed.
