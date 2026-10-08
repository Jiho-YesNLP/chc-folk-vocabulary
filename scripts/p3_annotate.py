"""LLM annotation of a frozen Stage-3 pool, under the three-way descriptor scheme.

Stage 3 has two annotation sets — `random` (the uniform sample; the coverage measurement)
and `ranked` (the top-N by hybrid score; the classifier's training set). Both are frozen
by ``p3_build_annotation_sets.py``; this script only *labels* what is already frozen, and
``--set`` selects which one. Pool construction lives entirely in the builder so the
random-before-ranked ordering can never be violated from here.

Pipeline (per ability, per set):
  {root}/{set}/pools/{uid}.jsonl                   (frozen Layer-1 stimulus)
    -> LLM labels each {folk_description | incidental_mention | off_topic};
       for descriptors, best_ability_code is constrained to the inventory (neighbour FPs)
    -> write the Layer-2 judgment file (the source of truth) and rebuild the
       derived lean view -- see "Layer 2" below

Layer 2 (source of truth): one JSONL per
(uid, annotator, prompt_version) at {root}/{set}/judgments/{uid}/{annotator}@{pv}.jsonl,
fields in JUDGMENT_FIELDS -- no text (joined against the Layer-1 pool by passage_id).
The lean view at {root}/{set}/views/{uid}.jsonl is DERIVED, rebuilt by materialize_lean()
from every annotator's Layer-2 file for that uid; it is not hand-authored and is safe to
delete/regenerate. Lean record schema (only what is consumed or diagnostic downstream):
  passage_id, text, source, seed_id, hybrid_score,
  label, best_ability_code, note, annotator
(the ability uid is the view filename's stem.)

For the in-conversation flow (a human or Claude reads the frozen pool and labels it
directly, rather than calling the OpenAI API), see p3_apply_inconv_labels.py.

Abilities are keyed by the composite uid ``{broad}-{narrow}`` (e.g. ``Gf-I``),
matching the retrieval output filenames and ``p2_retrieve_passages.py``. ``--ability``
also accepts a bare narrow mnemonic (e.g. ``FI``) when it is unambiguous.

Prompt versions: v1 is the original condensed prompt (kept so existing @v1 judgments stay
reproducible); v2 (default) mirrors the in-conversation instrument -- see the v2 section below
and data/raw/annotation/stage3-api_annotation_prompt_v2.md.

Usage:
  uv run python scripts/p3_annotate.py --set random --all --model gpt-4o      # v2
  uv run python scripts/p3_annotate.py --set random --ability Gf-I --dry-run  # show v2 prompt
  uv run python scripts/p3_annotate.py --set random --ability FI --limit 6    # smoke test
  uv run python scripts/p3_annotate.py --set random --all --prompt-version v1 --gaps-only
  uv run python scripts/p3_annotate.py --set ranked --summary-only            # rebuild the CSV
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import pathlib
import re
import sys
import time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data.inventory import read_inventory  # noqa: E402  taxonomy + project annotations, merged

_ws = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _ws.sub(" ", text.strip().lower())


# ── inventory: ability metadata + technical-label lexicon ──────────────────────

def load_inventory(path: pathlib.Path) -> dict:
    data = read_inventory(path)
    abilities: dict[str, dict] = {}
    labels: list[str] = []
    for broad in data["broad_abilities"]:
        labels.append(broad["name"])
        for narrow in broad.get("narrow_abilities", []):
            labels.append(narrow["name"])
            bare = narrow["code"]
            uid = narrow.get("uid") or f"{broad['code']}-{bare}"
            # Key by the composite uid so it matches the retrieval filenames
            # ({broad}-{narrow}_raw.jsonl) and resolve_ability's namespace.
            abilities[uid] = {
                "code": uid,
                "bare_code": bare,
                "name": narrow["name"],
                "broad_stratum": broad["code"],
                "broad_name": broad["name"],
                "definition": narrow.get("definition", ""),
                "discriminator": narrow.get("discriminator", ""),
            }
    return {"abilities": abilities, "labels": labels}


def build_label_matcher(labels: list[str]) -> re.Pattern:
    """Whole-phrase, case-insensitive matcher for exact CHC label strings."""
    phrases: set[str] = set()
    for lab in labels:
        for part in lab.split("/"):
            part = part.strip()
            if part:
                phrases.add(part)
    alts = sorted((re.escape(p) for p in phrases), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(alts) + r")\b", re.IGNORECASE)


# ── selection (identical to the proven pilot logic) ────────────────────────────

def select_candidates(raw_path: pathlib.Path, label_re: re.Pattern, top: int) -> tuple[list[dict], dict]:
    best: dict[str, dict] = {}
    total = 0
    with raw_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            total += 1
            pid = rec["passage_id"]
            cur = best.get(pid)
            if cur is None or rec["hybrid_score"] > cur["hybrid_score"]:
                best[pid] = rec

    uniq: dict[str, dict] = {}
    for rec in best.values():
        key = _norm(rec.get("text", ""))
        if not key:
            continue
        kept = uniq.get(key)
        if kept is None or rec["hybrid_score"] > kept["hybrid_score"]:
            uniq[key] = rec

    kept, prefiltered = [], 0
    for rec in uniq.values():
        if label_re.search(rec.get("text", "")):
            prefiltered += 1
            continue
        kept.append(rec)

    kept.sort(key=lambda r: r["hybrid_score"], reverse=True)
    pool = kept[:top]
    stats = {
        "raw_records": total,
        "unique_passages": len(best),
        "unique_after_text_dedup": len(uniq),
        "prefiltered_label_hits": prefiltered,
        "pool_size": len(pool),
    }
    return pool, stats


# Pool construction for both sets lives in p3_build_annotation_sets.py, which calls
# select_candidates() above once per ability and derives the random and ranked pools from
# it in that order. This module only annotates pools that are already frozen.


# ── the two annotation sets, and where their files live ───────────────────────
#
# All Stage-3 annotation data lives under ONE tree, organized set-first then layer:
#
#   {root}/{set}/pools/{uid}.jsonl                          Layer 1 — frozen stimulus
#   {root}/{set}/judgments/{uid}/{annotator}@{pv}.jsonl      Layer 2 — source of truth
#   {root}/{set}/views/{uid}.jsonl                           Layer 3 — derived view
#   {root}/{set}/summary.csv                                 per-ability dashboard
#
# The two sets have identical shape because they differ only in job, not in kind:
#
#   random — a uniform sample of the retrieval pool. The *measurement* instrument:
#            per-ability descriptor prevalence (the coverage analysis) and the held-out
#            precision number for the relevance classifier.
#   ranked — the top-N by hybrid score, minus whatever the random set drew. The
#            classifier's *training* set. Rank-biased by construction, so it is never
#            read as a coverage estimate.
#
# `rate` names each set's headline percentage. They are NOT interchangeable: a uniform
# sample estimates how much ability-descriptive language the corpus holds (prevalence); a top-N
# sample measures how well retrieval ranked it to the top (yield).

ANNOTATION_ROOT = "data/processed/annotation/track_a"

SETS = {
    "random": {"rate": "cad_prevalence_pct"},
    "ranked": {"rate": "cad_yield_pct"},
}

POOL_FIELDS = ("index", "passage_id", "text", "source", "seed_id", "hybrid_score", "pool_version")


class SetLayout:
    """Every path belonging to one annotation set. The single place the on-disk layout
    is defined -- scripts take `--root` + `--set` and ask this for paths, so a layout
    change lands here and nowhere else."""

    def __init__(self, root: pathlib.Path | str, name: str):
        if name not in SETS:
            raise KeyError(f"unknown set {name!r}; expected one of {sorted(SETS)}")
        self.root = pathlib.Path(root)
        self.name = name
        self.base = self.root / name
        self.rate = SETS[name]["rate"]

    @property
    def pools(self) -> pathlib.Path:
        return self.base / "pools"

    @property
    def judgments(self) -> pathlib.Path:
        return self.base / "judgments"

    @property
    def views(self) -> pathlib.Path:
        return self.base / "views"

    @property
    def summary(self) -> pathlib.Path:
        return self.base / "summary.csv"

    def pool(self, uid: str) -> pathlib.Path:
        return self.pools / f"{uid}.jsonl"

    def view(self, uid: str) -> pathlib.Path:
        return self.views / f"{uid}.jsonl"

    def judgment_dir(self, uid: str) -> pathlib.Path:
        return self.judgments / uid

    def judgment(self, uid: str, annotator_id: str, prompt_version: str) -> pathlib.Path:
        return self.judgment_dir(uid) / f"{annotator_id}@{prompt_version}.jsonl"

    def uids(self) -> list[str]:
        """Abilities with a frozen pool for this set."""
        return sorted(p.stem for p in self.pools.glob("*.jsonl")) if self.pools.exists() else []

    def mkdirs(self) -> None:
        for d in (self.pools, self.judgments, self.views):
            d.mkdir(parents=True, exist_ok=True)


def layouts(root: pathlib.Path | str = ANNOTATION_ROOT) -> dict[str, SetLayout]:
    return {name: SetLayout(root, name) for name in SETS}


def read_pool(path: pathlib.Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["index"])
    return rows


def write_pool(path: pathlib.Path, candidates: list[dict], pool_version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for i, c in enumerate(candidates):
            row = {
                "index": i,
                "passage_id": c["passage_id"],
                "text": c["text"],
                "source": c["source"],
                "seed_id": c.get("seed_id", ""),
                "hybrid_score": c["hybrid_score"],
                "pool_version": pool_version,
            }
            fh.write(json.dumps(row) + "\n")


def load_or_freeze_pool(path: pathlib.Path, build_fn, log=print) -> list[dict]:
    """Return the frozen pool at `path`, building+freezing it on first use.

    `build_fn()` -> (candidates, stats) is only called when no frozen pool exists yet —
    this is what makes a missing pool "the first annotation attempt" and every
    subsequent run reuse the identical, immutable stimulus set.
    """
    if path.exists():
        rows = read_pool(path)
        log(f"  [pool] frozen {path.name} v={rows[0]['pool_version'] if rows else '?'} n={len(rows)}")
        return rows
    candidates, stats = build_fn()
    pool_version = _dt.date.today().isoformat()
    write_pool(path, candidates, pool_version)
    log(f"  [pool] froze {path.name} v={pool_version} n={len(candidates)} ({stats})")
    return read_pool(path)


# ── frozen judgments (Layer 2: per-annotator, never overwritten by a different rater) ──
#
# One JSONL file per (uid, annotator_id, prompt_version), at
# {root}/{set}/judgments/{uid}/{annotator_id}@{prompt_version}.jsonl.
# A record carries only the judgment + join keys -- text is NOT duplicated here, it
# is looked up by passage_id against the Layer-1 pool. Re-running the SAME annotator
# overwrites only their own file; a different annotator or prompt_version gets a new
# file alongside it, so no rater's pass is ever clobbered.
#
# {root}/{set}/views/{uid}.jsonl is a DERIVED view — materialize_lean() rebuilds it
# from Layer 2 + the pool for downstream consumers (the eval-text leakage exclusion,
# write_summary). Views are not hand-authored and are always safe
# to delete and regenerate; Layer 2 is the thing to back up.

JUDGMENT_FIELDS = ("pool_id", "pool_version", "passage_id", "annotator_id", "annotator_type",
                   "prompt_version", "label", "best_ability_code", "confidence", "note", "timestamp")


def normalize_annotator(raw: str) -> tuple[str, str]:
    """('llm:gpt-4o', ...) -> ('llm-gpt-4o', 'llm'). Only LLM raters exist so far;
    a human pass should pass an annotator string starting with 'human'."""
    annotator_id = raw.replace(":", "-")
    annotator_type = "human" if raw.startswith("human") else "llm"
    return annotator_id, annotator_type


def write_judgments(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def read_judgments(path: pathlib.Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def judged_pids(layout: SetLayout, uid: str) -> set[str]:
    """passage_ids already judged for this uid by ANY annotator.

    Used by --gaps-only for resume semantics. Note it is deliberately annotator-blind: a
    second-provider IAA pass wants to re-judge passages the first annotator already saw,
    so such a pass must NOT use --gaps-only.
    """
    d = layout.judgment_dir(uid)
    out: set[str] = set()
    if d.exists():
        for jf in sorted(d.glob("*.jsonl")):
            for j in read_judgments(jf):
                out.add(j["passage_id"])
    return out


def pool_index(layout: SetLayout, uid: str) -> dict[str, dict]:
    """passage_id -> Layer-1 pool row, for joining judgments back to their stimulus."""
    p = layout.pool(uid)
    return {row["passage_id"]: row for row in read_pool(p)} if p.exists() else {}


def render_pool(layout: SetLayout, inv: dict, uid: str,
                unjudged_only: bool = False, limit: int = 0) -> tuple[str, list[dict]]:
    """The pool display the in-conversation annotator reads: ability header (definition,
    vs-neighbors line) then one tab-separated row per passage. Shared by p3_show_pool.py and
    the v2 API prompt so both raters see byte-identical stimuli. Returns (text, rows shown)."""
    ab = inv["abilities"].get(uid)
    rows = read_pool(layout.pool(uid))
    judged = judged_pids(layout, uid)

    out = ["", "=" * 78,
           f"{uid} — {ab['name'] if ab else '?'}  ({layout.name} set, n={len(rows)}, "
           f"{len(judged)} already judged)"]
    if ab:
        out.append(f"broad     : {ab['broad_stratum']} — {ab['broad_name']}")
        out.append(f"definition: {ab['definition']}")
        if ab.get("discriminator"):
            out.append(f"vs neighbors: {ab['discriminator']}")
    out += ["=" * 78, "idx\tjudged\tsource\ttext"]

    shown = []
    for r in rows:
        mark = "*" if r["passage_id"] in judged else ""
        if unjudged_only and mark:
            continue
        if limit and len(shown) >= limit:
            break
        text = " ".join(r["text"].split())
        out.append(f"{r['index']}\t{mark}\t{r['source']}\t{text}")
        shown.append(r)
    return "\n".join(out), shown


def materialize_lean(layout: SetLayout, uid: str) -> list[dict]:
    """Rebuild the flat lean view by joining every annotator's Layer-2 judgments for
    this uid against the Layer-1 pool. Layer 2 is the source of truth; this is a
    derived convenience file. Passage_ids are unioned across annotator files (each
    historical annotation session covers disjoint passage_ids); if two annotators
    ever judge the SAME passage_id, the first-seen (alphabetically by filename) wins
    and the conflict is printed -- proper adjudication is Layer 4 (gold/), not yet built.
    """
    jdir = layout.judgment_dir(uid)
    out_path = layout.view(uid)
    idx = pool_index(layout, uid)
    rows: list[dict] = []
    owner: dict[str, str] = {}
    if jdir.exists():
        for jf in sorted(jdir.glob("*.jsonl")):
            for j in read_judgments(jf):
                pid = j["passage_id"]
                pool_row = idx.get(pid)
                if pool_row is None:
                    print(f"  [warn] {uid}: {pid} in {jf.name} has no matching pool row; skipped")
                    continue
                # A rater is (annotator, prompt_version): the same model under v1 and v2 is two
                # raters, and must not put a passage in the view twice.
                rater = f"{j['annotator_id']}@{j.get('prompt_version', '')}"
                if pid in owner:
                    if owner[pid] != rater:
                        print(f"  [warn] {uid}: {pid} judged by both {owner[pid]} and "
                              f"{rater}; keeping {owner[pid]} (no gold/ adjudication yet)")
                    continue
                owner[pid] = rater
                rows.append(lean_record({
                    "passage_id": pid, "text": pool_row["text"], "source": pool_row["source"],
                    "seed_id": pool_row.get("seed_id", ""), "hybrid_score": pool_row["hybrid_score"],
                    "label": j["label"], "best_ability_code": j.get("best_ability_code"),
                    "note": j.get("note", ""), "annotator": j["annotator_id"],
                }, uid))
    rows.sort(key=lambda r: (r["hybrid_score"] is not None, r["hybrid_score"]), reverse=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return rows


# ── LLM annotation ─────────────────────────────────────────────────────────────

CRITERIA = """\
You classify a single naturalistic sentence retrieved as a possible *folk description* \
of the target CHC cognitive ability. Use EXACTLY one label:

- "folk_description" (INCLUDE): satisfies ALL of —
  (1a) describes a cognitive process/capacity/tendency that maps onto the target ability's
       definition, interpretable without technical vocabulary;
  (1b) everyday/observational/narrative register (not psychometric/clinical/academic prose);
  (1c) the ability is shown through behavior or experience, NOT named.
- "incidental_mention" (EXCLUDE): a relevant cognitive ability is present but
  (2a) peripheral to the main point, or (2b) too generic to tell which ability
  (could fit several), or (2c) purely an outcome judgment with no process described.
- "off_topic" (EXCLUDE): describes NO cognitive ability at all — retrieved only as a
  surface/similarity artifact. If you cannot name even a broad ability it could be about, it is off_topic.

Notes: metaphor/analogy that characterizes the process counts as folk_description. A sentence
may be a genuine folk description of a DIFFERENT ability than the target (a neighboring
false positive) — still label it folk_description but set is_target=false and name best_ability.
"""

SCHEMA_INSTRUCTION = """\
Return STRICT JSON only, no prose:
{"annotations": [
  {"index": <int>,
   "label": "folk_description" | "incidental_mention" | "off_topic",
   "rationale": "<= 25 words",
   "best_ability": "<CHC narrow ability name, or null>",
   "is_target": <true|false>}
]}
Echo every index you were given exactly once."""


def annotate_batch(client, model: str, ability: dict, batch: list[tuple[int, str]]) -> dict[int, dict]:
    sys_msg = (
        CRITERIA
        + f"\n\nTARGET ABILITY: {ability['name']} ({ability['code']}), under {ability['broad_name']}."
        + f"\nDefinition: {ability['definition']}"
        + (f"\nHow it differs from neighbors: {ability['discriminator']}" if ability.get("discriminator") else "")
        + (f"\n\nWhen label is folk_description, best_ability MUST be EXACTLY one of these CHC narrow "
           f"abilities (copy the name as written); use null if none fits:\n{ability['allowed_str']}"
           if ability.get("allowed_str") else "")
        + "\n\n" + SCHEMA_INSTRUCTION
    )
    lines = "\n".join(f"[{i}] {t}" for i, t in batch)
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Classify each sentence:\n{lines}"},
        ],
    )
    data = json.loads(resp.choices[0].message.content)
    return {int(a["index"]): a for a in data.get("annotations", [])}


def annotate_with_fallback(client, model, ability, batch):
    """Annotate a batch; on failure retry each item alone (isolates a flagged sentence)."""
    try:
        return annotate_batch(client, model, ability, batch), None
    except Exception as e:
        out: dict[int, dict] = {}
        for item in batch:
            try:
                out.update(annotate_batch(client, model, ability, [item]))
            except Exception:
                pass
        return out, str(e)


# ── LLM annotation, prompt v2: mirrors the in-conversation (Opus) instrument ───
#
# v1 above sends a condensed CRITERIA block, 20 passages per call, text only, each call
# stateless. v2 reproduces what the in-conversation rater worked from and how it worked:
#   - system prompt = data/raw/annotation/stage3-api_annotation_prompt_v2.md (the labeling
#     standard from docs/inconv_annotation_agent.md) + the FULL
#     data/raw/annotation/stage3-annotation_criteria.md;
#   - user turn = render_pool() output (the p3_show_pool.py display) for a whole group of
#     abilities, all 50 passages each, 3 abilities per call (the spec's batch size);
#   - the previous few (user, assistant) turns stay in context, as they did in the
#     conversation, so labels are calibrated across neighbouring abilities;
#   - a malformed or incomplete answer is sent back with the problems listed, as the apply
#     script's fail-closed error was for the in-conversation rater;
#   - unlike Opus's bare labels, each label is an object whose `reasoning` precedes `label`.
# The rendered system prompt is snapshotted per (set, prompt_version); a changed prompt under
# an existing version is refused, so a version string always names one exact instrument.

PROMPT_V2_PATH = "data/raw/annotation/stage3-api_annotation_prompt_v2.md"
CRITERIA_PATH = "data/raw/annotation/stage3-annotation_criteria.md"
PROMPT_BEGINS = "<!-- PROMPT BEGINS -->"

LABELS = ("folk_description", "incidental_mention", "off_topic")

# Models that reject an explicit temperature; learned on first rejection, per process.
_NO_TEMPERATURE: set[str] = set()


def build_system_prompt_v2(inv: dict) -> str:
    template = pathlib.Path(PROMPT_V2_PATH).read_text().split(PROMPT_BEGINS, 1)[1].lstrip("\n")
    ability_list = "\n".join(
        f"- {a['code']} — {a['name']} ({a['broad_stratum']} {a['broad_name']})"
        for a in sorted(inv["abilities"].values(), key=lambda a: a["code"]))
    return (template
            .replace("{{ABILITY_LIST}}", ability_list)
            .replace("{{CRITERIA}}", pathlib.Path(CRITERIA_PATH).read_text().strip()))


def snapshot_prompt(layout: SetLayout, prompt_version: str, prompt: str) -> pathlib.Path:
    """Freeze the exact system prompt a prompt_version denotes; refuse silent drift."""
    path = layout.base / "prompts" / f"{prompt_version}.md"
    if path.exists():
        if path.read_text() != prompt:
            sys.exit(f"{path} differs from the prompt now rendered (criteria or template "
                     f"edited?). Bump --prompt-version rather than reuse {prompt_version!r}.")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompt)
    return path


def validate_v2(data, expected: dict[str, list[int]], code_set: set) -> tuple[dict, list[str]]:
    """Check one response against the indices shown. Returns ({uid: {index: judgment}} for
    every uid that is fully valid, [problems])."""
    good: dict[str, dict] = {}
    problems: list[str] = []
    if not isinstance(data, dict):
        return good, ["top level must be a JSON object keyed by uid"]
    for uid, idxs in expected.items():
        got = data.get(uid)
        if not isinstance(got, dict):
            problems.append(f"{uid}: missing")
            continue
        errs: list[str] = []
        out: dict[int, dict] = {}
        for i in idxs:
            j = got.get(str(i))
            if not isinstance(j, dict):
                errs.append(f"index {i} missing")
                continue
            label, bac = j.get("label"), j.get("best_ability_uid")
            if label not in LABELS:
                errs.append(f"index {i} has invalid label {label!r}")
            elif label == "folk_description" and bac not in code_set:
                errs.append(f"index {i} folk_description needs a valid best_ability_uid, got {bac!r}")
            else:
                out[i] = {"label": label, "reasoning": j.get("reasoning", ""),
                          "best_ability_code": bac if label == "folk_description" else None}
        extra = sorted(set(got) - {str(i) for i in idxs})
        if extra:
            errs.append(f"indices not shown: {', '.join(extra)}")
        if errs:
            problems.append(f"{uid}: " + "; ".join(errs))
        else:
            good[uid] = out
    for uid in sorted(set(data) - set(expected)):
        problems.append(f"{uid}: not in this batch")
    return good, problems


def annotate_group_v2(client, model: str, system: str, history: list[dict], user_msg: str,
                      expected: dict[str, list[int]], code_set: set,
                      max_repairs: int = 2) -> tuple[dict, list[dict], list[str]]:
    """One call for a group of abilities, with up to `max_repairs` follow-up turns that list
    what was wrong. Returns (valid labels by uid, the turns to keep as context, problems left)."""
    turns = [{"role": "user", "content": user_msg}]
    good: dict[str, dict] = {}
    problems: list[str] = []
    for _ in range(max_repairs + 1):
        kwargs = {"model": model, "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": system}] + history + turns}
        if model in _NO_TEMPERATURE:
            resp = client.chat.completions.create(**kwargs)
        else:
            try:
                resp = client.chat.completions.create(temperature=0, **kwargs)
            except Exception as e:
                # Newer models accept only the default temperature. Learn once, not per call.
                if "temperature" not in str(e):
                    raise
                _NO_TEMPERATURE.add(model)
                resp = client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        content = choice.message.content or ""
        turns.append({"role": "assistant", "content": content})
        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            data, problems = None, [f"response is not valid JSON ({e})"]
        if data is not None:
            got, problems = validate_v2(data, expected, code_set)
            good.update(got)
        if choice.finish_reason == "length":
            problems.append("response was truncated; answer again")
        if not problems:
            break
        turns.append({"role": "user", "content":
                      "Your labels were rejected:\n- " + "\n- ".join(problems)
                      + "\nReturn the complete JSON again for every ability in this batch."})
    # Keep only the clean exchange as context for later groups: the original request and the
    # final answer, not the repair back-and-forth.
    return good, [turns[0], turns[-1] if turns[-1]["role"] == "assistant" else turns[-2]], problems


def own_complete(layout: SetLayout, uid: str, annotator_id: str, prompt_version: str) -> bool:
    """Has THIS annotator@version already labelled every passage of the pool?"""
    p = layout.judgment(uid, annotator_id, prompt_version)
    if not p.exists():
        return False
    return {j["passage_id"] for j in read_judgments(p)} >= {r["passage_id"] for r in read_pool(layout.pool(uid))}


def annotate_v2(client, model: str, inv: dict, uids: list[str], layout: SetLayout,
                prompt_version: str, per_call: int, context_batches: int,
                limit: int, dry_run: bool) -> None:
    system = build_system_prompt_v2(inv)
    annotator_id, annotator_type = normalize_annotator(f"llm:{model}")
    code_set = set(inv["abilities"])
    todo = [u for u in uids if not own_complete(layout, u, annotator_id, prompt_version)]
    if len(todo) < len(uids):
        print(f"[v2] skipping {len(uids) - len(todo)} abilities already complete for "
              f"{annotator_id}@{prompt_version}")
    if not dry_run:
        print(f"[v2] system prompt -> {snapshot_prompt(layout, prompt_version, system)}")

    history: list[dict] = []
    for g in range(0, len(todo), per_call):
        group = todo[g:g + per_call]
        blocks, expected, shown_rows = [], {}, {}
        for uid in group:
            text, rows = render_pool(layout, inv, uid, limit=limit)
            blocks.append(text)
            expected[uid] = [r["index"] for r in rows]
            shown_rows[uid] = rows
        user_msg = ("\n".join(blocks) + f"\n\n# {sum(map(len, expected.values()))} passages shown "
                    f"across {len(group)} abilities. Label every one.")
        if dry_run:
            print(f"{'#' * 78}\n# SYSTEM\n{'#' * 78}\n{system}\n{'#' * 78}\n# USER\n{'#' * 78}\n{user_msg}")
            return

        good, turns, problems = annotate_group_v2(client, model, system, history, user_msg,
                                                  expected, code_set)
        history = (history + turns)[-2 * context_batches:] if context_batches else []
        now = _dt.datetime.now(_dt.timezone.utc).isoformat()
        for uid in group:
            if uid not in good:
                print(f"  [fail] {uid}: not written -- {'; '.join(p for p in problems if p.startswith(uid)) or 'no valid labels'}")
                continue
            rows = [{"pool_id": uid, "pool_version": r["pool_version"], "passage_id": r["passage_id"],
                     "annotator_id": annotator_id, "annotator_type": annotator_type,
                     "prompt_version": prompt_version, "label": good[uid][r["index"]]["label"],
                     "best_ability_code": good[uid][r["index"]]["best_ability_code"],
                     "confidence": None, "note": good[uid][r["index"]]["reasoning"],
                     "timestamp": now} for r in shown_rows[uid]]
            write_judgments(layout.judgment(uid, annotator_id, prompt_version), rows)
            materialize_lean(layout, uid)
            dist = {k: sum(r["label"] == k for r in rows) for k in LABELS}
            print(f"  [{uid}] {len(rows)} -> {annotator_id}@{prompt_version}  {dist}")
        time.sleep(0.3)


# ── lean record + summary (shared with the migration) ──────────────────────────

LEAN_FIELDS = ("passage_id", "text", "source", "seed_id", "hybrid_score",
               "label", "best_ability_code", "note", "annotator")


def lean_record(rec: dict, ability_code: str) -> dict:
    """Project any annotated record (pilot or top100) to the lean schema."""
    label = rec.get("label")
    bac = rec.get("best_ability_code")
    if label != "folk_description":
        bac = None
    return {
        "passage_id": rec.get("passage_id", ""),
        "text": rec.get("text", ""),
        "source": rec.get("source", ""),
        "seed_id": rec.get("seed_id", ""),
        "hybrid_score": rec.get("hybrid_score"),
        "label": label,
        "best_ability_code": bac,
        "note": rec.get("note", rec.get("rationale", "")),
        "annotator": rec.get("annotator", ""),
    }


# ── the two annotation sets ────────────────────────────────────────────────────
#
# Each set has one job and one fixed depth applied to every ability. `view` is the
# derived lean-view suffix ({uid}_{view}.jsonl); `rate` names the headline percentage,
# which is NOT interchangeable between the sets:
#   random -> cad_prevalence_pct: a uniform sample of the pool, so this estimates how
#             much ability-descriptive language the corpus holds for it. The coverage number.
#   ranked -> cad_yield_pct: the top-N by hybrid score, so this measures how well
#             retrieval ranked ability descriptors to the top. A retrieval diagnostic only.

SETS = {
    "random": {"dir": "data/processed/random/track_a", "view": "random",
               "summary": "random_summary.csv", "rate": "cad_prevalence_pct"},
    "ranked": {"dir": "data/processed/annotations/track_a", "view": "annotated",
               "summary": "ranked_summary.csv", "rate": "cad_yield_pct"},
}


def write_summary(layout: SetLayout) -> pathlib.Path:
    """Scan every view in this set and write its per-ability dashboard CSV."""
    rows = []
    for f in sorted(layout.views.glob("*.jsonl")) if layout.views.exists() else []:
        code = f.stem
        n = cad = inc = off = on_t = neigh = 0
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab = r.get("label")
            n += 1
            if lab == "folk_description":
                cad += 1
                if r.get("best_ability_code") == code:
                    on_t += 1
                else:
                    neigh += 1
            elif lab == "incidental_mention":
                inc += 1
            elif lab == "off_topic":
                off += 1
        pct = lambda x: round(100 * x / n, 1) if n else 0.0
        rows.append((code, n, cad, on_t, neigh, inc, off,
                     pct(on_t), pct(inc), pct(off)))
    rows.sort(key=lambda r: r[0])
    out = layout.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    header = ("code,n,cad,cad_on_target,cad_neighbor,incidental,off_topic,"
              f"{layout.rate},incidental_pct,off_topic_pct")
    out.write_text(header + "\n" + "\n".join(",".join(str(c) for c in r) for r in rows) + "\n")
    print(f"[summary] {layout.name}: {len(rows)} abilities -> {out}")
    return out


# ── driver ─────────────────────────────────────────────────────────────────────

def resolve_ability(raw, name2code: dict, code_set: set) -> str | None:
    if not isinstance(raw, str):
        return None
    key = re.sub(r"\s*\(.*?\)\s*$", "", raw.strip()).lower()
    if key in name2code:
        return name2code[key]
    if raw.strip().upper() in code_set:
        return raw.strip().upper()
    return None


def annotate_ability(client, model, ability, pool, name2code, code_set, batch_size,
                     layout, prompt_version):
    dist = {"folk_description": 0, "incidental_mention": 0, "off_topic": 0, "error": 0}
    uid = ability["code"]
    annotator_id, annotator_type = normalize_annotator(f"llm:{model}")
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    rows: list[dict] = []
    for start in range(0, len(pool), batch_size):
        chunk = pool[start:start + batch_size]
        batch = [(start + j, c["text"]) for j, c in enumerate(chunk)]
        ann, err = annotate_with_fallback(client, model, ability, batch)
        if err:
            print(f"  [warn] batch {start}-{start + len(chunk) - 1} fell back "
                  f"(recovered {len(ann)}/{len(chunk)}): {err[:80]}")
        for j, c in enumerate(chunk):
            a = ann.get(start + j) or {}
            label = a.get("label")
            if label not in ("folk_description", "incidental_mention", "off_topic"):
                label = "error"
            bac = (resolve_ability(a.get("best_ability"), name2code, code_set)
                   if label == "folk_description" else None)
            rows.append({
                "pool_id": uid, "pool_version": c["pool_version"], "passage_id": c["passage_id"],
                "annotator_id": annotator_id, "annotator_type": annotator_type,
                "prompt_version": prompt_version,
                "label": label, "best_ability_code": bac, "confidence": None,
                "note": a.get("rationale", ""), "timestamp": now,
            })
            dist[label] += 1
        print(f"  [{min(start + len(chunk), len(pool))}/{len(pool)}] "
              + "  ".join(f"{k}={v}" for k, v in dist.items() if v))
        time.sleep(0.3)
    # Merge rather than overwrite: a resumed/partial run must not drop this annotator's
    # earlier judgments. Rows from THIS run win a passage_id collision (an explicit
    # re-annotation is a correction); anything they do not cover is preserved.
    jpath = layout.judgment(uid, annotator_id, prompt_version)
    if jpath.exists():
        fresh = {r["passage_id"] for r in rows}
        rows = [r for r in read_judgments(jpath) if r["passage_id"] not in fresh] + rows
    write_judgments(jpath, rows)
    materialize_lean(layout, uid)
    return dist


def main() -> None:
    ap = argparse.ArgumentParser(
        description="LLM annotation of a frozen Stage-3 pool (random or ranked set)")
    ap.add_argument("--set", choices=sorted(SETS), required=True,
                    help="which annotation set to label; selects the frozen pool, "
                         "the output tree, and the summary CSV")
    ap.add_argument("--ability", default=None, help="single uid, e.g. Gv-PI")
    ap.add_argument("--all", action="store_true", help="every ability with a frozen pool")
    ap.add_argument("--summary-only", action="store_true",
                    help="just rebuild this set's summary CSV from what is on disk")
    ap.add_argument("--inventory", default="data/raw/chc_taxonomy.json")
    ap.add_argument("--root", default=ANNOTATION_ROOT,
                    help="annotation tree root; all set paths derive from it")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--batch", type=int, default=20, help="v1 only: passages per call")
    ap.add_argument("--prompt-version", choices=["v1", "v2", "v3"], default="v2",
                    help="v1: condensed CRITERIA, 20 stateless passages per call. v2: the "
                         "in-conversation instrument (full criteria doc, whole pools, "
                         "3 abilities per call, rolling context, reasoning before label). "
                         "v3: v2 with the borderline-toward-descriptor rule removed")
    ap.add_argument("--abilities-per-call", type=int, default=3, help="v2 only")
    ap.add_argument("--context-batches", type=int, default=2,
                    help="v2 only: earlier (request, answer) turns kept in context; 0 = stateless")
    ap.add_argument("--dry-run", action="store_true",
                    help="v2 only: print the system prompt and first user turn; no API call")
    ap.add_argument("--gaps-only", action="store_true",
                    help="v1 only: label only passages no annotator has judged yet (resume). Do "
                         "NOT use for a second-provider IAA pass, which must re-judge the same "
                         "items. v2 always labels whole pools and resumes by skipping abilities "
                         "its own annotator@version file already covers.")
    ap.add_argument("--limit", type=int, default=0, help="cap pool for a smoke test")
    args = ap.parse_args()
    if args.prompt_version == "v2" and args.gaps_only:
        sys.exit("--gaps-only is v1 only; v2 resumes automatically per annotator@version.")

    layout = SetLayout(args.root, args.set)

    if args.summary_only:
        write_summary(layout)
        return
    layout.mkdirs()

    inv = load_inventory(pathlib.Path(args.inventory))
    abilities_by_name = sorted(inv["abilities"].values(), key=lambda a: a["name"])
    allowed_str = "; ".join(f"{a['name']} ({a['code']})" for a in abilities_by_name)
    # name2code maps any descriptor name (and the uid itself) -> composite uid,
    # the single namespace shared by filenames, inventory keys, and best_ability_code.
    name2code: dict[str, str] = {}
    bare2uids: dict[str, list[str]] = {}
    for a in inv["abilities"].values():
        name2code[a["name"].lower()] = a["code"]
        name2code[a["code"].lower()] = a["code"]
        for part in a["name"].split("/"):
            name2code[part.strip().lower()] = a["code"]
        bare2uids.setdefault(a["bare_code"], []).append(a["code"])
    code_set = set(inv["abilities"])

    def to_uid(token: str) -> str | None:
        """Resolve a CLI ability token (uid or unambiguous bare mnemonic) to a uid."""
        if token in inv["abilities"]:
            return token
        hits = bare2uids.get(token, [])
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            sys.exit(f"Ambiguous ability '{token}'; use a full uid: {', '.join(hits)}")
        return None

    if args.all:
        codes = layout.uids()
        if not codes:
            sys.exit(f"no frozen pools under {layout.pools}; run p3_build_annotation_sets.py first.")
    elif args.ability:
        resolved = to_uid(args.ability)
        codes = [resolved if resolved else args.ability]
    else:
        sys.exit("Specify --ability UID, --all, or --summary-only.")

    if args.prompt_version == "v2":
        uids = []
        for code in codes:
            if code not in inv["abilities"]:
                print(f"[skip] unknown ability code: {code}")
            elif not layout.pool(code).exists():
                print(f"[skip] no frozen pool: {layout.pool(code)} (run p3_build_annotation_sets.py)")
            else:
                uids.append(code)
        client = None
        if not args.dry_run:
            if not os.environ.get("OPENAI_API_KEY"):
                sys.exit("OPENAI_API_KEY not set in environment.")
            from openai import OpenAI
            client = OpenAI()
        # v2 path defaults to "v2"; an explicit --prompt-version overrides it so a
        # changed prompt can be stamped as a new version (snapshot_prompt refuses reuse).
        pv = args.prompt_version if args.prompt_version != "v1" else "v2"
        annotate_v2(client, args.model, inv, uids, layout, pv,
                    args.abilities_per_call, args.context_batches, args.limit, args.dry_run)
        if not args.dry_run:
            write_summary(layout)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set in environment.")
    from openai import OpenAI
    client = OpenAI()

    for code in codes:
        if code not in inv["abilities"]:
            print(f"[skip] unknown ability code: {code}")
            continue
        ppath = layout.pool(code)
        if not ppath.exists():
            print(f"[skip] no frozen pool: {ppath} (run p3_build_annotation_sets.py)")
            continue
        pool = read_pool(ppath)
        if args.gaps_only:
            have = judged_pids(layout, code)
            n_all = len(pool)
            pool = [c for c in pool if c["passage_id"] not in have]
            if not pool:
                print(f"[{code}] complete ({n_all} judged) — skipping")
                continue
            print(f"[{code}] {len(pool)} unlabeled of {n_all}")
        if args.limit:
            pool = pool[: args.limit]
        ability = dict(inv["abilities"][code], allowed_str=allowed_str)
        dist = annotate_ability(client, args.model, ability, pool, name2code, code_set,
                                args.batch, layout, args.prompt_version)
        print(f"[{code}] done -> {layout.view(code)}  {dist}")

    write_summary(layout)


if __name__ == "__main__":
    main()
