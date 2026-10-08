"""Versioned prompt + schema for the unified per-passage ability-descriptor extraction.

Single source of truth for Stage-4 Step A (scripts/p4_extract_descriptions.py). One
structured LLM call per passage returns *all five* extraction signals at once
(characterizing spans, conceptual schema, construal, register, evaluative framing),
so the 18k filtered passages are read by the model once rather than five times.

The passage has already cleared the relevance classifier as an ability descriptor of the
target ability; the model's job here is to *characterize how* the ability is described,
not to re-judge relevance — but it may flag a passage as off-target (classifier false
positive), which the deterministic build step can drop.

Two independent version stamps ride on every emitted record, because they answer
different questions and change at different times:

``PROMPT_VERSION`` -- which prompt text produced the annotation. Pure provenance.
    Bump it on any change to the prompt wording or the requested fields. It is
    NEVER rewritten afterwards: a record made by the v1 prompt stays v1 forever,
    even after its field names are migrated, because the stamp records what the
    model was actually asked.

``SCHEMA_VERSION`` -- which field-name generation the record's keys conform to.
    Bump it whenever a field is renamed or restructured, and rewrite it in place
    on existing records as part of that migration, so the stamp always describes
    the keys a reader will actually find. Generations so far:
      cad-v1  syntactic_frame + evaluative.valence
      cad-v2  syntactic_frame + evaluative.tone       (valence -> tone)
      cad-v3  construal       + evaluative.tone       (syntactic_frame -> construal)
"""

from __future__ import annotations

import json

PROMPT_VERSION = "extract-v2"   # prompt text that produced a record (provenance; never rewritten)
SCHEMA_VERSION = "cad-v3"       # field-name generation a record conforms to (rewritten by migrations)

# ── controlled vocabularies ──────────────────────────────────────────────────────
CONSTRUAL_TYPES = ("dispositional", "agentive", "spontaneous", "output_based", "other")
REGISTERS = ("attribute", "process")
RARITY = ("rare", "common", "unmarked")
TONE = ("positive", "neutral", "negative")
SPAN_POS = ("adjective", "phrase")

_CONSTRUAL_GLOSS = (
    "- dispositional: a standing trait of a person (\"she's the kind who...\", \"he's quick to...\")\n"
    "- agentive: a deliberate action the person worked through (\"he worked out...\", \"she pieced together...\")\n"
    "- spontaneous: something that happened to/for the person, unbidden (\"the ideas just came\", \"it popped into my head\")\n"
    "- output_based: framed through the product/result (\"her answer was...\", \"the drawing turned out...\")\n"
    "- other: none of the above fits"
)


def build_system_prompt(ability: dict) -> str:
    """ability: {name, code, broad_name, definition, discriminator} (from annotate.load_inventory)."""
    disc = f"\nHow it differs from neighbors: {ability['discriminator']}" if ability.get("discriminator") else ""
    return f"""\
You analyze a naturalistic sentence already identified as a *folk description* of one CHC \
cognitive ability. You extract HOW the ability is described in everyday language — never the \
technical construct. Do not invent content beyond the sentence.

TARGET ABILITY: {ability['name']} ({ability['code']}), under {ability['broad_name']}.
Definition (for your understanding only — do NOT echo its wording): {ability['definition']}{disc}

For each sentence, return:

1. "spans": the short folk phrases/words that characterize the ability AS SHOWN in the sentence.
   - Each span: {{"text": <verbatim or lightly trimmed substring>, "pos": "adjective" | "phrase"}}.
   - INCLUDE only language that describes the cognitive process/capacity/tendency itself
     (e.g. "ideas spark off each other", "links things that seem unrelated", "quick-witted").
   - EXCLUDE: the ability's name or any technical/clinical label; generic praise with no process
     ("amazing", "so smart"); topic words that are not about the ability. Use [] if none qualify.
2. "conceptual_schema": a <=5-word lay label for the underlying schema the description uses
   (e.g. "branching/traversal", "production volume", "mental rotation", "holding briefly").
   null if the sentence does not really characterize this ability.
3. "construal": one of — exactly one:
{_CONSTRUAL_GLOSS}
4. "register": "attribute" (a durable property predicated of a person) or
   "process" (a specific occasion on which the ability was exercised).
5. "evaluative": {{"rarity": "rare"|"common"|"unmarked",  (is the ability framed as exceptional or ordinary?)
                   "tone": "positive"|"neutral"|"negative",
                   "social_context": <=4-word phrase or null}}  (the social setting, if salient)
6. "on_target": true if the sentence genuinely characterizes {ability['name']}, false if it actually
   describes a different ability or nothing (a retrieval/classifier false positive).

Folk register only: characterize the behavior/experience, do not name the faculty.
"""


SCHEMA_INSTRUCTION = """\
Return STRICT JSON only, no prose:
{"annotations": [
  {"index": <int>,
   "spans": [{"text": "<str>", "pos": "adjective"|"phrase"}],
   "conceptual_schema": "<str>"|null,
   "construal": "dispositional"|"agentive"|"spontaneous"|"output_based"|"other",
   "register": "attribute"|"process",
   "evaluative": {"rarity": "rare"|"common"|"unmarked",
                  "tone": "positive"|"neutral"|"negative",
                  "social_context": "<str>"|null},
   "on_target": true|false}
]}
Echo every index you were given exactly once."""


def build_user_prompt(batch: list[tuple[int, str]]) -> str:
    lines = "\n".join(f"[{i}] {t}" for i, t in batch)
    return f"Analyze each sentence:\n{lines}"


# ── parsing / validation ─────────────────────────────────────────────────────────
def _one_of(value, allowed, default):
    return value if value in allowed else default


def _clean_spans(raw) -> list[dict]:
    spans = []
    if not isinstance(raw, list):
        return spans
    for s in raw:
        if isinstance(s, dict) and isinstance(s.get("text"), str) and s["text"].strip():
            spans.append({"text": s["text"].strip(), "pos": _one_of(s.get("pos"), SPAN_POS, "phrase")})
        elif isinstance(s, str) and s.strip():
            spans.append({"text": s.strip(), "pos": "phrase"})
    return spans


def normalize_annotation(a: dict) -> dict:
    """Coerce one raw LLM annotation dict into the validated record schema (enums enforced)."""
    ev = a.get("evaluative") or {}
    sc = ev.get("social_context")
    return {
        "spans": _clean_spans(a.get("spans")),
        "conceptual_schema": (a["conceptual_schema"].strip()
                              if isinstance(a.get("conceptual_schema"), str) and a["conceptual_schema"].strip()
                              else None),
        "construal": _one_of(a.get("construal"), CONSTRUAL_TYPES, "other"),
        "register": _one_of(a.get("register"), REGISTERS, "process"),
        "evaluative": {
            "rarity": _one_of(ev.get("rarity"), RARITY, "unmarked"),
            "tone": _one_of(ev.get("tone"), TONE, "neutral"),
            "social_context": sc.strip() if isinstance(sc, str) and sc.strip() else None,
        },
        "on_target": bool(a.get("on_target", True)),
    }


def parse_response(content: str, batch: list[tuple[int, str]]) -> dict[int, dict]:
    """Map response JSON to {index: normalized_annotation} for the indices in `batch`."""
    data = json.loads(content)
    valid = {i for i, _ in batch}
    out: dict[int, dict] = {}
    for a in data.get("annotations", []):
        try:
            idx = int(a["index"])
        except (KeyError, ValueError, TypeError):
            continue
        if idx in valid:
            out[idx] = normalize_annotation(a)
    return out
