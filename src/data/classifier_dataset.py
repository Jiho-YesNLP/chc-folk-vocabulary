"""Shared pieces of the relevance-classifier dataset: label ids, inventory and seed
loading, eval-text leakage exclusion, anchored example assembly, and passage-level splits.

Used by ``scripts/p3_build_soft_dataset.py``. An example pairs an *anchor* (ability name
plus a sample of its seed sentences) with a *target* passage, encoded as
``[CLS] anchor [SEP] target [SEP]``, and is labeled relative to the anchored ability.
"""

from __future__ import annotations

import json
import pathlib
import random
import re

from src.data.inventory import read_inventory  # taxonomy + project annotations, merged

# ── built-in delimiter tokens (already in the DistilBERT/BERT vocab) ─────────────
CLS_TOK = "[CLS]"
SEP_TOK = "[SEP]"

CAD = "folk_description"
NEG_LABELS = {"incidental_mention", "off_topic"}
VALID_LABELS = {CAD} | NEG_LABELS

# three-way class ids, judged *relative to the anchored ability*:
#   off_topic(0)         — no cognitive ability at all
#   incidental_mention(1) — an ability is present but not this one (incl. descriptor-for-a-neighbour
#                           under the wrong anchor, i.e. the false-positives)
#   folk_description(2)   — an ability descriptor of THIS ability
LABEL2ID = {"off_topic": 0, "incidental_mention": 1, "folk_description": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

_ws = re.compile(r"\s+")


def norm_text(text: str) -> str:
    return _ws.sub(" ", (text or "").strip().lower())


# ── inventory / seeds ───────────────────────────────────────────────────────────
def load_inventory(path: pathlib.Path) -> dict[str, dict]:
    """Return {uid: {code, name, definition, seeds:[...]}} for narrow abilities.

    Keyed by the composite uid ({broad}-{narrow}), matching the annotated-file
    names, p2_retrieve_passages.py, and the uid-named external seed files — so the
    ability_code carried by annotations resolves here directly.
    """
    inv = read_inventory(path)
    out: dict[str, dict] = {}
    for ba in inv["broad_abilities"]:
        for na in ba["narrow_abilities"]:
            seeds = na["seeds"]
            uid = na.get("uid") or f"{ba['code']}-{na['code']}"
            out[uid] = {
                "code": uid,
                "name": na["name"],
                "definition": na.get("definition", ""),
                "seeds": list(seeds),
            }
    return out


def apply_external_seeds(abilities: dict[str, dict], seed_dir: pathlib.Path) -> None:
    """Override inventory seeds with data/raw/seeds/track_a/{code}.jsonl when present."""
    if not seed_dir.exists():
        return
    for code, ab in abilities.items():
        f = seed_dir / f"{code}.jsonl"
        if not f.exists():
            continue
        texts = []
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts.append(rec["text"] if isinstance(rec, dict) else str(rec))
        if texts:
            ab["seeds"] = texts


# ── annotated data ──────────────────────────────────────────────────────────────
def eval_texts(layout) -> set[str]:
    """Normalized passage texts of the held-out eval set, for leakage exclusion.

    ``layout`` is a ``p3_annotate.SetLayout``; only its ``views`` directory is read.

    `p3_build_annotation_sets.py` guarantees random and ranked are disjoint *within* an
    ability, which is the right unit for annotation (a label is a property of the
    (ability, passage) pair). It does not prevent the same text from landing in ability
    Y's ranked pool and ability X's random pool. A cross-encoder can memorize text
    regardless of anchor, and the neighbour-descriptor rule (a passage annotated under Y is
    re-hosted as a positive under its true ability X) can even reproduce the same
    (ability, passage) pair on both sides. So exclude on text, globally.
    """
    if not layout.views.exists():
        raise SystemExit(f"no eval views under {layout.views} (pass --no-exclude-eval-text to skip)")
    return {norm_text(json.loads(l)["text"])
            for f in sorted(layout.views.glob("*.jsonl"))
            for l in f.read_text().splitlines() if l.strip()}


# ── example assembly ────────────────────────────────────────────────────────────
def build_anchor(name: str, seeds: list[str]) -> str:
    """Ability name + seeds, joined by the built-in [SEP] token (the text_a half)."""
    return f" {SEP_TOK} ".join([name, *seeds])


def make_example(
    ability: dict, target: str, label: int, label_name: str, example_type: str,
    source_label: str | None, origin_ability: str | None, passage_id: str,
    rng: random.Random, n_seeds: int,
) -> dict:
    pool = ability["seeds"]
    seeds = rng.sample(pool, min(n_seeds, len(pool)))
    anchor = build_anchor(ability["name"], seeds)
    return {
        "ability_code": ability["code"],
        "ability_name": ability["name"],
        "seeds": seeds,
        "anchor_text": anchor,
        "target_text": target,
        "input_text": f"{CLS_TOK} {anchor} {SEP_TOK} {target} {SEP_TOK}",
        "label": label,
        "label_name": label_name,
        "example_type": example_type,
        "source_label": source_label,
        "origin_ability": origin_ability,
        "passage_id": passage_id,
    }


def assign_splits(
    passage_ids: set[str], fracs: tuple[float, float, float], rng: random.Random
) -> dict[str, str]:
    ids = sorted(passage_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    split = {}
    for i, pid in enumerate(ids):
        if i < n_train:
            split[pid] = "train"
        elif i < n_train + n_val:
            split[pid] = "val"
        else:
            split[pid] = "test"
    return split
