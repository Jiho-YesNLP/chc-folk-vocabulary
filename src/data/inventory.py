"""Load the CHC ability inventory from its two source files.

The inventory is stored as two files so that published CHC content and this
project's additions never share a provenance record:

  data/raw/chc_taxonomy.json            the taxonomy as published by Schneider &
                                        McGrew (2018), transcribed verbatim from their
                                        2017 definitions sheet: codes, names, strata,
                                        descriptions, definitions, major/minor flags
  data/raw/chc_project_annotations.json this project's artifacts, keyed by uid:
                                        discriminator glosses, folk seeds, and the
                                        conceptual grouping used in reporting

``read_inventory`` joins them into the single merged shape every pipeline
script has always read (``{"meta": ..., "broad_abilities": [...]}``, with
``uid``, ``discriminator`` and ``seeds`` on each narrow ability
and ``conceptual_group`` on each broad ability), so callers only swap the
``json.loads`` call for this function.
"""

from __future__ import annotations

import json
import pathlib

TAXONOMY_PATH = "data/raw/chc_taxonomy.json"
ANNOTATIONS_NAME = "chc_project_annotations.json"

# Merged-view field order on a narrow ability, matching the pre-split inventory.
_NARROW_ORDER = ("code", "uid", "name", "definition", "major",
                 "discriminator", "seeds", "prior_definition")
_BROAD_ORDER = ("code", "name", "stratum", "conceptual_group", "description")


def ability_uid(broad_code: str, narrow_code: str) -> str:
    """Globally unique ability id, filesystem-safe: ``Gf-I``, ``Ga-U1U9``."""
    return f"{broad_code}-{narrow_code}"


def _ordered(entry: dict, order: tuple[str, ...]) -> dict:
    head = {k: entry[k] for k in order if k in entry}
    return {**head, **{k: v for k, v in entry.items() if k not in head}}


def read_inventory(taxonomy_path: str | pathlib.Path,
                   annotations_path: str | pathlib.Path | None = None) -> dict:
    """Return the merged inventory. The annotations file defaults to the taxonomy's sibling."""
    taxonomy_path = pathlib.Path(taxonomy_path)
    annotations_path = (pathlib.Path(annotations_path) if annotations_path
                        else taxonomy_path.with_name(ANNOTATIONS_NAME))
    tax = json.loads(taxonomy_path.read_text(encoding="utf-8"))
    ann = json.loads(annotations_path.read_text(encoding="utf-8"))

    broad_ann = ann["broad_abilities"]
    narrow_ann = ann["narrow_abilities"]
    seen: set[str] = set()
    broad_out = []
    for broad in tax["broad_abilities"]:
        if broad["code"] not in broad_ann:
            raise ValueError(f"{annotations_path.name}: no entry for broad ability {broad['code']}")
        narrows = []
        for narrow in broad["narrow_abilities"]:
            uid = ability_uid(broad["code"], narrow["code"])
            if uid not in narrow_ann:
                pending = ann["meta"].get("pending", {}).get("abilities", [])
                why = (" (listed as pending: seeds and discriminator not yet authored)"
                       if uid in pending else "")
                raise ValueError(f"{annotations_path.name}: no entry for narrow ability {uid}{why}")
            if not narrow_ann[uid].get("seeds"):
                raise ValueError(f"{annotations_path.name}: narrow ability {uid} has no seeds")
            seen.add(uid)
            narrows.append(_ordered({**narrow, "uid": uid, **narrow_ann[uid]}, _NARROW_ORDER))
        merged = _ordered({**broad, **broad_ann[broad["code"]]}, _BROAD_ORDER)
        merged.pop("narrow_abilities", None)
        broad_out.append({**merged, "narrow_abilities": narrows})

    unknown = sorted(set(narrow_ann) - seen)
    if unknown:
        raise ValueError(f"{annotations_path.name}: entries for abilities not in the taxonomy: {unknown}")
    return {"meta": {"taxonomy": tax["meta"], "project_annotations": ann["meta"]},
            "broad_abilities": broad_out}
