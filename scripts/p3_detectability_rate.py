"""Stage 3 — rate the 84 blinded ability definitions under the frozen detectability rubric (API).

OpenAI API (OPENAI_API_KEY). The rater sees only the frozen rubric (as the system prompt, read
from file so every rater gets byte-identical instructions) and one blinded definition at a
time; no ability name, code, stratum, or coverage result. Items are sent in the randomized
order of blind_definitions.json. Output: ratings_dir/v3_{tag}.jsonl, one JSON object per item
keyed by its opaque `id`. An existing output file is never overwritten.

The rubric's SHA-256 is checked against the frozen value before any call.

Non-OpenAI raters (Opus 5, Fable 5.1) were run as blind subagents given the same rubric and
definitions files and writing the same record format.

Usage:
  uv run python scripts/p3_detectability_rate.py configs/p3_detectability.yaml --model gpt-4o
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import pathlib
import sys
from concurrent.futures import ThreadPoolExecutor

import yaml

RUBRIC_SHA256 = "c45a72b2ce9409eeba124499ae3c5e4cf8277ceb1eb15916a5760f71120cec6b"


def rate(client, model: str, system: str, item: dict) -> dict:
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": f"Description of a human capacity:\n\n{item['definition']}"}]
    err = "no attempt"
    for kwargs in ({"temperature": 0}, {}):          # some newer models reject temperature
        for _ in range(3):
            try:
                r = client.chat.completions.create(model=model, response_format={"type": "json_object"},
                                                   messages=msgs, **kwargs)
                return {"id": item["id"], **json.loads(r.choices[0].message.content)}
            except Exception as e:                    # noqa: BLE001 — record and retry
                err = str(e)
                if "temperature" in err.lower() and kwargs:
                    break
    return {"id": item["id"], "category": "ERROR", "justification": err[:250]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None, help="output name v3_{tag}.jsonl (default: the model name)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    cfg = yaml.safe_load(pathlib.Path(args.config).read_text())

    rubric = pathlib.Path(cfg["rubric_path"]).read_bytes()
    if hashlib.sha256(rubric).hexdigest() != RUBRIC_SHA256:
        sys.exit(f"{cfg['rubric_path']} does not match the frozen v3 rubric; a changed rubric needs a new version")
    out = pathlib.Path(cfg["ratings_dir"]) / f"v3_{args.tag or args.model}.jsonl"
    if out.exists():
        sys.exit(f"{out} exists; ratings are never overwritten")
    items = json.loads(pathlib.Path(cfg["blind_definitions_path"]).read_text())

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(lambda it: rate(client, args.model, rubric.decode(), it), items))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in res) + "\n")
    print(f"{out}: {dict(collections.Counter(r['category'] for r in res))}")
    errs = [r for r in res if r["category"] == "ERROR"]
    if errs:
        print(f"{len(errs)} errors, e.g. {errs[0]['justification'][:200]}", file=sys.stderr)


if __name__ == "__main__":
    main()
