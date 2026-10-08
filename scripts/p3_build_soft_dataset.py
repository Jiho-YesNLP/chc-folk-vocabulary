"""Build the relevance-classifier dataset with soft labels from several annotators.

Examples are positives, annotated negatives, gold and synthetic false positives, with an
off-topic cap, eval-text leakage exclusion and a passage-level 80/10/10 split (shared pieces
in ``src/data/classifier_dataset.py``). Each example's target is the *average of the
annotators' labels* for that (anchor, passage) pair. Labels are read from Layer 2
(``{set}/judgments/{uid}/{annotator}.jsonl``) because the lean views keep only one
annotator per passage.

Each annotator's label is mapped relative to the anchor ability ``a``:

  folk_description, best_ability_code == a   -> folk_description
  folk_description for another ability       -> incidental_mention
  incidental_mention                         -> incidental_mention
  off_topic                                  -> off_topic

so annotators who agree give an ordinary one-hot target, and a folk/incidental split between
two annotators gives 0.5 / 0.5. Every row also keeps a hard ``label`` (argmax of the soft
target, ties broken toward folk_description, then incidental_mention) for reporting.
Train with ``p3_train_classifier.py --soft-labels``.

Usage:
  uv run python scripts/p3_build_soft_dataset.py --set ranked \
      --annotators opus-5.5-in-conversation@v2 llm-gpt-6-astra@v2 \
      --n-seeds 5 --fp-ratio 2 --out-dir data/processed/datasets/track_a/folk_classifier_v10_soft
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from collections import Counter, defaultdict

import p3_annotate as A
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.data import classifier_dataset as D  # noqa: E402  shared example assembly

CAD = D.CAD
L2I = D.LABEL2ID                      # {off_topic: 0, incidental_mention: 1, folk_description: 2}
I2L = {v: k for k, v in L2I.items()}
TIE_ORDER = [L2I[CAD], L2I["incidental_mention"], L2I["off_topic"]]


def rel_class(label: str, best: str | None, host: str, anchor: str) -> str:
    """One annotator's label for the passage, relative to `anchor`."""
    if label == CAD:
        return CAD if (best or host) == anchor else "incidental_mention"
    return label


def soft_target(judg: list[tuple[str, str | None]], host: str, anchor: str) -> list[float]:
    v = [0.0, 0.0, 0.0]
    for label, best in judg:
        v[L2I[rel_class(label, best, host, anchor)]] += 1.0 / len(judg)
    return [round(x, 4) for x in v]


def hard_of(soft: list[float]) -> int:
    m = max(soft)
    return next(i for i in TIE_ORDER if soft[i] == m)


def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    layout = A.SetLayout(args.root, args.set_name)
    inventory_path = pathlib.Path(args.inventory)
    abilities = D.load_inventory(inventory_path)
    D.apply_external_seeds(abilities, inventory_path.parent / "seeds" / "track_a")

    # ── join every annotator's Layer-2 judgments to the frozen pool text ───────────
    recs: dict[tuple[str, str], dict] = {}          # (host, pid) -> {text, judg: [(label, best)]}
    n_judg = Counter()
    for pool in sorted(layout.pools.glob("*.jsonl")):
        host = pool.stem
        text = {json.loads(l)["passage_id"]: json.loads(l)["text"]
                for l in pool.read_text().splitlines() if l.strip()}
        for ann in args.annotators:
            f = layout.judgment_dir(host) / f"{ann}.jsonl"
            if not f.exists():
                raise SystemExit(f"missing judgments: {f}")
            for l in f.read_text().splitlines():
                if not l.strip():
                    continue
                j = json.loads(l)
                if j["label"] not in D.VALID_LABELS:
                    continue
                best = j.get("best_ability_code")
                if best and (best not in abilities or not abilities[best]["seeds"]):
                    best = host              # unknown/seedless neighbour -> on-target, as in B
                key = (host, j["passage_id"])
                recs.setdefault(key, {"text": text[j["passage_id"]], "judg": []})["judg"].append((j["label"], best))
                n_judg[ann] += 1
    incomplete = sum(1 for r in recs.values() if len(r["judg"]) != len(args.annotators))
    print(f"judgments: {dict(n_judg)}   passages: {len(recs)}   missing an annotator: {incomplete}")

    # ── leakage exclusion on text (see D.eval_texts) ──────────────────────────────
    n_before = len(recs)
    drop = D.eval_texts(A.SetLayout(args.root, args.eval_set))
    recs = {k: v for k, v in recs.items() if D.norm_text(v["text"]) not in drop}
    n_dropped = n_before - len(recs)

    anchor_codes = sorted({h for h, _ in recs if abilities[h]["seeds"]})
    broad = lambda uid: uid.split("-")[0]

    def ex(anchor: str, key: tuple[str, str], etype: str, origin: str) -> dict:
        host, pid = key
        soft = soft_target(recs[key]["judg"], host, anchor)
        h = hard_of(soft)
        e = D.make_example(abilities[anchor], recs[key]["text"], h, I2L[h], etype,
                           None, origin, pid, rng, args.n_seeds)
        e["soft_label"] = soft
        e["annotator_labels"] = [list(j) for j in recs[key]["judg"]]
        return e

    # ── annotated examples under the host anchor, plus each named neighbour anchor ──
    host_ex, neigh_ex = [], []
    for key, r in recs.items():
        host = key[0]
        host_ex.append(ex(host, key, "annotated", host))
        for v in sorted({b for lab, b in r["judg"] if lab == CAD and b and b != host}):
            neigh_ex.append(ex(v, key, "neighbour_anchor", host))

    # off-topic cap: unanimous off-topic host examples, capped at ratio x incidental-majority ones
    pure_off = [e for e in host_ex if e["soft_label"][L2I["off_topic"]] == 1.0]
    rest = [e for e in host_ex if e["soft_label"][L2I["off_topic"]] < 1.0]
    n_inc = sum(1 for e in rest if e["label"] == L2I["incidental_mention"])
    cap = int(round(args.offtopic_ratio * n_inc))
    rng.shuffle(pure_off)
    kept_off = pure_off[:cap]
    examples = rest + kept_off + neigh_ex

    # ── synthetic false positives: folk-bearing passages under a different anchor ──
    folk_src = [(e["ability_code"], (e["origin_ability"], e["passage_id"])) for e in examples
                if e["soft_label"][L2I[CAD]] > 0]
    n_gold_fp = sum(1 for e in host_ex if any(lab == CAD and b and b != e["ability_code"]
                                              for lab, b in e["annotator_labels"]))
    n_syn = max(0, int(round(args.fp_ratio * len(folk_src))) - n_gold_fp)
    syn, attempts = 0, 0
    while syn < n_syn and folk_src and attempts < n_syn * 30:
        attempts += 1
        T, key = rng.choice(folk_src)
        cands = [c for c in anchor_codes if c != T]
        same = [c for c in cands if broad(c) == broad(T)]
        cands = same or cands
        b = rng.choice(cands)
        examples.append(ex(b, key, "false_positive", T))
        syn += 1

    # ── split by passage_id and write ──────────────────────────────────────────────
    split_map = D.assign_splits({e["passage_id"] for e in examples}, tuple(args.splits), rng)
    out = pathlib.Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    by_split = defaultdict(list)
    for e in examples:
        e["split"] = split_map[e["passage_id"]]
        by_split[e["split"]].append(e)
    for sp in ("train", "val", "test"):
        (out / f"{sp}.jsonl").write_text("".join(json.dumps(e) + "\n" for e in by_split[sp]))

    def summ(rows):
        mass = [round(sum(e["soft_label"][i] for e in rows), 1) for i in range(3)]
        return {"n": len(rows), "by_hard_label": dict(Counter(e["label_name"] for e in rows)),
                "soft_mass": dict(zip([I2L[i] for i in range(3)], mass)),
                "n_split_targets": sum(1 for e in rows if max(e["soft_label"]) < 1.0),
                "by_type": dict(Counter(e["example_type"] for e in rows))}
    meta = {"config": {"source_set": args.set_name, "annotators": args.annotators, "scheme": "three_way_soft",
                       "label2id": L2I, "n_seeds": args.n_seeds, "fp_ratio": args.fp_ratio,
                       "offtopic_ratio": args.offtopic_ratio, "splits": list(args.splits), "seed": args.seed},
            "totals": summ(examples), "splits": {sp: summ(by_split[sp]) for sp in ("train", "val", "test")},
            "eval_leakage_exclusion": {"eval_set": args.eval_set, "n_dropped_passages": n_dropped},
            "offtopic_cap": {"unanimous_off_available": len(pure_off), "kept": len(kept_off)},
            "false_positives": {"gold_host_examples": n_gold_fp, "synthetic": syn}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"eval-leak exclusion: dropped {n_dropped} passages")
    print(f"host examples {len(host_ex)} (off-topic kept {len(kept_off)}/{len(pure_off)}), "
          f"neighbour-anchor {len(neigh_ex)}, synthetic FP {syn}")
    print(json.dumps(meta["totals"], indent=1))
    for sp in ("train", "val", "test"):
        print(sp, meta["splits"][sp]["n"], meta["splits"][sp]["soft_mass"])
    print("written ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Soft-label descriptor-classifier dataset")
    ap.add_argument("--root", default=A.ANNOTATION_ROOT)
    ap.add_argument("--set", dest="set_name", default="ranked", choices=sorted(A.SETS))
    ap.add_argument("--eval-set", default="random", choices=sorted(A.SETS))
    ap.add_argument("--inventory", default="data/raw/chc_taxonomy.json")
    ap.add_argument("--annotators", nargs="+", required=True,
                    help="Layer-2 file stems, e.g. opus-5.5-in-conversation@v2 llm-gpt-6-astra@v2")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--fp-ratio", type=float, default=2.0)
    ap.add_argument("--offtopic-ratio", type=float, default=1.0)
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.1, 0.1])
    ap.add_argument("--seed", type=int, default=42)
    main(ap.parse_args())
