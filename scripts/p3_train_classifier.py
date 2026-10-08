"""Train / evaluate the seed-conditioned ability-descriptor relevance classifier.

A single cross-encoder (default microsoft/deberta-v3-base), shared across CHC
narrow abilities, scores whether a target sentence is an ability descriptor of an
ability. Each example is the pair (anchor, target):

    [CLS] <ability name> [SEP] <seed> [SEP] <seed> [SEP] <seed> [SEP] <target> [SEP]

so the model conditions on the ability's identity + a sample of its descriptor seeds
when judging the target. The anchor is rebuilt here from each record's
``ability_name`` + ``seeds`` using the *tokenizer's own* ``sep_token``, so the
data is backbone-agnostic — DeBERTa/BERT use ``[SEP]``, RoBERTa uses ``</s>``,
and the right separator is inserted automatically. Only built-in special tokens
are used — no vocab surgery.

On a small pilot set a base-size model has ~3x DistilBERT's parameters, so
``--freeze-layers N`` can freeze the embeddings + bottom N transformer blocks to
curb overfitting.

The data is class-imbalanced (descriptor positives are a minority), so the loss is
class-weighted by inverse frequency. Model selection is by positive-class F1 on
the validation split; the held-out test split is scored once at the end, with a
breakdown by example_type (positive / annotated_negative / false_positive) so
hard-negative behaviour is visible.

Requires the project GPU env (torch cu124); run on the server, not macOS.

Usage:
  # train + evaluate
  uv run python scripts/p3_train_classifier.py \
      --data-dir data/processed/datasets/track_a/cad_classifier \
      --out-dir  models/relevance_classifier

  # evaluate an existing checkpoint on the test split only
  uv run python scripts/p3_train_classifier.py --eval-only --out-dir models/relevance_classifier
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# ── data ────────────────────────────────────────────────────────────────────────
def read_split(data_dir: pathlib.Path, split: str) -> list[dict]:
    path = data_dir / f"{split}.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def pair_with_anchor_budget(tok, anchor: str, target: str, budget: int, max_len: int):
    """Encode (anchor, target) with the anchor cut to `budget` tokens, so every seed set
    gets the same room and a long target can never push seeds out (the default
    `truncation=True` trims the longer side, which with many seeds is the anchor)."""
    a_ids = tok(anchor, add_special_tokens=False)["input_ids"][:budget]
    t_ids = tok(target, add_special_tokens=False)["input_ids"]
    return tok.prepare_for_model(a_ids, t_ids, truncation="only_second", max_length=max_len)


class PairDataset(torch.utils.data.Dataset):
    """Tokenizes (anchor_text, target_text) pairs lazily."""

    def __init__(self, rows: list[dict], tokenizer, max_len: int, seed_token_budget: int = 0,
                 soft: bool = False):
        self.rows = rows
        self.tok = tokenizer
        self.max_len = max_len
        self.seed_token_budget = seed_token_budget
        self.soft = soft  # train on r["soft_label"] (annotator-averaged class distribution)

    def __len__(self) -> int:
        return len(self.rows)

    def _anchor(self, r: dict) -> str:
        # rebuild with the tokenizer's own separator so the data is backbone-agnostic
        seeds = r.get("seeds")
        if seeds is not None and r.get("ability_name"):
            sep = self.tok.sep_token or "[SEP]"
            return f" {sep} ".join([r["ability_name"], *seeds])
        return r["anchor_text"]  # fallback to the precomputed string

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        if self.seed_token_budget:
            # fixed budget for the anchor (name + seeds); the target is only cut by max_len
            enc = pair_with_anchor_budget(self.tok, self._anchor(r), r["target_text"],
                                          self.seed_token_budget, self.max_len)
        else:
            enc = self.tok(
                self._anchor(r),
                r["target_text"],
                truncation=True,
                max_length=self.max_len,
                padding=False,
            )
        enc["labels"] = [float(x) for x in r["soft_label"]] if self.soft else int(r["label"])
        return enc


# ── metrics (no sklearn) ────────────────────────────────────────────────────────
def binary_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, len(labels))
    return {
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def cad_vs_rest(labels: np.ndarray, preds: np.ndarray, cad_id: int) -> dict:
    """One-vs-rest metrics for the descriptor class — the deployment-relevant number
    (precision of 'called descriptor'; the held-out version is p3_eval_classifier.py).
    Works for binary or 3-way."""
    g = (np.asarray(labels) == cad_id).astype(int)
    p = (np.asarray(preds) == cad_id).astype(int)
    return binary_metrics(g, p)


def soft_cad_metrics(soft: np.ndarray, preds: np.ndarray, cad_id: int) -> dict:
    """Descriptor P/R/F1 against soft targets: each example counts as positive with weight
    equal to its descriptor mass, which equals pooling the annotators' hard labels."""
    g = soft[:, cad_id]
    p = (preds == cad_id).astype(float)
    tp, fp, fn = float((g * p).sum()), float(((1 - g) * p).sum()), float((g * (1 - p)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn}


def macro_f1(labels: np.ndarray, preds: np.ndarray, num_labels: int) -> float:
    f1s = []
    for c in range(num_labels):
        m = binary_metrics((np.asarray(labels) == c).astype(int), (np.asarray(preds) == c).astype(int))
        f1s.append(m["f1"])
    return sum(f1s) / len(f1s) if f1s else 0.0


def confusion_matrix(labels, preds, num_labels: int) -> list[list[int]]:
    M = [[0] * num_labels for _ in range(num_labels)]
    for g, p in zip(labels, preds):
        M[int(g)][int(p)] += 1
    return M


def make_compute_metrics(cad_id: int, num_labels: int):
    """Closure so the Trainer's compute_metrics knows the descriptor class + #classes."""
    def _cm(eval_pred) -> dict:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        if np.asarray(labels).ndim == 2:   # soft targets: expected counts, i.e. pooled annotators
            m = soft_cad_metrics(np.asarray(labels), preds, cad_id)
            return {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                    "accuracy": float(np.asarray(labels)[np.arange(len(preds)), preds].mean())}
        m = cad_vs_rest(np.asarray(labels), preds, cad_id)   # headline = descriptor class
        return {"f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "accuracy": float((preds == np.asarray(labels)).mean()),
                "macro_f1": macro_f1(labels, preds, num_labels)}
    return _cm


# ── weighted-loss trainer ────────────────────────────────────────────────────────
class WeightedTrainer(Trainer):
    def __init__(self, *a, class_weights=None, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None else None
        )
        loss = nn.functional.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def freeze_base_layers(model, n_freeze: int) -> None:
    """Freeze the embeddings + the bottom *n_freeze* transformer blocks (backbone-agnostic)."""
    if n_freeze <= 0:
        return
    base = model.base_model
    if hasattr(base, "embeddings"):
        for p in base.embeddings.parameters():
            p.requires_grad = False
    enc = getattr(base, "encoder", None)
    if enc is not None and hasattr(enc, "layer"):          # BERT / RoBERTa / DeBERTa
        layers = enc.layer
    else:
        tr = getattr(base, "transformer", None)            # DistilBERT
        layers = getattr(tr, "layer", []) if tr is not None else []
    for layer in list(layers)[:n_freeze]:
        for p in layer.parameters():
            p.requires_grad = False


def inverse_freq_weights(rows: list[dict], num_labels: int, soft: bool = False) -> torch.Tensor:
    from collections import Counter
    cnt = Counter()
    for r in rows:
        if soft:
            for c, w in enumerate(r["soft_label"]):
                cnt[c] += w
        else:
            cnt[int(r["label"])] += 1
    total = sum(cnt.values())
    # weight_c = total / (n_classes * count_c)
    return torch.tensor(
        [total / (num_labels * cnt[c]) if cnt.get(c) else 1.0 for c in range(num_labels)],
        dtype=torch.float,
    )


# ── reporting ────────────────────────────────────────────────────────────────────
def predict_logits(trainer: Trainer, ds: PairDataset) -> np.ndarray:
    return trainer.predict(ds).predictions


def report_by_type(rows: list[dict], preds: np.ndarray) -> dict:
    """Accuracy per example_type (each type is single-class, so report acc)."""
    buckets: dict[str, list[int]] = defaultdict(list)
    for r, p in zip(rows, preds):
        buckets[r["example_type"]].append(int(p == r["label"]))
    return {t: {"n": len(v), "acc": sum(v) / len(v)} for t, v in sorted(buckets.items())}


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_dir = pathlib.Path(args.data_dir)
    out_dir = pathlib.Path(args.out_dir)

    source = str(out_dir) if args.eval_only else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(source)

    # label scheme from the dataset meta (three-way by default; cad = the top class)
    meta = json.loads((data_dir / "meta.json").read_text()) if (data_dir / "meta.json").exists() else {}
    label2id = (meta.get("config", {}) or {}).get("label2id")
    test_rows = read_split(data_dir, "test")
    if args.eval_only:
        num_labels = None  # taken from the loaded checkpoint below
    else:
        num_labels = args.num_labels or (max(int(r["label"]) for r in test_rows
                                             + read_split(data_dir, "train")) + 1)
    test_ds = PairDataset(test_rows, tokenizer, args.max_len, args.seed_token_budget, args.soft_labels)

    if args.eval_only:
        model = AutoModelForSequenceClassification.from_pretrained(source)
        num_labels = model.config.num_labels
        cad_id = num_labels - 1
        trainer = WeightedTrainer(
            model=model,
            args=TrainingArguments(output_dir=str(out_dir), per_device_eval_batch_size=args.batch_size, report_to=[]),
            tokenizer=tokenizer,
            compute_metrics=make_compute_metrics(cad_id, num_labels),
        )
    else:
        cad_id = num_labels - 1
        train_rows = read_split(data_dir, "train")
        val_rows = read_split(data_dir, "val")
        train_ds = PairDataset(train_rows, tokenizer, args.max_len, args.seed_token_budget, args.soft_labels)
        val_ds = PairDataset(val_rows, tokenizer, args.max_len, args.seed_token_budget, args.soft_labels)

        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=num_labels)
        if label2id:  # store the names so filter_passages/eval can find the descriptor index
            model.config.label2id = label2id
            model.config.id2label = {int(v): k for k, v in label2id.items()}
        freeze_base_layers(model, args.freeze_layers)
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"model: {args.model_name}  num_labels: {num_labels}  cad_id: {cad_id}  "
              f"frozen bottom layers: {args.freeze_layers}")
        print(f"trainable params: {n_trainable:,} / {n_total:,}")
        weights = None if args.no_class_weights else inverse_freq_weights(train_rows, num_labels, args.soft_labels)
        if weights is not None:
            print(f"class weights: {[round(w,3) for w in weights.tolist()]}")

        from transformers import DataCollatorWithPadding
        targs = TrainingArguments(
            output_dir=str(out_dir),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            weight_decay=0.01,
            warmup_ratio=0.1,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=args.seed,
            report_to=[],
        )
        trainer = WeightedTrainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=make_compute_metrics(cad_id, num_labels),
            class_weights=weights,
        )
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

    # ── final test evaluation ──────────────────────────────────────────────────
    logits = predict_logits(trainer, test_ds)
    preds = np.argmax(logits, axis=-1)
    labels = np.array([int(r["label"]) for r in test_rows])
    cad = cad_vs_rest(labels, preds, cad_id)             # deployment metric (cad vs rest)
    if args.soft_labels:   # headline against the annotator-averaged targets; `labels` stay hard for the confusion matrix
        cad = {**soft_cad_metrics(np.array([r["soft_label"] for r in test_rows]), preds, cad_id), "hard_label_view": cad}
    overall_acc = float((preds == labels).mean())
    mf1 = macro_f1(labels, preds, num_labels)
    cm = confusion_matrix(labels, preds, num_labels)
    by_type = report_by_type(test_rows, preds)
    id2label = {int(v): k for k, v in (label2id or {0: "0", 1: "1"}).items()} if label2id \
        else {i: str(i) for i in range(num_labels)}

    print(f"\n=== TEST  (num_labels={num_labels}, descriptor class = {id2label.get(cad_id, cad_id)}) ===")
    print(f"CAD vs rest:  precision {cad['precision']:.3f}  recall {cad['recall']:.3f}  "
          f"f1 {cad['f1']:.3f}   |  macro_f1 {mf1:.3f}  acc {overall_acc:.3f}")
    print(f"descriptor confusion  tp={cad['tp']} fp={cad['fp']} fn={cad['fn']} tn={cad.get('tn', '-')}")
    print(f"{num_labels}x{num_labels} confusion (rows=gold, cols=pred; order {[id2label[i] for i in range(num_labels)]}):")
    for i in range(num_labels):
        print(f"  {id2label[i]:18s} {cm[i]}")
    print("by example_type (acc):")
    for t, v in by_type.items():
        print(f"  {t:20s} n={v['n']:4d}  acc={v['acc']:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "test_metrics.json").write_text(
        json.dumps({"num_labels": num_labels, "cad_id": cad_id,
                    "cad_vs_rest": cad, "macro_f1": mf1, "accuracy": overall_acc,
                    "confusion": cm, "id2label": id2label,
                    "by_example_type": by_type, "n_test": len(test_rows)}, indent=2)
    )
    print(f"\nmetrics -> {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train/eval the seed-conditioned descriptor classifier")
    ap.add_argument("--data-dir", default="data/processed/datasets/track_a/cad_classifier")
    ap.add_argument("--out-dir", default="models/relevance_classifier")
    ap.add_argument("--model-name", default="microsoft/deberta-v3-base")
    ap.add_argument("--num-labels", type=int, default=0,
                    help="0 = infer from data (3 for three-way, 2 for --binary datasets)")
    ap.add_argument("--freeze-layers", type=int, default=6,
                    help="freeze embeddings + bottom N transformer blocks (0 = full fine-tune)")
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--seed-token-budget", type=int, default=0,
                    help="cap the anchor (name + seeds) at this many tokens and never truncate it "
                         "further; 0 = truncate the longer side to --max-len")
    ap.add_argument("--soft-labels", action="store_true",
                    help="train on each row's soft_label (p3_build_soft_dataset.py) instead of label")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-class-weights", action="store_true",
                    help="disable inverse-frequency class weighting")
    ap.add_argument("--eval-only", action="store_true",
                    help="load --out-dir checkpoint and score the test split only")
    args = ap.parse_args()
    main(args)
