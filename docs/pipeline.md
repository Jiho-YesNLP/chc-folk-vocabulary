# Pipeline Reference

Setup and per-stage commands for reproducing the pipeline. See the [README](../README.md) for an overview.

The code is organized in six numbered stages: (1) inventory and rubric, (2) corpus and retrieval, (3) annotation and relevance filtering, (4) extraction and fingerprints, (5) joint embedding and structure analyses, (6) convergent validation. A script's prefix gives its stage (`p3_filter_passages.py` belongs to Stage 3), each config carries the same prefix (`configs/p3_filter.yaml`), and the `stageN-` prefix on instrument files does the same. Within a stage, sub-steps are lettered (Stage 4, Step A and Step B). The paper does not number stages: Stages 1–4 are its Resource Construction section (Section 3), RQ1's coverage tiers are measured on the Stage 3 random annotation set, Stage 5 holds the RQ2 analyses, and Stage 6 is RQ3.

## Paper results and the commands that produce them

Section, table and figure numbers refer to the submitted manuscript.

| Paper | Content | Script (config) | Output |
|---|---|---|---|
| §3.1, Table A1 | CHC ability inventory | none; committed input | `data/raw/chc_taxonomy.json`, `data/raw/chc_project_annotations.json` |
| §3.2–3.4 | Corpus, indexing, retrieval | `p2_download_reddit_subreddits.py`, `p2_build_index.py`, `p2_retrieve_passages.py` (`p2_*.yaml`) | `data/processed/retrieval/track_a/` |
| §3.5.1 | Annotation scheme and sets | `p3_build_annotation_sets.py`, `p3_annotate.py`, in-conversation protocol | `data/processed/annotation/track_a/` |
| §3.5.2 | Classifier training data | `p3_build_soft_dataset.py` | `data/processed/datasets/track_a/folk_classifier_v10_soft/meta.json` |
| §3.5.2 | Classifier evaluation and 0.55 threshold | `p3_eval_classifier.py` (`p3_filter.yaml`) | `results/classifier_eval/report.md` |
| §3.5.3 | Corpus filtering | `p3_filter_passages.py` (`p3_filter.yaml`) | `data/processed/corpus/track_a_v10/` |
| §3.6 | Extraction and fingerprint assembly | in-conversation protocol + `p4_merge_extract_annotations.py`; `p4_build_fingerprint.py` (`p4_fingerprint.yaml`) | `data/processed/fingerprints/track_a_opus55/` |
| Tables 1–2 (§5.1) | Corpus funnel, per-ability retention | `p3_corpus_stats.py` (`p3_corpus_stats.yaml`) | `results/corpus_stats_v10/report.md` |
| Table 3 (§5.2) | Inter-annotator agreement | `p3_annotation_stats.py` (`p3_annotation_stats.yaml`) | `results/annotation_stats/report.md` |
| Tables 4–5, B1 (§5.3) | Coverage by stratum, by tier, per ability | `p3_annotation_stats.py` (`p3_annotation_stats.yaml`) | `results/annotation_stats/report.md`, `coverage.csv` |
| Figure 3 (§5.3) | Correlates of absence | `p3_absence_routes_figure.py` (`p3_detectability.yaml`) | `results/figures/absence_routes.pdf` |
| Figure 4 (§5.3) | Coverage by PPIK group | `p3_coverage_ppik_figure.py` (`p3_annotation_stats.yaml`) | `results/figures/coverage_by_ppik.png` |
| Appendix C | Detectability ratings and test | `p3_detectability_rate.py`, `p3_detectability.py` (`p3_detectability.yaml`) | `results/detectability/report.md` |
| Tables 6–7, Figure 5 (§5.4) | Cohesion, label-blind recovery, dendrogram | `p5_embed_joint.py` (`p5_joint_embed.yaml`), `p5_cluster_viz.py` (`p5_cluster_viz.yaml`) | `results/track_a_cluster_joint_opus55/` |
| Figure 2 | Cluster explorer | `p5_export_cluster_web_data.py` (`p5_cluster_viz.yaml`) | `webapp/data/joint/` (screenshot of the site) |
| Table 8 (§5.5) | Discriminant validity | `p5_discriminant.py` (`p5_discriminant.yaml`) | `results/discriminant_opus55/summary.md` |
| Table 9 (§5.6) | Representation ablation | `p5_repr_ablation.py` (`p5_repr_ablation.yaml`) | `results/repr_ablation_opus55/repr_ablation.md` |
| Tables 10–11 (§5.7) | WJ IV validation, support split | `p6_validation_pilot.py`, `p6_validation_report.py` (`p6_validation_wj4.yaml`) | `results/validation_wj4_opus55/` |
| Table D1 | Example fingerprint (Gr-FI) | `p4_build_fingerprint.py` | `data/processed/fingerprints/track_a_opus55/Gr-FI.json` |

Figure 1 (pipeline diagram) is `docs/figures/pipeline.png`.

## Prerequisites

- Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).
- A CUDA GPU for the steps marked **(GPU)** below; on Linux `uv` installs the CUDA 12.4 build of PyTorch. All other steps run on CPU.
- `aria2c` and `zstd` for the Reddit download (Stage 2).
- `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for any OpenAI-compatible endpoint) for the API annotation and extraction steps.
- BGE-M3 (`BAAI/bge-m3`) is downloaded automatically on first use.

## Installation

```bash
git clone https://github.com/Jiho-YesNLP/chc-folk-vocabulary.git
cd chc-folk-vocabulary
uv sync --extra viz
cp .env.example .env
```

All commands below run from the repository root.

## Running the pipeline

### Stage 1 — Inventory and rubric

No commands: the ability inventory and `data/raw/annotation/stage3-annotation_criteria.md` are hand-authored, committed inputs. The inventory is two files. `data/raw/chc_taxonomy.json` transcribes, verbatim, Schneider & McGrew's 2017 definitions sheet (their abstract of the 2018 chapter) for the 13 broad abilities it does not mark as tentative and their 71 non-tentative narrow abilities; the tentative Gp, Go, Gh, Gk, Gei and Gf-RE, Gf-RP are excluded, and cross-listed abilities are kept once under one parent. `data/raw/chc_project_annotations.json` holds this project's additions: discriminator glosses, seed sentences, and conceptual groups. Scripts read the two joined through `src/data/inventory.py`, which refuses to load while any ability lacks an annotations entry. Configs point `inventory_path` at the taxonomy file; the annotations file is found next to it.

### Stage 2 — Corpus and retrieval

The corpus is the comment dumps of 54 subreddits (about 21 GB compressed, 131.4M comments) from a Pushshift-format "top 40k subreddits" torrent. Its Academic Torrents listing is no longer online, so the torrent file is committed as `data/raw/reddit.torrent`; the download script uses it instead of fetching one, and downloads from peers that still share the dump. `data/raw/reddit_dump_manifest.json` lists the 54 files with their torrent indices, exact sizes and comment counts, to check a download against.

```bash
# Acquire the raw Reddit .zst dumps (configs/p2_download_reddit.yaml lists the subreddits)
uv run python scripts/p2_download_reddit_subreddits.py configs/p2_download_reddit.yaml --list   # preview only
uv run python scripts/p2_download_reddit_subreddits.py configs/p2_download_reddit.yaml

# (GPU) Build the BGE-M3 hybrid index (dense FAISS + sparse)
uv run python scripts/p2_build_index.py configs/p2_index.yaml --inspect   # preflight scan only
uv run python scripts/p2_build_index.py configs/p2_index.yaml

# Retrieve per-ability candidate pools (label-free, seed-driven, debiased)
uv run python scripts/p2_retrieve_passages.py configs/p2_retrieval.yaml --ability Gr-FI   # smoke test
uv run python scripts/p2_retrieve_passages.py configs/p2_retrieval.yaml --all
# -> data/processed/retrieval/track_a/{uid}_raw.jsonl
```

### Stage 3 — Annotation, relevance classifier, filtering

Two frozen annotation sets per ability are built in one pass, with random drawn first so the two are disjoint. **random** is a uniform sample of the top-1,000 retrieval pool: the measurement set for coverage (RQ1), inter-annotator agreement, and the classifier's held-out evaluation. **ranked** is the top 50 by hybrid score: the classifier's training set. Each rater's judgments are stored in their own file, `{set}/judgments/{uid}/{annotator}@{prompt_version}.jsonl`, so raters never overwrite each other; `{set}/views/` and `summary.csv` are derived from them for display and keep one row per passage.

```bash
# Freeze both sets (no LLM call)
uv run python scripts/p3_build_annotation_sets.py --all
# -> data/processed/annotation/track_a/{random,ranked}/pools/{uid}.jsonl

# Rater 1: Claude Opus 5.5 in conversation, following docs/inconv_annotation_agent.md:
# print a batch of pools, label them, then merge the labels
uv run python scripts/p3_show_pool.py --set random --batch 3 --unjudged-only
uv run python scripts/p3_apply_inconv_labels.py --set random --labels labels.json \
    --annotator opus-5.5-in-conversation --prompt-version v2 --gaps-only
#   labels.json: {uid: {index: label | [label, best_ability_uid]}}

# Rater 2: gpt-6-astra through the API, same v2 instrument
# (system prompt: data/raw/annotation/stage3-api_annotation_prompt_v2.md + the criteria)
uv run python scripts/p3_annotate.py --set random --all --model gpt-6-astra --prompt-version v2
# -> data/processed/annotation/track_a/{set}/judgments/{uid}/{annotator}@v2.jsonl

# Progress and resume point, computed from disk
uv run python scripts/p3_annotation_status.py --set random
```

Repeat both raters on `--set ranked` for the classifier's training labels.

**Relevance classifier** (trained on the ranked set, evaluated on the random set):

```bash
# Soft labels: each target is the two raters' labels averaged
uv run python scripts/p3_build_soft_dataset.py --set ranked \
    --annotators opus-5.5-in-conversation@v2 llm-gpt-6-astra@v2 --n-seeds 5 --fp-ratio 2 \
    --out-dir data/processed/datasets/track_a/folk_classifier_v10_soft

# (GPU) Three seeds, deployed as an average (configs/p3_filter.yaml lists all three)
for s in 42 43 44; do
  uv run python scripts/p3_train_classifier.py --epochs 8 --freeze-layers 3 --soft-labels --seed $s \
      --data-dir data/processed/datasets/track_a/folk_classifier_v10_soft \
      --out-dir  models/relevance_classifier_v10/s$s
done

# (GPU for scoring) Held-out evaluation and threshold choice: scores the random set with every checkpoint
# in the filter config, averages them, pools both raters, sweeps 0.05-0.95 and takes the F1 maximum (0.55)
uv run python scripts/p3_eval_classifier.py configs/p3_filter.yaml
# -> results/classifier_eval/report.{json,md}; per-checkpoint scores cached in results/classifier_eval/scores/

# (GPU) Apply corpus-wide at the operating threshold
uv run python scripts/p3_filter_passages.py configs/p3_filter.yaml --ability Gr-FA   # smoke test
uv run python scripts/p3_filter_passages.py configs/p3_filter.yaml
# -> data/processed/corpus/track_a_v10/{uid}_filtered.jsonl, {uid}_scored.jsonl, coverage_report.json, filter_run_meta.json
```

`p3_eval_classifier.py` caches each checkpoint's scores, so once they exist the evaluation reruns on CPU; it needs both raters' random-set judgment files.

**Coverage, agreement, detectability, corpus statistics:**

```bash
uv run python scripts/p3_annotation_stats.py configs/p3_annotation_stats.yaml   # coverage (mean of the two raters) + agreement
uv run python scripts/p3_corpus_stats.py configs/p3_corpus_stats.yaml           # funnel + per-ability retention
uv run python scripts/p3_detectability_rate.py configs/p3_detectability.yaml --model gpt-4o   # API rater; never overwrites
uv run python scripts/p3_detectability.py configs/p3_detectability.yaml         # needs results/annotation_stats/coverage.csv
uv run python scripts/p3_absence_routes_figure.py configs/p3_detectability.yaml results/figures/absence_routes.pdf
uv run python scripts/p3_coverage_ppik_figure.py configs/p3_annotation_stats.yaml results/figures/coverage_by_ppik.png
# -> results/{annotation_stats,detectability}/report.{json,md}, results/corpus_stats_v10/report.{json,md}
```

The detectability instrument, blinded items and protocol are in `data/raw/detectability/` (see its README). The non-API rater (Fable 5.1) was run as a blind agent pointed at the same files. `p3_corpus_stats.py` needs the index metadata from Stage 2 for the index rows and reports them as unavailable otherwise.

### Stage 4 — Extraction and fingerprints

```bash
# Step A, API path: per-passage LLM extraction (prompt version pinned in configs/p4_extract.yaml)
uv run python scripts/p4_extract_descriptions.py configs/p4_extract.yaml --pilot --limit 10   # smoke test
uv run python scripts/p4_extract_descriptions.py configs/p4_extract.yaml --all [--overwrite]
# -> data/processed/descriptions/track_a/{uid}_extracted.jsonl

# Step A, in-conversation path (the run the paper reports: Claude Opus 5.5, extract-v2, corpus track_a_v10).
# Protocol: docs/stage4-inconv_extraction_prompt.md, rubric: docs/stage4-cad_extraction_rubric.md.
# p4_merge_extract_annotations.select_input gives the same capped candidate selection as the API path.
uv run python scripts/p4_extraction_status.py --pending   # worklist; defaults to the reported run
uv run python scripts/p4_merge_extract_annotations.py <uid> <annotations.json> \
    --annotator llm:claude-opus-5.5 --out-dir data/processed/descriptions/track_a_opus55 [--resume-from N]
# -> data/processed/descriptions/track_a_opus55/{uid}_extracted.jsonl

# Step B (GPU): deterministic fingerprint assembly (BGE-M3 for synonym merging only)
uv run python scripts/p4_build_fingerprint.py configs/p4_fingerprint.yaml --ability Gr-FA   # smoke test
uv run python scripts/p4_build_fingerprint.py configs/p4_fingerprint.yaml [--overwrite]
# -> data/processed/fingerprints/track_a_opus55/{uid}.json, fingerprints.json, fingerprint_report.md,
#    fingerprint_vectors.npz (full build only)
```

### Stage 5 — Joint embedding and structure analyses

```bash
# (GPU) Source-conditioned joint embedding: one vector per span [SEP] source sentence
uv run python scripts/p5_embed_joint.py configs/p5_joint_embed.yaml --ability Gr-FA   # smoke test
uv run python scripts/p5_embed_joint.py configs/p5_joint_embed.yaml [--overwrite]
# -> data/processed/clusters/track_a_opus55/joint_vectors.npz

# Cohesion metrics (kNN purity, k-means ARI), UMAP scatter and dendrogram
uv run python scripts/p5_cluster_viz.py configs/p5_cluster_viz.yaml
# -> results/track_a_cluster_joint_opus55/{metrics.json,metrics.md,scatter.png,dendrogram.png,similarity_heatmap.png}

# Representation ablation: span vs. source vs. span [SEP] source (CPU ok; encodings cached in data/interim/)
uv run python scripts/p5_repr_ablation.py configs/p5_repr_ablation.yaml
# -> results/repr_ablation_opus55/repr_ablation.{json,md}

# Discriminant analysis: which ability pairs folk language fails to separate
uv run python scripts/p5_discriminant.py configs/p5_discriminant.yaml
# -> results/discriminant_opus55/{pairs.csv,summary.json,summary.md}

# Cluster-explorer data for webapp/ (the site serves webapp/ as static files)
uv run python scripts/p5_export_cluster_web_data.py --config configs/p5_cluster_viz.yaml \
    --fingerprints data/processed/fingerprints/track_a_opus55/fingerprints.json --dataset joint
```

### Stage 6 — Convergent validation (WJ IV)

```bash
# Match each WJ IV test description to an ability by label, definition, fingerprint or seeds
# (frozen materials: data/raw/validation/wj4_operationalizations.json)
uv run python scripts/p6_validation_pilot.py configs/p6_validation_wj4.yaml

# Paper tables: candidate pools, support split, error analysis, zero-prevalence targets
uv run python scripts/p6_validation_report.py configs/p6_validation_wj4.yaml
# -> results/validation_wj4_opus55/{summary,report}.{json,md}, per_test.json, scores.npz
```

The report reads zero-prevalence targets from `results/annotation_stats/coverage.csv`, so run Stage 3's `p3_annotation_stats.py` first.
