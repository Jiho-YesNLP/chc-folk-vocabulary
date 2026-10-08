# Configs

One YAML per pipeline step. Each script takes its config as the first argument (`uv run python scripts/<script>.py configs/<config>.yaml`), and every path in a config is relative to the repository root. Full commands are in [`docs/pipeline.md`](../docs/pipeline.md).

**Naming.** Each config is named `p<stage>_<step>.yaml`, with the same stage prefix as the scripts that read it (`p3_filter.yaml` drives `p3_filter_passages.py` and `p3_eval_classifier.py`). `track_a` in data paths is the Reddit folk corpus, the only corpus in the study. Config names are model-agnostic: a config names a pipeline step, and the model it depends on is recorded inside it, in the informational `extraction_annotator` key (no script reads it) and in its paths. The `_opus55` suffix appears only on data and result directories, where it marks outputs of the Claude Opus 5.5 extraction (`data/processed/descriptions/track_a_opus55`), the run the manuscript reports; `_v10` marks the corpus filtered by the `relevance_classifier_v10` ensemble. `_wj4` marks the RQ3 validation against WJ IV.

## Configs by stage

| Stage | Config | Script | Output | Manuscript |
|---|---|---|---|---|
| 2 · Corpus | `p2_download_reddit.yaml` | `p2_download_reddit_subreddits.py` | `data/raw/corpus/track_a/reddit/` | Corpus Acquisition; corpus stats table |
| 2 · Index | `p2_index.yaml` | `p2_build_index.py` | `data/processed/index/track_a/` | Semantic Retrieval |
| 2 · Retrieval | `p2_retrieval.yaml` | `p2_retrieve_passages.py` | `data/processed/retrieval/track_a/` | Semantic Retrieval |
| 3 · Filter | `p3_filter.yaml` | `p3_filter_passages.py`, `p3_eval_classifier.py` | `data/processed/corpus/track_a_v10/` | Relevance Filtering |
| 3 · Annotation stats | `p3_annotation_stats.yaml` | `p3_annotation_stats.py`, `p3_coverage_ppik_figure.py` | `results/annotation_stats/` | RQ1 coverage tables; Inter-Annotator Reliability |
| 3 · Detectability | `p3_detectability.yaml` | `p3_detectability_rate.py`, `p3_detectability.py`, `p3_absence_routes_figure.py` | `data/processed/detectability/ratings_tax2018/`, `results/detectability/` | RQ1: detectability (exploratory); detectability appendix |
| 2–3 · Corpus stats | `p3_corpus_stats.yaml` | `p3_corpus_stats.py` | `results/corpus_stats_v10/` | Corpus Statistics tables |
| 4 · Extraction | `p4_extract.yaml` | `p4_extract_descriptions.py`, `p4_merge_extract_annotations.py`, `p4_extraction_status.py` | `data/processed/descriptions/…` | Fingerprint Construction, Step A |
| 4 · Fingerprints | `p4_fingerprint.yaml` | `p4_build_fingerprint.py` | `data/processed/fingerprints/track_a_opus55/` | Fingerprint Construction, Step B |
| 5 · Joint vectors | `p5_joint_embed.yaml` | `p5_embed_joint.py` | `data/processed/clusters/track_a_opus55/joint_vectors.npz` | Structure Recovery (input to every RQ2 analysis) |
| 5 · Cohesion | `p5_cluster_viz.yaml` | `p5_cluster_viz.py`, `p5_export_cluster_web_data.py` | `results/track_a_cluster_joint_opus55/`, `webapp/data/joint/` | RQ2: kNN purity, ARI, cluster and dendrogram figures |
| 5 · Ablation | `p5_repr_ablation.yaml` | `p5_repr_ablation.py` | `results/repr_ablation_opus55/` | RQ2: representation ablation |
| 5 · Discriminant | `p5_discriminant.yaml` | `p5_discriminant.py` | `results/discriminant_opus55/` | RQ2: which ability pairs collapse |
| 6 · Validation | `p6_validation_wj4.yaml` | `p6_validation_pilot.py`, then `p6_validation_report.py` | `results/validation_wj4_opus55/` | RQ3: validation tables and per-test breakdown |

Stage 1 (the ability inventory and annotation rubric) and the Stage 3 annotation and classifier scripts take command-line arguments, not configs.

## Notes

- **Shared settings must stay in sync.** The `[SEP]` template and encoder (BGE-M3) are shared by `p5_joint_embed.yaml`, `p5_repr_ablation.yaml` and `p6_validation_wj4.yaml`. `exclude_abilities` is empty in all four analysis configs (cohesion, ablation, discriminant, validation): the taxonomy keeps each cross-listed ability once (RS, WS under Grw; P under Gs), so there are no duplicates to exclude. Keep the four lists identical if one is ever added. `knn_k: 10` is shared by cohesion, ablation and discriminant. `random_seed: 17` is shared by cohesion, discriminant and validation; the ablation uses its own subsample seeds, `[42, 43, 44]`.
- **`p4_extract.yaml` serves two paths.** Its `corpus_dir`, `extract_cap`, `extract_floor` and `extract_seed` define which passages are extracted, for both the API path and the in-conversation path. Its `model: gpt-4o` and `output_dir` apply only to the API path. The manuscript's descriptions came from the in-conversation path (Claude Opus 5.5), written with `p4_merge_extract_annotations.py --out-dir data/processed/descriptions/track_a_opus55`.
- **`p3_filter.yaml` matches the manuscript's run** as recorded in `data/processed/corpus/track_a_v10/filter_run_meta.json`: the three `relevance_classifier_v10` checkpoints averaged, threshold 0.55, 5 seeds per anchor averaged over 4 draws. Anchors use the inventory's seeds; there is no external seed override. `p3_eval_classifier.py` takes the same config, so the held-out evaluation scores exactly what the filter applies.
