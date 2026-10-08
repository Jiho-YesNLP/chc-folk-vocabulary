# Detectability of Individual Differences — Frozen Instrument v3

A blind rating instrument that asks, for each CHC narrow ability, whether **individual differences** in that ability can be detected by an ordinary person using unaided senses, or whether detection requires apparatus. It is the non-linguistic predictor of folk-descriptor coverage reported in the paper's coverage results and its detectability appendix. The analysis is exploratory (see Limitations).

## Why this exists

An earlier predictor ("lexicalization": does ordinary English have a ready-made expression for this ability?) was discarded as circular: an LLM's judgment that English lacks a term is itself a distributional fact about English text, so correlating it with corpus prevalence measures the same quantity twice. Detectability replaces it because it is a claim about **measurement physics**: it would hold in a language that had never named any of these abilities, and is therefore constructively independent of the corpus outcome it predicts.

## Files

- `rubric_detectability_v3_FROZEN.txt`: the frozen instrument. SHA-256 `c45a72b2ce9409eeba124499ae3c5e4cf8277ceb1eb15916a5760f71120cec6b`, frozen 2026-09-23T14:37:29Z. **Do not edit.** Any change requires a new version number and a fresh run.
- `blind_definitions_tax2018.json`: the 71 current definitions (from `../chc_taxonomy.json`) as presented to raters: `{id, definition}` only, with ability name, code, uid and broad stratum stripped, in randomized order (seed 772). This is the reported run.
- `blind_key_tax2018.json`: maps opaque `item_NNN` ids back to ability uids. Held separately so raters never see it.
- `blind_definitions.json`, `blind_key.json`: the same for the earlier application to the 84 pre-revision definitions (seed 771), kept because the paper reports that run as a replication.

## Protocol

Each rater sees only the rubric and the blinded definitions, with no access to coverage results or the corpus. Raters read the rubric **from the file**, so all raters receive byte-identical instructions.

Categories are SPONTANEOUS / ELICITED / INSTRUMENTED. The load-bearing contrast is **INSTRUMENTED vs. rest**; the three-way scale is reported descriptively only, because it is not model-robust (see below).

## Results (71 current definitions)

One rater pair from two providers: GPT-4o (through the API) and Fable 5.1 (as a blind agent pointed at the same files). Neither annotated the corpus.

| rater pair | 3-way κ | INSTRUMENTED agreement | INSTRUMENTED κ |
|---|---|---|---|
| GPT-4o × Fable 5.1 | 0.432 | 67/71 | **0.769** |

- **INSTRUMENTED by both raters (n = 8):** Gps-MT, Gt-IT, Gt-R2, Gt-R4, Gt-R7, Gv-IL, Gv-PI, Gv-PN, with prevalence 0–2%.
- **Disagreements (4):** Ga-UP, Ga-US, Gt-R1, Gv-SR. In each, GPT-4o rated INSTRUMENTED (applying the resolution boundary strictly) and Fable 5.1 rated ELICITED (applying an exception).
- **Against coverage** (mean of the two corpus annotators): median prevalence 0.0% for the INSTRUMENTED class vs. 4.0% for the other 59 abilities (Mann-Whitney U = 70, one-sided p = 6.5e-04); zero prevalence in 5/8 vs. 9/59 (Fisher one-sided p = 0.0079).

**Gate, not dial.** INSTRUMENTED predicts whether an ability has *any* folk coverage and carries no information about how much: in the earlier 84-item run (below), the rank correlation with prevalence among abilities with non-zero coverage was null for every rater (ρ between −0.13 and 0.21, all p > 0.1). The current pipeline does not recompute it.

**The three-way scale is not model-robust.** Holding the rubric fixed and swapping raters moves 3-way κ widely while INSTRUMENTED κ stays high. The SPONTANEOUS/ELICITED boundary measures rater temperament more than the abilities, so only the INSTRUMENTED contrast carries a claim.

## Earlier run (84 pre-revision definitions)

The same frozen rubric was applied to the 84 definitions of the pre-revision inventory by six raters in three pairs (INSTRUMENTED κ = 0.64–0.82). It gave the same INSTRUMENTED class with Gt-R1 in place of Gps-MT, and the same zero-prevalence contrast. Its ratings are in `data/processed/detectability/ratings/`.

## Limitations

- **The rubric was developed while its author could see the coverage results.** Each revision is defensible on its own conceptual terms, but the instrument is exploratory until it is applied to a corpus or taxonomy it was not fitted against.
- **Raters are LLMs, and the reported class rests on one pair.** The construct is non-linguistic but the raters are language-derived. Human coding would address both.
- **Gt-R1 is an open case.** It is rated INSTRUMENTED by GPT-4o and by five of the six earlier raters, yet carries on-target descriptors. The likely reconciliation is that folk "reflexes" is a visible-action composite that lands on Gt-R1 as the nearest narrow ability.

## Reproduction

```
uv run python scripts/p3_annotation_stats.py configs/p3_annotation_stats.yaml      # coverage -> results/annotation_stats/coverage.csv
uv run python scripts/p3_detectability_rate.py configs/p3_detectability.yaml --model gpt-4o   # API raters; never overwrites
uv run python scripts/p3_detectability.py configs/p3_detectability.yaml            # kappa, Mann-Whitney, zero-prevalence table
```

Non-API raters (Fable 5.1) were run as blind agents pointed at the same rubric and definition files. Ratings are in `data/processed/detectability/ratings_tax2018/v3_{rater}.jsonl`; the report is written to `results/detectability/report.md`.
