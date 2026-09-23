# Reference check

Audited against the [paper](https://aclanthology.org/2026.acl-long.795.pdf), our current code, and the clean upstream checkout at [commit 2577f8c](https://github.com/SiunKim/diziner-ner/tree/2577f8ce7f06f0554e88b1931f9e0c9626c24f17). The paper and that code revision differ in places, so matching one does not establish equivalence to the other.

**Verdict:** the selection explanation and recorded pilot metrics are correct. The local implementation follows the broad paper method, but it is not an exact reproduction of the upstream code or its baseline prompts.

## What matches

| Mechanism | Local evidence | Scope of the match |
|---|---|---|
| Fixed label schema; shared and model-specific instructions; task goal | [`prompts.make_prompt`](../prompts.py) | Same configuration structure; locally authored wording |
| No supervisor refinements in iteration 0 | `initial_guidelines`, `save_baselines` | Matches upstream [`_create_ner_prompt`](https://github.com/SiunKim/diziner-ner/blob/2577f8ce7f06f0554e88b1931f9e0c9626c24f17/base_annotator.py#L202); not an identical prompt |
| Mean pairwise span F1 and normalized weights | [`metrics.analyze`](../metrics.py) | Paper §3.3; see upstream weighting difference below |
| Label conflict, conditional type confusion, boundary uncertainty | `metrics.disagreement` | Numerically matches upstream functions for the same vote distribution |
| Elite cumulative weight reaches 0.5 | `metrics.analyze` | Same cutoff; model-name tie-breaking and numerical tolerance are local |
| Four refinement stages | [`pipeline.refine`](../pipeline.py) | Pattern analysis → non-elite diagnosis → integration → organization; phase 2 makes one call per non-elite model, so not necessarily four API calls |
| Top-three iteration/model selection by mean agreement | `pilot`, `final_evaluation` | Follows paper §3.5; mean of three separate scores, not an ensemble prediction |
| Final round has no supervisor phases | `pilot` | Also present in upstream `main_experiments.py`: supervision runs only before the last iteration |
| Full preset’s refinement limits | [`config.full.json`](../config.full.json) | Matches Table 3’s Relaxed values; model pool, dataset, prompts, and encoder remain adaptations |

## Material differences from upstream code

| Area | Pinned upstream behavior | Local behavior and consequence |
|---|---|---|
| Automatic disagreement weights | [`calculate_auto_weights`](https://github.com/SiunKim/diziner-ner/blob/2577f8ce7f06f0554e88b1931f9e0c9626c24f17/utils_disagreement.py#L494) defaults to a 50/50 blend of separately normalized span-F1 and token Cohen’s-kappa weights when explicit weights are absent | Span-F1 only, following the paper. Can change elite membership, consensus, and refinement evidence. Other upstream agreement helpers use span-F1 only; there is no single universal upstream weighting path. |
| Supervisor error reference | [`compute_majority_voting`](https://github.com/SiunKim/diziner-ner/blob/2577f8ce7f06f0554e88b1931f9e0c9626c24f17/error_analysis.py#L171) uses unweighted token votes and first-encountered ties | Weighted BIO votes; ties prefer `O`, then lexical order. Local error counts are token-level and examples are restricted to hotspots; upstream builds richer entity/error documents. |
| Empty predictions | Upstream span-agreement helpers return F1 = 1 for two empty entity sets | F1 = 0; all-zero means use uniform weights. Prevents all-`O` answers from earning perfect selection scores. |
| Hotspots | Main upstream analysis defaults to the 80th percentile, includes all values at the threshold, and can bridge a one-token gap. A separate utility constant is 90; it is not the main pipeline default. | At most `ceil(0.2 × token_count)` ranked tokens, excluding zero scores; merges only adjacent selections. Ties can make the selected sets substantially different. |
| Grouping | [`kmeans_selection`](https://github.com/SiunKim/diziner-ner/blob/2577f8ce7f06f0554e88b1931f9e0c9626c24f17/lexical_diversity_grouping.py#L272) uses normalized embeddings, seed 42 each round, and random empty-cluster fallback | Pilot uses character TF-IDF; full uses a multilingual encoder. Seed is `seed + iteration`, fallback is deterministic, and the local pipeline requires complete disjoint groups. Only the centroid-representative approach is shared. |
| Annotation coordinates | Upstream asks for text/type/character offsets/confidence, then locates mentions by entity-text matching and filters undefined types | Requires token-index spans and fixed-schema labels; validates offsets, retries, and stops on failure. Baseline scores are comparable within this local experiment, not directly to published baseline scores. |
| BIO decoding | Upstream’s `utils_disagreement.extract_spans_from_bio` repairs orphan `I-` tags, but `utils_annotator.convert_bio_to_entities` drops an initial orphan and extends an active entity even if an `I-` type changes | Local `bio_spans` consistently repairs orphan/type-changing `I-` tags. Overlap assignment can match while downstream entity extraction differs. |
| Reporting and failure policy | Upstream has additional coalition diagnostics, optional model removal, optional gold-supervised analysis, and failure fallbacks | Simplified supervisor reports, fixed annotator pool, no gold-supervised refinement, and strict failure rather than accepting an invalid annotation. |

The paired `prompt_comparison` report is an extra local comparison: the highest-agreement changed prompt is tested against its same-model baseline. It is separate from top-three averaging. Schema definitions, financial task wording, retry feedback, and OpenRouter backends are also local choices.

## Stress-test evidence

[`tests/test_reference_audit.py`](../tests/test_reference_audit.py) checks the upstream commit and verifies the source bytes of inspected files. It executes only selected pure functions, without importing upstream API/config modules.

- 2,000 seeded random vote distributions plus five one-tag edge cases: disagreement formulas match within floating-point tolerance.
- 1,000 seeded entity-span cases: token-aligned character conversion and local overlap assignment produce identical BIO labels, including overlaps and repeated spans.
- Explicit counterexamples confirm differences in all-`O` agreement, hybrid weights, supervisor consensus, percentile ties, and the annotation BIO decoder.
- Recomputed every selected/baseline score in the recovered summary from saved predictions and local gold labels; all metrics and the documentation’s source hash match.
- Full suite with the upstream audit enabled: **80 tests passed**. No model calls were made. These tests verify the covered functions and artifacts, not every upstream execution path or live-model equivalence.

To repeat the optional audit, set `DIZINER_REFERENCE` to a clean checkout of the pinned commit when running `uv run pytest tests/test_reference_audit.py -q` from `other_diziner/`. Without that setting, upstream-dependent tests skip; artifact verification also skips if the recovered run is absent.

## Baseline and selection

A **baseline** is a model’s iteration-0 prompt. A **selected candidate** is a model plus a prompt iteration; weights never change. At each iteration, models annotate the same pilot examples and each model receives its mean strict-span F1 agreement with the others.

All candidates are ranked by agreement, descending. Ties use earlier iteration, then model name. The top three are tested against their own model’s baseline. A model may appear more than once, and iteration 0 can be selected. `selected_model.json` contains the first-ranked candidate. Test scores never influence this selection.

Five annotation rounds are numbered 0–4. The four supervisor phases refine instructions for the next round, so the final round has no phase files. Refinement is also skipped when no disagreement hotspots exist.

## Completed pilot

The recovered run used 40 pilot-training examples across five rounds and a separate 40-example English/Thai test set. [Recorded metrics](pilot-results.json) are derived from [the run summary](../runs/pilot-recovered/summary.json).

| Rank | Model / iteration | Pilot agreement | Baseline F1 | Selected F1 | Change |
|---|---|---:|---:|---:|---:|
| 1 | Gemma 3 12B / 4 | 10.97% | 2.85% | 5.34% | +2.49 pp |
| 2 | Mistral Small 3.2 / 4 | 5.97% | 24.26% | 26.74% | +2.49 pp |
| 3 | Mistral Small 3.2 / 0 | 5.82% | 24.26% | 24.26% | +0.00 pp |

The top-three mean rose from **17.12% to 18.78% F1 (+1.66 percentage points)**. Mistral iteration 4 performed best on test data at **26.74% F1**, although Gemma iteration 4 had the highest pilot agreement. For these selected candidates, pilot agreement did not rank test F1 correctly; this is not a correlation study across all candidates. The third candidate is the original Mistral baseline, so its delta is zero.

Mistral refinement reduced false positives from 172 to 115, while true positives fell from 49 to 46: precision improved and recall declined. Overall scores remain low, and 40 examples are insufficient to establish general improvement.

The run records **362 requests** and **$0.367 accounted cost**, including training, retries, and testing. Provider-reported cost is $0.293; 31 requests lacked reported costs and retain conservative accounting.

## Recovery and validation

Training originally stopped when Gemma repeatedly returned an empty span `[53,53)` for `IVL`. The recovery retained cached responses and request costs. More explicit exclusive-end feedback elicited `[53,54)`, and training completed in `runs/pilot-recovered`. See [recovery provenance](../runs/pilot-recovered/recovery.json).

Validation retries malformed JSON, unknown labels, and invalid token offsets, with six attempts per call. Repeated identical invalid outputs trigger a fresh request with correction feedback and no rejected assistant answer. Invalid spans are never silently repaired or accepted. Overlaps follow the normalization policy below.

## Separate language pilots

English-only and Thai-only runs apply the same method independently: filter cleaned training and test pools by the dataset's `lang` field before sampling or grouping, preserve original IDs, keep the shared schema fixed, and start each run from iteration-zero instructions. No prompt is transferred between languages. Each pilot uses three annotators, five rounds of eight training chunks and 40 same-language test chunks. These sample sizes and character TF-IDF grouping are explicit pilot adaptations. `test_size: null` with a language filter means the entire test subset for that language, not the entire bilingual split.

These presets share the local algorithm described above. They do not eliminate the documented differences from upstream or establish equivalence to the published benchmarks.

## Scope of overlap compatibility

Annotation now follows the assignment order in upstream `utils_annotator.convert_entities_to_bio`: start with all-O labels, apply each returned entity in response order, and let later entities overwrite earlier token labels. The local BIO decoder then produces flat spans (repairing orphan I tags as documented). Raw responses and normalized BIO/spans remain auditable. This representation policy does not infer the correct company mentions or repair semantic boundary errors.

The audit confirms BIO assignment equivalence for valid token spans mapped to character boundaries in space-joined text. It does not establish equivalence for arbitrary character offsets, text-matching repair, or all downstream BIO decoders. Nested, duplicate, reversed-order, and crossing spans are regression-tested. Unknown labels, out-of-range token offsets, and malformed JSON still retry and can stop the run: upstream text-based offset matching, undefined-type filtering, and all-O failure fallback have **not** been adopted. The completed recovered-run scores above use this normalization policy.
