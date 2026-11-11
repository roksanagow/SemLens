# SemLens improvement checklist

## Implemented in the current working tree

- [x] Honour explicit target offsets; validate spans and avoid unrelated-word fallback in data loading.
- [x] Find all occurrences of user-specified word forms, preserving matched text and stable occurrence IDs.
- [x] Load 1–10 corpora and show all of them together, with a consistent corpus colour map.
- [x] Keep corpus order explicit; let the user mark it as chronological.
- [x] Support single-corpus exploration/annotation without two-corpus scores.
- [x] Default to smaller-corpus size matching for score comparisons; allow unequal and repeated balanced sampling.
- [x] Explain reuse across sampling repetitions while keeping one point per occurrence in the plot.
- [x] Use absolute distance for per-definition scalar comparisons; display small nonzero scores accurately.
- [x] Confirmed: retain cosine usage-to-definition coordinates and absolute per-definition differences (user decision, 9 November 2026).
- [x] Invalidate stale plot coordinates/selections; preserve annotations when only embeddings or definitions change.
- [x] Support word and pooled-sentence representations and layer selection with a final-output default.
- [x] Retain late targets through overlapping context windows, with an explicit saved window policy.
- [x] Precompute persistent corpus embeddings, including source text, sentence index and model/layer metadata.
- [x] Estimate storage before preparation; offer float32/float16 and token/sentence-only caches.
- [x] Reopen cached corpora across sessions and select new words without running the encoder.
- [x] Save/reopen/export analysis snapshots and annotations.
- [x] Allow advanced definition templates and word/whole-input readout while preserving the basic workflow.
- [x] Make fetched and JSON-imported definitions directly editable; retain drafts during workspace navigation and use the edited entries when creating the definition space.
- [x] Show searchable weakest-counterpart rankings behind AMD in both directions, labelled by representation space.
- [x] Allow a definition axis against PC1/PC2; add optional corpus distributions and descriptive statistics.
- [x] Replace overlapping histogram bars with separate unfilled corpus lines using shared bins, count/proportion controls and comparable hovers.
- [x] Make the direct two-definition plot the default Definitions view, labelled Definition axes, with visible horizontal/vertical selectors and optional PC1/PC2 axes.
- [x] Keep advanced controls collapsed and document interpretation in help text and README.
- [x] Exercise extraction on the supplied testing_texts and embedding/cache equivalence on selected excerpts.
- [x] Make complete sentence columns horizontally scrollable.
- [x] Add light/dark sidebar display settings and green destination-labelled navigation buttons.
- [x] Save/reopen only the loaded word's embeddings and sentences, with size shown before saving; keep whole-corpus saving as the default.
- [x] Match the default definition readout to the usage representation after re-embedding.
- [x] Open diagnostics directly from their expanders, without extra show checkboxes.
- [x] Scale definition heatmaps and use readable short axes with full definitions in hovers and a key.
- [x] Restore coloured active-sense buttons and immediate click/lasso assignment, preserving undo and preventing stale-event replay.
- [x] Match sidebar/toolbar icons and chart backgrounds to the selected theme, including when Streamlit's underlying theme differs.
- [x] Give help buttons subtle theme-aware gradients, borders and hover/focus highlights; keep question-mark strokes visible and match help popup colours in both themes.
- [x] Highlight the active sense with a coloured button border; default to point selection with the ordinary arrow cursor.
- [x] Wrap usage hovers at approximately 75 characters, preserving the exact target highlight; place them above/below the point within the viewport and allow scrolling for unusually long sentences.
- [x] Default to `.txt` uploads and the dark theme; leave unannotated occurrences excluded from annotation review initially.
- [x] Scroll each AMD sentence cell independently in fixed-width columns; preserve search and ranking with pagination for long tables.
- [x] Show APD, PRT, AMD and SAMD heatmaps across every corpus pair in Explore, using the same distance/sampling settings as pair inspection.
- [x] Adapt definition similarity and correlation colour scales to observed values instead of fixing them to −1…1.
- [x] Select a demo direction: “bubble” in Hansard before, during and after the pandemic (user decision, 10 November 2026).
- [x] Add saved-file management: rename without overwriting another file, confirmed deletion, and reopening/editing/saving under the retained name. Apply management controls to saved word embeddings too.
- [x] Investigate newspapers for a broader “bubble” timeline including literal, financial, Covid and AI uses (user steering, 10 November 2026); record public sources and a proposed timeline.
- [x] Prepare a Guardian API collector with time-bin configuration, seeded article sampling, cached resumable requests, provenance and SemLens-ready CSV/text exports; test extraction, sampling, storage and API error handling offline.
- [x] Collect the Guardian “bubble” pilot: seven mini-corpora with 50 occurrences from 50 distinct articles each (350 total), source metadata, CSV/text exports and an import ZIP. Validated dates, exact offsets, article IDs, file hashes and SemLens loading. Flagged nine contexts for review without changing the sample. See [collection instructions](docs/guardian-collection.md).

## Discuss before further implementation

- [ ] Await user verification of the original LSCD score-display issue (not yet checked, 9 November 2026). If it remains, reproduce the specific view/settings; tiny-value rounding, single-definition collapse and stale settings have already been addressed.
- [ ] Review the collected Guardian “bubble” pilot and its nine flagged contexts; agree definitions, annotations and the presentation's qualitative comparisons. The [newspaper plan](docs/demo-bubble-news.md) records the seven-period pilot; the earlier [Hansard proposal](docs/demo-hansard-bubble.md) remains an alternative. The dataset is available, but no semantic-change findings are established yet.
- [ ] Evaluate decoder-only extraction: original target token, sentence-final token, and a target-specific appended prompt. No default is selected yet.
- [ ] Reconcile the local research repo's Euclidean AMD/projection and reciprocal-cosine PRT with the paper's formulations before claiming exact reproduction.
- [ ] Consider additional descriptive views only after using the existing axes and optional distribution plots.

## Follow-up engineering options

- [ ] Resume interrupted whole-corpus preparation (currently incomplete collections are removed and must be restarted).
- [ ] More linguistic sentence segmentation and automatic morphological expansion; current form matching is explicit and raw-text splitting is heuristic.
- [ ] Broader model/language validation and guidance on useful intermediate layers.
- [ ] More efficient very-large-corpus matching; current SAMD still needs the full pairwise distance matrix.
