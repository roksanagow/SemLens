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
- [x] Accept negative layer numbers counting back from the final output (`-2` = penultimate); report the available range for an invalid layer.
- [x] Default **Maximum occurrences per corpus** to `all`, accepting `all` or a positive number.
- [x] Remember corpora switched off in a plot legend across projections, definition axes, workspaces and annotation actions; leave them out of selection.
- [x] AMD inspector: choose the corpus pair inside the panel (scores follow), compact three-row scrolling tables, and a closest-usages lookup for one chosen sentence across the other corpora.
- [x] Per-definition table: fixed-width pinned definition column, readable score names, and APD/PRT alongside AMD/SAMD (absolute distance, by necessity for scalars).
- [x] Blind annotation: uniform colour and no corpus in hovers, with corpus names kept in the legend for hiding/showing.
- [x] Stop the help-icon styling from shrinking buttons that have help text (the sense buttons rendered as 24px circles).
- [x] Apply XL-LEXEME's `<t>`/`</t>` target markers automatically for word representations and for the template word in definitions; record the setting, with an opt-out. Saved corpus caches stay unmarked.
- [x] Pool vectors as each sentence is encoded instead of holding every token vector until the end.
- [x] Keep Manual input definition drafts and the chosen definition source across workspace navigation.
- [x] Fit distribution bins to the observed distances; print scores in every heatmap cell up to ten corpora.
- [x] Fit LDA on a limited number of principal components so shuffled corpus labels no longer separate.
- [x] Refresh the sidebar summary and corpus order immediately after loading data.
- [x] Definition anchors as `D1`, `D2`, … in the theme's text colour, with full definitions in hovers.
- [x] Wiktionary: parts of speech with a filter, bracketed usage labels, and German, French, Spanish and Swedish editions.
- [x] `setup.sh` no longer copies a file that is not in the repository.
- [x] Sense × corpus frequency table, chart and download in Annotation; type-to-highlight search in the annotation plot.
- [x] Highlight the target occurrence in hovers and evidence tables; bring it into view in scrolling cells.
- [x] Keep extra CSV columns as usage metadata in saved analyses and the annotation export.
- [x] Heatmaps follow corpora switched off in the legend; change-over-time chart for chronological corpora.
- [x] Full metric names and formulas in help and under heatmaps; projection help; share of variation on PCA axes.
- [x] Accessibility pass: first-screen guide, glossary, plain-language help on every control, example dataset.
- [x] Input options: wildcard forms, exact-case matching, context division (sentences, lines, paragraphs; blank lines and common abbreviations handled), other text encodings, texts kept for another word.
- [x] Review: frequency per million words, word-form counts, sortable and searchable concordance.
- [x] Model presets; display settings for point size, opacity, sequential palette, figure format, order by name.
- [x] Plot options: PCA components, UMAP and t-SNE settings following the vector distance, search in every plot, CSV downloads of plot positions, scores and definition distances.
- [x] One-line launcher: `semlens` command (`uvx --from git+… semlens`); app packaged with its example data; installed copies save to `~/.semlens` or `SEMLENS_HOME`.
- [x] Graded annotation (off by default, in a collapsed **Annotation options** box): a rating for each sense on a user-set scale (default 4), rating picker, graded outlines, mean-rating table and rating threshold for frequencies, `rating` / `sense:<name>` export and import.
- [x] Several senses per usage (off by default, in both normal and graded annotation): click toggles, lasso adds, nested outlines, per-sense list columns, per-sense counts with side-by-side bars, and a guarded switch-off that offers to keep only the first sense.
- [x] Continuous time: a time column in CSV/TSV (years, numbers, ISO dates), corpus names as times, periods of equal length or equal size (no corpus column needed), the Over time view (time against PC1/PC2 or a definition, with a moving average), colour by time, dates in hovers, times in saved analyses and CSVs. The example dataset now carries its publication dates.
- [x] Load previous annotations from the Annotation workspace: a saved analysis (local or file) or an annotation CSV onto the loaded usages, matched by usage ID, then sentence and position, then sentence; with nothing loaded, opens a saved analysis whole.
- [x] Different words: a forms question and form-to-word table in Data & Model; a global Corpus/Word switch with a Words/Corpora shown filter at the top of Explore, Definitions and Annotation; Words by corpus; Words over time; sentence-level definitions by default; `word` column in exports.
- [x] Annotation: list (concordance) annotation, remaining-only view, class renaming and merging, annotation import.
- [x] Cost estimates and safeguards before heavy work (model downloads, embedding, saving corpora, large comparisons, Discover): confirmation above 2 GB of disk, 40% of memory or ten minutes; blocked when less than 5 GB of disk would remain or above 75% of memory; an optional five-second speed trial, and speeds measured by every run.
- [x] Remembered settings between sessions (`settings.json`, changed values only, with a restore button); embedding settings no longer reset when leaving Data & Model, and neither section's settings reset when visiting the other.
- [x] Speed: SAMD matching filters already-matched pairs in chunks (same result, about twice as fast, so Explore's first view of seven corpora takes half the time); scatter figures are kept between reruns unless something they show changes (idle clicks about twice as fast); Discover's word-count summaries are computed once.
- [x] Discover maps: words of the quick first pass in their aligned static embeddings, and in Results the usages of chosen words (seeds ringed) or the words as the average of their usages, with Explore's PCA, UMAP and t-SNE. word2vec is the quick first pass's default.
- [x] Page tabs as one bar on the first line of both sections (the open tab a shade darker); grey example text in empty boxes; standard terms (usage, embedding, subword token) in Discover's help and both glossaries.
- [x] Discover related words, a second section of the app: corpora (text files, a table, the texts of Analyse a word, saved corpora, saved discoveries), restriction to contexts containing given words, seeds as words or usages (a table, a saved analysis, the open analysis's annotations), an optional static first pass (PPMI + SVD or word2vec, aligned by Procrustes) that can shortlist the words to embed, a capped even sample with estimates for several caps, XLM-RoBERTa base layer 8 word vectors or masked slots, kNN-AMD and the share near the concept per word and corpus, background correction, spelling notes, marking and accepting with optional addition to the seeds and incremental re-scoring, and a hand-off to Analyse a word as a Different-words analysis.

## Usability plan (from the walkthrough of 24 November 2026; questions settled 24 November 2026)

Numbers refer to the walkthrough's findings. Done already: **5**, no Deploy button (`toolbarMode` in `.streamlit/config.toml` and in the `semlens` launcher). Not planned: **13** (long help is fine when it carries necessary information), **16** (hidden corpora are not a problem).

Settled: the start page has two sentences on what SemLens does; the example ships with its embeddings; the AMD baseline is leave-one-out and is worded "AMD is 1.4× the AMD within each corpus"; the score animations are built from PNGs the user supplies (2–4 per score), and (?) tooltips can show animated GIFs (tested); autosave as described below; the Discover quick path is presented as the simple alternative to steps 3 and 4; **"sense" everywhere** (on screen and in files, replacing "sense class" and the earlier "class"); corpora named by years are ordered by year automatically; the first-time guide is for Annotation only for now.

### Priority 1 · Done first
- [ ] **No blank bar at the top.** `toolbarMode = "minimal"` (no ⋮ menu; the browser's own print remains) and the header made transparent with no height, with less space above the page. Tested: the button that reopens a collapsed sidebar stays visible and works.

### Priority 1 · Protect the user's work
- [ ] **18 Autosave and recovery.** Save the open analysis automatically: the usages and embeddings once, when they are loaded or embedded (16-bit); afterwards only the small parts that change (annotations, classes, definitions, settings) as a JSON file written atomically on each change. Keep only the latest session (and the one before it), replacing older ones. On the next start, offer "Continue where you left off?" (never automatic). Recovery only reads files: no model run, no embedding, nothing heavy is repeated. Before each write, check free disk against the 5 GB reserve; when short, skip the embeddings (keep annotations keyed by usage ID) and say so once. The same for Discover's marks, which are saved already.
- [ ] **19 Warn before discarding.** Loading other usages, re-embedding or replacing definitions when annotations, classes or definitions exist: name what will be lost ("4 definitions and 120 annotated usages") and offer to save first.

### Priority 1 · First visit and orientation
- [ ] **1, 4, 9 Start page**, shown on the first visit and then only from a "Start page" link in the sidebar: two sentences on what SemLens does (for people who arrive from the landing page, the `uvx` command, a colleague or a workshop rather than the GitHub page); the two sections with one line each (Analyse a word / Discover related words); three choices (Try the example · Use my own texts · Open saved work); the **Advanced options** switch (off by default). A "Don't show this again" choice is remembered with the settings.
- [ ] **2 Example that opens at once**: ship the example's XL-LEXEME embeddings in 16-bit (0.7 MB; the definitions' 16 KB), so the example opens in Explore without a model download. Writing new definitions or embedding one's own data still downloads the model, and the page says so before it does.
- [ ] **3 Status line** under the tabs in both sections, with the next step: Analyse a word — usages loaded · embedded (model) · definitions · annotations; Discover — corpora read · seeds · embedded · scored. Each step's message names the next ("Now explore the usages →"). Remove the contradictory "Analysis restored" + "No analyses saved locally yet".

### Priority 2 · Less on screen at once
- [ ] **9, 10 Advanced options switch** (start page and sidebar, remembered). Hidden when off: model, layer, context window, batch size, target markers, storage precision, pre-computed and saved-embedding data sources, sampling mode, repetitions, sampling seed, vector distance, UMAP/t-SNE settings, Discover's filters, cap table, model, layer, representation, *k*, background correction and ranking options. The sidebar keeps the display settings (theme, palette, point size and opacity, figure format) for everyone; score settings move next to the scores in Explore, and corpus order to Data & Model, both as advanced settings.
- [ ] **8 Second-level tabs.** Explore: Map · Scores · Compare two corpora · Over time. Discover Results: Ranking · Map · Read usages · Save & open in SemLens. Annotation: the plot and the class buttons first, options below.
- [ ] **Discover maps open by default** (the usage map in Results; the word map in the quick first pass).
- [ ] **6 AMD as the headline score** in Explore, with a sentence in words ("2020–21 differs most from 1999–2003"); APD, PRT and SAMD under "More scores"; formulas only in the help.
- [ ] **7 AMD against AMD within each corpus**, in a collapsed "Compared with AMD within each corpus": within a corpus, each usage's distance to its nearest *other* usage of the same corpus (leave-one-out), averaged; the two corpora's values averaged; shown as "AMD is 1.4× the AMD within each corpus". Computed on the same sampled usages as the score. Not for APD or PRT.
- [ ] **6 (later) Animated help for each score**: a short GIF per score (APD, PRT, AMD, SAMD) with ten usages, five in each of two corpora, assembled from 2–4 PNGs per score supplied by the user and shown inside the score's (?) tooltip (tested: tooltips show animated GIFs).

### Priority 3 · Words and guidance
- [ ] **11 Plain names first, technical name in brackets**, explained in the (?) help: e.g. "Closeness to the seeds (kNN-AMD)", "Annotate on: the usage map / the definitions map (full embeddings / definition space)", "Details: model, layer, dimensions" instead of "Representation: word · 1024 dimensions".
- [ ] **12 One vocabulary throughout**: "usage" (not occurrence or use), "sense" (not sense class or class; on screen and in files, whose columns are already `sense` and `sense:<name>`), "Data & Model" (also in headings); interface, help, glossaries, README. Corpora named by years (1999, 1999–2003) are ordered by year automatically, in both sections.
- [ ] **14 Step-by-step annotation guide** for first-time users, at the top of Annotation, with "Don't show this again" (or leave it for next time): ① name a sense ② choose it ③ click or lasso usages ④ read the counts below. No "use definitions as senses".
- [ ] (later) The same first-time guide for Explore, Definitions and Discover, if Annotation's proves useful.
- [ ] **15 Discover marking**: ✓ fits / ✗ does not fit on each usage; marks are recorded by default, with a visible switch "Add the usages I mark to the seeds".
- [ ] **17 Discover**: a "Find related words with the default settings" button on the Seeds step (sample, embed and score in one go, after showing the cost), presented as the simple path, and steps 3 and 4 as the alternative for choosing the words to embed and the settings (with the quick first pass's map); one **Save & export** place in Results: the discovery's name and that it is saved automatically, how to reopen it, and every download (scores, marked usages, word decisions), with opening the words in Analyse a word.

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
- [ ] Discover: multi-word seeds and candidates (phrases such as *echo chamber*), and test the method on a large general corpus; the prototype used only the 350 Guardian *bubble* articles.
- [ ] Discover: rarer uses of frequent words can be missed under a cap; consider sampling more occurrences of words whose uses are spread widely, or a second pass for the top candidates without a cap.
- [ ] Discover: negative seeds (uses marked as not fitting) are recorded but not yet used in scoring.
- [ ] Speed with very large analyses: an annotation click still rebuilds the annotation plot (about 1 s at 30,000 usages), and every interaction reruns the whole workspace. Streamlit fragments could rerun only the part of a page that changed.
