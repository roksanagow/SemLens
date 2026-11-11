# SemLens

Interactive exploration of word meaning across **one to ten corpora**, using contextual embeddings, definitions, usage comparisons and human sense annotation.

All loaded corpora appear together in a shared plot, coloured by corpus. With one corpus, you can explore and annotate usages. With several corpora, the plot continues to show all of them while separate selectors choose the two corpora for a score comparison.

## Install and launch

Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
streamlit run app/app.py
```

On Windows, activate with `.venv\Scripts\activate`.

The first use of an encoder may download its model files from Hugging Face. The default is `pierluigic/xl-lexeme`. Other Hugging Face encoders or local model directories can be selected under advanced settings.

## A short workflow

1. In **Data & Model**, enter a target word and optional comma-separated forms, such as `run` with `runs, running, runner, runners`.
2. Upload `.txt` corpora (the default), paste sentences, or load a CSV/TSV. Give each corpus a distinct name. Click **Load usages** and review the highlighted target occurrences.
3. Choose **Target word** or **Whole sentence**, then embed the usages. Model, layer and context-window controls are inside **Advanced embedding settings**.
4. In **Explore**, inspect every corpus together. PCA is the default; UMAP, t-SNE and LDA are optional.
5. In **Definitions**, enter or import definitions and create a definition space. **Definition axes** starts with the first two definitions on the horizontal and vertical axes; use the axis selectors to choose other definitions or PC1/PC2.
6. In **Annotation**, create sense classes; each becomes a button. The active sense has a coloured border. Point selection is on by default, with a standard arrow cursor: click an occurrence to assign it immediately. Lasso selection is also available. You can clear a selection and undo annotation changes.
7. Expand **Save or export this analysis** to save locally or download a portable analysis file.

The default theme is **Dark**; choose **Light** or **Dark** in the sidebar. Green buttons at the bottom right lead to the next workspace. Open expandable diagnostic sections to load their content directly. In the AMD pair tables, each sentence cell scrolls horizontally while the neighbouring column stays in place. Other sentence tables scroll horizontally across the grid. Annotation review initially shows only annotated occurrences; tick **Include unannotated occurrences** to include the rest.

## Inputs and target identification

Supported inputs:

| Source | Contents |
| --- | --- |
| Paste sentences | One sentence per line, for 1–10 named corpora |
| Raw corpora | Up to ten UTF-8 text files; each file is a named corpus |
| CSV/TSV | `sentence`, `corpus`; optional `start`, `end`, `sentence_id` |
| Pre-computed vectors | `.pt` or `.npy` with shape `[occurrences, dimensions]`, plus matching metadata rows |
| Saved corpora | Persistent corpus embeddings with their source text and sentence index |
| Saved word embeddings | Only the current word's usage vectors, matching sentences and extraction settings |
| Saved analysis | A `.semlens` snapshot with usages, vectors, definitions and annotations |

Matching is case insensitive and finds every occurrence of the explicit forms. Forms are user-defined: you may intentionally include related lexemes such as *runner*. SemLens does not silently substitute a vaguely similar word.

CSV `start` and `end` are **zero-based character offsets**, with an exclusive end. They are authoritative, including when the target has an unexpected surface form. Invalid, empty or partially specified spans produce an error. Without positions, all exact occurrences are found; pre-computed vectors require explicit positions if a row contains several matching occurrences, to preserve alignment.

The review panel and plot hover highlight the exact selected occurrence. Hover sentences wrap at approximately 75 characters and appear above or below their point, depending on available screen space. Unusually long sentences can be scrolled inside the hover box; no text is truncated. Repeated occurrences in one sentence remain separate observations; whole-sentence mode gives those occurrences the same context representation.

Raw-text sentence splitting is a lightweight punctuation heuristic. Abbreviations, quotations and historical orthography may need correction; use prepared sentences or CSV for precise context boundaries. Source files are retained unchanged in saved corpus collections, alongside the extracted sentences.

## Corpus order and sampling

Expand **Display & comparison settings** in the sidebar to set corpus order and indicate whether it is chronological. Remove and re-add names in the order control, then apply it. The input order is used initially; alphabetical order is never assumed to be chronological.

Sampling applies to **score comparisons**, independently of the shared plot:

- **Equal sizes** (default): randomly select as many occurrences from each compared corpus as the smaller corpus contains.
- **All occurrences**: allow unequal sizes. SAMD matches only the smaller number of occurrences; surplus occurrences in the larger corpus remain unmatched.
- **Repeated equal-size samples**: repeat balanced selection without replacement within each repetition. A usage may participate in several repetitions, but remains one plotted occurrence. Mean, minimum, maximum and standard deviation describe variation between samples, **not a confidence interval**. When both corpora have the same size, repetitions use the same occurrences.

A saved seed makes sampling reproducible. An optional input cap reduces the occurrences loaded and plotted; zero means no cap. Large uncapped comparisons can take substantial time and memory, particularly SAMD.

## Representations and layers

**Target word** averages the selected target's subword-token representations. **Whole sentence** averages the encoder's text-token representations, excluding padding and special tokens. This is contextual mean pooling, not a separately trained sentence-embedding model's output head.

The default layer `-1` uses the encoder's final output, preserving SemLens's original extraction convention. `0` selects input embeddings, and positive integers select intermediate transformer layers. Final output and a model's last intermediate hidden-state tensor are not necessarily identical; the saved metadata records the chosen extraction source.

Long sentences are encoded in overlapping windows. A token found in overlapping windows uses its first encoded occurrence. This preserves late targets instead of silently replacing truncated targets. The window size is part of the embedding configuration; long-context representations should be interpreted with that windowing policy in mind.

Decoder-only and encoder-decoder extraction strategies remain outside the supported workflow pending evaluation.

## Persistent corpus embeddings

To prepare a collection for future sessions:

1. Paste or upload the **complete corpora** in Data & Model. A target word need not be selected for this preparation step.
2. Expand **Prepare reusable corpus embeddings**.
3. Leave **Embedding save scope** at **Whole corpus** (the default). Choose **Word and sentence representations** or **Sentence representations only**, and a storage precision.
4. Click **Estimate storage**. The estimate includes vectors, copied source text, extracted sentences and an allowance for indexing. It may require downloading the tokenizer/configuration, but does not run the encoder.
5. Click **Compute & save corpus embeddings**.
6. Later, choose **Saved corpora**, select the collection, and enter any target and forms. Word selection does not call the encoder again.

A token cache supports both target-span and pooled sentence representations. A sentence-only cache is smaller and cannot recover individual word representations. The default float32 storage preserves the computed precision; float16 halves vector storage with rounding.

Each immutable collection saves one selected layer and records the model ID, available model revision, tokenizer class, layer/readout, dimensions, precision, context window and windowing policy. Another model or unsaved layer needs a new collection. Interrupted computations are not published as usable caches; restart the preparation if it was interrupted.

Collections are stored under:

```text
.semlens/corpora/<collection-id>/
    index.sqlite    # original corpus text, sentence text/positions, embedding settings
    vectors.npy     # memory-mapped numeric vectors
    offsets.npy     # token character offsets, for token caches
```

These files persist across application restarts and are excluded from Git. The collection directory is self-contained: it can be moved, and reopened using **Open a corpus cache from another location**. Keep its files together. NumPy arrays use memory mapping so the entire vector collection need not be loaded to inspect a word.

The numeric storage estimate is approximately:

```text
number of saved vectors × embedding dimensions × bytes per value
```

For one 1,024-dimensional layer, float16 needs approximately 205 MB for 100,000 sentence vectors, or 6.14 GB for 3 million token vectors, before text/index overhead. Float32 doubles the numeric payload.

For a smaller file, choose **Current word only** after embedding the loaded usages. This saves exactly those occurrences (including any applied selection cap), their complete sentences, target offsets, forms, vectors and model/layer/representation metadata. It stores the computed word or sentence representation, without copying unrelated source text. **Prepare word embedding file** shows the actual compressed size before saving or downloading; it does not rerun the encoder. Local files live in `.semlens/words/<name>.semlens` and reopen through **Saved word embeddings** after restarting the app. Reusing a name replaces that file. For another word or unsaved representation, return to the original corpora or a complete corpus cache.

## Saving an analysis

**Prepare analysis file** displays the actual exported file size before saving. **Save analysis locally** writes `.semlens/projects/<name>.semlens`; **Download analysis** provides a portable copy. An opened analysis retains its filename, so you can edit definitions, annotations or settings and save an updated snapshot. Change **Analysis name** to save a separate copy. If a different saved analysis already uses that name, explicitly select **Replace existing saved analysis** before saving.

Under **Data source → Saved analysis**, select a file and open **Manage saved analysis** to rename or delete it. Renaming preserves its contents and will not overwrite another file. Deletion requires a separate confirmation and removes only the selected local snapshot; an analysis already open in this session and separately saved corpus embeddings stay available. The same controls are available under **Saved word embeddings**. To edit the analysis contents, open it, make changes in the workspaces, then prepare and save it again.

Reopen through **Data source → Saved analysis**. A snapshot includes:

- The current target, forms, occurrence IDs, exact sentences and character positions.
- Current usage embeddings and their model/layer settings.
- Definitions, definition vectors and formatting settings.
- Sense classes and annotations, linked to occurrences.
- Corpus order, sampling/distance settings, and source text when available.

A snapshot preserves the current analysis. Complete token caches remain separately available through **Saved corpora**; they are not copied into every snapshot. The snapshot's current analysis can still be reopened without its corpus cache or original uploads.

Annotation CSV export can include unannotated occurrences. Changing representation or definitions preserves existing annotations on the same occurrences. Changing the target data only retains annotations whose occurrence identity still matches.

## Definition space and interpretation

By default, definitions use the input `word: definition` and match the usage representation: target-word extraction for word usages, whole-input pooling for sentence usages. Each usage then becomes a vector of **cosine distances to the definitions**: smaller means closer. These coordinates are distances, not probabilities or verified sense assignments.

Fetched Wiktionary definitions and JSON imports appear in an editable text box, with one definition per line. Revise, add or remove entries, then click **Create definition space** to apply them. Draft edits survive navigation between workspaces during the session; existing plots continue to use the last created space until you apply the changes. Fetching again replaces the current dictionary draft.

Under **Advanced definition settings**, choose `word: definition`, `definition: word`, the definition alone, or a template using `{word}` and `{definition}`. Independently choose target-word or whole-input pooling. A template preview shows the actual input. With sentence usage representations, the initial definition readout defaults to whole-input pooling. The encoder and layer are shared with the usage embeddings.

**Definition axes** is the default view in Definitions. With two or more definitions, it initially plots distance to definition 1 against distance to definition 2. The **Horizontal axis** and **Vertical axis** selectors support any definition or PC1/PC2 fitted on the joint definition space. With just one definition, the vertical axis defaults to PC1. PCA, UMAP, t-SNE and LDA remain available under **Plot view**. PCA and LDA definition anchors are transformed with the same fitted model as the usages. Anchors are not added to UMAP/t-SNE plots.

**Definition distributions and descriptive statistics** shows a separate coloured line for each corpus, connecting counts or proportions in 30 shared distance bins. Lines have distinct dash styles and no filled areas or smoothing. Hover to compare the corpora within a bin; click legend entries to hide/show a corpus or double-click to isolate one. The table reports count, mean, median and population standard deviation of definition distance. These describe model distances, not sense frequencies.

The similarity and correlation heatmaps scale with the number of definitions. Their colour limits adapt to the observed values, so small differences remain visible; consult the numeric values and colour bars when comparing matrices. Short axis labels (`D1`, `D2`, …) refer to a complete definition key below; hovering a cell displays both definitions.

LDA deliberately uses corpus labels to seek separation. With two corpora the view uses LD1 plus residual PC1; with more corpora it uses LD1/LD2 when available. Definition weights operate on **distance features**: positive means increasing distance increases LD1. Interpret weights alongside corpus positions and mean distances; a sign alone is not evidence that a sense became more frequent.

## Scores and usage evidence

| Score | What it compares |
| --- | --- |
| APD | Every cross-corpus usage pair; also reflects variation within the corpora |
| PRT | The two corpus centroids |
| AMD | Each usage's closest counterpart, averaged in both directions |
| SAMD | A greedy one-to-one matching |
| Directional AMD | Closest counterparts for one corpus looking into the other |

The vector distance setting chooses cosine or Euclidean distance for multidimensional comparisons. **Per-definition scores always use absolute distance `|x-y|`**, because cosine distance between positive scalar features collapses to zero. A definition space with just one definition also uses absolute distance for its scores. The embedding-to-definition projection itself remains cosine distance.

Scores are computed in the full representation, not on the plotted 2D coordinates. Small nonzero scores are displayed in scientific notation instead of rounded to zero. Cosine PRT is labelled undefined if a corpus centroid has no direction.

With two or more corpora, **Explore** displays APD, PRT, AMD and SAMD as four heatmaps covering every corpus pair. Each uses its own colour scale and the sidebar's distance and sampling settings. Hover a cell for its score, sample sizes and sample min–max. The diagonal is left blank. Balanced sampling matches sizes separately for each pair; sample sizes can therefore differ across the matrix. Reordering corpora or reversing the pair selectors preserves the scores for a given pair and seed. All-pair results are cached, but an initial comparison of ten corpora requires 45 pair calculations.

Expand **Inspect the usage pairs behind AMD** for ranked tables in both directions, with the weakest counterparts first. Type part of a sentence to search either member of a pair. Each sentence has its own horizontal scroll area; tables show up to 50 pairs per page, with ranking and search covering all pairs. When sampling is repeated, the sample selector uses the exact occurrences underlying that repetition's scores. The displayed distances and scores use the same representation and distance setting. Explore and Definitions each have an inspector because the nearest counterparts can change between full embeddings and definition space; the panel title identifies its space.

A high directional score identifies weak counterparts in the chosen comparison. Whether this reflects semantic emergence/loss, topical variation or corpus composition requires inspection of the usages.

## Development and verification

For the planned newspaper demo, [Guardian collection instructions](docs/guardian-collection.md) explain how to search for `bubble`/`bubbles` and export small, dated corpora with source metadata. A developer API key is required for live collection.

```bash
pip install -e ".[all]"
python -m pytest -q
```

Tests cover explicit offsets, word forms, one/many corpora, scalar distances, balanced sampling, LDA, shared plots, annotation invalidation, cache reuse and saved-analysis round trips. Optional integration tests use a small locally cached real encoder without network downloads. When present, the user-supplied files in `testing_texts/` exercise corpus extraction and selected-excerpt embedding/cache equivalence.

The core library is in `semlens/`; Streamlit workspaces are separated into `app/ui_data.py`, `app/ui_explore.py`, `app/ui_definitions.py` and `app/ui_annotations.py`.

## Citation

```bibtex
@inproceedings{goworek-dubossarsky-2026-rethinking,
    title = "Rethinking Metrics for Lexical Semantic Change Detection",
    author = "Goworek, Roksana and Dubossarsky, Haim",
    booktitle = "Proceedings of the 6th International Workshop on Computational Approaches to Language Change (LChange'26)",
    year = "2026",
    pages = "147--161",
    url = "https://aclanthology.org/2026.lchange-1.13",
}
```

The tool builds on [Rethinking LSCD Metrics](https://github.com/roksanagow/Rethinking_LSCD_Metrics) and [projecting_sentences](https://github.com/roksanagow/projecting_sentences). The local research repository and paper describe different distance conventions in some places; exact reproduction of historical experiment results remains to be reconciled.

## License

MIT
