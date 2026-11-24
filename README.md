# SemLens

> **New here?** The [SemLens page](https://roksanagow.github.io/SemLens/) walks through setup and the tool with screenshots.
>
> Problems or questions? Email Roksana Goworek at [r.goworek@qmul.ac.uk](mailto:r.goworek@qmul.ac.uk) or [open an issue](https://github.com/roksanagow/SemLens/issues).

Interactive exploration of word meaning across **one to ten corpora**, using contextual embeddings, definitions, usage comparisons and human sense annotation.

All loaded corpora appear together in a shared plot, coloured by corpus. With one corpus, you can explore and annotate usages. With several corpora, the plot continues to show all of them while separate selectors choose the two corpora for a score comparison.

## Install and launch

### One line, no download of the repository

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
uvx --python 3.12 --from git+https://github.com/roksanagow/SemLens semlens
```

This fetches SemLens and its requirements into a private environment and opens the app in your browser. The first run takes several minutes, mostly for PyTorch; later runs start at once. Everything runs on your computer: the language model is downloaded to it on first use, and saved corpora, embeddings and analyses go to a `.semlens` folder in your home folder.

<details>
<summary>Options for the one-line launcher</summary>

- Anything after `semlens` is passed to Streamlit, for example `semlens --server.port 8502`.
- To keep saved files somewhere else, set `SEMLENS_HOME` to a folder before launching: `SEMLENS_HOME=~/Documents/SemLens uvx … semlens`.
- To pick up a newer version later, add `--refresh` after `uvx`.
- The command prints the folder it saves to when it starts.

</details>

### From a copy of the repository

Python 3.10 or newer. Run these from the top folder of the repository (the one that contains `pyproject.toml`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
streamlit run app/app.py
```

On Windows, use `python -m venv .venv` and activate with `.venv\Scripts\activate`. Run this way, saved files go to `.semlens` inside the repository folder.

<details>
<summary>Or let the setup script do it (macOS and Linux)</summary>

```bash
bash setup.sh --run
```

This creates `.venv`, installs everything and launches the app. Without `--run` it installs only. It also answers Streamlit's first-run request for an email address with a blank one, so the app starts without that prompt; with the manual steps above, press Enter when Streamlit asks.

</details>

<details>
<summary>If a command is not found, or the app does not start</summary>

- **`python: command not found`**: macOS and many Linux systems only have `python3`, which is why the steps above use it. Once the environment is activated, `python` works too.
- **`No such file: app/app.py`, or `pip install -e` fails**: you are not in the top folder of the repository. `cd` into the folder that contains `pyproject.toml` and try again.
- **`streamlit: command not found`**: the environment is not active in this terminal. Run `source .venv/bin/activate` again.

</details>

The first use of an encoder downloads its model files from Hugging Face and keeps them for later runs. The default, `pierluigic/xl-lexeme` (about 560 million parameters), is a 2.2 GB download; because its repository holds only the older `.bin` weights, transformers also saves a converted copy, so it takes about 4.5 GB of disk space. Other encoders are chosen under **Advanced embedding settings** → **Language model**: any Hugging Face encoder by name or page link, or a model from GitHub or elsewhere once its files are downloaded into a folder (give the folder's path). Below the choice, SemLens estimates the download and disk space from the model's parameters (roughly 4 GB per billion parameters), looking typed-in models up on Hugging Face. Embedding needs about as much free memory as the model's size.

## A short workflow

The first screen opens with **New here? How SemLens works**, and the sidebar has a **Glossary** of the terms used. To try the tool before preparing data, choose **Example: “bubble” in the Guardian, 1999–2026** under **Data source**, load it and embed it.

1. In **Data & Model**, enter a target word and optional comma-separated forms, such as `run` with `runs, running, runner, runners`.
2. Upload `.txt` corpora (the default), paste sentences, or load a CSV/TSV. Give each corpus a distinct name. Click **Load usages** and review the highlighted target occurrences.
3. Choose **Target word** or **Whole sentence**, then embed the usages. Model, layer and context-window controls are inside **Advanced embedding settings**.
4. In **Explore**, inspect every corpus together. PCA is the default, with each axis labelled by its share of the variation; UMAP, t-SNE and LDA are optional. The **(?)** beside **Plot view** explains what each view shows.
5. In **Definitions**, enter or import definitions and create a definition space. **Definition axes** starts with the first two definitions on the horizontal and vertical axes; use the axis selectors to choose other definitions or PC1/PC2.
6. In **Annotation**, create sense classes; each becomes a button. The active sense has a coloured border. Point selection is on by default, with a standard arrow cursor: click an occurrence to assign it immediately. Lasso selection is also available. You can clear a selection and undo annotation changes. Turn on **Blind annotation** to annotate without seeing which corpus a usage comes from. **Annotation options** (collapsed) lets a usage have several senses and lets each sense be rated.
7. Expand **Save or export this analysis** to save locally or download a portable analysis file.

The default theme is **Light**; choose **Light** or **Dark** in the sidebar. Tables and list editors follow Streamlit's own theme, which is light; to make them dark as well, start SemLens with `--theme.base dark` (after `semlens` or `streamlit run app/app.py`). Green buttons at the bottom right lead to the next workspace. Open expandable diagnostic sections to load their content directly. In the AMD pair tables, each sentence cell scrolls horizontally while the neighbouring column stays in place. Other sentence tables scroll horizontally across the grid. Annotation review initially shows only annotated occurrences; tick **Include unannotated occurrences** to include the rest.

**Remembered settings.** Display and comparison settings (theme, palette, distance, sampling, point size and opacity, figure format), the embedding settings in Data & Model (representation, model, layer, context window, batch size, target markers) and Discover's settings (filters, cap, model, layer, representation, ranking options) are kept in `settings.json` in SemLens's storage folder (`.semlens` in the repository, `~/.semlens` for an installed copy, or `SEMLENS_HOME`), and come back the next time SemLens opens. Only settings changed from their defaults are written, so improved defaults in later versions still reach you. Words, corpora, seeds and annotations are not remembered this way: save an analysis or a discovery for those. **Remembered settings** at the bottom of the sidebar restores every default.

Click a corpus name in a plot legend to hide or show its usages; double-click to show that corpus alone. The choice is remembered when you change the projection or the definition axes, move between Explore, Definitions and Annotation, or annotate. A hidden corpus stays in the legend, greyed out, and is left out of click and lasso selection.

**Blind annotation** (Annotation workspace) draws every usage in the same colour and removes the corpus from the hover text. Corpus names remain in the legend and still hide or show their usages. LDA is unavailable while blind, because it arranges usages by corpus.

## Discover related words

The sidebar's **Section** switch opens a second part of the app, **Discover related words**. Instead of one word, it looks at every word of the corpora and ranks them by how close their usages are to a set of **seeds**: words you choose, or particular usages, such as sentences annotated with one sense in SemLens. It answers questions like “which words were used like *bubble* (a closed social group) in each period?”. It works in five steps, each explained on screen.

1. **Corpora**: text files (one per corpus, up to ten), a CSV/TSV table with a text column and a corpus column, the texts already loaded in Analyse a word, or a saved corpus collection. **Only contexts containing** keeps only the sentences (or paragraphs) with given words or phrases, which can be other than the seeds; everything after it is then quicker. **Read the corpora** counts every word.
2. **Seeds**: seed words (commas between them, Enter to count them; `cocoon*` for any ending), optionally only from some corpora (to ask which words in older texts did the job of a modern term), and seed usages from a table (`sentence`, optional `start`/`end`), a saved analysis or the analysis open in Analyse a word (chosen sense classes, optionally a minimum rating). Seeds from several different words work best; with one or two, the page says why.
3. **Quick first pass** (optional): one static embedding per word and corpus, from word2vec (the default) or PPMI + SVD (often steadier below about a million words per corpus), rotated so that the corpora share one space (orthogonal Procrustes). **Map of the words** draws the top words and the seed words (as stars) in two dimensions (PCA, UMAP or t-SNE, as in Explore), coloured by corpus; since the corpora share one space, a line can join each word across them. It takes seconds and no language model, finds words that appear in the same texts as much as words used alike, and can shortlist the words that step 4 embeds.
4. **Sample & embed**: words are filtered (minimum occurrences, length, English function words, words you list) and each keeps at most a set number of usages per corpus (the **cap**, 100 by default), drawn at random from all of them. A table shows what caps from 25 to all would mean in occurrences and disk space, and a note says what the cap risks: rarer senses of a frequent word may be missed. The model is XLM-RoBERTa base by default (layer 8 of 12, about two thirds of the way up). Each usage is represented by the embedding of the word itself (**Target word**), or of the model's mask token put in its place (**Masked target**, slower: one model pass per usage). **Prepare the sample** shows exactly what will be embedded and what it costs before anything starts. A saved corpus collection holding word embeddings can be reused without running a model. The embeddings are kept on disk as a **discovery**, which can be reopened later.
5. **Results**: every sampled usage is compared with its *k* nearest seeds (10 by default), not with their average, so a word can rank high when it resembles only some seeds. For each word, in each corpus and over all corpora:
   - **kNN-AMD**: the average distance from its usages to their nearest seeds; lower is closer.
   - **Near the concept**: the share of its usages at least as close to the seeds as a typical seed is to the other seed words. A word with one fitting sense among others shows up here. It needs seeds from at least two words.
   - **Spelling**: a note when the word shares a subword token with a seed word. Models build a word split into subword tokens partly from those pieces, so *bust* (bu+st) can resemble *bubble* (bu+bble) for its spelling alone; such words can be hidden.
   - **Background correction** (off by default, suggested for seeds from one or two words): each usage's distance to the seeds minus its distance to usages in general, so that words close to everything lose their advantage.

   A heatmap scores the top words in each corpus. **Map of the words and their usages** draws the sampled usages of chosen words with the seeds ringed, coloured by word or by corpus, with each sentence in its hover; or the words themselves, each as the average of its usages. **Read a word's usages** lists a word's sampled usages, nearest the seeds first, each with its nearest seed; mark usages as fitting or not, and accept or reject the word. Choose beforehand whether marked usages and accepted words **join the seeds**: if they do, **Update the results** compares only the new seeds, in seconds. Marks, decisions and seeds are saved with the discovery and can be downloaded as CSV.

**Analyse these words in SemLens** opens chosen words (up to ten) in Analyse a word as a Different-words analysis: at once, with the sampled usages and Discover's embeddings, or with every usage in the corpora, to embed in Data & Model with the model of your choice (XL-LEXEME, say).

How the method was chosen: on the 350 Guardian *bubble* articles, seeding part of a known group (currencies, months, family terms, foods) and checking whether the rest came up in the top 20, target-word embeddings from layer 8 found 3/3, 7/7, 8/8 and 7/8; masked-target embeddings found 1/3, 7/7, 7/8 and 0/8; static PPMI 2/3, 2/7, 1/8 and 0/8. With seeds from one word, target-word embeddings favour words spelled like it, hence the spelling notes and the background correction.

## Inputs and target identification

Supported inputs:

| Source | Contents |
| --- | --- |
| Paste sentences | One sentence per line, for 1–10 named corpora |
| Raw corpora | Up to ten plain-text files; each file is a named corpus. UTF-8 is expected; UTF-16 and Windows-1252 files are read too, with a note |
| Example | 350 Guardian sentences containing *bubble*, in seven periods, with each article's publication date as its time |
| CSV/TSV | `sentence`, `corpus`; optional `start`, `end`, `sentence_id`, and a time column (see [Time](#time)). Any other columns (date, title, URL, …) are kept and returned in the annotation export |
| Pre-computed vectors | `.pt` or `.npy` with shape `[occurrences, dimensions]`, plus matching metadata rows |
| Saved corpora | Persistent corpus embeddings with their source text and sentence index |
| Saved word embeddings | Only the current word's usage vectors, matching sentences and extraction settings |
| Saved analysis | A `.semlens` snapshot with usages, vectors, definitions and annotations |

Matching finds every occurrence of the forms you list, as whole words. Forms are user-defined: you may intentionally include related lexemes such as *runner*, and a form ending in `*` matches any ending (`runn*` finds *running* and *runner*). The target word itself is entered in full, because it names the word in definitions. SemLens does not silently substitute a vaguely similar word. Case is ignored unless you tick **Match upper and lower case exactly** under **Usage selection options**, which separates *May* from *may*.

### Time

Usages can carry a point in time as well as a corpus.

- **From a CSV/TSV:** after choosing the file, pick its **Time column**. A column called `date`, `year` or `time` is picked automatically. It may hold years (`1999`), other numbers, or ISO dates (`2020-05-10`, `2020-05`, `2020-05-10T02:14:27Z`); dates become decimal years, so 1 July 2020 is about 2020.5. Dates written as `10/05/2020` are not read, because day and month order is ambiguous. Rows without a readable time are kept, counted in a note, and left out of time views.
- **Periods instead of a corpus column:** with a time column, the corpus column is optional. Without one (or by choosing **Periods formed from the time column**), usages are grouped into periods that act as the corpora for scores, heatmaps and frequencies: 2–10 periods of **Equal length** or with **About equal numbers of usages**. Dates are grouped by calendar year, so periods read *1999–2004*. Usages without a time form a last period, *No time*. Periods formed this way are marked chronological.
- **From corpus names:** for raw text or pasted sentences, tick **Use corpus names as times** under **Usage selection options** when every corpus is named by a year (`1999`) or a range (`1999–2003`, which becomes its middle, 2001).

With times loaded, every plot workspace gains two things:

- **Over time** (under **Plot view**) puts time on the horizontal axis and PC1, PC2 or the distance to a chosen definition on the vertical one. **Plot options** holds a moving average (the mean position of about a tenth of the usages shown, centred on each) and a small sideways spread for usages that share a year. Both are on by default.
- **Colour points by: Time** (in **Plot options** of the other views) shades usages from earliest (dark) to latest (light), with a colour bar. The legend then shows grey entries, which still hide and show each corpus.

Hovers show each usage's date or year. Time views and time colours are unavailable in blind annotation, because they would reveal the period. Times are saved with an analysis and included in the plot and annotation CSVs; they are not part of a usage's identity, so adding a time column keeps existing annotations matched.

CSV `start` and `end` are **zero-based character offsets**, with an exclusive end. They are authoritative, including when the target has an unexpected surface form. Invalid, empty or partially specified spans produce an error. Without positions, all exact occurrences are found; pre-computed vectors require explicit positions if a row contains several matching occurrences, to preserve alignment.

**Review loaded occurrences** shows, for each corpus, how many occurrences were loaded and, for pasted or uploaded text, how many were found, the corpus size in words and the frequency per million words. It lists the word forms found, and gives a concordance: one line per usage with the target in the middle, which can be filtered by corpus or by words in the sentence and sorted by the word before or after the target.

Pasted or uploaded texts stay available for the rest of the session: return to Data & Model, enter another word and load again without supplying them a second time.

The review panel and plot hover highlight the exact selected occurrence. Hover sentences wrap at approximately 75 characters and appear above or below their point, depending on available screen space. Unusually long sentences can be scrolled inside the hover box; no text is truncated. Repeated occurrences in one sentence remain separate observations; whole-sentence mode gives those occurrences the same context representation.

For uploaded text, **How the text is divided into contexts** (under **Usage selection options**) offers three choices. *Automatically* ends a sentence at `.`, `!`, `?` and at a blank line, so a heading does not run into the paragraph below; titles such as *Mr.* and *Dr.*, *e.g.*, *i.e.* and numbered references such as *No. 10* do not end one. *One sentence per line* suits text that is already split. *One paragraph per context* keeps blocks separated by a blank line whole, for when a sentence is too little context. The automatic splitter remains a heuristic: other abbreviations, quotations and historical orthography may need correction, and a CSV gives exact boundaries. Source files are retained unchanged in saved corpus collections, alongside the extracted sentences.

### Different words

The target word and its other forms can also be **different words to compare**, such as *bubble* and *balloon*, or a singular and its plural.

- **The question:** as soon as another form is typed (or, for the example and saved analyses, once they are loaded), Data & Model asks **Are these forms of one word, or different words to compare?** The default, **Forms of one word**, analyses every usage together. The answer can also be changed under **Display & comparison settings** in the sidebar, and is saved with an analysis.
- **Which word each form belongs to:** with **Different words**, a table lists each form you typed with its word, and, once usages are loaded, how many each found. Each form starts as its own word; give two forms the same word to count them as one (capitals are ignored when names are compared). A wildcard such as `balloon*` is always one word. A usage belongs to the form it matched: spelled exactly the same, then ignoring capitals, then the longest wildcard. Usages from a table whose form matches no entry are grouped as *other forms*. Editing the table regroups at once, with no loading or embedding again, and annotations stay. Up to 10 words.
- **Colour and compare by:** a switch at the top of Explore, Definitions and Annotation chooses **Corpus** or **Word**. Whichever is chosen does everything corpora do elsewhere: colours, legend, LDA, scores and heatmaps, the comparison of two groups, definition distributions and sense frequencies. The switch is global: changing it in one workspace changes all three.
- **Showing some words or corpora:** beside the switch, **Words shown** (when comparing corpora) or **Corpora shown** (when comparing words) limits the usages drawn and compared. Show one word to see how it is used across the corpora; show one corpus to compare the words within it. Hovers name the other dimension (the word, or the corpus).
- **Words by corpus** (Explore): how often each word occurs in each corpus, as counts and shares, with a stacked chart (shares, or totals) and, for raw text loaded without a maximum per corpus, per million words.
- **Words over time** (Explore, for three or more chronological corpora): one chosen score at a time, for each word against the period before or the first period, and the distance between one word and each other word, period by period. Scores also depend on how many usages are compared, so a word with few usages per period scores higher from that alone: read each line against itself.
- **Definitions** default to **Definition only** read as **Whole input** (a sentence-level embedding), since one set of definitions serves all the words. They can still be changed. Usages remain word-level vectors, so distances to definitions are larger overall than with word-level definitions; they stay comparable between usages and words.
- **Annotation:** annotations always belong to the loaded usages, whatever is grouped or shown; the export adds a `word` column. In blind annotation, words keep their colours (each sentence shows its word anyway) while corpus names and time views stay hidden.

## Corpus order and sampling

Expand **Display & comparison settings** in the sidebar to set corpus order and indicate whether it is chronological. Remove and re-add names in the order control, then apply it, or click **Order by name**, which puts names beginning with a year into time order. The same panel sets the colour palette (including a sequential one that runs dark to light in corpus order), point size and opacity, and whether the camera button on each plot saves PNG or SVG. The input order is used initially; alphabetical order is never assumed to be chronological.

Sampling applies to **score comparisons**, independently of the shared plot:

- **Equal sizes** (default): randomly select as many occurrences from each compared corpus as the smaller corpus contains.
- **All occurrences**: allow unequal sizes. SAMD matches only the smaller number of occurrences; surplus occurrences in the larger corpus remain unmatched.
- **Repeated equal-size samples**: repeat balanced selection without replacement within each repetition. A usage may participate in several repetitions, but remains one plotted occurrence. Mean, minimum, maximum and standard deviation describe variation between samples, **not a confidence interval**. When both corpora have the same size, repetitions use the same occurrences.

A saved seed makes sampling reproducible. **Maximum occurrences per corpus** (under **Usage selection options**) defaults to `all`; enter a number such as `500` to load and plot a reproducible random subset of that size from each corpus. Large uncapped comparisons can take substantial time and memory, particularly SAMD.

## Representations and layers

**Target word** averages the selected target's subword-token representations. **Whole sentence** averages the encoder's text-token representations, excluding padding and special tokens. This is contextual mean pooling, not a separately trained sentence-embedding model's output head.

### Target markers (XL-LEXEME)

XL-LEXEME was trained with the target wrapped in two special tokens: `… the dot-com <t>bubble</t> burst`. When **Target word** representations are embedded with a model whose tokenizer has both tokens, SemLens inserts them around each target automatically; the vector is still the mean of the target's own subword tokens. Definitions created afterwards are marked in the same way: the word that the template adds is wrapped (`<t>bubble</t>: a small pocket of gas…`), never a word inside the definition text. The template preview under **Advanced definition settings** shows the exact input.

Markers do not apply to **Whole sentence** representations, to models without the two tokens, or to **Saved corpora**: a corpus cache is encoded once for every word, before any target is chosen. To reproduce the earlier behaviour, untick **Mark the target word when the model supports it** under **Advanced embedding settings**. Saved analyses record whether markers were used, and their definitions follow that setting.

On the Guardian *bubble* sample, marking the target raised the mean cosine distance between usages from 0.10 to 0.39, and usages picked out by literal keywords (soap, bath, champagne, …) fell nearest a “pocket of gas” definition in 40 of 46 cases instead of 17. That is one word, one sample and a keyword proxy rather than annotation; compare both settings on your own data if the choice matters.

Under **Advanced embedding settings**, **Language model** lists XL-LEXEME (the default), XLM-RoBERTa large and base, and English BERT, with **Another model…** for any other Hugging Face encoder or local folder.

The default layer `-1` uses the encoder's final output, preserving SemLens's original extraction convention. Other negative values count back from the top: `-2` is the penultimate layer, `-3` the one before it, whatever the model's depth. `0` selects input embeddings, and positive integers count transformer layers from the bottom. A layer outside the model's range gives an error naming the available range. Final output and a model's last intermediate hidden-state tensor are not necessarily identical; the saved metadata records the chosen extraction source.

Long sentences are encoded in overlapping windows. A token found in overlapping windows uses its first encoded occurrence. This preserves late targets instead of silently replacing truncated targets. The window size is part of the embedding configuration; long-context representations should be interpreted with that windowing policy in mind.

Decoder-only and encoder-decoder extraction strategies remain outside the supported workflow pending evaluation.

## Heavy work: estimates and safeguards

Before anything that can take a while or use a lot of the computer, SemLens shows what it will cost: the download, the disk space, the memory and the time.

- **Where:** the first use of a model (its download), **Embed loaded usages**, **Compute & save corpus embeddings**, every heavy step of Discover (reading the corpora, the quick first pass, embedding the sample, comparing with the seeds), and comparisons of large corpora (scores, heatmaps, the comparison of two corpora, words over time), whose distance matrices grow with the product of the two corpora's sizes (SAMD keeps about 16 bytes per pair of usages: 30,000 × 30,000 usages is about 14 GB). UMAP and t-SNE on more than 20,000 usages show a warning.
- **Small actions** start straight away, with the estimate shown below the button.
- **Large actions** need a tick, **I understand, continue**: more than 2 GB of disk space, more than 40% of the computer's memory, or more than ten minutes.
- **Actions that could freeze the computer or fill its disk are not started**: when they would leave less than 5 GB of free disk, or need more than 75% of the computer's memory. The message says how to make the action smaller, such as a maximum number of occurrences per corpus, Equal sizes sampling, a smaller batch, float16 storage or a smaller model. A model download that would fill the disk is refused in the same way.
- **Times:** before a model has been used, the time is a rough estimate from its number of parameters. For a long job with a downloaded model, **Measure the speed on this computer (about 5 seconds)** times it on a few of your sentences; every embedding run also measures the speed, so later estimates in the same session are measured ones.

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

### Annotation aids

**Annotate in a list** (below the plot) shows the usages as concordance lines with a sense to choose on each, for annotating by reading rather than by position in the plot. Lines can be searched, sorted by a neighbouring word and limited to usages without a sense. **Show only usages without a sense** does the same for the plot. Under **Manage classes and annotations**, a class can be renamed (renaming onto an existing class merges them) or removed.

**Load previous annotations** (top of the Annotation workspace) brings back earlier work without going to Data & Model. With usages loaded, it puts the annotations of a saved analysis (from this computer or a `.semlens` file) or of an annotation CSV onto them: a CSV exported earlier, edited in a spreadsheet, or prepared by a colleague on the same data. Usages are matched by usage ID; failing that, by sentence and target position (so annotations survive a change of corpus names or periods); failing that, by the sentence alone when it holds the word once. Anything ambiguous is skipped and counted. Loaded annotations are added to the ones made so far, a usage annotated in both keeping the loaded ones, unless **Replace the annotations made so far** is ticked; classes from the file are added, options the file needs (several senses, ratings) are switched on, and Undo restores the previous state. With nothing loaded yet, the same box opens a saved analysis whole, with its usages, vectors, definitions and annotations.

**Highlight usages containing** (Annotation workspace) rings every usage whose sentence contains the text you type, for example `AI, dot-com, specul*`. It matches whole words or phrases and ignores case; commas separate alternatives and a trailing `*` matches any ending. A ringed point can be clicked or lassoed like any other.

**Sense frequencies by corpus** appears below the plot once usages are annotated: a table of each sense in each corpus, as percentages of that corpus's annotated usages or as counts, a stacked chart in corpus order (shares, or totals with **Show totals instead of shares in the chart**), and a CSV download. It reports how many usages are annotated in each corpus, because frequencies from a partly annotated corpus describe only what has been annotated. It is hidden while blind annotation is on.

### Several senses and graded annotation

Two options in the collapsed **Annotation options** box change what an annotation holds. Both are off by default, so annotation starts as one sense per usage.

- **Allow several senses per usage:** clicking a usage adds the active sense, or takes it off again if the usage already has it; a lasso adds the active sense to every usage inside. Each sense is drawn as its own outline, nested around the point in the order the senses were given. **Remove the active sense from selected** and **Remove all senses from selected** replace **Clear selected**. Switching the option off while some usages have several senses asks first, and offers to keep only the first sense of each (undo brings the others back).
- **Graded annotation:** each sense a usage has also gets a rating, from 1 (the usage barely has this sense) to the top of the scale (it clearly does). **Points on the scale** sets the top, from 2 to 10; the default is 4. Under **Giving ratings**, the default is **Type a rating after each click or lasso**: a small box opens with the cursor already in it, so a rating is a click, a digit and Enter. Clicking a usage that already has the sense shows its rating, and the box can also remove the sense. Escape or Cancel assigns nothing. The other choice, **Use a default rating**, shows **Default rating** buttons above the plot instead, and every click or lasso gives that rating, which is quicker for long runs of similar usages. Weaker ratings draw thinner, paler outlines. Switching grading off hides the ratings without deleting them, and the scale cannot be made smaller than a rating already in use.

The hover in the Annotation workspace lists each usage's senses (and ratings). In **Annotate in a list**, one sense per usage is chosen in the Sense column (with a Rating column when graded); with several senses, each sense has its own column: a tick box, or, when graded, a rating, `—` for not that sense and `✓` for a sense given before grading was switched on.

In **Sense frequencies by corpus**, a usage with several senses counts once under each, so a corpus's senses can add up to more than 100% (and a stacked bar can reach above it); a row counts the usages with several senses. The chart is always stacked; **Show totals instead of shares in the chart** switches it from shares of each corpus's annotated usages (every bar the same height) to numbers of usages, so a corpus with more annotated usages has a taller bar. With grading, **Count a sense from rating** counts only ratings at or above a threshold, and a **Mean rating by corpus** table (with its own CSV) gives the mean rating of each sense in each corpus.

The annotation CSV keeps a `sense` column, with several senses joined by `; `. Graded annotation of one sense per usage adds a `rating` column; with several senses or grading, each sense also has a `sense:<name>` column holding its rating, or 1/0 when ungraded. **Load previous annotations** reads all of these, and a file that needs several senses or ratings switches those options on, with a note. Saved analyses keep the ratings and the options; analyses saved before these options existed open unchanged.

Annotation CSV export can include unannotated occurrences. Extra columns from an uploaded usage table are included in the export and saved with an analysis; apart from the time column, they are not shown in hovers. Changing representation or definitions preserves existing annotations on the same occurrences. Changing the target data only retains annotations whose occurrence identity still matches.

## Definition space and interpretation

By default, definitions use the input `word: definition` and match the usage representation: target-word extraction for word usages, whole-input pooling for sentence usages. Each usage then becomes a vector of **cosine distances to the definitions**: smaller means closer. These coordinates are distances, not probabilities or verified sense assignments.

Typed definitions, fetched Wiktionary definitions and JSON imports all appear in an editable text box, with one definition per line. Revise, add or remove entries, then click **Create definition space** to apply them. Draft edits survive navigation between workspaces during the session; existing plots continue to use the last created space until you apply the changes. Fetching again replaces the current dictionary draft.

Wiktionary definitions come with their part of speech: when an entry has several, **Parts of speech to include** narrows the list (and rebuilds the draft). Usage labels are kept in brackets, as in `(economics) A period of intense speculation…`; delete any you do not want embedded. Under **Dictionary settings**, the edition is the language the definitions are written in (`en`, `de`, `fr`, `es`, `sv` and others with the same page layout), and the section heading is the language of the word as that edition writes it. For English glosses of a German word, keep the `en` edition and enter `German`.

Under **Advanced definition settings**, choose `word: definition`, `definition: word`, the definition alone, or a template using `{word}` and `{definition}`. Independently choose target-word or whole-input pooling. A template preview shows the actual input. With sentence usage representations, the initial definition readout defaults to whole-input pooling. The encoder and layer are shared with the usage embeddings.

**Definition axes** is the default view in Definitions. With two or more definitions, it initially plots distance to definition 1 against distance to definition 2. The **Horizontal axis** and **Vertical axis** selectors support any definition or PC1/PC2 fitted on the joint definition space. With just one definition, the vertical axis defaults to PC1. PCA, UMAP, t-SNE and LDA remain available under **Plot view**. **Plot options**, below it, holds the settings of the current view: which principal components to plot; UMAP's number of neighbours, minimum distance and seed; t-SNE's perplexity and seed. UMAP and t-SNE find neighbours with the vector distance chosen in the sidebar. The same box has a text search that rings matching usages and a download of the plotted positions as CSV. PCA and LDA definition anchors are transformed with the same fitted model as the usages. Anchors are drawn as stars labelled `D1`, `D2`, … in the order of the definition list, with the full definition in the hover. They are not added to UMAP/t-SNE plots.

**Definition distributions and descriptive statistics** shows a separate coloured line for each corpus, connecting counts or proportions in shared distance bins. The bins span the distances actually observed for that definition, and their number (8 to 30) follows the size of the smallest corpus. Lines have distinct dash styles and no filled areas or smoothing. Hover to compare the corpora within a bin; click legend entries to hide/show a corpus or double-click to isolate one. The table reports count, mean, median and population standard deviation of definition distance. These describe model distances, not sense frequencies.

The similarity and correlation heatmaps scale with the number of definitions. Their colour limits adapt to the observed values, so small differences remain visible; consult the numeric values and colour bars when comparing matrices. Short axis labels (`D1`, `D2`, …) refer to a complete definition key below; hovering a cell displays both definitions.

LDA deliberately uses corpus labels to seek separation. With two corpora the view uses LD1 plus residual PC1; with more corpora it uses LD1/LD2 when available. Fitted on every embedding dimension, LDA separates any labelling of a few hundred usages, including randomly shuffled corpus labels. SemLens therefore fits it on the leading principal components, about one per 20 usages, and says so under the plot; a small definition space is used as it is. Definition weights operate on **distance features**: positive means increasing distance increases LD1. Interpret weights alongside corpus positions and mean distances; a sign alone is not evidence that a sense became more frequent.

## Scores and usage evidence

Below, $A$ and $B$ are the usages of the two corpora and $d$ is the distance chosen in the sidebar. The same names and formulas appear in the **(?)** beside each score and under each heatmap.

| Score | Name | Formula | What it compares |
| --- | --- | --- | --- |
| APD | Average pairwise distance | $\frac{1}{\lvert A\rvert\lvert B\rvert}\sum_{a\in A}\sum_{b\in B}d(a,b)$ | Every cross-corpus usage pair; also reflects variation within the corpora |
| PRT | Prototype distance | $d(\bar a,\bar b)$ | The two corpus averages |
| AMD | Average minimum distance | $\frac12\left(\frac{1}{\lvert A\rvert}\sum_{a}\min_{b}d(a,b)+\frac{1}{\lvert B\rvert}\sum_{b}\min_{a}d(a,b)\right)$ | Each usage's closest counterpart, averaged in both directions |
| SAMD | Symmetric average minimum distance | $\frac{1}{\lvert M\rvert}\sum_{(a,b)\in M}d(a,b)$ | A one-to-one matching $M$, built by matching the closest remaining pair first |
| Directional AMD | Average minimum distance, one direction | $\frac{1}{\lvert A\rvert}\sum_{a}\min_{b}d(a,b)$ | Closest counterparts for one corpus looking into the other |

The vector distance setting chooses cosine or Euclidean distance for multidimensional comparisons. **Per-definition scores always use absolute distance `|x-y|`**, because cosine distance between positive scalar features collapses to zero, and Euclidean distance in one dimension is the absolute difference. The per-definition table reports APD, PRT, AMD, SAMD and both directional AMDs for each definition; there PRT is the difference between the two corpus means. Its definition column has a fixed width and stays in place while the scores scroll; double-click a definition to read it in full. A definition space with just one definition also uses absolute distance for its scores. The embedding-to-definition projection itself remains cosine distance.

Scores are computed in the full representation, not on the plotted 2D coordinates. Small nonzero scores are displayed in scientific notation instead of rounded to zero. Cosine PRT is labelled undefined if a corpus centroid has no direction.

Scores for all pairs, the usage-to-definition distances and the sense frequencies can each be downloaded as CSV from the place where they are shown.

With two or more corpora, **Explore** displays APD, PRT, AMD and SAMD as four heatmaps covering every corpus pair, with the score printed in each cell. Each uses its own colour scale and the sidebar's distance and sampling settings. Hover a cell for its score, sample sizes and sample min–max. Corpora switched off in the plot legend are left out of the heatmaps, so the legend also chooses which corpora are compared. When the corpus order is marked as chronological and at least three corpora are shown, **Change over time** plots each score for every period against the one before it and against the first. The diagonal is left blank. Balanced sampling matches sizes separately for each pair; sample sizes can therefore differ across the matrix. Reordering corpora or reversing the pair selectors preserves the scores for a given pair and seed. All-pair results are cached, but an initial comparison of ten corpora requires 45 pair calculations.

Expand **Inspect the usage pairs behind AMD** and choose one of two views.

**Weakest counterparts between two corpora** gives ranked tables in both directions, with the weakest counterparts first. With three or more corpora, choose the pair inside the panel; the scores above follow the same pair. Each table shows about three rows at a time and scrolls for the rest. Type part of a sentence to search either member of a pair. Each sentence has its own horizontal scroll area; tables hold up to 50 pairs per page, with ranking and search covering all pairs.

**Closest usages to a chosen sentence** takes one usage (search for part of its sentence, then pick it from the list) and lists the usages nearest to it in each of the other corpora, three per corpus by default. It searches every loaded occurrence, including any left out of an equal-size score sample.
 When sampling is repeated, the sample selector uses the exact occurrences underlying that repetition's scores. The displayed distances and scores use the same representation and distance setting. Explore and Definitions each have an inspector because the nearest counterparts can change between full embeddings and definition space; the panel title identifies its space.

In these tables, and in plot hovers, the target occurrence is highlighted, and each scrolling sentence cell opens with its target in view.

A high directional score identifies weak counterparts in the chosen comparison. Whether this reflects semantic emergence/loss, topical variation or corpus composition requires inspection of the usages.

## Development and verification

For the planned newspaper demo, [Guardian collection instructions](docs/guardian-collection.md) explain how to search for `bubble`/`bubbles` and export small, dated corpora with source metadata. A developer API key is required for live collection.

```bash
pip install -e ".[all]"
python -m pytest -q
```

`semlens` (or `python -m app.launch`) starts the app from any folder once the package is installed.

Tests cover explicit offsets, word forms, one/many corpora, scalar distances, balanced sampling, LDA, shared plots, target markers, dictionary parsing, legend visibility, blind annotation, several and graded senses (storage, toggling, list edits, frequencies, export and import), time parsing and periods, the Over time view and time colouring, sense frequencies, annotation invalidation, cache reuse and saved-analysis round trips. Optional integration tests use a small locally cached real encoder without network downloads. When present, the user-supplied files in `testing_texts/` exercise corpus extraction and selected-excerpt embedding/cache equivalence.

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
