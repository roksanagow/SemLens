# SemLens talk: slides and captions

Word: *bubble* in the Guardian, 1999–2026 (7 periods × 50 usages, XL-LEXEME).
Screenshots are in `screenshots/`, light theme, 2× resolution.

For each slide: **Title** is the slide heading, **Caption** is the line that
goes on the slide, **Say** is the speaker note. Slides marked *(optional)* can
be dropped to stay inside 15 minutes; the others run to about 18.

Slides 3, 8 and 14 are the short concept slides. They sit directly before the
screens they explain, so that each idea is used within a minute of being
introduced.

> **Check before presenting.** The sense labels behind slides 18–19 are a
> draft made by Claude from reading each sentence, not your annotation. They
> are in `draft_senses.csv`; correct them in the app (open
> `bubble_talk.semlens`) and retake those two screenshots, or present them as
> a draft.

---

## Part 1 · The question and the data (2 min)

### 1. Title (no screenshot)
**Title:** SemLens: looking at how a word's meaning differs across corpora
**Caption:** One word, many sentences, several periods. What changed, and where is the evidence?
**Say:** The running example is *bubble*. In 1999 it was mostly dot-com. In 2020 the government told us to form one.

### 2. `01_start.png` *(optional)*
**Title:** A tool for linguists, in the browser
**Caption:** Four steps: load usages, explore them, compare them with definitions, annotate senses.
**Say:** No code. You give it a word and one text per corpus, or a table of sentences. The example in this talk ships with the tool.

### 3. Concept: from a sentence to a point (no screenshot)
**Title:** Each usage becomes a vector
**Caption:** A language model reads the whole sentence and returns numbers for the target word in that context. Similar uses get similar numbers.
**Say:** The model is XL-LEXEME, trained to judge whether a word means the same in two sentences. Distance between two usages is the cosine distance between their vectors. Everything that follows is built from those distances.

### 4. `02_concordance.png`
**Title:** Start where a corpus linguist starts
**Caption:** A concordance of every occurrence, searchable and sortable by the neighbouring word.
**Say:** Here filtered to lines containing *burst* or *pop* and sorted by the word before the target: *asset bubble*, *Bitcoin bubble*, *dot.com bubble*, *debt bubble*. The highlighted word is the exact occurrence that gets a vector.

---

## Part 2 · Seeing the usages (4 min)

### 5. `03_explore_hover.png`
**Title:** 350 usages in one picture
**Caption:** Each point is one sentence, coloured by period. Hover to read it.
**Say:** This is a PCA view: the two directions along which the usages differ most, 40% and 15% of the variation. Three regions are visible before we know what they are. Hovering shows the sentence; this one is the *AI bubble*.

### 6. `04_explore_search.png`
**Title:** Where do the tech bubbles sit?
**Caption:** Typing words rings every usage whose sentence contains them: 27 usages, 13 of them from 1999–2003.
**Say:** *AI*, *dot-com*, *internet*, *tech* all land in the same lower-left arm, across 25 years. The model was not told these are financial; it placed them together from context.

### 7. `05_explore_legend.png`
**Title:** Compare two periods by hiding the rest
**Caption:** Only 1999–2003 and 2020–2021 shown. The ringed usages mention *support*, *travel*, *Covid* or *household*: none in the first period, 9 in the second.
**Say:** Click a period in the legend to hide it; the choice stays when you change the view. The blue arm at the lower left is nearly empty of purple, and the purple points gather at the top.

---

## Part 3 · Measuring the difference (5 min)

### 8. Concept: four scores (no screenshot, or crop the formulas from `06_heatmaps.png`)
**Title:** Four ways to turn two clouds of points into one number
**Caption:**
- **APD**, average pairwise distance: how far apart is a typical cross-period pair?
- **PRT**, prototype distance: how far apart are the two period averages?
- **AMD**, average minimum distance: how far is each usage from its *closest* counterpart in the other period?
- **SAMD**, symmetric AMD: the same, but each usage may be matched only once.

**Say:** APD and PRT describe the periods as wholes. AMD and SAMD ask whether each usage has a counterpart, so they react when a sense appears or disappears even if the average barely moves. AMD can be read in each direction separately.

### 9. `06_heatmaps.png`
**Title:** Every period against every other
**Caption:** One heatmap per score, with its formula underneath. 2020–2021 stands out from the earliest periods on PRT, AMD and SAMD.
**Say:** The brightest PRT cell is 1999–2003 against 2020–2021 (0.05). Each map has its own colour scale, so compare cells within a map, not across maps.

### 10. `07_change_over_time.png`
**Title:** The same scores read along the timeline
**Caption:** Solid: each period against the one before. Dashed: each period against 1999–2003.
**Say:** Against the first period, PRT climbs to a peak in 2020–2021 and then falls back part of the way. Against the period before, it stays low: no single step was large, but the drift adds up.

### 11. `08_compare_scores.png` *(optional)*
**Title:** One pair of periods in detail
**Caption:** 2016–2019 against 2020–2021. AMD is larger from 2020–2021 back to 2016–2019 (0.096) than the other way (0.069).
**Say:** That asymmetry says the later period contains usages with no close earlier counterpart, more than the reverse. That is what a new sense looks like.

### 12. `09_weakest_pairs.png` *(optional)*
**Title:** Read the sentences behind the number
**Caption:** The pairs that contribute most to AMD, weakest counterpart first, in both directions.
**Say:** Among the 2020–2021 usages with the poorest earlier match is the school *“bubbles” system*. Names such as the chimp *Bubbles* also surface here, which is useful: they are noise you can now see and exclude.

### 13. `10_nearest_ai.png` and `11_nearest_support.png` (side by side or two slides)
**Title:** Old sense, new referent, versus new sense
**Caption (AI):** The closest usage to *AI bubble* in every earlier period is another speculative bubble: dot.com, dotcom, property, internet, Bitcoin. Distances 0.01–0.02.
**Caption (support):** The closest earlier usages to *support bubbles* are social-enclosure bubbles, five to ten times further away (0.10–0.14). The nearest match of all is *support bubble* in 2022–2023 (0.07).
**Say:** This is the centre of the talk. *AI bubble* is new vocabulary but not a new meaning: the model finds it a near-identical neighbour in 1999. *Support bubble* has no such neighbour before 2020; the nearest things are the Westminster-bubble kind of use, which is plausibly where the sense came from.

---

## Part 4 · Definitions as axes (3 min)

### 14. Concept + `12_definitions_input.png`
**Title:** Definition space: measure every usage against the senses you expect
**Caption:** Write one definition per line. The model reads each one, and every usage is described by its distance to each definition.
**Say:** The axes of the earlier picture had no meaning. Here each axis is a sense you wrote down, so position can be read directly. The definitions are hypotheses: edit them and rebuild in seconds.

### 15. `12_b_wiktionary.png` *(optional)*
**Title:** Or start from a dictionary
**Caption:** Wiktionary entries fetched for the word, filtered by part of speech, then edited by hand.
**Say:** 35 senses come back for *bubble*, with labels such as *(economics)* and *(figurative)* kept. It works for other language editions too.

### 16. `13_definition_axes.png`
**Title:** Financial sense against social sense
**Caption:** Horizontal: distance to “asset prices rise beyond a sustainable value”. Vertical: distance to “an insulated social environment”. Stars mark the definitions.
**Say:** Usages near the left are financial, usages near the bottom are social, and the literal ones are far from both at the top right. Note D3, the pandemic-group definition: it sits next to the social definition, so the model sees those two senses as close. That is a finding about the senses, and a caution about the definitions.

### 17. `14_definition_distributions.png`
**Title:** How close is each period to one definition?
**Caption:** Distance to the financial definition, by period. In 1999–2003 a third of usages are almost on top of it; the median distance rises from 0.16 to 0.48 by 2024–2026.
**Say:** These are distances, not sense counts. To get counts, someone has to decide which usage is which sense. That is the last step.

*(Optional extra: `15_per_definition_scores.png`, the four scores computed along each definition separately, 1999–2003 against 2020–2021. The largest shift in the average is along the pandemic-group and social definitions, the smallest along the literal one.)*

---

## Part 5 · Annotation: the linguist decides (4 min)

### 18. `16_blind_lasso.png`
**Title:** Annotate without seeing the period
**Caption:** Blind mode removes the period colours and labels. One lasso assigned 83 usages to “financial / speculative”; hovering checks any of them.
**Say:** You define the sense classes. Click a point to label it, or lasso a group and then check it by reading. Blind mode keeps you from labelling by date. The tool never assigns a sense itself.

### 19. `17_annotate_list.png` *(optional)*
**Title:** Or annotate by reading
**Caption:** The same usages as a concordance with a sense to choose on each line. Plot and list stay in step.
**Say:** For people who would rather read than point. Sorting by the neighbouring word brings *housing bubble*, *dotcom bubble* together so they can be labelled in a run.

### 20. `18_annotated_plot.png`
**Title:** Senses over the model's picture
**Caption:** All 350 usages labelled (draft labels). Diamonds show the sense, the dot inside shows the period.
**Say:** The regions we saw at the start turn out to be senses: financial at the lower left, literal on the right, social enclosure at the upper left, with the pandemic sense scattered through and beside it. The model's geometry and a reader's judgement largely agree, and where they do not is where to look.

### 21. `19_sense_percentages.png` (or `19_b_sense_counts.png`)
**Title:** The result: sense frequencies by period
**Caption:** Financial falls from 56% of usages (1999–2003) to 12% (2024–2026). Social enclosure rises from 4% to 30%. The pandemic sense is 44% in 2020–2021 and 4% by 2024–2026. (Draft labels, 50 usages per period.)
**Say:** This table is the linguistic claim, and every cell traces back to sentences you can read. It exports as a table, with the Guardian date and link for each sentence.

### 22. Close (no screenshot)
**Title:** What the tool is for
**Caption:** The model proposes structure; the linguist reads, labels and decides. Works with any language model and any language it covers.
**Say:** Three things to take away. The picture is a way to find sentences worth reading. The scores summarise it, and each can be opened into its sentence pairs. The annotation is yours.

---

## Optional extras: other options in the tool

These show settings beyond the defaults. In the deck they sit after the screenshot they extend, and their notes start with `[OPTIONAL SLIDE]`.

| Screenshot | Title | Caption |
|---|---|---|
| `01_b_data_sources.png` | Many ways to bring data in | Raw text files, pasted sentences, a table of usages, corpora or vectors saved earlier, vectors computed elsewhere, or a whole saved analysis. |
| `01_c_upload_options.png` | Choosing which occurrences to load | Other forms take wildcards (bubbl*). The options limit occurrences per corpus, match case exactly, and choose whether a context is a sentence, a line or a paragraph. |
| `01_d_model_settings.png` | The language model, with defaults that work | XL-LEXEME is the default. Other models, the layer and the context length are in an expandable box. |
| `05_b_other_views.png` | Other views of the same usages | PCA, UMAP, t-SNE and LDA. Each view's settings are in Plot options, and the plotted coordinates download as a CSV. |
| `05_c_display_settings.png` | Display settings | A sequential palette for periods in time order, the distance used for all scores, point size and opacity, PNG or SVG figures, and corpus order. |
| `12_c_definition_settings.png` | How the model reads a definition | The input format and the representation can be changed. The preview shows exactly what the model receives, with the target word marked. |

The deck is `SemLens talk - metrics + def space.pptx` in this folder: your six opening slides, a section slide, 28 screenshot slides (the heatmaps are split over two) and your closing slide. Each screenshot slide has its title, caption and speaker note in the notes, and red rectangles that are ordinary shapes you can move or delete.

## Timing

| Part | Slides | Minutes |
|---|---|---|
| 1 · Question and data | 1–4 | 2 |
| 2 · Seeing the usages | 5–7 | 4 |
| 3 · Measuring the difference | 8–13 | 5 |
| 4 · Definitions as axes | 14–17 | 3 |
| 5 · Annotation | 18–22 | 4 |

Dropping the optional slides (2, 11, 12, 15, 19) brings it to about 15 minutes.

## Numbers quoted above, and where they come from

- Search counts (slides 6, 7): the line under the search box in each screenshot.
- Scores (slides 9–11): cosine distance, full embeddings, equal samples of 50 per period.
- Closest usages (slide 13): the Distance column in the two screenshots.
- Definition distances (slide 17): the Median distance column.
- Sense shares (slide 21): `draft_senses.csv`, labelled by Claude from each sentence in context. Classes: financial / speculative (including non-financial hype that is said to burst, such as “the Harry Potter bubble”), literal, social enclosure, Covid contact group, other (names, *bubble and squeak*, *speech bubble*, the verb in figurative use).
