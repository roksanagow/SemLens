# “Bubble” across newspaper periods

The user's 10 November 2026 follow-up expands the demo beyond Hansard: investigate many periods and meanings, including literal bubbles, Covid contact groups and AI investment. A seven-period Guardian API pilot has now been collected; the historical extensions below remain candidates for discussion.

The user subsequently requested API searches and time-binned mini-corpora. The [collector and loading instructions](guardian-collection.md) describe the completed pilot: 50 sampled articles/usages per bin, 350 total. The local files are in `.semlens/datasets/guardian_bubble/`. Nine contexts are flagged for manual review; no sense annotations or semantic-change conclusions have been assigned.

## Recommended starting point: the Guardian

The [Guardian Open Platform](https://open-platform.theguardian.com/) advertises content dating back to 1999. Its [developer access](https://open-platform.theguardian.com/access/) currently offers article text for non-commercial uses with a free registered API key, limited to 500 calls per day and one call per second. The pilot used the user's locally configured key. Public access should not be confused with a public-domain corpus that can be redistributed without conditions.

Keeping a consistent source across the main comparison limits some changes in collection method, although editorial composition can still change. The actual API pilot spans sections and production offices and includes Observer material: 47 rows are labelled The Observer, 141 The Guardian, and 162 theguardian.com. The CSV retains publication and section metadata so this composition can be inspected.

Examples establishing that the range of uses is worth investigating:

- [Literal soap bubbles, 2000](https://www.theguardian.com/science/2000/mar/23/technology1).
- [Dot-com investment bubbles, 2001](https://www.theguardian.com/media/2001/feb/11/business.internet).
- [Filter bubbles, 2016](https://www.theguardian.com/media/2016/jul/03/facebook-bubble-brexit-filter).
- [Support bubbles, 2020](https://www.theguardian.com/politics/2020/jun/11/support-bubbles-what-is-new-policy-in-england-and-how-does-it-work).
- [AI investment bubbles, 2025](https://www.theguardian.com/technology/2025/dec/01/ai-bubble-us-economy).

These are existence checks, not evidence about relative frequencies. Search for `bubble` and `bubbles` throughout each period instead of collecting only the collocations we expect to find.

One possible seven-corpus timeline, within SemLens's ten-corpus limit:

| Period | Questions to investigate |
| --- | --- |
| 1999–2003 | Literal uses and dot-com speculation |
| 2004–2009 | Financial and housing uses |
| 2010–2015 | Financial uses alongside information/social insulation |
| 2016–2019 | Filter, political and social uses before Covid |
| 2020–2021 | Pandemic contact groups alongside older uses |
| 2022–2023 | Persistence, retrospective Covid references and other uses |
| 2024–10 November 2026 | AI investment alongside the other senses |

These unequal-length windows were used for the initial pilot and can be revised after inspecting it. Literal bubbles occur in the sample across periods. Financial and figurative uses should not be assumed absent from earlier periods.

## Extending much further back

| Source | Available historical material | Practical tradeoff |
| --- | --- | --- |
| [American Stories](https://huggingface.co/datasets/dell-research-harvard/AmericanStories) | Article text extracted from U.S. newspapers; available years within 1774–1963, with gaps | Article boundaries and dates are provided. Coverage and OCR quality need checking; switching from U.S. historical papers to a modern British paper changes more than time. |
| [Chronicling America](https://guides.loc.gov/chronicling-america/additional-features) | Library of Congress digitised historical U.S. newspapers | Searchable page OCR and public data access. Page text can mix articles and advertisements; inspect original scans and extract coherent contexts. |
| [Trove](https://www.library.gov.au/research/research-guides-0/australian-newspapers-research-guide/australian-newspapers-online) | Australian newspapers, including substantial coverage from 1803–1955 | Public historical newspaper access; title coverage, OCR quality and the change in national variety need attention. |

For a first demo I would use the Guardian timeline, then explore a separate historical newspaper comparison if a longer history is the priority. We can still show both analyses during the presentation without treating a change of publication and country as evidence of semantic change.

## Sampling and interpretation

The pilot retains publication date, article ID, title, section and source URL alongside the sentence and exact target span, with last-modified dates separate. It samples article ranks with seed 42 and selects one occurrence per retained article. Each bin has 50 selected occurrences; this balances articles rather than sampling all occurrences uniformly. `validation.json` reports section/publication composition, and `review_candidates.csv` flags duplicate text and possible heading/list/promotional contexts. The sample contains 350 distinct articles and 349 distinct sentence texts.

A useful hypothesis is that *AI bubble* continues the existing financial-bubble sense while changing its subject matter. Covid contact-group uses may present a different semantic contrast. This lets students investigate whether embeddings and LSCD scores reflect a new sense, a new topic within a sense, or changes in sense frequencies. Word and sentence representations may behave differently. Treat these as questions to test through usage inspection and annotation, not conclusions encoded in the demo labels.
