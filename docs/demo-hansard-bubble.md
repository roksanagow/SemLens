# Proposed student demo: “bubble” in Hansard

The user selected this direction on 10 November 2026. The periods and collection design below are proposals for discussion; a demo dataset has not been assembled or embedded.

Later that day the user broadened the idea to newspaper corpora across more periods and meanings. See the [newspaper proposal](demo-bubble-news.md); this Hansard plan remains an alternative.

## Question and periods

How does the pandemic use of *bubble* relate to established uses, and what happens to that use afterwards?

| Proposed corpus | Purpose |
| --- | --- |
| 2017–2019 | Establish the earlier range of uses |
| 2020–2021 | Examine pandemic contact-group uses alongside existing meanings |
| 2022–2024 | Investigate persistence, retrospective references and competing uses |

These are practical comparison windows, not a claim about an exact pandemic end date. Initially search for both `bubble` and `bubbles`. Decide separately how to handle hyphenated compounds and verbal uses; do not remove inconvenient examples merely because they do not fit the expected story.

## Evidence that makes this worth investigating

These public records establish that relevant uses exist; they do not establish frequencies or a change trajectory:

- [20 March 2019, No-deal EU Exit Preparations](https://hansard.parliament.uk/Commons/2019-03-20/debates/AD043295-65EC-4E63-9917-E206FAD31A17/No-DealEUExitPreparations): a reference to the Westminster bubble.
- [15 December 2020, Healthcare Support Services: Conception to Age Two](https://hansard.parliament.uk/commons/2020-12-15/debates/989C6502-BC2B-4E13-9954-BB0519D7694A/HealthcareSupportServicesConceptionToAgeTwo): support bubbles as household contact arrangements.
- [16 January 2024, Westminster Hall](https://hansard.parliament.uk/commons/2024-01-16/debates/517b977f-823b-49f5-a082-6bde96477f6b/WestminsterHall): a debate on online filter bubbles. This also illustrates why one long debate could dominate a small sample.

Possible annotation distinctions to discuss include literal bubbles, economic bubbles, social or political insulation, information filtering, and pandemic contact groups. Let the actual usages inform the distinctions and retain an uncertain/other option.

## Collection plan to agree

Use Commons spoken debates, keeping Commons Chamber and Westminster Hall identifiable. Avoid mixing speeches with written answers or statements without a deliberate decision. [Hansard search](https://hansard.parliament.uk/search) supports date and House filters; explicitly set the dates because its default search covers the last five years.

Collect complete contributions behind the search matches before extracting sentences. Preserve date, speaker, debate, venue, contribution identifier and source URL in a provenance file. Keep the sentence text and target offsets in the SemLens input. Deduplicate the same contribution appearing on both a debate page and a whole-day page while preserving genuinely distinct occurrences.

First count usable occurrences by period, year and debate. Then choose a manageable, reproducible sample, provisionally up to 150 occurrences per period. Review whether a handful of debates or speakers dominates it before settling the sampling rule. A word-only saved embedding file would make the eventual demo quick to reopen without encoding whole parliamentary corpora.

The repeatable collection route still needs verification: public records were accessible through web search, but a direct scripted request to the search webpage returned HTTP 403 during this check. No complete retrieval or frequency counts are claimed here.

## Classroom sequence

1. Explore all three periods on one plot and inspect the all-pair score matrices.
2. Read the weakest-counterpart pairs and annotate a small set of occurrences.
3. Compare candidate definitions using a definition axis against PC1, then the corpus histograms.
4. Discuss where the plots, scores and human readings agree or disagree.

The substantive question is whether we see a distinct conventional use, a change in the mixture of existing uses, topical concentration, or some combination. A 2024 reference to a 2020 support bubble would show continued reference to that meaning, but would not by itself establish continued productive use. We should inspect the later usages before deciding the story to present.
