# Collecting the Guardian “bubble” mini-corpora

The first pilot collection is complete: **350 occurrences from 350 articles**, split equally across seven time bins. The collection files are kept locally in `.semlens/datasets/guardian_bubble/`. The usage table for SemLens has since been moved into the repository as `testing_texts/bubble_usages.csv`; the note on redistribution at the end of this page applies to that copy too. New live collection requires a Guardian developer API key; cached responses can be reused.

## Pilot collected on 10 November 2026

| Period | API search matches | Selected occurrences/articles |
| --- | ---: | ---: |
| 1999–2003 | 3,612 | 50 |
| 2004–2009 | 6,663 | 50 |
| 2010–2015 | 8,847 | 50 |
| 2016–2019 | 5,081 | 50 |
| 2020–2021 | 4,618 | 50 |
| 2022–2023 | 2,105 | 50 |
| 2024–10 November 2026 | 2,965 | 50 |

These are API hit counts before exact-form/content checks, across unequal-length periods. They are not sense frequencies. The latest period grew from 2,963 to 2,965 hits during a pause; its old responses were archived under `superseded/` and that period was refreshed before sampling. Earlier periods remained consistent and were reused from cache.

All 350 rows passed checks for unique article IDs, corpus sizes, date bounds, exact target positions, source URLs, file hashes and loading with SemLens. The API includes Observer material alongside Guardian material; publication/section summaries are in `validation.json`. Nine rows in `review_candidates.csv` are flagged for duplicate text, heading/list-style contexts or possible promotional content. They remain in the pilot for manual review; the sample has not been curated to favour particular senses.

## Set up access

Register for a key through the [Guardian Open Platform](https://open-platform.theguardian.com/access/). Its developer tier offers article text for non-commercial use, with limits of one call per second and 500 calls per day.

Save the key locally in `.streamlit/secrets.toml`:

```toml
guardian_api_key = "YOUR_KEY"
```

That file is ignored by Git. Alternatively, provide `GUARDIAN_API_KEY` in the environment. The collector does not accept keys on the command line, print them, or include them in cache keys or manifests.

A local JSON key file is also supported at `.streamlit/guardian-key.json`:

```json
{"api_key": "YOUR_KEY"}
```

Use the actual key from the registration email or the `api-key` parameter in an authenticated API URL. Saving the JSON returned by an API search saves article results, not the key. Environment and TOML settings take precedence over the JSON key file.

## Search and build

From the SemLens repository, first retrieve counts:

```bash
.venv/bin/python scripts/guardian_corpora.py --counts-only
```

Then build the samples:

```bash
.venv/bin/python scripts/guardian_corpora.py
```

The default is up to **50 articles and 50 target occurrences per bin**, using the seven periods in [guardian_bubble_bins.json](../scripts/guardian_bubble_bins.json). The last period ends on 10 November 2026. No embeddings are computed during collection.

To choose other bins or sample sizes, use `--bins path/to/bins.json`, `--per-bin 100`, and a new `--output` directory. Each bin needs `name`, `start` and `end`; dates are inclusive. Bins must be chronological, non-overlapping and number at most ten. Change the final cutoff explicitly when including newer articles.

## Search and sampling rules

The [official search documentation](https://open-platform.theguardian.com/documentation/md/content_search.md) specifies the query and date parameters. The collector uses:

- Query: `"bubble" OR "bubbles"`, restricted to article bodies with `query-fields=body`.
- English articles (`tag=type/article`, `lang=en`), across sections and production offices.
- Explicit publication-date bounds, with `use-date=published`; `lastModified` is retained separately.
- Pages of 50 results, ordered oldest first so article ranks have a defined order. Selection itself follows a seeded random permutation of ranks across the whole bin, not the first results or the highest-relevance results.

For each candidate, it extracts paragraph text, checks for exact whole-word forms, and chooses one matching occurrence using the saved seed. Article IDs are deduplicated. Missing body text, out-of-bin dates, non-articles and results without the exact forms are excluded and replaced from the remaining candidate ranks where possible. No sense or topic filters are applied. The text may contain Guardian and other publications represented in the API; the publication field is retained for inspection.

This is an **article-balanced sample of target usages**, not an occurrence-weighted sample or a general newspaper corpus. A sentence can contain several target occurrences, but the CSV identifies exactly the selected one. Counts reported as `api_matches` are search-engine hits before these local checks.

## Output and loading into SemLens

Everything is saved under `.semlens/datasets/guardian_bubble/` by default, outside version control:

| File | Contents |
| --- | --- |
| `counts.json` | Search-hit totals and date bounds for each bin |
| `bubble_usages.csv` | All mini-corpora, with corpus labels, exact target offsets, sentence IDs, dates and source metadata |
| `mini_corpora/<bin>.txt` | One selected usage sentence per line, for convenient reading or text upload |
| `articles/<bin>.jsonl` | Cleaned body text and provenance for each sampled article |
| `manifest.json` | Completed-run configuration, per-bin counts, sampling notes and output hashes |
| `cache/` | Successful API responses, request parameters without the key, and retrieval timestamps |
| `run.json` | Configuration used to protect against mixing different runs |

**Recommended:** choose **Upload CSV / TSV** in SemLens, enter `bubble` as the target and `bubbles` as another form, then upload `bubble_usages.csv` (now at `testing_texts/bubble_usages.csv`). Its date, title, section, publication, byline and URL columns are kept and returned in the annotation export. This preserves the intended 50 selected occurrences per complete bin and exact sentence boundaries. The `.txt` files are convenient, but the app may find additional occurrences in a sentence and applies its own sentence splitter.

Inspect the summaries for bins with fewer than the requested number of usages. Review sentence boundaries, source dates, sections and a selection of articles before presenting: splitting is heuristic, republication can affect dates, and the sense distribution is deliberately not predetermined.

## Resuming and limits

Requests are spaced at least 1.05 seconds apart. The default budget is 400 new requests per invocation; this is not a tracker for other uses of the same key or earlier invocations that day. HTTP 429 stops collection rather than repeatedly retrying. Successful responses remain cached. Rerun with the same settings and output directory to resume, including offline reconstruction when all necessary responses are cached.

Changing the configuration requires a different output directory. If the API's reported result count changes during sampling, the collector stops and asks for a new output directory rather than silently mixing ranks from inconsistent result sets. A completed manifest is written only after successful export; counts alone do not mean the corpora have been built.

The raw responses preserve the retrieved snapshot for reproducibility. Public API access does not make the article text public domain; keep the collected text local and consult the access terms before distributing a demo dataset.
