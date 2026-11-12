"""Definition sourcing, formatting, and embedding.

Supports three definition sources:

1. **Manual input** — user provides a list of definition strings.
2. **Wiktionary** — fetched via the MediaWiki API (multilingual).
3. **JSON file** — load from a pre-existing definitions file.

Definitions are formatted as ``"word: definition text"`` before embedding,
matching the convention from Goworek & Dubossarsky (2026).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from .data_loading import find_occurrences


# ---------------------------------------------------------------------------
# Definition formatting
# ---------------------------------------------------------------------------

def format_definitions(word: str, raw_definitions: list[str]) -> list[str]:
    """Prepend ``'word: '`` to each definition for contextualised embedding.

    Strips whitespace and deduplicates.  Skips if already formatted.
    """
    formatted = []
    seen = set()
    prefix = f"{word.lower()}:"

    for d in raw_definitions:
        d = d.strip()
        if not d:
            continue
        if d.lower().startswith(prefix):
            key = d.lower()
            text = d
        else:
            key = f"{word}: {d}".lower()
            text = f"{word}: {d}"
        if key not in seen:
            seen.add(key)
            formatted.append(text)

    return formatted


# ---------------------------------------------------------------------------
# Wiktionary fetching (MediaWiki API)
# ---------------------------------------------------------------------------

WIKTIONARY_API_URL = "https://{lang}.wiktionary.org/w/api.php"


def fetch_wiktionary_entries(
    word: str,
    language: str = "en",
    target_language_name: str | None = None,
) -> list[tuple[str, str]]:
    """Fetch ``(part of speech, definition)`` pairs from Wiktionary via the MediaWiki API.

    Parameters
    ----------
    word : str
        The word to look up.
    language : str
        Wiktionary edition language code (``"en"``, ``"de"``, ``"fr"``, etc.).
    target_language_name : str or None
        The heading of the language section to read, as that edition writes it
        (``"English"`` or ``"German"`` on en.wiktionary, ``"Deutsch"`` on
        de.wiktionary). If no section has that heading, the edition's own
        language section is used.
    """
    import requests

    url = WIKTIONARY_API_URL.format(lang=language)
    params = {
        "action": "parse",
        "page": word,
        "prop": "wikitext",
        "format": "json",
    }
    headers = {
        "User-Agent": "SemLens/0.1 (https://github.com/roksanagow/SemLens; academic research tool)",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(
            f"Wiktionary lookup failed for '{word}': {data['error'].get('info', 'unknown error')}"
        )

    wikitext = data["parse"]["wikitext"]["*"]
    return _parse_wikitext_entries(wikitext, target_language_name or "English", edition=language)


def fetch_wiktionary_definitions(
    word: str,
    language: str = "en",
    target_language_name: str | None = None,
) -> list[str]:
    """Definition strings only (without the ``word:`` prefix); see ``fetch_wiktionary_entries``."""
    return [definition for _, definition in fetch_wiktionary_entries(word, language, target_language_name)]


# Part-of-speech headings, by their first word(s), across the editions SemLens
# reads (English, German, French, Spanish, Swedish and others with the same layout).
_POS_HEADINGS = {
    "noun", "verb", "adjective", "adverb", "proper noun", "pronoun", "preposition", "conjunction",
    "determiner", "interjection", "numeral", "particle", "article", "phrase", "prepositional phrase",
    "idiom", "proverb", "prefix", "suffix", "abbreviation", "initialism", "acronym", "contraction",
    "participle", "postposition", "classifier", "symbol",
    "substantiv", "adjektiv", "adverb", "eigenname", "pronomen", "präposition", "konjunktion", "interjektion", "partikel",
    "nom", "nom commun", "nom propre", "verbe", "adjectif", "adverbe", "pronom", "préposition", "conjonction",
    "interjection", "locution nominale", "locution verbale", "locution adjectivale", "locution adverbiale",
    "sustantivo", "verbo", "adjetivo", "adverbio", "pronombre", "preposición", "conjunción", "interjección",
    "nombre", "sostantivo", "aggettivo", "avverbio", "substantivo",
}


def _heading(line: str) -> tuple[int, str] | None:
    match = re.match(r"^(={2,6})\s*(.*?)\s*\1\s*$", line.strip())
    if not match:
        return None
    # Editions that write headings as templates: {{S|nom|fr}}, {{Wortart|Substantiv|Deutsch}},
    # {{langue|fr}} name the section in the first argument; {{sustantivo femenino|es}} in the name.
    def named(template):
        parts = [part.strip() for part in template.group(1).split("|")]
        if parts[0].casefold() in {"s", "wortart", "langue", "lengua", "sprache"}:
            return parts[1] if len(parts) > 1 else ""
        return parts[0]
    text = re.sub(r"\{\{([^{}]*)\}\}", named, match.group(2))
    return len(match.group(1)), _clean_wikitext(text, labels=False)


def _is_pos(heading: str) -> bool:
    words = re.sub(r"[^\w\s-]", " ", heading.casefold()).split()
    return any(" ".join(words[:n]) in _POS_HEADINGS for n in (3, 2, 1) if len(words) >= n)


def _parse_wikitext_entries(wikitext: str, target_language: str, edition: str = "en") -> list[tuple[str, str]]:
    """Extract ``(part of speech, definition)`` pairs from one language section.

    Reads numbered ``#`` items (English, French, Swedish and most editions),
    ``:[1]`` items under ``{{Bedeutungen}}`` (German) and ``;1:`` items (Spanish).
    """
    lines = wikitext.split("\n")
    sections = [(i, h[1]) for i, line in enumerate(lines) if (h := _heading(line)) and h[0] == 2]
    if not sections:
        return []

    def matches(name: str, heading: str) -> bool:
        return bool(name) and re.search(rf"(?<!\w){re.escape(name.casefold())}(?!\w)", heading.casefold()) is not None

    # The requested heading, else the edition's own language (French and Spanish
    # head it with the language code; elsewhere it is the first section).
    chosen = next((k for k, (_, heading) in enumerate(sections) if matches(target_language, heading)), None)
    if chosen is None:
        chosen = next((k for k, (_, heading) in enumerate(sections) if matches(edition, heading)), None)
    if chosen is None:
        if edition == "en":
            return []
        chosen = 0
    begin = sections[chosen][0] + 1
    finish = sections[chosen + 1][0] if chosen + 1 < len(sections) else len(lines)

    entries: list[tuple[str, str]] = []
    pos = None
    in_meanings = False  # German: inside a {{Bedeutungen}} block
    for line in lines[begin:finish]:
        stripped = line.strip()
        heading = _heading(stripped)
        if heading:
            pos = heading[1].split(",")[0].strip() if _is_pos(heading[1]) else None
            in_meanings = False
            continue
        if pos is None:
            continue
        if stripped.startswith("{{"):
            in_meanings = stripped.casefold().startswith("{{bedeutungen")
            if in_meanings:
                continue
        raw = None
        if re.match(r"^#[^#*:]", stripped):
            raw = stripped.lstrip("# ")
        elif in_meanings and (match := re.match(r"^:+\[[\d\w, –-]+\]\s*(.+)", stripped)):
            raw = match.group(1)
        elif (match := re.match(r"^;\s*\d+\s*(.*?):\s*(.+)", stripped)):
            raw = f"{match.group(1)} {match.group(2)}"
        if raw is None:
            continue
        definition = _clean_wikitext(raw)
        if len(re.sub(r"\([^)]*\)", "", definition).strip(" .")) > 2:
            entries.append((pos, definition))
    return entries


def _parse_wikitext_definitions(wikitext: str, target_language: str) -> list[str]:
    """Definition strings from one language section of English-layout wikitext."""
    return [definition for _, definition in _parse_wikitext_entries(wikitext, target_language)]


_LABEL_TEMPLATES = {"lb", "lbl", "label", "tlb", "term-label", "k", "lexique", "csem", "ámbito", "tagg", "kontext"}
_LINK_TEMPLATES = {"l", "m", "link", "mention", "ll", "l-self", "m-self", "w", "wp", "pedia", "lang", "polytonic", "ü", "term"}
_GLOSS_TEMPLATES = {"gloss", "gl", "qualifier", "qual", "q", "i", "sense", "s"}
_PLAIN_TEMPLATES = {"non-gloss definition", "non-gloss", "non gloss", "n-g", "ng", "ngd", "unsupported", "nowrap", "small"}
_DROPPED_TEMPLATES = {"senseid", "anchor", "attention", "rfdef", "rfc", "rfv-sense", "rfd-sense", "c", "cln", "topics",
                      "catlangname", "defdate", "cite", "ref", "refn", "r", "quelle", "source", "réf", "référence nécessaire"}
_LABEL_JOINERS = {"chiefly", "mostly", "mainly", "usually", "especially", "now", "sometimes", "often", "and", "or", "of", "in", "_"}
_LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}(-[A-Za-z]+)?$")


def _join_labels(labels: list[str]) -> str:
    out = ""
    for label in labels:
        if label == "_":
            out += " "
            continue
        if out and not out.endswith(" "):
            previous = out.split()[-1].rstrip(",").casefold()
            out += " " if previous in _LABEL_JOINERS or label.casefold() in {"and", "or"} else ", "
        out += label
    return out.strip()


def _resolve_template(body: str, labels: bool) -> str:
    parts = [part.strip() for part in body.split("|")]
    name = parts[0].casefold()
    args = [a for a in parts[1:] if a and not re.match(r"^[\w-]+\s*=", a)]
    words = [a for a in args if not _LANGUAGE_CODE.match(a)]
    if name in _DROPPED_TEMPLATES:
        return ""
    if name in _LABEL_TEMPLATES:
        return f"({_join_labels(words)})" if labels and words else ""
    if name in _LINK_TEMPLATES:
        return words[-1] if name in {"w", "wp", "pedia"} and words else (words[1] if len(words) > 1 else words[0] if words else "")
    if name in _GLOSS_TEMPLATES:
        return f"({', '.join(words)})" if words else ""
    if name in _PLAIN_TEMPLATES:
        return words[0] if words else ""
    if name.endswith(" of") or name.endswith(" de"):
        return f"{parts[0][:1].upper()}{parts[0][1:]} {words[0]}" if words else ""
    if not words:
        # A bare label such as {{figuré|fr}} or {{familier}}.
        return f"({parts[0]})" if labels and re.match(r"^[\w' -]{3,30}$", parts[0]) else ""
    return words[-1]


def _clean_wikitext(text: str, labels: bool = True) -> str:
    """Turn a line of wikitext into plain text; context labels become ``(label)``."""
    text = re.sub(r"<!--.*?-->", "", text)
    text = re.sub(r"<ref[^>]*/>|<ref[^>]*>.*?</ref>", "", text, flags=re.IGNORECASE | re.DOTALL)
    # Resolve templates from the innermost outwards.
    for _ in range(20):
        resolved = re.sub(r"\{\{([^{}]*)\}\}", lambda m: _resolve_template(m.group(1), labels), text)
        if resolved == text:
            break
        text = resolved
    # [[link|display]] → display, [[link]] → link; trailing letters stay attached.
    text = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("'''", "").replace("''", "")
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,;:])", r"\1", text)
    return text.lstrip(":;, ").strip()


# ---------------------------------------------------------------------------
# JSON file loading
# ---------------------------------------------------------------------------

def load_definitions_from_json(
    path: str | Path,
    word: str | None = None,
) -> dict[str, list[str]]:
    """Load definitions from a JSON file.

    The file should map ``{word: [def1, def2, ...]}`` (matching the format
    used in the Rethinking_LSCD_Metrics repository).

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.
    word : str or None
        If given, return only definitions for that word.  Otherwise return all.

    Returns
    -------
    dict[str, list[str]]
        Mapping of word → list of raw definition strings.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if word is not None:
        # Try exact match, then without trailing info (e.g. "word_nn")
        key = word if word in data else word.split("_")[0]
        if key not in data:
            raise KeyError(f"Word '{word}' not found in {path}")
        return {word: data[key]}

    return data


# ---------------------------------------------------------------------------
# Definition embedding
# ---------------------------------------------------------------------------

def definition_inputs(word: str, definitions: list[str], *, template="{word}: {definition}", readout="word"):
    """Build inputs and target positions from an explicit template.

    The position of {word} is tracked during formatting, so a word occurring
    inside the gloss cannot accidentally become the readout for a suffix template.
    """
    import string
    fields=[field for _,field,_,_ in string.Formatter().parse(template) if field is not None]
    if any(field not in {"word", "definition"} for field in fields) or "definition" not in fields:
        raise ValueError("Use {definition} and optionally {word} in the template.")
    if readout not in {"word", "sentence"}:
        raise ValueError("Choose word or sentence readout.")
    if readout=="word" and fields.count("word") != 1:
        raise ValueError("Target-word readout needs exactly one {word} placeholder.")
    inputs,spans=[],[]
    for text in definitions:
        prefix=word+":"
        gloss=text[len(prefix):].strip() if text.casefold().startswith(prefix.casefold()) else text.strip()
        result=""
        span=None
        for literal,field,format_spec,conversion in string.Formatter().parse(template):
            result+=literal
            if field is not None:
                if format_spec or conversion:
                    raise ValueError("Template placeholders cannot use format specifications or conversions.")
                value=word if field=="word" else gloss
                if field=="word":
                    span=(len(result),len(result)+len(value))
                result+=value
        if not result.strip():
            raise ValueError("Definition input cannot be empty.")
        inputs.append(result)
        spans.append(span if span is not None else (0,len(result)))
    return inputs,spans


def embed_definitions(loaded_model, word: str, formatted_definitions: list[str], *, batch_size=16,
                      max_length=128, layer=-1, template="{word}: {definition}", readout="word",
                      target_markers: bool = False) -> Tensor:
    """Embed definitions the same way as the usages they will be compared with.

    With ``target_markers`` and a word readout, the word that the template adds
    (never a word inside the gloss) is wrapped in the model's marker tokens,
    matching usages that were embedded with markers.
    """
    from .embeddings import extract_word_embeddings
    inputs,spans=definition_inputs(word,formatted_definitions,template=template,readout=readout)
    return extract_word_embeddings(loaded_model,inputs,spans,batch_size=batch_size,max_length=max_length,
                                   layer=layer,representation=readout,
                                   target_markers=bool(target_markers) and readout=="word")
