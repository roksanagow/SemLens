"""Different words in one analysis: which word each usage is, and views grouped by word.

The target word and each other form typed are *entries*. Every usage belongs
to the entry it matched: the entry spelled exactly as found, else the same
ignoring case, else the longest wildcard (``balloon*``) it starts with. A word
is one or more entries: by default each entry is its own word, and entries
given the same word name are merged. Usages whose form matches no entry
(possible with offsets from a table) form one group, "other forms".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .data_loading import TargetWordData, UsageInstance

OTHER_FORMS = "other forms"
MAX_WORDS = 10


def entry_of(form: str, entries: Sequence[str]) -> str | None:
    """The typed entry a found form belongs to, or None."""
    for entry in entries:
        if not entry.endswith("*") and entry == form:
            return entry
    folded = form.casefold()
    for entry in entries:
        if not entry.endswith("*") and entry.casefold() == folded:
            return entry
    best = None
    for entry in entries:
        if entry.endswith("*") and folded.startswith(entry[:-1].casefold()):
            if best is None or len(entry) > len(best):
                best = entry
    return best


def default_word(entry: str) -> str:
    """An entry's word name before any merging: the entry, without a wildcard."""
    return entry.rstrip("*").strip() or entry


def word_names(entries: Sequence[str], mapping: Mapping[str, str] | None = None) -> dict[str, str]:
    """Entry → word name. Names that differ only in capitals are one word, written as first given."""
    mapping = mapping or {}
    display: dict[str, str] = {}
    names = {}
    for entry in entries:
        name = str(mapping.get(entry) or "").strip() or default_word(entry)
        names[entry] = display.setdefault(name.casefold(), name)
    return names


def usage_words(data: TargetWordData, mapping: Mapping[str, str] | None = None) -> tuple[list[str], list[str]]:
    """The word of every usage, and the words in the order of the entries (other forms last)."""
    entries = data.forms
    names = word_names(entries, mapping)
    labels = []
    for usage in data.usages:
        entry = entry_of(usage.matched_form, entries)
        labels.append(names[entry] if entry is not None else OTHER_FORMS)
    present = set(labels)
    order = [w for w in dict.fromkeys(names.values()) if w in present]
    if OTHER_FORMS in present and OTHER_FORMS not in order:
        order.append(OTHER_FORMS)
    return labels, order


def entry_counts(data: TargetWordData, entries: Sequence[str]) -> dict[str, int]:
    """How many loaded usages each entry found."""
    counts = {entry: 0 for entry in entries}
    for usage in data.usages:
        entry = entry_of(usage.matched_form, entries)
        if entry in counts:
            counts[entry] += 1
    return counts


@dataclass
class GroupedUsage(UsageInstance):
    """A usage shown under another group (a word, say), keeping its original identity."""
    original_id: str = ""

    @property
    def usage_id(self) -> str:
        return self.original_id or super().usage_id


def regroup(data: TargetWordData, labels: Sequence[str], order: Sequence[str],
            keep: Sequence[int] | None = None) -> TargetWordData:
    """The usages at ``keep`` (all by default), each under its label in ``labels`` as its corpus.

    Usage IDs stay those of the original usages, so annotations and exports
    still refer to the same occurrences.
    """
    keep = range(len(data)) if keep is None else keep
    usages = []
    for i in keep:
        u = data.usages[i]
        usages.append(GroupedUsage(u.sentence, u.word, u.char_start, u.char_end, labels[i], u.sentence_id,
                                   dict(u.metadata), u.time, original_id=u.usage_id))
    present = {usage.corpus for usage in usages}
    return TargetWordData(data.word, usages, [g for g in order if g in present], list(data.forms), data.time_label)
