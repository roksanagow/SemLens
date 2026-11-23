"""Sense annotations: each usage has one or more senses, each with an optional rating.

An annotation is stored as ``{sense: rating}``, in the order the senses were
given; the rating is an integer from 1 up to the scale's top point, or ``None``
when annotation is not graded. Older files stored a single sense name per
usage; :func:`normalise_label` reads those as ``{sense: None}``.
"""
from __future__ import annotations

import math
from typing import Mapping

SENSE_SEPARATOR = "; "
DEFAULT_SCALE = 4
MIN_SCALE, MAX_SCALE = 2, 10


def parse_rating(value) -> int | None:
    """A rating from a number or text; blank means no rating."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"A rating must be a whole number, not {value!r}.") from None
    if not math.isfinite(number) or number != int(number) or number < 1:
        raise ValueError(f"A rating must be a whole number of at least 1, not {value!r}.")
    return int(number)


def normalise_label(value) -> dict[str, int | None]:
    """``{sense: rating or None}`` from a sense name, a list of names, or a mapping."""
    if value is None:
        return {}
    if isinstance(value, str):
        value = value.strip()
        return {value: None} if value else {}
    if isinstance(value, Mapping):
        label = {}
        for sense, rating in value.items():
            sense = str(sense).strip()
            if sense:
                label[sense] = parse_rating(rating)
        return label
    if isinstance(value, (list, tuple)):
        return {str(s).strip(): None for s in value if str(s).strip()}
    raise ValueError(f"Cannot read an annotation from {value!r}.")


def normalise_annotations(mapping: Mapping | None) -> dict[int, dict[str, int | None]]:
    """Usage position → label, dropping usages that end up with no sense."""
    out = {}
    for index, value in (mapping or {}).items():
        label = normalise_label(value)
        if label:
            out[int(index)] = label
    return out


def senses_of(label) -> list[str]:
    return list(normalise_label(label))


def with_several_senses(annotations: Mapping) -> int:
    """How many usages carry more than one sense."""
    return sum(len(label) > 1 for label in annotations.values())


def highest_rating(annotations: Mapping) -> int:
    """The largest rating in use, or 0 when nothing is rated."""
    return max((rating for label in annotations.values() for rating in label.values() if rating is not None), default=0)


def has_ratings(annotations: Mapping) -> bool:
    return highest_rating(annotations) > 0


def counts_towards(rating: int | None, threshold: int) -> bool:
    """Whether a sense counts at a rating threshold; an unrated sense always counts."""
    return rating is None or rating >= threshold


def label_text(label: Mapping, *, graded: bool = False) -> str:
    """Readable form: ``financial; social``, or ``financial (3); social (1)`` when graded."""
    parts = []
    for sense, rating in label.items():
        parts.append(f"{sense} ({rating})" if graded and rating is not None else sense)
    return SENSE_SEPARATOR.join(parts)


def merge_ratings(first: int | None, second: int | None) -> int | None:
    """Keep the stronger rating when two labels for the same sense are merged."""
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)
