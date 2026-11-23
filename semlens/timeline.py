"""Continuous time: reading years and dates, and grouping usages into periods.

Times are plain numbers. Dates become decimal years (1 July 2020 is about
2020.5), so years and dates share one axis and other numbers work as well.
"""
from __future__ import annotations

import math
import re
from datetime import date, datetime, timezone
from typing import Sequence

PERIOD_METHODS = ("width", "count")
UNDATED = "No time"

_YEAR_MONTH = re.compile(r"^(\d{4})-(\d{1,2})$")
_RANGE = re.compile(r"^\s*(-?\d{1,4})\s*(?:-|–|—|to)\s*(-?\d{1,4})\s*$")


def decimal_year(moment: date | datetime) -> float:
    """A date or time as a year with a fraction: the share of the year that has passed."""
    if isinstance(moment, datetime):
        if moment.tzinfo is not None:
            moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
        start = datetime(moment.year, 1, 1)
        end = datetime(moment.year + 1, 1, 1)
        return moment.year + (moment - start).total_seconds() / (end - start).total_seconds()
    start = date(moment.year, 1, 1)
    end = date(moment.year + 1, 1, 1)
    return moment.year + (moment - start).days / (end - start).days


def parse_time(value) -> float | None:
    """A number, or an ISO date (2020, 2020-05, 2020-05-10, 2020-05-10T02:14:27Z), as a number.

    Returns ``None`` for blank values; raises ``ValueError`` for text that is
    neither. Day-first or month-first dates such as 10/05/2020 are not read,
    because they are ambiguous.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        if not math.isfinite(number):
            raise ValueError(f"Not a usable time: {text!r}.")
        return number
    match = _YEAR_MONTH.match(text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return decimal_year(date(year, month, 1))
    iso = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        moment = datetime.fromisoformat(iso)
    except ValueError:
        try:
            return decimal_year(date.fromisoformat(text))
        except ValueError:
            raise ValueError(f"Not a usable time: {text!r}. Use a number or a date such as 2020-05-10.") from None
    return decimal_year(moment)


def parse_times(values: Sequence) -> tuple[list[float | None], list[int]]:
    """Times for a column of values, and the positions of values that could not be read.

    Unreadable values become ``None``, like blanks, so one stray cell does not
    stop a whole table from loading.
    """
    times, unreadable = [], []
    for position, value in enumerate(values):
        try:
            times.append(parse_time(value))
        except ValueError:
            times.append(None)
            unreadable.append(position)
    return times, unreadable


def time_from_corpus_name(name: str) -> float | None:
    """A time from a corpus name: a year or number ("1999"), or the middle of a range ("1999–2003")."""
    match = _RANGE.match(name)
    if match:
        first, last = int(match.group(1)), int(match.group(2))
        if first <= last:
            return (first + last) / 2
    try:
        return parse_time(name)
    except ValueError:
        return None


def _whole(values: Sequence[float]) -> bool:
    return all(float(v).is_integer() for v in values)


def _format(value: float, whole: bool) -> str:
    return str(int(math.floor(value))) if whole else f"{value:.4g}"


def period_names(edges: Sequence[float], whole: bool) -> list[str]:
    """Names such as 1999–2003 for the periods between consecutive edges.

    With whole numbers (years), a period runs from its first edge to the year
    before the next one.
    """
    names = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        if whole:
            # The upper edge is the first year of the next period (or the year
            # after the last one), so a period ends the year before it.
            end = math.ceil(hi) - 1
            start = math.ceil(lo)
            names.append(str(start) if end <= start else f"{start}–{end}")
        else:
            names.append(f"{_format(lo, False)}–{_format(hi, False)}")
    if len(set(names)) < len(names):
        names = [f"{lo:.6g}–{hi:.6g}" for lo, hi in zip(edges[:-1], edges[1:])]
    return names


def looks_like_years(times: Sequence[float | None]) -> bool:
    """Whether every known time lies between the years 1000 and 3000."""
    known = [t for t in times if t is not None]
    return bool(known) and all(1000 <= t < 3000 for t in known)


def assign_periods(times: Sequence[float | None], count: int = 5, method: str = "width",
                   *, calendar_years: bool | None = None) -> tuple[list[str], list[str]]:
    """A period name for every usage, and the period names in time order.

    ``method`` is ``"width"`` (periods of equal length) or ``"count"`` (about
    the same number of usages in each). With ``calendar_years`` (the default
    when every time looks like a year), dates are grouped by calendar year, so
    periods read 1999–2004 rather than splitting a year. Usages without a time
    go to a last period called "No time". Periods with no usages are left out.
    """
    if method not in PERIOD_METHODS:
        raise ValueError(f"Unknown way to form periods: {method!r}.")
    if calendar_years is None:
        calendar_years = looks_like_years(times)
    if calendar_years:
        times = [None if t is None else float(math.floor(t)) for t in times]
    known = sorted(t for t in times if t is not None)
    if not known:
        raise ValueError("No usage has a usable time, so periods cannot be formed.")
    count = max(1, int(count))
    whole = _whole(known)
    lo, hi = known[0], known[-1]
    if hi == lo:
        edges = [lo, hi]
    elif method == "width":
        if whole:
            # Whole years: equal spans of whole years, so no year is split.
            span = math.ceil((hi - lo + 1) / count)
            edges = [lo + k * span for k in range(count)] + [hi + 1]
            edges = [e for k, e in enumerate(edges) if k == len(edges) - 1 or e <= hi]
            edges[-1] = hi + 1
        else:
            edges = [lo + (hi - lo) * k / count for k in range(count + 1)]
    else:
        # Each period starts at the usage that begins its share of the ranks.
        starts = sorted(set(known[(k * len(known)) // count] for k in range(count)))
        edges = starts + [hi + 1] if whole else sorted(set(starts + [hi]))
        if len(edges) < 2:
            edges = [lo, hi]
    if whole and edges[-1] == hi:
        edges[-1] = hi + 1
    names = period_names(edges, whole)

    def period_of(t: float) -> int:
        for k in range(len(edges) - 1):
            upper_inclusive = (k == len(edges) - 2) and not whole
            if edges[k] <= t < edges[k + 1] or (upper_inclusive and t == edges[k + 1]):
                return k
        return len(edges) - 2

    labels = [UNDATED if t is None else names[period_of(t)] for t in times]
    order = [n for n in names if n in labels] + ([UNDATED] if UNDATED in labels else [])
    return labels, order
