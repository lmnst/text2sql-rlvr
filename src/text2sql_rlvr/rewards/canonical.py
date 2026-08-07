"""Turn raw SQLite values into hashable keys that compare the way we want.

Two result sets that are semantically the same routinely differ in Python:
``COUNT`` returns ``int`` while ``SUM`` over the same column returns ``float``,
and any arithmetic introduces float noise in the last few digits. Canonical
keys collapse exactly those differences and nothing else -- ``NULL`` stays
distinct from ``0`` and from ``''``, and large integers stay exact.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Any

#: Above 2**53 a float can no longer represent every integer, so an integral
#: float that large is not safe to treat as the integer it prints as.
_INT_EXACT_LIMIT = 2**53

#: Significant digits kept for non-integral floats. 6 absorbs accumulated
#: rounding from a few arithmetic ops without merging genuinely different values.
DEFAULT_FLOAT_SIG = 6

NULL_KEY: Hashable = ("null",)


def canon_value(value: Any, *, float_sig: int = DEFAULT_FLOAT_SIG) -> Hashable:
    """Map one cell to a hashable canonical key."""
    if value is None:
        return NULL_KEY
    if isinstance(value, bool):
        return ("i", int(value))
    if isinstance(value, int):
        return ("i", value)
    if isinstance(value, float):
        if math.isnan(value):
            return ("f", "nan")
        if math.isinf(value):
            return ("f", "inf" if value > 0 else "-inf")
        if value.is_integer() and abs(value) < _INT_EXACT_LIMIT:
            return ("i", int(value))
        return ("f", f"{value:.{float_sig}g}")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return ("b", bytes(value).hex())
    if isinstance(value, str):
        return ("s", value)
    return ("o", str(value))


def canon_row(row: tuple[Any, ...], *, float_sig: int = DEFAULT_FLOAT_SIG) -> tuple[Hashable, ...]:
    """Canonicalise one row, preserving column order."""
    return tuple(canon_value(v, float_sig=float_sig) for v in row)


def canon_rows(
    rows: tuple[tuple[Any, ...], ...] | list[tuple[Any, ...]],
    *,
    float_sig: int = DEFAULT_FLOAT_SIG,
) -> tuple[tuple[Hashable, ...], ...]:
    """Canonicalise a result set, preserving both row order and multiplicity."""
    return tuple(canon_row(tuple(row), float_sig=float_sig) for row in rows)
