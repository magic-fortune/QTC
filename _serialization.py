"""Canonical serialization shared by IR and evidence records."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from enum import Enum
from fractions import Fraction
from typing import Any


def to_data(value: Any) -> Any:
    """Convert a typed QTC value to deterministic JSON-compatible data."""

    if dataclasses.is_dataclass(value):
        return {
            field.name: to_data(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, tuple):
        return [to_data(item) for item in value]
    if isinstance(value, list):
        return [to_data(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): to_data(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (set, frozenset)):
        return sorted((to_data(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value in canonical serialization: {type(value)!r}")


def canonical_json(value: Any) -> str:
    """Serialize *value* with stable keys and no insignificant whitespace."""

    return json.dumps(to_data(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    """Return the SHA-256 digest used by evidence and plan references."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
