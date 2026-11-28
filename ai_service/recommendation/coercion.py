"""Shared strict numeric, boolean, and metadata coercion helpers for Stage 4 and Stage 5."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Mapping
from typing import Any


class InvalidNumericMetadataError(ValueError):
    """Raised when non-blank catalog metadata contains booleans, NaN/Inf, or invalid values."""


_TRUE_BOOL_TOKENS = frozenset({"true", "1", "yes"})
_FALSE_BOOL_TOKENS = frozenset({"false", "0", "no"})

OPTIONAL_FLOAT_METADATA_BOUNDS: dict[str, tuple[float, float | None]] = {
    "annual_premium": (0.0, None),
    "sum_insured_min": (0.0, None),
    "sum_insured": (0.0, None),
    "deductible_amount": (0.0, None),
    "copay_pct": (0.0, 100.0),
    "claim_settlement_ratio": (0.0, 100.0),
    "no_claim_bonus_pct": (0.0, 200.0),
    "room_rent_limit_value": (0.0, None),
}

OPTIONAL_INT_METADATA_MINS: dict[str, int] = {
    "max_renewal_age": 0,
    "max_family_members": 0,
    "initial_waiting_period_days": 0,
    "ped_waiting_period_months": 0,
    "specific_disease_waiting_months": 0,
    "maternity_waiting_months": 0,
    "network_hospitals_count": 0,
    "min_tenure_years": 0,
    "max_tenure_years": 0,
}

TYPE_METADATA_NUMERIC_BOUNDS: dict[str, tuple[float, float | None]] = {
    "vehicle_age_limit_years": (0.0, None),
    "life_cover_min": (0.0, None),
    "life_cover_max": (0.0, None),
    "policy_term_years_min": (0.0, None),
    "policy_term_years_max": (0.0, None),
    "premium_payment_term_years": (0.0, None),
}


def _is_bool_scalar(value: Any) -> bool:
    """Return True for Python ``bool`` and NumPy ``bool_`` scalars."""
    return isinstance(value, bool) or (
        type(value).__module__ == "numpy" and type(value).__name__ in ("bool", "bool_")
    )


def is_missing_or_blank(value: Any) -> bool:
    """Return True only for None, empty/whitespace strings, or float NaN (e.g., blank CSV cells).

    Explicitly returns False for booleans, float infinity, non-blank strings
    such as ``'NaN'`` or ``'n/a'``, and Decimal NaN/Inf so malformed values are caught.
    """
    if value is None:
        return True
    if _is_bool_scalar(value):
        return False
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def parse_optional_bool(value: Any) -> bool | None:
    """Normalize native booleans, 0/1 ints, and recognized string tokens; return None otherwise.

    Unlike Python's ``bool(value)``, unrecognized strings (such as ``'invalid'`` or
    ``'maybe'``) and non-binary integers/objects return ``None`` rather than
    evaluating silently to ``True``.
    """
    if is_missing_or_blank(value):
        return None
    if _is_bool_scalar(value):
        return bool(value)
    if isinstance(value, numbers.Integral):
        return bool(value) if value in (0, 1) else None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_BOOL_TOKENS:
            return True
        if lowered in _FALSE_BOOL_TOKENS:
            return False
        return None
    return None


def parse_bool(value: Any, *, default: bool = False) -> bool:
    """Normalize supported boolean values with a fallback default for missing or invalid tokens."""
    parsed = parse_optional_bool(value)
    return default if parsed is None else parsed


def parse_strict_optional_float(value: Any, *, field_name: str = "value") -> float | None:
    """Parse an optional numeric attribute strictly, rejecting booleans and non-finite inputs."""
    if is_missing_or_blank(value):
        return None
    if _is_bool_scalar(value):
        raise InvalidNumericMetadataError(
            f"Boolean value is invalid for numeric field '{field_name}'."
        )
    try:
        converted = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise InvalidNumericMetadataError(
            f"Invalid numeric value for field '{field_name}'."
        ) from exc
    if not math.isfinite(converted):
        raise InvalidNumericMetadataError(
            f"Non-finite numeric value is invalid for field '{field_name}'."
        )
    return converted


def parse_strict_optional_int(value: Any, *, field_name: str = "value") -> int | None:
    """Parse an optional integer attribute strictly, rejecting booleans, decimals, and strings."""
    numeric = parse_strict_optional_float(value, field_name=field_name)
    if numeric is None:
        return None
    if not numeric.is_integer():
        raise InvalidNumericMetadataError(
            f"Non-integer value is invalid for integer field '{field_name}'."
        )
    return int(numeric)


def parse_bounded_optional_float(
    value: Any,
    *,
    field_name: str = "value",
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Parse optional finite float strictly and enforce inclusive [min_value, max_value] bounds."""
    parsed = parse_strict_optional_float(value, field_name=field_name)
    if parsed is None:
        return None
    if min_value is not None and parsed < min_value:
        raise InvalidNumericMetadataError(
            f"Numeric value for '{field_name}' must be >= {min_value}; got {parsed}."
        )
    if max_value is not None and parsed > max_value:
        raise InvalidNumericMetadataError(
            f"Numeric value for '{field_name}' must be <= {max_value}; got {parsed}."
        )
    return parsed


def parse_bounded_optional_int(
    value: Any,
    *,
    field_name: str = "value",
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Parse optional integer strictly and enforce inclusive [min_value, max_value] bounds."""
    parsed = parse_strict_optional_int(value, field_name=field_name)
    if parsed is None:
        return None
    if min_value is not None and parsed < min_value:
        raise InvalidNumericMetadataError(
            f"Integer value for '{field_name}' must be >= {min_value}; got {parsed}."
        )
    if max_value is not None and parsed > max_value:
        raise InvalidNumericMetadataError(
            f"Integer value for '{field_name}' must be <= {max_value}; got {parsed}."
        )
    return parsed


def parse_safe_optional_policy_float(policy: Mapping[str, Any], field_name: str) -> float | None:
    """Parse an optional float catalog field safely, returning None when missing or invalid."""
    min_val, max_val = OPTIONAL_FLOAT_METADATA_BOUNDS.get(field_name, (None, None))
    try:
        return parse_bounded_optional_float(
            policy.get(field_name),
            field_name=field_name,
            min_value=min_val,
            max_value=max_val,
        )
    except InvalidNumericMetadataError:
        return None


def parse_safe_optional_policy_int(policy: Mapping[str, Any], field_name: str) -> int | None:
    """Parse an optional integer catalog field safely, returning None when missing or invalid."""
    min_val = OPTIONAL_INT_METADATA_MINS.get(field_name)
    try:
        return parse_bounded_optional_int(
            policy.get(field_name),
            field_name=field_name,
            min_value=min_val,
        )
    except InvalidNumericMetadataError:
        return None


def parse_strict_metadata_mapping(
    raw: Any,
    *,
    field_name: str = "type_specific_metadata",
) -> dict[str, Any]:
    """Parse ``type_specific_metadata`` as a dictionary or raise ``InvalidNumericMetadataError``."""
    if is_missing_or_blank(raw):
        return {}
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, (str, bytes)):
        try:
            decoded = json.loads(raw)
        except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
            raise InvalidNumericMetadataError(f"Malformed JSON object in '{field_name}'.") from exc
        if not isinstance(decoded, dict):
            raise InvalidNumericMetadataError(
                f"Metadata field '{field_name}' must be a JSON object."
            )
        return {str(k): v for k, v in decoded.items()}
    raise InvalidNumericMetadataError(
        f"Unsupported type '{type(raw).__name__}' for metadata field '{field_name}'."
    )


def validate_policy_optional_numeric_fields(policy: Mapping[str, Any]) -> list[str]:
    """Validate optional numeric policy columns and metadata keys using canonical Stage 4 bounds."""
    invalid_fields: list[str] = []

    for field_name, (min_val, max_val) in OPTIONAL_FLOAT_METADATA_BOUNDS.items():
        if field_name not in policy:
            continue
        try:
            parse_bounded_optional_float(
                policy.get(field_name),
                field_name=field_name,
                min_value=min_val,
                max_value=max_val,
            )
        except InvalidNumericMetadataError:
            invalid_fields.append(field_name)

    for field_name, min_val in OPTIONAL_INT_METADATA_MINS.items():
        if field_name not in policy:
            continue
        try:
            parse_bounded_optional_int(
                policy.get(field_name),
                field_name=field_name,
                min_value=min_val,
            )
        except InvalidNumericMetadataError:
            invalid_fields.append(field_name)

    try:
        meta = parse_strict_metadata_mapping(
            policy.get("type_specific_metadata"), field_name="type_specific_metadata"
        )
    except InvalidNumericMetadataError:
        invalid_fields.append("type_specific_metadata")
        meta = {}

    for meta_key, (min_val, max_val) in TYPE_METADATA_NUMERIC_BOUNDS.items():
        if meta_key not in meta:
            continue
        qualified_name = f"type_specific_metadata.{meta_key}"
        try:
            parse_bounded_optional_float(
                meta.get(meta_key),
                field_name=qualified_name,
                min_value=min_val,
                max_value=max_val,
            )
        except InvalidNumericMetadataError:
            invalid_fields.append(qualified_name)

    return invalid_fields


def find_invalid_mandatory_policy_bounds(
    *,
    min_age: int,
    max_age: int,
    prem_min: float,
    prem_max: float,
    sum_max: float,
) -> list[str]:
    """Return domain-integrity rule failures for core mandatory policy bounds."""
    invalid_bounds: list[str] = []
    if min_age < 0 or max_age > 120 or min_age > max_age:
        invalid_bounds.append("entry_age_range")
    if prem_min < 0 or prem_max < prem_min:
        invalid_bounds.append("annual_premium_range")
    if sum_max <= 0:
        invalid_bounds.append("sum_insured_max")
    return invalid_bounds
