from __future__ import annotations

import hashlib
import json
import math

from pydantic import BaseModel

CANONICALIZATION_PROFILE = "SSB-CJ1"


class CanonicalizationError(ValueError):
    pass


def _normalize(value: object, active_containers: set[int] | None = None) -> object:
    active = active_containers if active_containers is not None else set()
    if isinstance(value, BaseModel):
        return _normalize(
            value.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=False,
                exclude_unset=False,
                exclude_defaults=False,
                round_trip=False,
            ),
            active,
        )
    if value is None or type(value) is str or type(value) is bool or type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not canonical JSON")
        if value == 0.0:
            return 0.0
        return value
    if type(value) is list:
        container_id = id(value)
        if container_id in active:
            raise CanonicalizationError("cyclic JSON containers are not canonical")
        active.add(container_id)
        try:
            return [_normalize(item, active) for item in value]
        finally:
            active.remove(container_id)
    if type(value) is dict:
        container_id = id(value)
        if container_id in active:
            raise CanonicalizationError("cyclic JSON containers are not canonical")
        active.add(container_id)
        try:
            normalized: dict[str, object] = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise CanonicalizationError("canonical JSON object keys must be strings")
                normalized[key] = _normalize(item, active)
            return normalized
        finally:
            active.remove(container_id)
    raise CanonicalizationError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: object) -> bytes:
    normalized = _normalize(value)
    try:
        encoded = json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return encoded.encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError) as error:
        raise CanonicalizationError("value cannot be encoded as canonical JSON") from error


def sha256_ref(value: object) -> str:
    digest = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return f"sha256:{digest}"
