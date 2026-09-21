"""Analysis-side admission rules for technically excluded confirmatory cells."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ApprovedTechnicalExclusionPolicy:
    """A freeze-bound cap for pre-behavior provider exclusions.

    The custody audit must separately establish that each excluded record is a
    genuine pre-behavior provider failure. Analysis only admits the bounded,
    already-approved projection of that audit.
    """

    allowed_error_codes: tuple[str, ...]
    max_total_exclusions: int
    max_exclusions_per_primary_cell: int

    def __post_init__(self) -> None:
        if (
            type(self.allowed_error_codes) is not tuple
            or not self.allowed_error_codes
            or any(type(code) is not str or not code for code in self.allowed_error_codes)
            or tuple(sorted(set(self.allowed_error_codes))) != self.allowed_error_codes
        ):
            raise ValueError("approved exclusion error codes must be sorted and unique")
        if type(self.max_total_exclusions) is not int or self.max_total_exclusions < 1:
            raise ValueError("max_total_exclusions must be a positive exact integer")
        if (
            type(self.max_exclusions_per_primary_cell) is not int
            or not 1 <= self.max_exclusions_per_primary_cell < 3
        ):
            raise ValueError(
                "max_exclusions_per_primary_cell must preserve at least one planned repeat"
            )


__all__ = ["ApprovedTechnicalExclusionPolicy"]
