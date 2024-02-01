"""Documentation-friendly enum classes."""

from typing import Self

from strenum import LowercaseStrEnum


class StrEnum(LowercaseStrEnum):
    """StrEnum which renders only its value in a string context."""

    @classmethod
    def parse(cls, value: str) -> Self:
        """Parse a string into a StrEnum of proper type."""
        return cls(value)

    def __repr__(self) -> str:  # noqa: D105
        return f"'{self.value}'"
