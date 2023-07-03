"""Documentation-friendly enum classes."""

from strenum import LowercaseStrEnum


class StrEnum(LowercaseStrEnum):
    """StrEnum which renders only its value in a string context."""

    def __repr__(self) -> str:  # noqa: D105
        return f"'{self.value}'"
