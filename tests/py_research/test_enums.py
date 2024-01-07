"""Test enums module."""

from py_research.enums import StrEnum


class TestEnum(StrEnum):
    """Test enum."""

    __test__ = False

    VALUE1 = "value1"
    VALUE2 = "value2"


def test_str_enum_repr():
    """Test the __repr__ method of StrEnum."""
    assert repr(TestEnum.VALUE1) == "'value1'"
    assert repr(TestEnum.VALUE2) == "'value2'"
