"""Test hashing module."""

from datetime import date
from decimal import Decimal
from fractions import Fraction

from py_research.hashing import gen_str_hash


def test_gen_str_hash():
    """Test gen_str_hash."""
    # Test with different types of numbers
    assert len(gen_str_hash(123, length=10)) == 10
    assert len(gen_str_hash(123.456, length=10)) == 10
    assert len(gen_str_hash(complex(1, 2), length=10)) == 10
    assert len(gen_str_hash(Decimal("123.456"), length=10)) == 10
    assert len(gen_str_hash(Fraction(1, 2), length=10)) == 10

    # Test with a string
    assert len(gen_str_hash("test", length=10)) == 10

    # Test with a date
    assert len(gen_str_hash(date.today(), length=10)) == 10

    # Test with raw_str=True
    assert gen_str_hash("test", length=10, raw_str=True) == "test"
