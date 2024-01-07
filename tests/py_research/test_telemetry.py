"""Test telemetry module."""

from py_research.telemetry import get_logger


def test_get_logger():
    """Test get_logger."""
    # Test with no arguments
    logger = get_logger()
    logger.info("test")

    # Test with a name
    logger = get_logger("test")
    logger.info("test")
    assert logger.name == "test"
