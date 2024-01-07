"""Test telemetry module."""

import structlog
from py_research.telemetry import get_logger


def test_get_logger():
    """Test get_logger."""
    # Test with no arguments
    logger = get_logger()
    assert isinstance(logger, structlog.stdlib.BoundLogger)

    # Test with a name
    logger = get_logger("test")
    assert isinstance(logger, structlog.stdlib.BoundLogger)
    assert logger.name == "test"
