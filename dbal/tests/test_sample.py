"""Sample test module.

This is provided as an example for testing your library.
Find out more: https://pytest.org
"""

import pytest


def test_sample():
    """Sample for tests with pytest module"""
    value = 10
    assert value > 0
    with pytest.raises(ZeroDivisionError):
        value = value / 0
    assert value == 10
