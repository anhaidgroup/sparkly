"""Unit tests for sparkly.index.index_base module."""

import pytest
from sparkly.index import index_base


class TestIndex:
    """Tests for Index abstract base class."""

    def test_index_cannot_be_instantiated(self):
        """Test that Index cannot be instantiated directly (it's abstract)."""
        with pytest.raises(TypeError):
            index_base.Index()
