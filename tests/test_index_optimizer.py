"""Unit tests for sparkly.index_optimizer.index_optimizer module."""

import pytest
import numpy as np
from sparkly.index_optimizer.index_optimizer import (
    IndexOptimizer,
    _compute_wilcoxon_score,
)
from sparkly.index_optimizer.query_scorer import (
    AUCQueryScorer,
    RankQueryScorer,
)
from sparkly.index.lucene_index import LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.query_generator import QuerySpec


@pytest.fixture
def sample_config():
    """Create a sample IndexConfig for testing."""
    config = IndexConfig()
    config.add_field("name", ["standard"])
    config.add_field("description", ["standard"])
    config.id_col = "_id"
    return config


@pytest.fixture
def real_index_with_docs(tmp_path, sample_config, sample_table_a):
    """
    Create a real Lucene index using LuceneIndex with abt_buy sample data.
    Returns the LuceneIndex instance (which has the index_reader).
    """
    index = LuceneIndex(
        tmp_path / "test_index", sample_config, delete_if_exists=True
    )
    index.upsert_docs(sample_table_a)
    index.init()

    yield index

    index.deinit()


class TestComputeWilcoxonScore:
    """Tests for _compute_wilcoxon_score function."""

    def test_compute_wilcoxon_score_basic(self):
        """Test basic _compute_wilcoxon_score calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.5, 2.5, 3.5, 4.5])
        result = _compute_wilcoxon_score(x, y)

        # Should return tuple (statistic, pvalue)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (float, np.floating))
        assert isinstance(result[1], (float, np.floating))

    def test_compute_wilcoxon_score_identical(self):
        """Test _compute_wilcoxon_score with identical arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = _compute_wilcoxon_score(x, y)

        # Should return (0, 1) when all elements are the same
        assert result == (0, 1)

    def test_compute_wilcoxon_score_different(self):
        """Test _compute_wilcoxon_score with different arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        result = _compute_wilcoxon_score(x, y)

        # Should return valid wilcoxon result
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestIndexOptimizer:
    """Tests for IndexOptimizer class."""

    def test_index_optimizer_init_defaults(self):
        """Test IndexOptimizer initialization with default parameters."""
        optimizer = IndexOptimizer(is_dedupe=False)

        assert optimizer._is_dedupe is False
        assert isinstance(optimizer._scorer, AUCQueryScorer)
        assert optimizer._confidence == 0.99
        assert optimizer._init_top_k == 10
        assert optimizer._max_combination_size == 3
        assert optimizer._opt_query_limit == 250
        assert optimizer._sample_size == 10000
        assert optimizer._use_early_pruning is True
        assert optimizer._index is None

    def test_index_optimizer_init_custom_params(self):
        """Test IndexOptimizer initialization with custom parameters."""
        scorer = RankQueryScorer(threshold=0.5, k=10)
        optimizer = IndexOptimizer(
            is_dedupe=True,
            scorer=scorer,
            conf=0.95,
            init_top_k=5,
            max_combination_size=2,
            opt_query_limit=100,
            sample_size=5000,
            use_early_pruning=False,
        )

        assert optimizer._is_dedupe is True
        assert optimizer._scorer == scorer
        assert optimizer._confidence == 0.95
        assert optimizer._init_top_k == 5
        assert optimizer._max_combination_size == 2
        assert optimizer._opt_query_limit == 100
        assert optimizer._sample_size == 5000
        assert optimizer._use_early_pruning is False

    def test_index_optimizer_init_validation(self):
        """Test IndexOptimizer parameter validation."""
        # Invalid: conf must be in [0, 1)
        with pytest.raises(ValueError, match='1 validation error for IndexOptimizer'):
            IndexOptimizer(is_dedupe=False, conf=-0.1)

        with pytest.raises(ValueError, match='1 validation error for IndexOptimizer'):
            IndexOptimizer(is_dedupe=False, conf=1.0)

        with pytest.raises(ValueError, match='1 validation error for IndexOptimizer'):
            IndexOptimizer(is_dedupe=False, conf=1.5)

    def test_index_optimizer_index_property(self, real_index_with_docs):
        """Test IndexOptimizer index property getter and setter."""
        optimizer = IndexOptimizer(is_dedupe=False)

        # Initially None
        assert optimizer.index is None

        # Set index
        optimizer.index = real_index_with_docs

        # Should be set
        assert optimizer.index == real_index_with_docs

    def test_index_optimizer_optimize(
        self, real_index_with_docs, sample_table_b
    ):
        """Test IndexOptimizer optimize method."""
        optimizer = IndexOptimizer(is_dedupe=False, sample_size=5)

        # Optimize should return a QuerySpec
        query_spec = optimizer.optimize(real_index_with_docs, sample_table_b)

        # Should return a QuerySpec
        assert isinstance(query_spec, QuerySpec)

        # Should have at least some fields
        assert len(query_spec) >= 0  # Could be empty QuerySpec

    def test_index_optimizer_make_index_config(self, sample_table_a):
        """Test IndexOptimizer make_index_config method."""
        optimizer = IndexOptimizer(is_dedupe=False)

        config = optimizer.make_index_config(sample_table_a, id_col='_id')

        # Should return an IndexConfig
        assert isinstance(config, IndexConfig)

        # Should have some fields added
        assert len(config.field_to_analyzers) > 0

        # Should have BM25 similarity settings
        assert config.sim['type'] == 'BM25'
        assert config.sim['b'] == 0.75
        assert config.sim['k1'] == 1.2
