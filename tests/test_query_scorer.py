"""Unit tests for sparkly.index_optimizer.query_scorer module."""

import pytest
import numpy as np
from sparkly.index_optimizer import query_scorer
from sparkly.index import QueryResult


class TestScoreQueryResults:
    """Tests for score_query_results function."""

    def test_score_query_results_basic(self):
        """Test basic score_query_results functionality."""
        # Create mock query results
        query_results = [
            QueryResult(
                id1_list=[1, 2, 3],
                scores=np.array([10.0, 8.0, 5.0], dtype=np.float32),
                search_time=0.1
            ),
            QueryResult(
                id1_list=[4, 5],
                scores=np.array([9.0, 7.0], dtype=np.float32),
                search_time=0.2
            ),
        ]

        results = query_scorer.score_query_results(query_results)

        assert isinstance(results, list)
        assert len(results) == 2
        # Each result should be a float (from score_query_result)
        assert all(isinstance(r, (float, np.floating)) for r in results)


class TestScoreQueryResult:
    """Tests for score_query_result function."""

    def test_score_query_result_basic(self):
        """Test basic score_query_result functionality."""
        scores = np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32)
        result = query_scorer.score_query_result(scores)

        # Should return normalized AUC of scores normalized by first score
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_score_query_result_empty_scores(self):
        """Test score_query_result with empty scores."""
        scores = []
        result = query_scorer.score_query_result(scores)

        # Empty array should return 1.0
        assert result == 1.0

    def test_score_query_result_single_score(self):
        """Test score_query_result with single score."""
        scores = [10.0]
        result = query_scorer.score_query_result(scores)

        # Single score should return 1.0
        assert result == 1.0

    def test_score_query_result_scalar(self):
        """Test score_query_result with scalar (0-d array)."""
        scores = np.array(10.0)
        result = query_scorer.score_query_result(scores)

        # Scalar should return 1.0
        assert result == 1.0

    def test_score_query_result_drop_first(self):
        """Test score_query_result with drop_first=True."""
        scores = np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32)
        result_with_drop = query_scorer.score_query_result(
            scores, drop_first=True
        )
        result_without_drop = query_scorer.score_query_result(
            scores, drop_first=False
        )

        # Results should be different
        assert result_with_drop != result_without_drop
        # Both should be valid floats
        assert isinstance(result_with_drop, (float, np.floating))
        assert isinstance(result_without_drop, (float, np.floating))

    def test_score_query_result_two_scores(self):
        """Test score_query_result with exactly two scores."""
        scores = np.array([10.0, 8.0], dtype=np.float32)
        result = query_scorer.score_query_result(scores)

        # Should compute norm_auc on normalized scores
        assert isinstance(result, (float, np.floating))
        assert result > 0


class TestScoreQueryResultSum:
    """Tests for score_query_result_sum function."""

    def test_score_query_result_sum_basic(self):
        """Test basic score_query_result_sum functionality."""
        scores = np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32)
        result = query_scorer.score_query_result_sum(scores)

        # Should return sum / (norm_auc^2)
        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_score_query_result_sum_empty(self):
        """Test score_query_result_sum with empty scores."""
        scores = []
        result = query_scorer.score_query_result_sum(scores)

        # Empty should return 0.0
        assert result == 0.0

    def test_score_query_result_sum_single_score(self):
        """Test score_query_result_sum with single score."""
        scores = [10.0]
        result = query_scorer.score_query_result_sum(scores)

        # Single score should return 0.0
        assert result == 0.0

    def test_score_query_result_sum_scalar(self):
        """Test score_query_result_sum with scalar."""
        scores = np.array(10.0)
        result = query_scorer.score_query_result_sum(scores)

        # Scalar should return 0.0
        assert result == 0.0

    def test_score_query_result_sum_two_scores(self):
        """Test score_query_result_sum with exactly two scores."""
        scores = np.array([10.0, 8.0], dtype=np.float32)
        result = query_scorer.score_query_result_sum(scores)

        # Should compute sum / (norm_auc^2)
        assert isinstance(result, (float, np.floating))
        assert result > 0


class TestUpdateSpec:
    """Tests for _update_spec function."""

    def test_update_spec_basic(self):
        """Test basic _update_spec functionality."""
        base = {'name': ['name.standard']}
        result = query_scorer._update_spec(base, 'name', 'name.3gram')

        # Should add new path
        assert 'name.3gram' in result['name']
        assert 'name.standard' in result['name']
        # Original should not be modified
        assert 'name.3gram' not in base['name']

    def test_update_spec_new_field(self):
        """Test _update_spec with new field."""
        base = {'name': ['name.standard']}
        result = query_scorer._update_spec(
            base, 'description', 'description.standard'
        )

        # Should add new field
        assert 'description' in result
        assert result['description'] == ['description.standard']
        # Original should not be modified
        assert 'description' not in base

    def test_update_spec_duplicate_path(self):
        """Test _update_spec doesn't add duplicate paths."""
        base = {'name': ['name.standard']}
        result = query_scorer._update_spec(base, 'name', 'name.standard')

        # Should not duplicate
        assert result['name'].count('name.standard') == 1


class TestComputeWilcoxonScore:
    """Tests for compute_wilcoxon_score function."""

    def test_compute_wilcoxon_score_basic(self):
        """Test basic compute_wilcoxon_score calculation."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([1.5, 2.5, 3.5, 4.5])
        result = query_scorer.compute_wilcoxon_score(x, y)

        # Should return tuple (statistic, pvalue)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], (float, np.floating))
        assert isinstance(result[1], (float, np.floating))

    def test_compute_wilcoxon_score_identical(self):
        """Test compute_wilcoxon_score with identical arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        result = query_scorer.compute_wilcoxon_score(x, y)

        # Should return (0, 1) when all elements are the same
        assert result == (0, 1)

    def test_compute_wilcoxon_score_different(self):
        """Test compute_wilcoxon_score with different arrays."""
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([2.0, 3.0, 4.0])
        result = query_scorer.compute_wilcoxon_score(x, y)

        # Should return valid wilcoxon result
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestQueryScorer:
    """Tests for QueryScorer abstract base class."""

    def test_query_scorer_cannot_be_instantiated(self):
        """Test that QueryScorer cannot be instantiated directly."""
        with pytest.raises(TypeError):
            query_scorer.QueryScorer()


class TestAUCQueryScorer:
    """Tests for AUCQueryScorer class."""

    def test_auc_query_scorer_init(self):
        """Test AUCQueryScorer initialization (no parameters)."""
        scorer = query_scorer.AUCQueryScorer()
        assert isinstance(scorer, query_scorer.AUCQueryScorer)
        assert isinstance(scorer, query_scorer.QueryScorer)

    def test_auc_query_scorer_score_query_result(self):
        """Test AUCQueryScorer score_query_result method."""
        scorer = query_scorer.AUCQueryScorer()
        query_result = QueryResult(
            id1_list=[1, 2, 3],
            scores=np.array([10.0, 8.0, 5.0], dtype=np.float32),
            search_time=0.1
        )

        result = scorer.score_query_result(
            query_result, None, drop_first=False
        )

        assert isinstance(result, (float, np.floating))
        assert result > 0

    def test_auc_query_scorer_score_query_result_drop_first(self):
        """Test AUCQueryScorer score_query_result with drop_first."""
        scorer = query_scorer.AUCQueryScorer()
        query_result = QueryResult(
            id1_list=[1, 2, 3],
            scores=np.array([10.0, 8.0, 5.0], dtype=np.float32),
            search_time=0.1
        )

        result_drop = scorer.score_query_result(
            query_result, None, drop_first=True
        )
        result_no_drop = scorer.score_query_result(
            query_result, None, drop_first=False
        )

        assert result_drop != result_no_drop

    def test_auc_query_scorer_score_query_results(self):
        """Test AUCQueryScorer score_query_results method."""
        scorer = query_scorer.AUCQueryScorer()
        query_results = [
            QueryResult(
                id1_list=[1, 2],
                scores=np.array([10.0, 8.0], dtype=np.float32),
                search_time=0.1
            ),
            QueryResult(
                id1_list=[3, 4],
                scores=np.array([9.0, 7.0], dtype=np.float32),
                search_time=0.2
            ),
        ]

        results = scorer.score_query_results(
            query_results, None, drop_first=False
        )

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, (float, np.floating)) for r in results)


class TestRankQueryScorer:
    """Tests for RankQueryScorer class."""

    def test_rank_query_scorer_init(self):
        """Test RankQueryScorer initialization with threshold and k."""
        scorer = query_scorer.RankQueryScorer(threshold=0.5, k=10)

        assert isinstance(scorer, query_scorer.RankQueryScorer)
        assert isinstance(scorer, query_scorer.QueryScorer)
        assert scorer._threshold == 0.5
        assert scorer._k == 10

    def test_rank_query_scorer_score_query_result(self):
        """Test RankQueryScorer score_query_result method."""
        scorer = query_scorer.RankQueryScorer(threshold=0.5, k=10)
        query_result = QueryResult(
            id1_list=[1, 2, 3, 4],
            scores=np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32),
            search_time=0.1
        )

        result = scorer.score_query_result(query_result, None)

        # Should return rank (number of scores >= threshold)
        assert isinstance(result, (int, np.integer))
        assert result >= 0

    def test_rank_query_scorer_score_query_result_empty(self):
        """Test RankQueryScorer with empty scores."""
        scorer = query_scorer.RankQueryScorer(threshold=0.5, k=10)
        query_result = QueryResult(
            id1_list=[], scores=np.array([], dtype=np.float32), search_time=0.1
        )

        result = scorer.score_query_result(query_result, None)

        # Should return k when scores are empty
        assert result == 10

    def test_rank_query_scorer_score_query_result_single(self):
        """Test RankQueryScorer with single score."""
        scorer = query_scorer.RankQueryScorer(threshold=0.5, k=10)
        query_result = QueryResult(
            id1_list=[1],
            scores=np.array([10.0], dtype=np.float32),
            search_time=0.1
        )

        result = scorer.score_query_result(query_result, None)

        # Should return k when scores < 2
        assert result == 10

    def test_rank_query_scorer_score_query_results(self):
        """Test RankQueryScorer score_query_results method."""
        scorer = query_scorer.RankQueryScorer(threshold=0.5, k=10)
        query_results = [
            QueryResult(
                id1_list=[1, 2],
                scores=np.array([10.0, 8.0], dtype=np.float32),
                search_time=0.1
            ),
            QueryResult(
                id1_list=[3, 4],
                scores=np.array([9.0, 7.0], dtype=np.float32),
                search_time=0.2
            ),
        ]

        results = scorer.score_query_results(query_results, None)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, (int, np.integer)) for r in results)
