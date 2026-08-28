"""Unit tests for sparkly.search module."""

import pytest
from sparkly.search import Searcher, search, search_gen
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


class TestSearcher:
    """Tests for Searcher class."""

    def test_searcher_init_defaults(self, real_index_with_docs):
        """Test Searcher initialization with default parameters."""
        searcher = Searcher(real_index_with_docs)

        assert searcher._index == real_index_with_docs
        assert searcher._search_chunk_size == 500

    def test_searcher_init_custom_chunk_size(self, real_index_with_docs):
        """Test Searcher initialization with custom search_chunk_size."""
        custom_chunk_size = 1000
        searcher = Searcher(
            real_index_with_docs, search_chunk_size=custom_chunk_size
        )

        assert searcher._index == real_index_with_docs
        assert searcher._search_chunk_size == custom_chunk_size

    def test_searcher_init_validation(self, real_index_with_docs):
        """Test Searcher parameter validation (type checking)."""
        # Invalid: search_chunk_size must be positive
        with pytest.raises(Exception):  # Pydantic validation error
            Searcher(real_index_with_docs, search_chunk_size=0)

        with pytest.raises(Exception):  # Pydantic validation error
            Searcher(real_index_with_docs, search_chunk_size=-1)

    def test_searcher_get_full_query_spec(self, real_index_with_docs):
        """Test Searcher get_full_query_spec method."""
        searcher = Searcher(real_index_with_docs)

        query_spec = searcher.get_full_query_spec()

        # Should return a QuerySpec
        assert isinstance(query_spec, QuerySpec)

        # Should match expected spec based on config:
        # name -> name.standard, description -> description.standard
        expected_spec = QuerySpec({
            'name': {'name.standard'},
            'description': {'description.standard'}
        })
        assert query_spec == expected_spec

    def test_searcher_search(
        self, real_index_with_docs, sample_table_b
    ):
        """Test Searcher search method."""
        searcher = Searcher(real_index_with_docs)

        # Create a query spec
        query_spec = QuerySpec({
            'name': {'name.standard'},
            'description': {'description.standard'}
        })

        # Perform search
        results = searcher.search(
            sample_table_b, query_spec, limit=5, id_col='_id'
        )

        # Check that results is a DataFrame
        assert results is not None

        # Check schema
        columns = results.columns
        assert 'id2' in columns
        assert 'id1_list' in columns
        assert 'scores' in columns
        assert 'search_time' in columns

        # Check that we get results
        result_count = results.count()
        assert result_count > 0

        # Check that each row has the expected structure
        sample_row = results.limit(1).collect()[0]
        assert hasattr(sample_row, 'id2')
        assert hasattr(sample_row, 'id1_list')
        assert hasattr(sample_row, 'scores')
        assert hasattr(sample_row, 'search_time')

        # Check types
        assert isinstance(sample_row.id1_list, list)
        assert isinstance(sample_row.scores, list)
        assert isinstance(sample_row.search_time, float)

    def test_searcher_search_missing_id_column(
        self, real_index_with_docs, sample_table_b
    ):
        """Searcher search raises when id_col is not in search_df."""
        searcher = Searcher(real_index_with_docs)
        query_spec = QuerySpec({'name': {'name.standard'}})

        with pytest.raises(ValueError, match="missing id column"):
            searcher.search(
                sample_table_b, query_spec, limit=5, id_col='no_such_col'
            )

    def test_searcher_search_rejects_duplicate_ids(
        self, real_index_with_docs, sample_table_b
    ):
        """Searcher search raises when search_df ids are duplicated."""
        searcher = Searcher(real_index_with_docs)
        query_spec = QuerySpec({'name': {'name.standard'}})
        duplicated = sample_table_b.union(sample_table_b)

        with pytest.raises(ValueError, match="must be unique"):
            searcher.search(duplicated, query_spec, limit=5, id_col='_id')

    def test_searcher_search_rejects_null_ids(
        self, real_index_with_docs, sample_table_b, spark_session
    ):
        """Searcher search raises when search_df ids contain nulls."""
        import pyspark.sql.functions as F

        searcher = Searcher(real_index_with_docs)
        query_spec = QuerySpec({'name': {'name.standard'}})
        with_null = sample_table_b.withColumn(
            '_id',
            F.when(F.col('_id') == sample_table_b.first()['_id'], None)
            .otherwise(F.col('_id'))
        )

        with pytest.raises(ValueError, match="nulls are present"):
            searcher.search(with_null, query_spec, limit=5, id_col='_id')

    def test_searcher_search_validate_ids_disabled(
        self, real_index_with_docs, sample_table_b
    ):
        """Searcher search skips id validation when validate_ids is False."""
        searcher = Searcher(real_index_with_docs)
        query_spec = QuerySpec({'name': {'name.standard'}})
        duplicated = sample_table_b.union(sample_table_b)

        results = searcher.search(
            duplicated, query_spec, limit=5, id_col='_id', validate_ids=False
        )

        assert results.count() == duplicated.count()


class TestSearchFunctions:
    """Tests for module-level search functions."""

    def test_search(self, real_index_with_docs):
        """Test search function."""
        # Create sample search records
        search_recs = [
            {'name': 'Widget', 'description': 'Blue widget'},
            {'name': 'Gadget', 'description': 'Red gadget'}
        ]

        query_spec = QuerySpec({
            'name': {'name.standard'},
            'description': {'description.standard'}
        })

        results = search(real_index_with_docs, query_spec, 5, search_recs)

        # Should return a list
        assert isinstance(results, list)
        assert len(results) == len(search_recs)

        # Each result should be a QueryResult
        from sparkly.index import QueryResult
        for result in results:
            assert isinstance(result, QueryResult)
            assert hasattr(result, 'id1_list')
            assert hasattr(result, 'scores')
            assert hasattr(result, 'search_time')

    def test_search_gen(self, real_index_with_docs):
        """Test search_gen function."""
        # Create sample search records
        search_recs = [
            {'name': 'Widget', 'description': 'Blue widget'},
            {'name': 'Gadget', 'description': 'Red gadget'}
        ]

        query_spec = QuerySpec({
            'name': {'name.standard'},
            'description': {'description.standard'}
        })

        results = list(
            search_gen(real_index_with_docs, query_spec, 5, search_recs)
        )

        # Should return a list of QueryResults
        assert isinstance(results, list)
        assert len(results) == len(search_recs)

        # Each result should be a QueryResult
        from sparkly.index import QueryResult
        for result in results:
            assert isinstance(result, QueryResult)
            assert hasattr(result, 'id1_list')
            assert hasattr(result, 'scores')
            assert hasattr(result, 'search_time')

        # search_gen should be a generator
        gen = search_gen(real_index_with_docs, query_spec, 5, search_recs)
        assert hasattr(gen, '__iter__')
        assert hasattr(gen, '__next__')
