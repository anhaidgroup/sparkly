"""Unit tests for sparkly.query_generator.lucene_query_generator."""

import pytest

from sparkly.index.lucene_index import LuceneIndex
from sparkly.index_config import IndexConfig
from sparkly.query_generator import QuerySpec
from sparkly.query_generator.lucene_query_generator import (
    LuceneQueryGenerator,
)


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


@pytest.fixture
def query_generator(real_index_with_docs):
    """Create a LuceneQueryGenerator instance for testing."""
    index = real_index_with_docs
    return index._query_gen


class TestLuceneQueryGenerator:
    """Tests for LuceneQueryGenerator class."""

    def test_lucene_query_generator_init(self, real_index_with_docs):
        """Test LuceneQueryGenerator initialization."""
        index = real_index_with_docs
        generator = index._query_gen

        assert isinstance(generator, LuceneQueryGenerator)
        assert generator._config.to_dict() == index.config.to_dict()
        assert generator._query_builder is not None

    def test_lucene_query_generator_generate(self, query_generator):
        """Test LuceneQueryGenerator generate_query method."""
        doc = {"name": "Widget", "description": "Blue widget"}
        query_spec = QuerySpec(
            {
                "name": {"name.standard"},
                "description": {"description.standard"},
            }
        )

        query = query_generator.generate_query(doc, query_spec)

        assert query is not None
        # Query should be a BooleanQuery
        from org.apache.lucene.search import BooleanQuery
        assert isinstance(query, BooleanQuery)

    def test_lucene_query_generator_generate_with_real_data(
        self, query_generator, sample_table_a
    ):
        """Test generate_query with real abt_buy data."""
        # Get a real document from the indexed data
        sample_doc = (
            sample_table_a.select("name", "description")
            .limit(1)
            .toPandas()
            .iloc[0]
            .to_dict()
        )
        query_spec = QuerySpec(
            {
                "name": {"name.standard"},
                "description": {"description.standard"},
            }
        )

        query = query_generator.generate_query(sample_doc, query_spec)

        assert query is not None
        from org.apache.lucene.search import BooleanQuery
        assert isinstance(query, BooleanQuery)

    def test_lucene_query_generator_generate_query_clauses(
        self, query_generator
    ):
        """Test LuceneQueryGenerator generate_query_clauses method."""
        doc = {"name": "Widget", "description": "Blue widget"}
        query_spec = QuerySpec(
            {
                "name": {"name.standard"},
                "description": {"description.standard"},
            }
        )

        clauses = query_generator.generate_query_clauses(doc, query_spec)

        assert isinstance(clauses, dict)
        assert ("name", "name.standard") in clauses
        assert ("description", "description.standard") in clauses
        # Each clause should be a Query
        from org.apache.lucene.search import Query
        assert isinstance(
            clauses[("name", "name.standard")], Query
        )
