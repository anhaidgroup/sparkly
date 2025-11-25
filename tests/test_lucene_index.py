"""Unit tests exercising uncovered helpers in sparkly.index.lucene_index."""

import numpy as np
import pandas as pd
import pytest

from sparkly.index.lucene_index import LuceneIndex, _DocumentConverter
from sparkly.index_config import IndexConfig
from sparkly.query_generator import QuerySpec


@pytest.fixture
def sample_config():
    cfg = IndexConfig()
    cfg.add_field("name", ["standard"])
    cfg.add_field("description", ["standard"])
    cfg.add_concat_field("full_text", ["name", "description"], ["standard"])
    return cfg


@pytest.fixture
def lucene_index_instance(tmp_path, sample_config):
    return LuceneIndex(
        tmp_path / "index",
        sample_config,
        delete_if_exists=True,
    )


class TestDocumentConverter:
    """Tests for _DocumentConverter class."""

    def test_format_columns_expands_fields(self, sample_config):
        """
        _format_columns should concat fields and create analyzer-specific cols.
        """
        converter = _DocumentConverter(sample_config)
        df = pd.DataFrame(
            {
                "_id": [1, 2],
                "name": ["Widget", "Gadget"],
                "description": ["Blue", "Red"],
            }
        )

        formatted = converter._format_columns(df.copy())

        assert formatted.index.name == "_id"
        assert set(formatted.columns) == {
            "name.standard",
            "description.standard",
            "full_text.standard",
        }
        assert formatted.loc[1, "full_text.standard"] == "Widget Blue"


class TestLuceneIndex:
    """Tests for helper methods on LuceneIndex."""

    def test_arg_check_config_requires_fields(self, tmp_path):
        """_arg_check_config should fail when no fields are defined."""
        empty_config = IndexConfig()

        with pytest.raises(ValueError):
            LuceneIndex(
                tmp_path / "index",
                empty_config,
                delete_if_exists=True,
            )

    def test_arg_check_upsert_missing_required_field(
        self,
        lucene_index_instance,
    ):
        """_arg_check_upsert raises when analyzed fields are missing."""
        df = pd.DataFrame(
            {
                "_id": [1],
                "description": ["missing name"],
            }
        )

        with pytest.raises(ValueError) as excinfo:
            lucene_index_instance._arg_check_upsert(df)

        assert "missing columns" in str(excinfo.value)

    def test_arg_check_upsert_requires_integer_ids(
        self,
        lucene_index_instance,
    ):
        """_arg_check_upsert enforces integer id_col types."""
        df = pd.DataFrame(
            {
                "_id": ["not-int"],
                "name": ["Widget"],
                "description": ["Blue"],
            }
        )

        with pytest.raises(TypeError) as excinfo:
            lucene_index_instance._arg_check_upsert(df)

        assert "id_col must be integer type" in str(excinfo.value)

    def test_chunk_df_respects_chunk_size(self, lucene_index_instance):
        lucene_index_instance._index_build_chunk_size = 2
        df = pd.DataFrame(
            {
                "_id": [1, 2, 3, 4, 5],
                "name": ["a", "b", "c", "d", "e"],
                "description": ["x"] * 5,
            }
        )

        chunks = list(lucene_index_instance._chunk_df(df))

        assert [len(chunk) for chunk in chunks] == [2, 2, 1]
        assert chunks[0].equals(df.iloc[0:2])

    def test_num_indexed_docs_uses_reader_count(self, lucene_index_instance):
        """num_indexed_docs returns the count from _index_reader."""

        class DummyReader:
            def __init__(self, count):
                self._count = count

            def numDocs(self):
                return self._count

        lucene_index_instance._index_reader = DummyReader(7)
        lucene_index_instance.init = lambda: None

        assert lucene_index_instance.num_indexed_docs() == 7

    def test_get_full_query_spec_basic(self, lucene_index_instance):
        """get_full_query_spec returns analyzer-specific field names."""
        spec = lucene_index_instance.get_full_query_spec()

        assert spec == QuerySpec(
            {
                "name": {"name.standard"},
                "description": {"description.standard"},
                "full_text": {"full_text.standard"},
            }
        )

    def test_get_full_query_spec_cross_fields(self, lucene_index_instance):
        """cross_fields adds concat analyzers to source fields."""
        spec = lucene_index_instance.get_full_query_spec(cross_fields=True)

        assert spec["name"] == {"name.standard", "full_text.standard"}
        assert spec["description"] == {
            "description.standard",
            "full_text.standard",
        }
        assert spec["full_text"] == {"full_text.standard"}

    def test_get_full_query_spec_reads_metadata_when_config_missing(
        self,
        lucene_index_instance,
    ):
        """When _config is None, the spec is rebuilt from stored metadata."""
        lucene_index_instance._config = None
        spec = lucene_index_instance.get_full_query_spec()

        assert "name" in spec
        assert spec["full_text"] == {"full_text.standard"}


class TestLuceneIndexSparkData:
    """Ensure Spark DataFrames work end-to-end."""

    def test_upsert_to_pandas_spark_dataframe(self, tmp_path, sample_table_a):
        """Small dataframe (< 25000) converts to pandas, single-threaded."""
        config = IndexConfig()
        config.add_field("name", ["standard"])
        config.add_field("description", ["standard"])
        config.id_col = "_id"

        index = LuceneIndex(
            tmp_path / "spark-index",
            config,
            delete_if_exists=True,
        )
        index.upsert_docs(sample_table_a)

        assert index.is_built
        assert index.num_indexed_docs() == sample_table_a.count()

    def test_upsert_already_built_index(self, tmp_path, sample_table_a):
        """Upserting an already built index should not rebuild the index."""
        config = IndexConfig()
        config.add_field("name", ["standard"])
        config.add_field("description", ["standard"])
        config.id_col = "_id"

        index = LuceneIndex(
            tmp_path / "spark-index",
            config,
            delete_if_exists=True,
        )
        index.upsert_docs(sample_table_a)

        assert index.is_built
        assert index.num_indexed_docs() == sample_table_a.count()

        index.upsert_docs(sample_table_a)

        assert index.is_built
        assert index.num_indexed_docs() == sample_table_a.count()

    def test_upsert_medium_dataframe_parallel_spark(
        self, tmp_path, medium_table_a
    ):
        """
        > 25k rows with force_distributed=True uses spark parallel build.
        """
        config = IndexConfig()
        config.add_field("name", ["standard"])
        config.add_field("description", ["standard"])
        config.id_col = "_id"
        
        index = LuceneIndex(
            tmp_path / "medium-index",
            config,
            delete_if_exists=True,
        )

        index.upsert_docs(medium_table_a, force_distributed=True, show_progress_bar=True)
        
        assert index.is_built
        
        num_docs = index.num_indexed_docs()
        assert num_docs == medium_table_a.count()

    def test_delete_docs_single_id(self, tmp_path, sample_table_a):
        """delete_docs should delete the documents from the index."""
        config = IndexConfig()
        config.add_field("name", ["standard"])
        config.add_field("description", ["standard"])
        config.id_col = "_id"

        index = LuceneIndex(
            tmp_path / "delete-index",
            config,
            delete_if_exists=True,
        )
        index.upsert_docs(sample_table_a)
        assert index.is_built
        initial_count = sample_table_a.count()
        assert index.num_indexed_docs() == initial_count

        # Get an actual ID from the DataFrame to delete
        actual_ids = (
            sample_table_a.select("_id")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        id_to_delete = actual_ids[0]

        deleted_count = index.delete_docs([id_to_delete])
        assert deleted_count == 1  # Should have deleted 1 document
        assert index.num_indexed_docs() == initial_count - 1

    def test_delete_docs_multiple_ids(self, tmp_path, sample_table_a):
        """delete_docs should delete the documents from the index."""
        config = IndexConfig()
        config.add_field("name", ["standard"])
        config.add_field("description", ["standard"])
        config.id_col = "_id"

        index = LuceneIndex(
            tmp_path / "delete-index",
            config,
            delete_if_exists=True,
        )
        index.upsert_docs(sample_table_a)
        assert index.is_built
        initial_count = sample_table_a.count()
        assert index.num_indexed_docs() == initial_count

        # Get an actual ID from the DataFrame to delete
        actual_ids = (
            sample_table_a.select("_id")
            .rdd.map(lambda r: r[0])
            .collect()
        )
        ids_to_delete = pd.Series([actual_ids[0], actual_ids[1]])

        deleted_count = index.delete_docs(ids_to_delete)
        assert deleted_count == 2  # Should have deleted 2 documents
        assert index.num_indexed_docs() == initial_count - 2


@pytest.fixture
def abt_lucene_index(tmp_path_factory, sample_table_a):
    """
    Build a real Lucene index from the abt_buy sample Spark DataFrame to
    exercise search-related methods end-to-end.
    """
    tmp_dir = tmp_path_factory.mktemp("abt_index")
    config = IndexConfig()
    config.add_field("name", ["standard"])
    config.add_field("description", ["standard"])
    config.id_col = "_id"

    index = LuceneIndex(tmp_dir, config, delete_if_exists=True)
    index.upsert_docs(sample_table_a)
    index.init()

    docs_pdf = (
        sample_table_a.select("_id", "name", "description")
        .toPandas()
        .reset_index(drop=True)
    )
    query_spec = QuerySpec({"name": ["name.standard"]})

    yield index, docs_pdf, query_spec

    index.deinit()


class TestLuceneIndexWithRealData:
    """Integration-style tests that rely on PySpark and PyLucene."""

    def test_search_returns_results(self, abt_lucene_index):
        index, docs_pdf, query_spec = abt_lucene_index
        doc = docs_pdf.iloc[0][["name", "description"]].to_dict()

        result = index.search(doc, query_spec, limit=5)

        assert result.ids is not None
        assert len(result.ids) >= 1
        assert len(result.scores) == len(result.ids)

    def test_search_many_dataframe_shape(self, abt_lucene_index):
        index, docs_pdf, query_spec = abt_lucene_index
        docs_subset = docs_pdf.head(2)[["name", "description"]]

        result_df = index.search_many(docs_subset, query_spec, limit=5)

        assert list(result_df.index) == list(docs_subset.index)
        assert all(
            isinstance(row.ids, np.ndarray)
            for _, row in result_df.iterrows()
        )

    def test_id_to_lucene_id_round_trip(self, abt_lucene_index):
        index, docs_pdf, _ = abt_lucene_index
        target_id = int(docs_pdf["_id"].iloc[0])

        lucene_doc_id = index.id_to_lucene_id(target_id)

        assert isinstance(lucene_doc_id, int)
        # Fetching back through id_to_lucene_id should succeed repeatedly
        assert index.id_to_lucene_id(target_id) == lucene_doc_id

    def test_score_docs_returns_scores(self, abt_lucene_index):
        index, docs_pdf, query_spec = abt_lucene_index
        ids = docs_pdf["_id"].head(2).tolist()
        doc = docs_pdf.iloc[0][["name", "description"]].to_dict()
        query = index._query_gen.generate_query(doc, query_spec)
        queries = {("name", "name.standard"): query}

        scores_df = index.score_docs(ids, queries)

        assert scores_df.shape[0] >= len(ids)
        assert index.config.id_col in scores_df.columns
        assert ("name", "name.standard") in scores_df.columns
