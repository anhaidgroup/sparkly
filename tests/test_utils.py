"""Unit tests for sparkly.utils module."""

import pytest
import time
import logging
import numpy as np
import pandas as pd
from zipfile import ZipFile

from sparkly.utils import (
    get_index_name,
    Timer,
    get_logger,
    auc,
    norm_auc,
    atomic_unzip,
    zip_dir,
    invoke_task,
    persisted,
    is_persisted,
    is_null,
    type_check,
    type_check_iterable,
    repartition_df,
    spark_to_pandas_stream,
    check_tables_manual,
    check_tables_auto,
)


class TestGetIndexName:
    """Tests for get_index_name function."""

    def test_get_index_name_basic(self):
        """Test basic get_index_name functionality."""
        # Converts to lowercase and replaces hyphens
        result = get_index_name("My-Index-Name")
        assert result == "my_index_name"

        # Already lowercase, no hyphens
        result = get_index_name("simple_name")
        assert result == "simple_name"

    def test_get_index_name_with_postfixes(self):
        """Test get_index_name with postfixes."""
        result = get_index_name("index", "v1", "test")
        assert result == "index_v1_test"

        result = get_index_name("My-Index", "production")
        assert result == "my_index_production"

        # No postfixes
        result = get_index_name("index")
        assert result == "index"


class TestTimer:
    """Tests for Timer class."""

    def test_timer_get_interval(self):
        """Test Timer get_interval method."""
        timer = Timer()
        # First interval should be very small (just creation time)
        interval1 = timer.get_interval()
        assert interval1 >= 0

        # Sleep and get next interval
        time.sleep(0.1)
        interval2 = timer.get_interval()
        assert interval2 >= 0.09  # Should be close to sleep time

        # Next interval should be small again
        interval3 = timer.get_interval()
        assert interval3 < 0.01  # Should be very small

    def test_timer_get_total(self):
        """Test Timer get_total method."""
        timer = Timer()
        total1 = timer.get_total()
        assert total1 >= 0

        time.sleep(0.1)
        total2 = timer.get_total()
        assert total2 >= total1
        assert total2 >= 0.1

    def test_timer_set_start_time(self):
        """Test Timer set_start_time method."""
        timer = Timer()
        time.sleep(0.1)
        old_total = timer.get_total()
        assert old_total >= 0.1

        # Reset start time
        timer.set_start_time()
        new_total = timer.get_total()
        assert new_total < 0.05  # Should be very small after reset


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_basic(self):
        """Test basic get_logger functionality."""
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_module"
        assert logger.level == logging.DEBUG  # Default level

    def test_get_logger_with_level(self):
        """Test get_logger with custom log level."""
        logger = get_logger("test_module", level=logging.INFO)
        assert isinstance(logger, logging.Logger)
        assert logger.level == logging.INFO


class TestAuc:
    """Tests for auc function."""

    def test_auc_basic(self):
        """Test basic auc calculation."""
        # Array sorted in descending order
        x = np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32)
        result = auc(x)

        assert isinstance(result, (float, np.floating))
        assert result > 0

        # Single element
        x = np.array([10.0], dtype=np.float32)
        result = auc(x)
        assert isinstance(result, (float, np.floating))


class TestNormAuc:
    """Tests for norm_auc function."""

    def test_norm_auc_basic(self):
        """Test basic norm_auc calculation."""
        # Array sorted in descending order
        x = np.array([10.0, 8.0, 5.0, 3.0], dtype=np.float32)
        result = norm_auc(x)

        assert isinstance(result, (float, np.floating))
        assert result > 0
        # Normalized AUC should be less than or equal to max value
        assert result <= 10.0

        # Compare with auc - norm_auc should be auc / len(x)
        auc_result = auc(x)
        expected = auc_result / len(x)
        assert abs(result - expected) < 0.001


class TestAtomicUnzip:
    """Tests for atomic_unzip function."""

    def test_atomic_unzip_basic(self, tmp_path):
        """Test basic atomic_unzip functionality."""
        # Create a test zip file
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "test_file.txt").write_text("test content")

        zip_file = tmp_path / "test.zip"
        with ZipFile(zip_file, 'w') as zf:
            zf.write(test_dir / "test_file.txt", "test_file.txt")

        # Unzip to output location
        output_loc = tmp_path / "output"
        atomic_unzip(str(zip_file), str(output_loc))

        # Check that files were extracted
        assert output_loc.exists()
        assert (output_loc / "test_file.txt").exists()
        assert (output_loc / "test_file.txt").read_text() == "test content"

    def test_atomic_unzip_already_exists(self, tmp_path):
        """Test atomic_unzip when output already exists."""
        output_loc = tmp_path / "output"
        output_loc.mkdir()
        (output_loc / "existing_file.txt").write_text("existing")

        zip_file = tmp_path / "test.zip"
        with ZipFile(zip_file, 'w') as zf:
            zf.writestr("new_file.txt", "new content")

        # Should return early if output exists
        atomic_unzip(str(zip_file), str(output_loc))

        # Existing file should still be there
        assert (output_loc / "existing_file.txt").exists()
        # New file should not be extracted
        assert not (output_loc / "new_file.txt").exists()


class TestZipDir:
    """Tests for zip_dir function."""

    def test_zip_dir_basic(self, tmp_path):
        """Test basic zip_dir functionality."""
        # Create test directory with files
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")
        subdir = test_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_text("content3")

        # Zip the directory
        zip_file = zip_dir(test_dir)

        # Check that zip file was created
        assert zip_file.exists()
        assert zip_file.suffix == '.zip'

        # Verify contents
        with ZipFile(zip_file, 'r') as zf:
            names = zf.namelist()
            assert "file1.txt" in names
            assert "file2.txt" in names
            assert "subdir/file3.txt" in names

    def test_zip_dir_with_outfile(self, tmp_path):
        """Test zip_dir with specified output file."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("content")

        outfile = tmp_path / "custom.zip"
        zip_file = zip_dir(test_dir, outfile=outfile)

        assert zip_file == outfile
        assert zip_file.exists()


class TestInvokeTask:
    """Tests for invoke_task function."""

    def test_invoke_task_basic(self):
        """Test basic invoke_task functionality."""
        def add(a, b):
            return a + b

        # Create a delayed task (joblib format)
        task = (add, (1, 2), {})
        result = invoke_task(task)

        assert result == 3

    def test_invoke_task_with_kwargs(self):
        """Test invoke_task with keyword arguments."""
        def multiply(x, y=2):
            return x * y

        task = (multiply, (5,), {'y': 3})
        result = invoke_task(task)

        assert result == 15


class TestPersisted:
    """Tests for persisted context manager."""

    def test_persisted_basic(self, sample_table_a):
        """Test basic persisted functionality."""
        # Use persisted context manager
        with persisted(sample_table_a) as persisted_df:
            # DataFrame should be persisted
            assert is_persisted(persisted_df)

        # After context, should be unpersisted
        assert not is_persisted(sample_table_a)

    def test_persisted_none(self):
        """Test persisted with None DataFrame."""
        # Should handle None gracefully
        with persisted(None) as df:
            assert df is None


class TestIsPersisted:
    """Tests for is_persisted function."""

    def test_is_persisted_basic(self, sample_table_a):
        """Test basic is_persisted functionality."""
        # Initially not persisted
        assert not is_persisted(sample_table_a)

        # Persist it
        sample_table_a.persist()
        assert is_persisted(sample_table_a)

        # Unpersist
        sample_table_a.unpersist()
        assert not is_persisted(sample_table_a)


class TestIsNull:
    """Tests for is_null function."""

    def test_is_null_basic(self):
        """Test basic is_null functionality."""
        # None should be null
        assert is_null(None) is True

        # Regular values should not be null
        assert is_null(0) is False
        assert is_null("") is False
        assert is_null([]) is False

        # NaN should be null
        assert is_null(float('nan')) is True
        assert is_null(np.nan) is True

        # pd.NA should be null
        assert is_null(pd.NA) is True


class TestTypeCheck:
    """Tests for type_check function."""

    def test_type_check_valid(self):
        """Test type_check with valid types."""
        # Should not raise for valid types
        type_check(5, 'value', int)
        type_check("test", 'value', str)
        type_check([1, 2], 'value', list)

    def test_type_check_invalid(self):
        """Test type_check with invalid types."""
        with pytest.raises(TypeError, match='value must be type'):
            type_check(5, 'value', str)

        with pytest.raises(TypeError, match='value must be type'):
            type_check("test", 'value', int)


class TestTypeCheckIterable:
    """Tests for type_check_iterable function."""

    def test_type_check_iterable_valid(self):
        """Test type_check_iterable with valid types."""
        # Valid: list of ints
        type_check_iterable([1, 2, 3], 'values', list, int)

        # Valid: tuple of strings
        type_check_iterable(('a', 'b'), 'values', tuple, str)

    def test_type_check_iterable_invalid_container(self):
        """Test type_check_iterable with invalid container type."""
        with pytest.raises(TypeError, match='values must be type'):
            type_check_iterable([1, 2], 'values', dict, int)

    def test_type_check_iterable_invalid_elements(self):
        """Test type_check_iterable with invalid element types."""
        with pytest.raises(TypeError, match='all elements of values'):
            type_check_iterable([1, 2, "3"], 'values', list, int)


class TestRepartitionDf:
    """Tests for repartition_df function."""

    def test_repartition_df_basic(self, sample_table_a):
        """Test basic repartition_df functionality."""
        original_count = sample_table_a.count()

        # Repartition into chunks
        repartitioned = repartition_df(sample_table_a, part_size=2)

        # Should have partitions
        assert repartitioned.rdd.getNumPartitions() > 0

        # Count should be the same
        assert repartitioned.count() == original_count

    def test_repartition_df_with_by_column(
        self, sample_table_a, spark_session
    ):
        """Test repartition_df with by column."""
        import pyspark.sql.functions as F

        # Add a category column to partition by
        df = sample_table_a.withColumn(
            "category", (F.col("_id") % 2).cast("int")
        )
        original_count = df.count()

        # Repartition by category
        repartitioned = repartition_df(df, part_size=2, by="category")

        # Should have partitions
        assert repartitioned.rdd.getNumPartitions() > 0
        assert repartitioned.count() == original_count

    def test_repartition_df_small_dataframe(self, sample_table_a):
        """Test repartition_df with small DataFrame."""
        original_count = sample_table_a.count()

        # Should still work with small DataFrame
        repartitioned = repartition_df(sample_table_a, part_size=10)
        assert repartitioned.count() == original_count


class TestSparkToPandasStream:
    """Tests for spark_to_pandas_stream function."""

    def test_spark_to_pandas_stream_basic(self, sample_table_a):
        """Test basic spark_to_pandas_stream functionality."""
        original_count = sample_table_a.count()

        # Convert to pandas stream with chunk size
        batches = list(spark_to_pandas_stream(sample_table_a, chunk_size=2))

        # Should have batches
        assert len(batches) > 0

        # Each batch should be a pandas DataFrame
        for batch in batches:
            assert isinstance(batch, pd.DataFrame)

        # Total rows should match
        total_rows = sum(len(batch) for batch in batches)
        assert total_rows == original_count

    def test_spark_to_pandas_stream_with_custom_by(
        self, sample_table_a
    ):
        """Test spark_to_pandas_stream with custom by column."""
        # Rename _id to custom_id for testing
        df = sample_table_a.withColumnRenamed("_id", "custom_id")
        original_count = df.count()

        # Stream with custom by column
        batches = list(
            spark_to_pandas_stream(df, chunk_size=2, by="custom_id")
        )

        # Should work
        assert len(batches) > 0
        total_rows = sum(len(batch) for batch in batches)
        assert total_rows == original_count


class TestLocalParquetToSparkDf:
    """Tests for local_parquet_to_spark_df function."""

    def test_local_parquet_to_spark_df_basic(self, abt_buy_data_path):
        """Test basic local_parquet_to_spark_df functionality."""
        from sparkly.utils import local_parquet_to_spark_df

        # Use existing parquet file from test data
        parquet_file = abt_buy_data_path / "table_a.parquet"

        # Convert to Spark DataFrame
        df = local_parquet_to_spark_df(parquet_file)

        # Should be a Spark DataFrame
        assert hasattr(df, 'count')
        assert hasattr(df, 'columns')

        # Should have data
        count = df.count()
        assert count > 0

        # Should have expected columns
        assert '_id' in df.columns
        assert 'name' in df.columns

    def test_local_parquet_to_spark_df_schema(self, abt_buy_data_path):
        """Test local_parquet_to_spark_df preserves schema."""
        from sparkly.utils import local_parquet_to_spark_df

        parquet_file = abt_buy_data_path / "table_a.parquet"

        df = local_parquet_to_spark_df(parquet_file)

        # Check schema types
        schema = df.schema
        assert len(schema.fields) > 0

        # _id should be integer type
        id_field = [f for f in schema.fields if f.name == '_id'][0]
        assert 'LongType' in str(id_field.dataType) or 'IntegerType' in str(
            id_field.dataType
        )


class TestCheckTablesManual:
    """Tests for check_tables_manual (id columns only, no superset check)."""

    # --- Pandas: valid cases ---

    def test_check_tables_manual_pandas_valid(self):
        """check_tables_manual passes with valid pandas DataFrames (superset not required)."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame(
            {"_id": [10, 20], "name": ["x", "y"], "extra": [0, 0]}
        )
        check_tables_manual(table_a, "_id", table_b, "_id")  # no raise

    def test_check_tables_manual_pandas_same_columns(self):
        """check_tables_manual passes when table_b has same columns as table_a."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        check_tables_manual(table_a, "_id", table_b, "_id")  # no raise

    def test_check_tables_manual_pandas_table_b_not_superset(self):
        """check_tables_manual passes when table_b columns are not superset (no superset check)."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"], "foo": [0, 0]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})  # no foo
        check_tables_manual(table_a, "_id", table_b, "_id")  # no raise

    # --- Pandas: missing id column ---

    def test_check_tables_manual_pandas_table_a_missing_id_column(self):
        """check_tables_manual raises when table_a is missing the id column."""
        table_a = pd.DataFrame({"name": ["a", "b"]})  # no _id
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="table_a: missing id column"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    def test_check_tables_manual_pandas_table_b_missing_id_column(self):
        """check_tables_manual raises when table_b is missing the id column."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame({"name": ["x", "y"]})  # no _id
        with pytest.raises(ValueError, match="table_b: missing id column"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    # --- Pandas: empty ---

    def test_check_tables_manual_pandas_table_a_empty(self):
        """check_tables_manual raises when table_a is empty."""
        table_a = pd.DataFrame({"_id": [], "name": []})
        table_b = pd.DataFrame({"_id": [10], "name": ["x"]})
        with pytest.raises(ValueError, match="table_a: empty dataframe"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    def test_check_tables_manual_pandas_table_b_empty(self):
        """check_tables_manual raises when table_b is empty."""
        table_a = pd.DataFrame({"_id": [1], "name": ["a"]})
        table_b = pd.DataFrame({"_id": [], "name": []})
        with pytest.raises(ValueError, match="table_b: empty dataframe"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    # --- Pandas: nulls and types ---

    def test_check_tables_manual_pandas_nulls_in_table_a_id(self):
        """check_tables_manual raises when table_a id column has nulls."""
        table_a = pd.DataFrame({"_id": [1, np.nan], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    def test_check_tables_manual_pandas_nulls_in_table_b_id(self):
        """check_tables_manual raises when table_b id column has nulls."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, np.nan], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    def test_check_tables_manual_pandas_id_column_int64_nullable(self):
        """check_tables_manual passes with pandas nullable Int64 id column (e.g. from parquet)."""
        table_a = pd.DataFrame({"_id": pd.array([1, 2], dtype="Int64"), "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": pd.array([10, 20], dtype="Int64"), "name": ["x", "y"]})
        check_tables_manual(table_a, "_id", table_b, "_id")  # no raise

    def test_check_tables_manual_pandas_id_column_wrong_type(self):
        """check_tables_manual raises when id column is not int32/int64 or Int32/Int64."""
        table_a = pd.DataFrame(
            {"_id": [1.0, 2.0], "name": ["a", "b"]}
        )  # float
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="must be int32/int64 or Int32/Int64"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    def test_check_tables_manual_pandas_id_column_not_unique(self):
        """check_tables_manual raises when id column is not unique."""
        table_a = pd.DataFrame({"_id": [1, 1], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="must be unique"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    # --- Type mismatches ---

    def test_check_tables_manual_mixed_pandas_spark(self, sample_table_a):
        """check_tables_manual raises when table_a is pandas and table_b is Spark."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = sample_table_a  # Spark, from conftest
        with pytest.raises(TypeError, match="both be pandas DataFrames"):
            check_tables_manual(table_a, "_id", table_b, "_id")

    def test_check_tables_manual_mixed_spark_pandas(self, sample_table_a):
        """check_tables_manual raises when table_a is Spark and table_b is pandas."""
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(TypeError, match="both be Spark DataFrames"):
            check_tables_manual(sample_table_a, "_id", table_b, "_id")

    def test_check_tables_manual_invalid_type(self):
        """check_tables_manual raises when table_a is not DataFrame."""
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(TypeError, match="table_a must be pandas or Spark DataFrame"):
            check_tables_manual([1, 2, 3], "_id", table_b, "_id")

    # --- Spark: valid and invalid ---

    def test_check_tables_manual_spark_valid(self, sample_table_a, sample_table_b):
        """check_tables_manual passes with valid Spark DataFrames."""
        check_tables_manual(sample_table_a, "_id", sample_table_b, "_id")  # no raise

    def test_check_tables_manual_spark_table_a_missing_id_column(self, sample_table_b):
        """check_tables_manual raises when Spark table_a is missing id column."""
        table_a_no_id = sample_table_b.select("name", "description", "price")
        with pytest.raises(ValueError, match="table_a: missing id column"):
            check_tables_manual(table_a_no_id, "_id", sample_table_b, "_id")

    def test_check_tables_manual_spark_table_b_not_superset(self, sample_table_a, sample_table_b):
        """check_tables_manual passes when Spark table_b has fewer columns (no superset check)."""
        table_b_small = sample_table_b.select("_id", "name")
        table_a_has_more = sample_table_a.select("_id", "name", "description")
        check_tables_manual(table_a_has_more, "_id", table_b_small, "_id")  # no raise

    def test_check_tables_manual_spark_nulls_in_table_a_id(self, spark_session, sample_table_b):
        """check_tables_manual raises when Spark table_a id column has nulls."""
        table_a_with_null = spark_session.createDataFrame(
            [("a", 1), ("b", None)], schema=["name", "_id"]
        )
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_manual(table_a_with_null, "_id", sample_table_b, "_id")

    def test_check_tables_manual_spark_nulls_in_table_b_id(self, spark_session, sample_table_a):
        """check_tables_manual raises when Spark table_b id column has nulls."""
        table_b_with_null = spark_session.createDataFrame(
            [("x", 10), ("y", None)], schema=["name", "_id"]
        )
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_manual(sample_table_a, "_id", table_b_with_null, "_id")


class TestCheckTablesAuto:
    """Tests for check_tables_auto (id columns + table_b columns superset of table_a)."""

    # --- Pandas: valid cases ---

    def test_check_tables_auto_pandas_valid(self):
        """check_tables_auto passes with valid pandas DataFrames and superset columns."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame(
            {"_id": [10, 20], "name": ["x", "y"], "extra": [0, 0]}
        )
        check_tables_auto(table_a, "_id", table_b, "_id")  # no raise

    def test_check_tables_auto_pandas_same_columns(self):
        """check_tables_auto passes when table_b has same columns as table_a."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        check_tables_auto(table_a, "_id", table_b, "_id")  # no raise

    # --- Pandas: missing id column ---

    def test_check_tables_auto_pandas_table_a_missing_id_column(self):
        """check_tables_auto raises when table_a is missing the id column."""
        table_a = pd.DataFrame({"name": ["a", "b"]})  # no _id
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="table_a: missing id column"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    def test_check_tables_auto_pandas_table_b_missing_id_column(self):
        """check_tables_auto raises when table_b is missing the id column."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame({"name": ["x", "y"]})  # no _id
        with pytest.raises(ValueError, match="table_b: missing id column"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    # --- Pandas: empty ---

    def test_check_tables_auto_pandas_table_a_empty(self):
        """check_tables_auto raises when table_a is empty."""
        table_a = pd.DataFrame({"_id": [], "name": []})
        table_b = pd.DataFrame({"_id": [10], "name": ["x"]})
        with pytest.raises(ValueError, match="table_a: empty dataframe"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    def test_check_tables_auto_pandas_table_b_empty(self):
        """check_tables_auto raises when table_b is empty."""
        table_a = pd.DataFrame({"_id": [1], "name": ["a"]})
        table_b = pd.DataFrame({"_id": [], "name": []})
        with pytest.raises(ValueError, match="table_b: empty dataframe"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    # --- Pandas: nulls and types ---

    def test_check_tables_auto_pandas_nulls_in_table_a_id(self):
        """check_tables_auto raises when table_a id column has nulls."""
        table_a = pd.DataFrame({"_id": [1, np.nan], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    def test_check_tables_auto_pandas_nulls_in_table_b_id(self):
        """check_tables_auto raises when table_b id column has nulls."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, np.nan], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    def test_check_tables_auto_pandas_id_column_int64_nullable(self):
        """check_tables_auto passes with pandas nullable Int64 id column (e.g. from parquet)."""
        table_a = pd.DataFrame({"_id": pd.array([1, 2], dtype="Int64"), "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": pd.array([10, 20], dtype="Int64"), "name": ["x", "y"]})
        check_tables_auto(table_a, "_id", table_b, "_id")  # no raise

    def test_check_tables_auto_pandas_id_column_wrong_type(self):
        """check_tables_auto raises when id column is not int32/int64 or Int32/Int64."""
        table_a = pd.DataFrame(
            {"_id": [1.0, 2.0], "name": ["a", "b"]}
        )  # float
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="must be int32/int64 or Int32/Int64"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    def test_check_tables_auto_pandas_id_column_not_unique(self):
        """check_tables_auto raises when id column is not unique."""
        table_a = pd.DataFrame({"_id": [1, 1], "name": ["a", "b"]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(ValueError, match="must be unique"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    # --- Pandas: column superset ---

    def test_check_tables_auto_pandas_table_b_not_superset(self):
        """check_tables_auto raises when table_b columns are not superset of table_a."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"], "foo": [0, 0]})
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})  # no foo
        with pytest.raises(ValueError, match="must be a superset"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    # --- Type mismatches ---

    def test_check_tables_auto_mixed_pandas_spark(self, sample_table_a):
        """check_tables_auto raises when table_a is pandas and table_b is Spark."""
        table_a = pd.DataFrame({"_id": [1, 2], "name": ["a", "b"]})
        table_b = sample_table_a  # Spark, from conftest
        with pytest.raises(TypeError, match="both be pandas DataFrames"):
            check_tables_auto(table_a, "_id", table_b, "_id")

    def test_check_tables_auto_mixed_spark_pandas(self, sample_table_a):
        """check_tables_auto raises when table_a is Spark and table_b is pandas."""
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(TypeError, match="both be Spark DataFrames"):
            check_tables_auto(sample_table_a, "_id", table_b, "_id")

    def test_check_tables_auto_invalid_type(self):
        """check_tables_auto raises when table_a is not DataFrame."""
        table_b = pd.DataFrame({"_id": [10, 20], "name": ["x", "y"]})
        with pytest.raises(TypeError, match="table_a/table_b must be pandas or Spark DataFrames"):
            check_tables_auto([1, 2, 3], "_id", table_b, "_id")

    # --- Spark: valid and invalid ---

    def test_check_tables_auto_spark_valid(self, sample_table_a, sample_table_b):
        """check_tables_auto passes with valid Spark DataFrames and superset columns."""
        check_tables_auto(sample_table_a, "_id", sample_table_b, "_id")  # no raise

    def test_check_tables_auto_spark_table_a_missing_id_column(self, sample_table_b):
        """check_tables_auto raises when Spark table_a is missing id column."""
        table_a_no_id = sample_table_b.select("name", "description", "price")
        with pytest.raises(ValueError, match="table_a: missing id column"):
            check_tables_auto(table_a_no_id, "_id", sample_table_b, "_id")

    def test_check_tables_auto_spark_table_b_not_superset(self, sample_table_a, sample_table_b):
        """check_tables_auto raises when Spark table_b columns are not superset of table_a."""
        table_b_small = sample_table_b.select("_id", "name")
        table_a_has_more = sample_table_a.select("_id", "name", "description")
        with pytest.raises(ValueError, match="must be a superset"):
            check_tables_auto(table_a_has_more, "_id", table_b_small, "_id")

    def test_check_tables_auto_spark_nulls_in_table_a_id(self, spark_session, sample_table_b):
        """check_tables_auto raises when Spark table_a id column has nulls."""
        table_a_with_null = spark_session.createDataFrame(
            [("a", 1), ("b", None)], schema=["name", "_id"]
        )
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_auto(table_a_with_null, "_id", sample_table_b, "_id")

    def test_check_tables_auto_spark_nulls_in_table_b_id(self, spark_session, sample_table_a):
        """check_tables_auto raises when Spark table_b id column has nulls."""
        table_b_with_null = spark_session.createDataFrame(
            [("x", 10), ("y", None)], schema=["name", "_id"]
        )
        with pytest.raises(ValueError, match="nulls are present in the id column"):
            check_tables_auto(sample_table_a, "_id", table_b_with_null, "_id")
