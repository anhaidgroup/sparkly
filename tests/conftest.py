"""Pytest configuration and shared fixtures."""
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"builtin type .* has no __module__ attribute",
    category=DeprecationWarning,
)

# Import after warnings filter to suppress PyLucene deprecation warnings
import pytest  # noqa: E402
from pathlib import Path  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from sparkly.utils import init_jvm  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def init_pylucene_jvm():
    """
    Initialize PyLucene JVM before any tests run.
    Session-scoped and autouse=True ensures it runs once for all tests.
    """
    init_jvm()


@pytest.fixture(scope="session")
def spark_session():
    """
    Create a SparkSession for testing.
    Session-scoped to reuse across tests for efficiency.
    """
    spark = SparkSession.builder \
        .appName("sparkly-tests") \
        .getOrCreate()

    yield spark
    spark.stop()


@pytest.fixture(scope="session")
def abt_buy_data_path():
    """
    Path to the abt_buy dataset directory.
    Session-scoped since the data doesn't change.
    Used by sample fixtures to read from the actual dataset.
    """
    # Get the path relative to the tests directory
    tests_dir = Path(__file__).parent
    repo_root = tests_dir.parent
    data_path = repo_root / "examples" / "data" / "abt_buy"
    return data_path


@pytest.fixture
def sample_table_a(spark_session, abt_buy_data_path):
    """
    Small sample Spark DataFrame for unit tests (5 rows with actual matches).
    Selects rows from table_a that have matches in the gold standard.
    Faster than using the full abt_buy dataset, but uses real data structure.

    Schema: _id, name, description, price
    """
    import pandas as pd
    table_a_path = abt_buy_data_path / "table_a.parquet"
    gold_path = abt_buy_data_path / "gold.parquet"

    # Get first 5 gold pairs to find matching IDs
    gold = pd.read_parquet(gold_path).head(5)
    a_ids = gold['id1'].tolist()

    # Filter table_a to get rows that match
    table_a = pd.read_parquet(table_a_path)
    sample_a = table_a[table_a['_id'].isin(a_ids)]

    return spark_session.createDataFrame(sample_a)


@pytest.fixture
def sample_table_b(spark_session, abt_buy_data_path):
    """
    Small sample Spark DataFrame for unit tests (5 rows with actual matches).
    Selects rows from table_b that have matches in the gold standard.
    Faster than using the full abt_buy dataset, but uses real data structure.

    Schema: _id, name, description, price
    """
    import pandas as pd
    table_b_path = abt_buy_data_path / "table_b.parquet"
    gold_path = abt_buy_data_path / "gold.parquet"

    # Get first 5 gold pairs to find matching IDs
    gold = pd.read_parquet(gold_path).head(5)
    b_ids = gold['id2'].tolist()

    # Filter table_b to get rows that match
    table_b = pd.read_parquet(table_b_path)
    sample_b = table_b[table_b['_id'].isin(b_ids)]

    return spark_session.createDataFrame(sample_b)


@pytest.fixture
def sample_gold(spark_session, abt_buy_data_path):
    """
    Small sample gold standard matches for unit tests (first 5 matching pairs).
    These pairs correspond to the rows selected in sample_table_a and
    sample_table_b.

    Schema: id1, id2 (matching pairs)
    """
    import pandas as pd
    gold_path = abt_buy_data_path / "gold.parquet"
    # Read first 5 matching pairs from actual parquet file
    df_pandas = pd.read_parquet(gold_path).head(5)
    return spark_session.createDataFrame(df_pandas)


@pytest.fixture
def medium_table_a(spark_session, abt_buy_data_path):
    """
    Medium-sized Spark DataFrame (~26k rows) to test local parallel build path.
    Threshold: > 25000 rows triggers parallel local build.
    Uses a smaller size to avoid memory issues.
    Schema: _id, name, description, price
    """
    import logging
    import pyspark.sql.functions as F
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Creating medium_table_a fixture - generating 26000 rows")
        
        df = spark_session.range(26000).withColumn(
            'name', F.concat(F.lit('name'), F.col('id').cast('string'))
        ).withColumn(
            'description',
            F.concat(F.lit('description'), F.col('id').cast('string'))
        ).withColumn('_id', F.col('id') + 1).drop('id')
        
        logger.info("medium_table_a fixture created successfully")
        return df
    except Exception as e:
        logger.error(f"Failed to create medium_table_a: {e}", exc_info=True)
        raise
