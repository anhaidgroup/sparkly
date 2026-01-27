"""Unit tests for sparkly.analysis module."""


class TestAnalyze:
    """Tests for analyze function."""

    def test_analyze_basic(self):
        """Test basic analyze functionality."""
        from sparkly.analysis import analyze
        from org.apache.lucene.analysis.standard import StandardAnalyzer
        analyzer = StandardAnalyzer()
        text = "Hello, world!"
        terms = analyze(analyzer, text)
        assert terms == ["hello", "world"]

    def test_analyze_with_offset(self):
        """Test analyze with offset information."""
        from sparkly.analysis import analyze
        from org.apache.lucene.analysis.standard import StandardAnalyzer
        analyzer = StandardAnalyzer()
        text = "Hello, world!"
        terms = analyze(analyzer, text, with_offset=True)
        assert terms == [("hello", 0, 5), ("world", 7, 12)]


class TestGetShingleAnalyzer:
    """Tests for get_shingle_analyzer function."""

    def test_get_shingle_analyzer(self):
        """Test get_shingle_analyzer returns correct analyzer."""
        from sparkly.analysis import get_shingle_analyzer
        from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
        analyzer = get_shingle_analyzer()
        assert analyzer is not None
        assert isinstance(analyzer, ShingleAnalyzerWrapper)


class TestGetStandardAnalyzerNoStopWords:
    """Tests for get_standard_analyzer_no_stop_words function."""

    def test_get_standard_analyzer_no_stop_words(self):
        """Test get_standard_analyzer_no_stop_words returns analyzer."""
        from sparkly.analysis import get_standard_analyzer_no_stop_words
        from org.apache.lucene.analysis.standard import StandardAnalyzer
        analyzer = get_standard_analyzer_no_stop_words()
        assert analyzer is not None
        assert isinstance(analyzer, StandardAnalyzer)


class TestPythonAlnumTokenFilter:
    """Tests for PythonAlnumTokenFilter class."""

    def test_python_alnum_token_filter_accept(self):
        """Test PythonAlnumTokenFilter accept method."""
        from sparkly.analysis import PythonAlnumTokenFilter
        from org.apache.lucene.analysis.standard import StandardAnalyzer
        analyzer = StandardAnalyzer()
        stream = analyzer.tokenStream("contents", "hello world")
        alnum_token_filter = PythonAlnumTokenFilter(stream)
        try:
            alnum_token_filter.reset()
            while alnum_token_filter.incrementToken():
                term = alnum_token_filter.termAtt.toString()
                assert term.isalnum()
            alnum_token_filter.end()
        finally:
            alnum_token_filter.close()


class TestStrippedGram3Analyzer:
    """Tests for StrippedGram3Analyzer class."""

    def test_stripped_gram3_analyzer_create_components(self):
        """Test StrippedGram3Analyzer createComponents method."""
        from sparkly.analysis import StrippedGram3Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = StrippedGram3Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)

    def test_stripped_gram3_analyzer_init_reader(self):
        """Test StrippedGram3Analyzer initReader method."""
        from sparkly.analysis import StrippedGram3Analyzer
        from org.apache.lucene.analysis.pattern import PatternReplaceCharFilter
        from java.io import StringReader
        analyzer = StrippedGram3Analyzer()
        reader = analyzer.initReader("contents", StringReader("hello world"))
        assert reader is not None
        assert isinstance(reader, PatternReplaceCharFilter)


class TestGram3Analyzer:
    """Tests for Gram3Analyzer class."""

    def test_gram3_analyzer_create_components(self):
        """Test Gram3Analyzer createComponents method."""
        from sparkly.analysis import Gram3Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = Gram3Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)


class TestGram2Analyzer:
    """Tests for Gram2Analyzer class."""

    def test_gram2_analyzer_create_components(self):
        """Test Gram2Analyzer createComponents method."""
        from sparkly.analysis import Gram2Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = Gram2Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)


class TestGram4Analyzer:
    """Tests for Gram4Analyzer class."""

    def test_gram4_analyzer_create_components(self):
        """Test Gram4Analyzer createComponents method."""
        from sparkly.analysis import Gram4Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = Gram4Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)


class TestUnfilteredGram3Analyzer:
    """Tests for UnfilteredGram3Analyzer class."""

    def test_unfiltered_gram3_analyzer_create_components(self):
        """Test UnfilteredGram3Analyzer createComponents method."""
        from sparkly.analysis import UnfilteredGram3Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = UnfilteredGram3Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)


class TestUnfilteredGram5Analyzer:
    """Tests for UnfilteredGram5Analyzer class."""

    def test_unfiltered_gram5_analyzer_create_components(self):
        """Test UnfilteredGram5Analyzer createComponents method."""
        from sparkly.analysis import UnfilteredGram5Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = UnfilteredGram5Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)


class TestStandardEdgeGram36Analyzer:
    """Tests for StandardEdgeGram36Analyzer class."""

    def test_standard_edge_gram36_analyzer_create_components(self):
        """Test StandardEdgeGram36Analyzer createComponents method."""
        from sparkly.analysis import StandardEdgeGram36Analyzer
        from org.apache.lucene.analysis import Analyzer
        analyzer = StandardEdgeGram36Analyzer()
        components = analyzer.createComponents("contents")
        assert components is not None
        assert isinstance(components, Analyzer.TokenStreamComponents)
