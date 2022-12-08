# needed for correct import paths to be found
import lucene
from org.apache.lucene.analysis.tokenattributes import CharTermAttribute, OffsetAttribute
from org.apache.lucene.analysis import Analyzer
from org.apache.pylucene.analysis import PythonAnalyzer,  PythonFilteringTokenFilter
from org.apache.lucene.analysis.standard import StandardAnalyzer, StandardTokenizer
from org.apache.lucene.analysis.shingle import ShingleAnalyzerWrapper
from org.apache.lucene.analysis.ngram import  NGramTokenizer, EdgeNGramTokenFilter
from org.apache.lucene.analysis.core import LowerCaseFilter
from org.apache.lucene.analysis import CharArraySet
from org.apache.lucene.analysis.pattern import PatternReplaceCharFilter
from java.util.regex import Pattern

def _fetch_terms_with_offsets(obj):
    termAtt = obj.getAttribute(CharTermAttribute.class_)
    offsetAtt = obj.getAttribute(OffsetAttribute.class_)
    try:
        obj.clearAttributes()
        obj.reset()
        while obj.incrementToken():
            yield (termAtt.toString(), offsetAtt.startOffset(), offsetAtt.endOffset())
    finally:
        obj.end()
        obj.close()

def _fetch_terms(obj):
    termAtt = obj.getAttribute(CharTermAttribute.class_)
    try:
        obj.clearAttributes()
        obj.reset()
        while obj.incrementToken():
            yield termAtt.toString() 
    finally:
        obj.end()
        obj.close()


def analyze_generator(analyzer, text, with_offset=False):
    """
    Apply the analyzer to the text and return the tokens, optionally with offsets
    
    Parameters
    ----------
    analyzer : 
        The lucene analyzer to be applied
    text : str
        the text that will be analyzer
    with_offset : bool
        if true, return the offsets with the tokens in the form 
        (TOKEN, START_OFFSET, END_OFFSET)

    Returns
    -------
    generator of str or tuples
        a list of tokens potentially with offsets
    """

    stream = analyzer.tokenStream("contents", text)

    if with_offset:
        terms = _fetch_terms_with_offsets(stream)
    else:
        terms = _fetch_terms(stream)

    return terms

def analyze(analyzer, text, with_offset=False):
    """
    Apply the analyzer to the text and return the tokens, optionally with offsets
    
    Parameters
    ----------
    analyzer : 
        The lucene analyzer to be applied
    text : str
        the text that will be analyzer
    with_offset : bool
        if true, return the offsets with the tokens in the form 
        (TOKEN, START_OFFSET, END_OFFSET)

    Returns
    -------
    list of str or tuples
        a list of tokens potentially with offsets
    """

    return list(analyze_generator(analyzer, text, with_offset))


def get_shingle_analyzer():
    return ShingleAnalyzerWrapper(2, 3)

def get_standard_analyzer_no_stop_words():
    return StandardAnalyzer(CharArraySet.EMPTY_SET)

class PythonAlnumTokenFilter(PythonFilteringTokenFilter):
    def __init__(self, tokenStream):
        super().__init__(tokenStream)
        self.termAtt = self.addAttribute(CharTermAttribute.class_)

    def accept(self):
        return self.termAtt.toString().isalnum()



class StrippedGram3Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)


    def createComponents(self, fieldName):
        
        src = NGramTokenizer(3,3)
        res = LowerCaseFilter(src)
        # these chars already stripped out
        #res = PythonAlnumTokenFilter(res)

        return Analyzer.TokenStreamComponents(src, res)

    def initReader(self, fieldName, reader):
        pat = Pattern.compile("[^A-Za-z0-9]")
        return PatternReplaceCharFilter(pat, '', reader)

class Gram3Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)


    def createComponents(self, fieldName):
        
        src = NGramTokenizer(3,3)
        res = PythonAlnumTokenFilter(src)
        res = LowerCaseFilter(res)

        return Analyzer.TokenStreamComponents(src, res)

class Gram2Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)


    def createComponents(self, fieldName):
        
        src = NGramTokenizer(2,2)
        res = PythonAlnumTokenFilter(src)
        res = LowerCaseFilter(res)

        return Analyzer.TokenStreamComponents(src, res)

class Gram4Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)


    def createComponents(self, fieldName):
        
        src = NGramTokenizer(4,4)
        res = PythonAlnumTokenFilter(src)
        res = LowerCaseFilter(res)

        return Analyzer.TokenStreamComponents(src, res)

class UnfilteredGram3Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)


    def createComponents(self, fieldName):
        
        src = NGramTokenizer(3,3)
        res = LowerCaseFilter(src)

        return Analyzer.TokenStreamComponents(src, res)

class UnfilteredGram5Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)


    def createComponents(self, fieldName):
        
        src = NGramTokenizer(5,5)
        res = LowerCaseFilter(src)

        return Analyzer.TokenStreamComponents(src, res)

class StandardEdgeGram36Analyzer(PythonAnalyzer):

    def __init__(self):
        PythonAnalyzer.__init__(self)

    def createComponents(self, fieldName):
        src = StandardTokenizer()
        res = LowerCaseFilter(src)
        res = PythonAlnumTokenFilter(res)
        res = EdgeNGramTokenFilter(res, 3, 6, False)

        return Analyzer.TokenStreamComponents(src, res)
