from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import PorterStemmer

class PorterTokenizer(object):
    """Custom PorterTokenizer for TfidfVectorizer"""

    def __init__(self):
        self.stemmer = PorterStemmer()

    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc)]

class Vectorizer(object):
    """Vecotizer wrapper for sklearn TfidfVectorizer.

    Allows passing of custom tokenizer

    TODO: add more custom tokenizers"""

    def __init__(self,
                 tokenizer=None,
                 encoding='utf-8',
                 stop_words='english',
                 ngram_range=None):
        self.tokenizers = {'porter': PorterTokenizer()}
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizers[tokenizer],
                                          encoding=encoding,
                                          stop_words=stop_words,
                                          ngram_range=ngram_range)

    def fit(self, X):
        self.vectorizer.fit(X)
        return self

    def fit_transform(self, X):
        return self.vectorizer.fit_transform(X)
