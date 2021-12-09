import pickle
import re
import time

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
)

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t)
            for t in doc.split()
            if len(t) >= 2
            and re.match("[a-z].*", t)
            and re.match(TOKEN_PATTERN, t)
        ]


def save(data, name):
    pickle.dump(
        data, open(f"../../models/neutral_models/vectorizers/{name}.pkl", "wb")
    )


def tfidf(data, name, max_df=0.80, min_df=600, vocab=False):
    """
    Text to matrix using term frequency inverse document frequency

    Parameters
    -----------
    data: array
        an array with each index being text
    name: str
        name in which to save files
    max_df: float [0, 1]
        ceiling of percentage of indices a word appears in
    min_df: int
        minimum times a word must appear
    vocab: Bool
        return vocabulary dictionary

    Returns
    --------
    tfidf: sparse matrix
    (optional) vocab: dict
    """
    tfidf_vectorizer = TfidfVectorizer(
        input="content",
        analyzer="word",
        tokenizer=LemmaTokenizer(),
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1, 2),
    )
    start = time.time()
    tfidf = tfidf_vectorizer.fit_transform(data)
    print(f"Time to vectorize: {(time.time() - start): .2f}")
    print(f"Length of vocabulary: {len(tfidf_vectorizer.get_feature_names())}")
    save(tfidf_vectorizer.vocabulary_, f"{name}_tfidf_features")
    save(tfidf, f"{name}_tfidf_matrix")
    if vocab:
        return tfidf, tfidf_vectorizer.vocabulary_
    return tfidf


def count(data, name, max_df=0.80, min_df=700, vocab=False):
    """
    Text to matrix using count of terms

    Parameters
    -----------
    data: array
        an array with each index being text
    name: str
        name in which to save files
    max_df: float [0, 1]
        ceiling of percentage of indices a word appears in
    min_df: int
        minimum times a word must appear
    vocab: Bool
        return vocabulary dictionary

    Returns
    --------
    tfidf: sparse matrix
    (optional) vocab: dict
    """
    count_vectorizer = CountVectorizer(
        input="content",
        analyzer="word",
        tokenizer=LemmaTokenizer(),
        max_df=max_df,
        min_df=min_df,
        ngram_range=(1, 2),
    )
    start = time.time()
    count = count_vectorizer.fit_transform(data)
    print(f"Time to vectorize: {(time.time() - start): .2f}")
    print(f"Length of vocabulary: {len(count_vectorizer.get_feature_names())}")
    save(count_vectorizer.vocabulary_, f"{name}_count_features")
    save(count, f"{name}_count_matrix")
    if vocab:
        return count, count_vectorizer.vocabulary_
    return count
