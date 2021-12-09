import pickle

import numpy as np
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.linear_model import LogisticRegression

from notebooks.utilities import data, ml_stats, sampling
from notebooks.utilities import train_test_split as tts
from notebooks.utilities import vectorize


df = data.neutral()


df_sampled = sampling.undersample(df)


def logreg(X_train, y_train, X_test, name, C=1.0, solver="saga", max_iter=100):
    clf = LogisticRegression(
        C=C, random_state=13, solver=solver, max_iter=max_iter, n_jobs=-1
    )
    clf.fit(X_train, y_train.argmax(axis=1))
    pickle.dump(
        clf, open(f"../../models/neutral_models/{name}.pkl", "wb",),
    )
    return clf.predict(X_test), clf


def l1_logreg(X_train, y_train, X_test, name, C=1.0, solver="saga", max_iter=100):
    clf = LogisticRegression(
        C=C, random_state=13, solver=solver, max_iter=max_iter, n_jobs=-1, penalty="l1"
    )
    clf.fit(X_train, y_train.argmax(axis=1))
    pickle.dump(
        clf, open(f"../../models/neutral_models/{name}.pkl", "wb",),
    )
    return clf.predict(X_test), clf


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "logreg", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model = logreg(X_train, y_train, X_test, "logreg_tfidf", C=5.0, solver="saga")


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "l1_logreg", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model = l1_logreg(
    X_train, y_train, X_test, "logreg_tfidf", C=5.0, solver="saga"
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "logreg", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model = logreg(X_train, y_train, X_test, "logreg_count", max_iter=500)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "l1_logreg", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model = l1_logreg(X_train, y_train, X_test, "logreg_count", max_iter=500)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)



