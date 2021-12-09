import pickle

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.svm import LinearSVC

from notebooks.utilities import data, ml_stats, sampling
from notebooks.utilities import train_test_split as tts
from notebooks.utilities import vectorize


df = data.neutral()


df_sampled = sampling.undersample(df)


def linsvc(
    X_train,
    y_train,
    X_test,
    name,
    penalty="l2",
    loss="squared_hinge",
    C=1.0,
    max_iter=1000,
):
    svm = LinearSVC(penalty=penalty, loss=loss, C=C, max_iter=max_iter, dual=False)
    clf = CalibratedClassifierCV(svm)
    clf.fit(X_train, y_train.argmax(axis=1))
    pickle.dump(
        clf, open(f"../../models/neutral_models/vectorizers/{name}.pkl", "wb",),
    )
    return clf.predict(X_test), clf


def l1_linsvc(
    X_train,
    y_train,
    X_test,
    name,
    penalty="l2",
    loss="squared_hinge",
    C=1.0,
    max_iter=1000,
):
    svm = LinearSVC(penalty=penalty, loss=loss, C=C, max_iter=max_iter, dual=False)
    clf = CalibratedClassifierCV(svm)
    clf.fit(X_train, y_train.argmax(axis=1))
    pickle.dump(
        clf, open(f"../../models/neutral_models/vectorizers/{name}.pkl", "wb",),
    )
    return clf.predict(X_test), clf


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "linearsvc", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model = linsvc(
    X_train, y_train, X_test, "linearsvc_tfidf", penalty="l1", C=10.0
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "l1_linearsvc", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model = linsvc(
    X_train, y_train, X_test, "l1_linearsvc_tfidf", penalty="l1", C=10.0
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "linearsvc", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model = linsvc(
    X_train, y_train, X_test, "linearsvc_count", penalty="l2", C=100.0
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "linearsvc", max_df=0.90, min_df=100, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model = linsvc(
    X_train, y_train, X_test, "linearsvc_count", penalty="l2", C=100.0
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)



