import pickle

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)

from notebooks.utilities import data, ml_stats, sampling
from notebooks.utilities import train_test_split as tts
from notebooks.utilities import vectorize


df = data.neutral()


df_sampled = sampling.undersample(df)


def mlp(
    X_train, y_train, X_test, y_test, name, class_weight=None, epochs=15, batch_size=64,
):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        class_weight=class_weight,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    print(model.evaluate(X_train, y_train))
    model.save(f"../../models/neutral_models/{name}")
    return np.argmax(model.predict(X_test), axis=1), model, history


def sig_mlp(
    X_train, y_train, X_test, y_test, name, class_weight=None, epochs=15, batch_size=64,
):
    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation="relu"))
    model.add(Dense(64, activation="sigmoid"))
    model.add(Dense(8, activation="sigmoid"))
    model.add(Dense(y_train.shape[1], activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    history = model.fit(
        X_train,
        y_train,
        class_weight=class_weight,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
    )
    print(model.evaluate(X_train, y_train))
    model.save(f"../../models/neutral_models/{name}")
    return np.argmax(model.predict(X_test), axis=1), model, history


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "mlp", max_df=0.9, min_df=50, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model, history = mlp(X_train, y_train, X_test, y_test, "mlp_tfidf", epochs=10)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "sig_mlp", max_df=0.9, min_df=50, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model, history = sig_mlp(
    X_train, y_train, X_test, y_test, "sig_mlp_tfidf", epochs=10
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "mlp", max_df=0.9, min_df=50, vocab=True
)


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model, history = mlp(X_train, y_train, X_test, y_test, "mlp_count", epochs=10)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "sig_mlp", max_df=0.9, min_df=50, vocab=True
)


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model, history = mlp(
    X_train, y_train, X_test, y_test, "sig_mlp_count", epochs=10
)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)



