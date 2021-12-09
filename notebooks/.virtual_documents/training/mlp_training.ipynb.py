import pickle

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from ml_project.notebooks.utilities import data, ml_stats, sampling
from ML.Moderation.notebooks.utilities import train_test_split as tts
from ML.Moderation.notebooks.utilities import vectorize
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)


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


tfidf_vec, vocab = vectorize.tfidf(
    df_sampled["text"], "mlp", max_df=0.9, min_df=50, vocab=True
)


y = tts.encode_outputs(df_sampled["class"])


X_train, X_test, y_train, y_test = tts.split(tfidf_vec, y)


y_pred, model, history = mlp(X_train, y_train, X_test, y_test, "mlp_tfidf", epochs=2)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


vocab_tfidf = pickle.load(
    open("../../models/neutral_models/vectorizers/mlp_tfidf_features.pkl", "rb")
)
transformer = TfidfTransformer()
loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=vocab_tfidf)
test_tfidf = transformer.fit_transform(
    loaded_vec.fit_transform(
        np.array(
            [
                "what are you a homo",
                "I love you",
                "I go to the mall",
                "fuck your mom bitch",
                "hello",
                "I have food",
                "you are gay as fuck",
                "niggers",
                "blessjesus",
            ]
        )
    )
)
preds = model.predict(test_tfidf.toarray())
print(preds.argmax(axis=1))
print(preds)


count_vec, vocab = vectorize.count(
    df_sampled["text"], "mlp", max_df=0.9, min_df=50, vocab=True
)


X_train, X_test, y_train, y_test = tts.split(count_vec, y)


y_pred, model, history = mlp(X_train, y_train, X_test, y_test, "mlp_count", epochs=3)


ml_stats.class_accuracies(y_test, y_pred)


ml_stats.confusion_matrix_plot(y_test, y_pred, 2)


ml_stats.stats(y_test, y_pred, 2)


vocab_tfidf = pickle.load(
    open("../../models/neutral_models/vectorizers/mlp_count_features.pkl", "rb")
)
loaded_vec = CountVectorizer(decode_error="replace", vocabulary=vocab_tfidf)
test_count = loaded_vec.fit_transform(
    np.array(
        [
            "what are you a homo",
            "I love you",
            "I go to the mall",
            "fuck your mom bitch",
            "hello",
            "I have food",
            "you are gay as fuck",
            "niggers",
            "blessjesus",
        ]
    )
)
from tensorflow import keras
model = keras.models.load_model("../../models/neutral_models/mlp_tfidf")
preds = model.predict(test_count.toarray())
print(preds.argmax(axis=1))
print(preds)



