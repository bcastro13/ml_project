from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split


def encode_outputs(y):
    """
    'One-hot encode' numerical labels

    Parameters
    -----------
    y: array
        an array of numerical labels

    Returns
    --------
    array
        a (len(y), number of categories) shape array
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    encoded_y = encoder.transform(y)
    return np_utils.to_categorical(encoded_y)


def split(vector, y, test_size=0.2, random_state=13):
    """
    Train test split of vector.

    Parameters
    -----------
    vector: sparse matrix:
        sparse matrix from either tfidf or count vectorizer
    y: array
        an array of labels
    train_size: float
        percentage of data to be used in training
    random_state: int
        seed used to maintain same train-test-split

    Returns
    --------
    X_train: array
    X_test: array
    y_train: array
    y_test: array
    """
    return train_test_split(
        vector.toarray(), y, test_size=test_size, random_state=random_state
    )
