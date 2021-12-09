import unittest
from ..utilities import train_test_split
from scipy.sparse import csr_matrix


class TrainTestSplitTest(unittest.TestCase):
    def test_encode_outputs(self):
        y = train_test_split.encode_outputs([1, 0, 1, 1, 0])
        self.assertEqual(len(y[0]), 2)

    def test_split(self):
        X_train, X_test, y_train, y_test = train_test_split.split(
            csr_matrix((5, 1)), [1, 2, 3, 4, 5]
        )
        self.assertEqual(len(X_train), 4)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(X_test), 1)
        self.assertEqual(len(y_test), 1)
