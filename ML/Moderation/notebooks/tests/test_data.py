import unittest
from ..utilities.data import complete, neutral, hate

PATH = "data/processed/processed_stopwords.csv"


class DataTest(unittest.TestCase):
    def test_complete(self):
        self.assertEqual(
            sorted(list(complete(PATH)["class"].unique())), [0, 1, 2]
        )

    def test_neutral(self):
        self.assertEqual(sorted(list(neutral(PATH)["class"].unique())), [0, 1])

    def test_hate(self):
        self.assertEqual(sorted(list(hate(PATH)["class"].unique())), [0, 1])
