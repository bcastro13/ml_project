import unittest
from ..utilities.sampling import undersample
import pandas as pd


class SamplingTest(unittest.TestCase):
    def test_undersample(self):
        self.assertLessEqual(
            len(undersample(pd.DataFrame([0, 0, 1], columns=["class"]))), 3
        )
