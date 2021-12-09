from unittest import TestCase
from pipeline.voting_classifier import Voting

TEST_TEXT = ["hello", "I love you!"]
TOTAL_MODELS = 8


class PipelineTest(TestCase):
    def test_voting_classifier_setup(self):
        classifier = Voting("models/neutral_models")
        self.assertGreater(len(classifier.models), 0)

        classifier = Voting("models/neutral_models", ["logreg", "mlp"])
        print(classifier.models)
        self.assertGreater(len(classifier.models), 0)
        self.assertLess(len(classifier.models), TOTAL_MODELS)

    def test_predict_proba(self):
        classifier = Voting("models/neutral_models", ["linearsvc"])
        predictions = classifier.predict_proba(TEST_TEXT)
        self.assertEqual(predictions.shape[1], 2)
        self.assertEqual(len(predictions), len(TEST_TEXT))

    def test_predict(self):
        classifier = Voting("models/neutral_models", ["linearsvc"])
        predictions = classifier.predict(TEST_TEXT)
        self.assertEqual(len(predictions), len(TEST_TEXT))
