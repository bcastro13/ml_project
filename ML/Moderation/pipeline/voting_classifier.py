import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow import keras
import json
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class _LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t)
            for t in doc.split()
            if len(t) >= 2
            and re.match("[a-z].*", t)
            and re.match(re.compile(r"(?u)\b\w\w+\b"), t)
        ]


class Voting:
    def __init__(self, path, ignore=[], root_path=""):
        self.models = []
        self.path = path
        self.root_path = root_path
        for filename in os.listdir(path):
            if any(model in filename for model in ignore):
                continue
            if os.path.isdir(f"{path}/{filename}"):
                if any(
                    ".pb" in file_dir
                    for file_dir in os.listdir(f"{path}/{filename}")
                ):
                    self.models.append(f"{path}/{filename}")
            elif ".pkl" in filename:
                self.models.append(f"{path}/{filename}")

    def __extract_model_name(self, path):
        return path.split("/")[-1].split(".")[0]

    def __find_model_vectorizer(self, model_name):
        for filename in os.listdir(f"{self.path}/vectorizers"):
            if model_name in filename and "features" in filename:
                return f"{os.getcwd()}/{self.path}/vectorizers/{filename}"

    def __load_models(self, path):
        has_predict_proba = False
        if ".pkl" in path:
            model = pickle.load(open(f"{os.getcwd()}/{path}", "rb"))
            has_predict_proba = True
        else:
            model = keras.models.load_model(f"{os.getcwd()}/{path}")
        return model, has_predict_proba

    def __vectorize(self, vectorizer_path, text):
        vocab = pickle.load(open(vectorizer_path, "rb"))
        if "count" in vectorizer_path:
            vectorizer = CountVectorizer(
                decode_error="replace",
                vocabulary=vocab,
                tokenizer=_LemmaTokenizer(),
            )
        elif "tfidf" in vectorizer_path:
            vectorizer = TfidfVectorizer(
                decode_error="replace",
                vocabulary=vocab,
                tokenizer=_LemmaTokenizer(),
            )
        sparse_matrix = vectorizer.fit_transform(text)
        return sparse_matrix.toarray()

    def __weighted_avg(self, preds):
        with open(f"{self.root_path}data/models_metrics.json", "r") as file:
            metrics = json.load(file)

        unweighted_probs = []
        for key in preds.keys():
            # weight by accuracy
            # unweighted_probs.append(
            #     (metrics[key]["accuracy"] * preds[key]).tolist()
            # )
            unweighted_probs.append(
                [
                    metrics[key]["f1_0"] * preds[key][0],
                    metrics[key]["f1_1"] * preds[key][1],
                ]
            )
        return np.mean(np.array(unweighted_probs), axis=0)

    def ___expand_contractions(self, text):
        def replace(match):
            return contractions[match.group(0)]

        with open(f"{self.root_path}data/contractions.json", "r") as file:
            contractions = json.load(file)

        contractions_re = re.compile("(%s)" % "|".join(contractions.keys()))
        return contractions_re.sub(replace, text)

    def __make_translation(self, text, pattern):
        """
        Remove pattern from string of text

        Parameters
        ----------
        text: str
            text to be cleaned
        pattern: str
            text to remove

        Returns
        -------
        str
        """
        translator = str.maketrans("", "", pattern)
        return text.translate(translator)

    def __clean_text(self, text):
        cleaned_text = []
        for words in text:
            words = words.lower().strip()
            words = self.___expand_contractions(words)
            words = self.__make_translation(words, string.punctuation)
            words = self.__make_translation(words, string.digits)
            stop_words = set(stopwords.words("english"))
            word_tokens = words.split()
            words = " ".join(
                [w for w in word_tokens if not w.lower() in stop_words]
            )
            cleaned_text.append(words)
        return cleaned_text

    def predict_proba(self, text):
        """
        Predict probability of text belongs to each classifier

        Parameters
        ----------
        text: [str]
            list of strings to classify

        Return
        ------
        2D np.array which each column representing probability of class
        """
        preds = []
        text_probs = {}
        text = self.__clean_text(text)
        for model_path in self.models:
            model, has_predict_proba = self.__load_models(model_path)
            model_name = self.__extract_model_name(model_path)
            vectorizer_path = self.__find_model_vectorizer(model_name)
            sparse_matrix = self.__vectorize(vectorizer_path, text)

            if has_predict_proba:
                model_preds = model.predict_proba(sparse_matrix)
            else:
                model_preds = model.predict(sparse_matrix)

            for ind in range(len(text)):
                if text_probs.get(ind):
                    text_probs[ind][model_name] = model_preds[ind]
                else:
                    text_probs[ind] = {model_name: model_preds[ind]}
        for key in text_probs.keys():
            preds.append(self.__weighted_avg(text_probs[key]))
        return np.array(preds)

    def predict(self, text):
        """
        Predict that class that a text belongs to

        Parameters
        ----------
        text: [str]
            list of strings to classify

        Returns
        -------
        np.array of class prediction for each text
        """
        return np.argmax(self.predict_proba(text), axis=1)


if __name__ == "__main__":
    from rich import print

    TEXT = [
        "I love you",
        "Eugene Callahan",
        "Astros met with the agents for free agent CF Starling Marte. "
        + "Seems like a potential fit but there’s competition, including "
        + "Phillies, Mets, Yanks, Marlins and others.",
        "Is that 12 or just a nigga in an impala?",
        "Fire Mullen. Take everyone’s scholarship. Burn down Gainesville "
        + "and shoot me in the face. Im turning off this game. Somebody "
        + "please tell me if we win. Slip roofies in my drink if we lose",
        "Fuck your life, bing bong",
        "President Biden refers to baseball legend Satchel Paige as "
        + "'the great negro at the time.'",
        "EVERYONE APOLOGIZE TO ME AND THANK ME BECAUSE THE GAME OF THE "
        + "YEAR WAS NEVER A FUCKING DOUBT",
    ]

    classifier = Voting(
        "ML/Moderation/models/neutral_models", ["linearsvc"], "ML/Moderation/"
    )
    predictions = classifier.predict(TEXT)
    probas = classifier.predict_proba(TEXT)
    print(TEXT)
    print(predictions)
    print(probas)
