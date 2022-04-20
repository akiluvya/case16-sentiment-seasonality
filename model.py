# General imports
import collections
import numpy as np
import argparse
import pickle
import re

# Feature extraction/processing
from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import stopwords

# Models
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from scipy.stats import norm

from model import *


class BaseTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, Y):
        return self

    def transform(self, X):
        return X


# Returns the emoticons in the tweet
class GetEmoticons(BaseTransformer):
    def transform(self, X):
        def transform_sentence(x):
            # Regex that matches emoticons proposed by Jesse Sweetland:
            # stackoverflow.com/questions/28077049/regex-matching-emoticons
            res = re.findall(r"(\:\w+\:|\<[\/\\]?3|[\(\)\\\D|\*\$][\-\^]"
                             "?[\:\;\=]|[\:\;\=B8][\-\^]?[3DOPp\@\$\*\\\)"
                             "\(\/\|])(?=\s|[\!\.\?]|$)", x)
            return res

        return [transform_sentence(x) for x in X]


# Returns the number of capitalized letters in the tweet
class Capitalization(BaseTransformer):
    def transform(self, X):
        def transform_sentence(x):
            diff = float(len([a != b for a, b in zip(x, x.lower())]))
            return [diff / max(1, len(x))]

        return [transform_sentence(x) for x in X]


# Cleans the tweet by removing special characters, transforming to lowercase,
# erasing trailing spaces, removing stop-words and applying stemming.
class GetWords(BaseTransformer):
    def fit(self, X, y):
        self.stemmer = EnglishStemmer(ignore_stopwords=False)
        self.stopwords = set(stopwords.words('english'))
        return self

    def transform(self, X):
        def transform_sentence(x):
            x = re.sub('[^a-zA-Z]+', ' ', x).lower().split(' ')
            x = [tk for tk in x if len(tk) > 0]
            x = [tk for tk in x if tk not in self.stopwords]
            x = [re.sub(r'([a-z])\1+', r'\1', tk) for tk in x]
            x = [self.stemmer.stem(tk) for tk in x]
            return x

        return [transform_sentence(x) for x in X]


# Computes a binary dictionary that specifies if a certain type of punctuation
# appears in the tweet. These symbols are usually used to emphasize something
# (e.g. !, ?, ..., !!!, ???)
class Punctuation(BaseTransformer):
    def fit(self, X, y):
        self.punctuation_of_interest = ['!', '?', '...', '!!!', '???']
        return self

    def transform(self, X):
        def transform_sentence(x):
            return [k in x for k in self.punctuation_of_interest]

        return [transform_sentence(x) for x in X]


class FilterUncommonWords(BaseTransformer):
    def __init__(self, min_occurrences=5, min_abs_polarity=0.1):
        self.min_occurrences = min_occurrences
        self.min_abs_polarity = min_abs_polarity

    def fit(self, X, Y):
        self.dict_ = {}
        polarity_ = {}

        for x, y in zip(X, Y):
            for tk in set(x):
                self.dict_[tk] = self.dict_.get(tk, 0) + 1
                polarity_[tk] = polarity_.get(tk, 0) + (2 * y - 1)
        self.dict_ = {tk: c for tk, c in self.dict_.items()
                      if c > self.min_occurrences and
                      abs(float(polarity_[tk]) /
                          self.dict_[tk]) > self.min_abs_polarity}
        return self

    def transform(self, X):
        return [[tk for tk in x if tk in self.dict_] for x in X]


class PriorEncoding(BaseTransformer):
    def fit(self, X, Y):
        self.dict_ = {}
        self.polarity_ = {}

        for x, y in zip(X, Y):
            for tk in set(x):
                self.dict_[tk] = self.dict_.get(tk, 0) + 1
                self.polarity_[tk] = self.polarity_.get(tk, 0) + (2 * y - 1)

        self.polarity_ = {tk: float(self.polarity_[tk]) / self.dict_[tk]
                          for tk in self.dict_}
        return self

    def transform(self, X):
        def transform_sentence(x):
            if len(x) == 0:
                return [np.nan] * 8
            else:
                polarities = [self.polarity_.get(tk, 0) for tk in x]
                neg_ = [p for p in polarities if p < 0]
                pos_ = [p for p in polarities if p > 0]

                return [np.min(polarities),
                        np.max(polarities),
                        np.mean(polarities),
                        len(neg_),
                        len(pos_),
                        np.mean(neg_) if len(neg_) > 0 else np.nan,
                        np.mean(pos_) if len(pos_) > 0 else np.nan,
                        abs(np.mean(neg_)) > np.mean(pos_)
                        if len(neg_) > 0 and len(pos_) > 0 else np.nan
                        ]

        return [transform_sentence(x) for x in X]


class NaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.labels = list(set(y))

        # Split the observations according to their classes
        observations_by_class = self.get_observations_by_class(X,
                                                               y)
        # Compute the probability of each class P(class)
        n = len(y)
        self.p_class = np.zeros(len(self.labels))
        for k in observations_by_class:
            self.p_class[k] = len(observations_by_class[k]) / float(n)

        # Compute the distribution for each class
        self.mean_ = {l: np.mean(observations_by_class[l], axis=0)
                      for l in self.labels}
        self.std_ = {l: np.std(observations_by_class[l], axis=0)
                     for l in self.labels}
        for l in self.labels:
            self.std_[l][self.std_[l] == 0] = 1

        return self

    def predict(self, X):
        def gaussian_prob(value, label, ft):
            return norm.pdf(value, self.mean_[label][ft],
                            self.std_[label][ft])

        def predict_instance(instance):
            ps = [np.prod([gaussian_prob(v, cl, i)
                           for i, v in enumerate(instance)]) *
                  self.p_class[cl] for cl in self.labels]
            ps = zip(ps, self.labels)
            return max(ps, key=lambda x: x[0])[1]

        return np.asarray([predict_instance(i) for i in X])

    def get_observations_by_class(self, instances, labels):
        return {l: instances[labels == l] for l in self.labels}