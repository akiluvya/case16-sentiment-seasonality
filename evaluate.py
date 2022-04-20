# General imports
import numpy as np
import argparse
import pickle
import csv
import os
import re

# Feature extraction/processing
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline, FeatureUnion
#from sklearn.preprocessing import Imputer
#from sklearn.preprocessing import SimpleImputer
from sklearn.impute import SimpleImputer

# Assessment
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from model import *


def get_args():
    parser = argparse.ArgumentParser(
                        description="Validate and train a sentime analysis "
                                    "predictive algorithm.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', metavar="I", nargs='?',
                        default='Sentiment Analysis Dataset.csv',
                        help='Path to the input file')
    parser.add_argument('--output', metavar="O", nargs='?',
                        default="model.pickle", help='Output filename')
    return parser.parse_args()


args = get_args()

f = open(args.input)
X = list(csv.reader(f))[1:]
f.close()

Y = np.asarray([int(x[1]) for x in X])
X = [x[3] for x in X]

p1 = Pipeline([('preprocess', GetWords()),
               ('dict', FilterUncommonWords(min_abs_polarity=0.1)),
               ('priors', PriorEncoding()),
               ])
p2 = Pipeline([('preprocess', GetEmoticons()),
               ('dict', FilterUncommonWords(min_abs_polarity=0.3)),
               ('priors', PriorEncoding()),
               ])
pp = FeatureUnion([('text', p1),
                   ('emoticons', p2),
                   ('punctuation', Punctuation()),
                   ('capitalization', Capitalization()),
                   ])

clf = Pipeline([('features', pp),
                #('imputer', Imputer()),
                ('imputer', SimpleImputer()),
                ('variance', VarianceThreshold()),
                ('model', NaiveBayes()),
                ])

Xtr, Xts, Ytr, Yts = train_test_split(X, Y, test_size=0.2)
clf.fit(Xtr, Ytr)
print('Accuracy: %f' % accuracy_score(Yts, clf.predict(Xts)))

f = open(args.output, 'wb')
pickle.dump(clf, f)
f.close()