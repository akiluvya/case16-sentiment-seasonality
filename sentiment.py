# General imports
import collections
import numpy as np
import argparse
import pickle
import csv
import os
import re


def get_args():
    parser = argparse.ArgumentParser(
                        description="Validate and train a sentime analysis "
                                    "predictive algorithm.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', metavar="I", nargs='?',
                        default='Sentiment Analysis Dataset.csv',
                        help='Path to the input file')
    parser.add_argument('--model', metavar="M", nargs='?',
                        default='model.pickle',
                        help='Path to the model file')
    parser.add_argument('--output', metavar="O", nargs='?',
                        default="predictions.csv", help='Output filename')
    return parser.parse_args()

args = get_args()
f = open(args.model, 'rb')
clf = pickle.loads(f.read())
f.close()

#f = open(args.input)
#tweets = f.readlines()
#tweets = [x.strip() for x in tweets]
#f.close()

#f = open(args.output, 'w')
def get_sentiment(text):
    w = clf.predict([text])
    if w[0] == 1 : return 1 # Positive
    return 0 #Negative

#f.close()

#print(get_sentiment("This taste eeew"))