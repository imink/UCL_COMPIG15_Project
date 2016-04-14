# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm, cross_validation
from pprint import pprint
from datetime import *
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
from sklearn.learning_curve import validation_curve
from sklearn.grid_search import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import style
from nltk.stem.lancaster import LancasterStemmer

import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import re


class FeatureExtraction(object):
    """docstring for FeatureExtrac"""
    def __init__(self):
        super(FeatureExtraction, self).__init__()


    # remove the digital number
    # def removeDigits(self, string):
    #     results = ''.join(i for i in string if not i.isdigit())
    #     return results


    def tdidf_vectorizer(self, processed_data):
        # print data_content

        print "Initialise TFIDF feature_extraction function"
        print '=' * 50
        # Initialise the feature_extraction func
        vectorizer = TfidfVectorizer(min_df=1,max_df = 0.6, stop_words='english', max_features = 4000)
        # vectorizer = CountVectorizer(ngram_range=(1,5) ,min_df=1,max_df = 0.6, stop_words='english',preprocessor=removeDigits, max_features = 4000, lowercase=True)

        # fit x train data
        vectorized_data = vectorizer.fit_transform(processed_data)
        #pprint (vectorizer.get_feature_names()[:100])
        return vectorized_data




    def tf_vectorizer(self, processed_data):
        # print data_content
        print "Initialise TF feature_extraction function"
        print '=' * 50
        # Initialise the feature_extraction func
        vectorizer = CountVectorizer(ngram_range=(1,5) ,min_df=1,max_df = 0.6, stop_words='english', max_features = 4000)

        # fit x train data
        vectorized_data = vectorizer.fit_transform(processed_data)
        #pprint (vectorizer.get_feature_names()[:100])
        return vectorized_data

    # Freature Presence feature extraction
    def fp_vectorizer(self, processed_data):
        binarizer = Binarizer(threshold = 5)
        vectorized_data = binarizer.fit_transform(processed_data)
        return vectorized_data

















