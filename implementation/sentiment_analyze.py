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


from data_process import DataProcess
from feature_extraction import FeatureExtraction
from evaluation.svm import EvalSVM

# Init the object
data_process = DataProcess()
feature_extraction = FeatureExtraction()


data_content, data_lable = data_process.load_data('dataset/5000_new_train.csv')


processed_data = data_process.pre_process(data_content)
processed_data = data_process.lemmatizer(processed_data)

pprint (processed_data)

vectorized_data = feature_extraction.tfidf_vectorizer(processed_data)

# vectorizer = TfidfVectorizer(min_df=1,max_df = 0.6, stop_words='english', preprocessor=removeDigits, max_features = 4000)
# vectorized_data = vectorizer.fit_transform(processed_data)

a_train, a_test, b_train, b_test = train_test_split(vectorized_data, data_lable, test_size=0.33, random_state=42)

# init svm
svm = EvalSVM(0.1, 10) 
clf = svm.init_classifier()
clf = svm.fit_train_data(clf, a_train, b_train)

svm.eval_output(clf, a_train, b_train, a_test, b_test)
svm.accuracy(b_test)

# svm.parameter_turning(a_train, b_train)

