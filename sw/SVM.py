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

# load the initial dataset
def load_data():

    print "Read the dataset"
    print '=' * 50
    cols = ['ItemID', 'Sentiment', 'SentimentSource', 'SentimentText']
    df = pd.read_csv('5000_train.csv', sep=',', header=None, names=cols)
    data_label = list(df.Sentiment)
    data_content = list(df.SentimentText)

    # return the data content and their label
    return data_content, data_label

# pre process the data
def pre_process(data):

    print "Pre-process the data"
    print '=' * 50
    filtered_data = []
    for temp in data:
        # to lowercase
        temp = temp.lower()
        # remove the &quot
        temp = re.sub(r'^&quot;', '', temp, flags=re.MULTILINE)

        words = ""
        temp_list = temp.split()
        # remove http/www/#/@
        for i in temp_list:
            if i.find("http") == -1 and i.find("www") == -1 and i.find("#") == -1 and i.find("@") == -1:
                # handle the multiplication
                word = ''.join(ch for ch, sencond in itertools.groupby(i))
                words = words + " " + word
        filtered_data.append(words)
    return filtered_data

def lemmatizer(data):
    filtered_data = []
    lmtzr = LancasterStemmer()
    for item in data:
        word_list = re.sub("[^\w]", " ", item).split()
        #word_list = item.split()
        words = ""
        for word in word_list:
            words = words + " " + lmtzr.stem(word)
        filtered_data.append(words)
    return filtered_data

# remove the digital number
def removeDigits(s):
    results = ''.join(i for i in s if not i.isdigit())
    return results


def Accuracy(predict_label, test_label):

    positive = 1
    negative = 0

    pos_count = 0
    neg_count = 0

    for i, j in zip(predict_label, test_label):

        if i == "1" and j == "1":
            pos_count += 1

        if i == "0" and j == "0":
            neg_count += 1

    sample_sum = len(predict_label)
    # print sample_sum

    # print zip(predict_label, test_label)

    print "pos sum",pos_count, "neg sum", neg_count

    accuracy = float(pos_count + neg_count)/sample_sum
    print accuracy


def vectorizer(filtered_data):
    # print data_content

    print "Initialise feature_extraction function"
    print '=' * 50
    # Initialise the feature_extraction func
    vectorizer = TfidfVectorizer(min_df=1,max_df = 0.6, stop_words='english', preprocessor=removeDigits, max_features = 4000)
    # vectorizer = CountVectorizer(ngram_range=(1,5) ,min_df=1,max_df = 0.6, stop_words='english',preprocessor=removeDigits, max_features = 4000, lowercase=True)

    # fit x train data
    data_idf_set = vectorizer.fit_transform(filtered_data)
    #pprint (vectorizer.get_feature_names()[:100])
    return data_idf_set

def SVM(a_train, b_train):

    clf = svm.SVC(kernel = 'rbf', gamma=0.1, C=10)
    print "SVM configuration... \n\n", clf
    print('=' * 50)
    clf.fit(a_train, b_train)
    b_predict = clf.predict(a_test)

    return b_predict, clf

def evaluation(b_predict, b_test):

    # print "predict_label", b_predict
    # print "actual label", b_test


    print "Number of %d has been predicted" % len(b_predict), '\n\n'

    print "The results shows below"


    scores = cross_validation.cross_val_score(clf, a_train, b_train, cv=10)
    print('=' * 80)
    print "Cross Validation is \n",scores
    print("Mean Accuracy of Cross Validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    report = classification_report(b_test, b_predict, target_names = ['Negative', 'Positive'])
    print('=' * 80)
    print "Accuracy Report Table: \n\n", report


if __name__ == "__main__":
    data_content, data_label = load_data()

    filtered_data = lemmatizer(data_content)
    filtered_data = pre_process(filtered_data)

    data_idf_set = vectorizer(filtered_data)

    a_train, a_test, b_train, b_test = train_test_split(data_idf_set, data_label, test_size=0.33, random_state=42)

    b_predict, clf = SVM(a_train, b_train)

    evaluation(b_predict, b_test)














"""
Grid Search Section
Exhausted search of predefined parameter
"""

print('=' * 80)
print "Grid Seach For Best Estimator"

parameters = {'C':(0.2,0.5,1,2,3,4,5,10),
			  'gamma':(0.2,0.5,1,2,3,4,5,10)}

C_range = 10. ** np.arange(-2, 9)
gamma_range = 10. ** np.arange(-5, 4)

param_grid = dict(gamma=gamma_range, C=C_range)


gs_clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, n_jobs=-1)

# Fit and train the train data
gs_clf = gs_clf.fit(a_train,b_train)
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

# Print the score for each parameters
for param_name in sorted(parameters.keys()):
	print("%s: %r" % (param_name, best_parameters[param_name]))

print "Score is "
print score


print("The best classifier is: ", gs_clf.best_estimator_)


# plot the scores of the grid
# grid_scores_ contains parameter settings and scores
score_dict = gs_clf.grid_scores_

# We extract just the scores
scores = [x[1] for x in score_dict]
scores = np.array(scores).reshape(len(C_range), len(gamma_range))




'''
plot
'''


# Useing MatLab like graph style
style.use('ggplot')
# # Make a nice figure
# plt.figure(figsize=(8, 6))
# plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.95)
# plt.imshow(scores, interpolation='nearest', cmap=plt.cm.spectral)
# plt.xlabel('gamma')
# plt.ylabel('C')
# plt.colorbar()
# plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
# plt.yticks(np.arange(len(C_range)), C_range)
# plt.show()