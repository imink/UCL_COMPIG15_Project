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
import os
import re
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


# Useing MatLab like graph style
style.use('ggplot')

#pre process the data

def removeDigits(s):
	results = ''.join(i for i in s if not i.isdigit())
	return results

def pre_process():
	for temp in data_content:
		temp=temp.lower()
		temp = re.sub(r'^&quot;', '', temp, flags=re.MULTILINE)
		cache=""
		temp_list = temp.split()
		for i in temp_list:
			if i.find("http")==-1 and i.find("www.")==-1 and i.find("#")==-1 and i.find("@")==-1:
				cache+=i+" "
		temp = cache
		cache=""
		print temp



def load_data():
	cols = ['ItemID','Sentiment','SentimentSource','SentimentText']
	# df = pd.read_csv('5000_data.csv', skipinitialspace=True, usecols=fileds)
	df = pd.read_csv('10000_data.csv', sep=',', header=None, names=cols);
	# print df.head()


	# df_iterators = df.iterrows()
	# for index, row in df_iterators:
	# 	print row.ItemID
	data_label = list(df.Sentiment)
	data_content = list(df.SentimentText)

	return data_content, data_label





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





data_content, data_label = load_data()

pre_process()

# print data_content

print "Initialise feature_extraction function"
# Initialise the feature_extraction func
# vectorizer = TfidfVectorizer(min_df=1,max_df = 0.6, stop_words='english',preprocessor=Remove_Digits, max_features = 4000)
vectorizer = CountVectorizer(ngram_range=(1,3) ,min_df=1,max_df = 0.6, stop_words='english',preprocessor=removeDigits, max_features = 4000, lowercase=True)


# fit x train data
data_idf_set = vectorizer.fit_transform(data_content)
# idf = vectorizer._tfidf.idf_


# print("n_samples: %d, n_features: %d" % data_idf_set.shape)


pprint (vectorizer.get_feature_names()[:100])


# pprint (dict(zip(vectorizer.get_feature_names()[:100], idf)))
# print vectorizer

a_train, a_test, b_train, b_test = train_test_split(data_idf_set, data_label, test_size=0.33, random_state=42)

# print("n_samples: %d, n_features: %d" % a_test.shape)


clf = svm.SVC(kernel = 'rbf', gamma=0.1, C=10)
print "svm configuration... \n", clf
clf.fit(a_train, b_train)


b_predict = clf.predict(a_test)

# print "predict_label", b_predict
# print "actual label", b_test


print "Number of %d has been predicted" % len(b_predict)




"""
Analyse the Results
"""
print('=' * 80)
print "The results shows below"


Accuracy(b_predict, b_test) 
print "Confusion matrix is \n",confusion_matrix(b_test, b_predict)






# scores = cross_validation.cross_val_score(clf, a_train, b_train, cv=10)
# print('=' * 80)
# print "Cross Validation is \n",scores
# print("Mean Accuracy of Cross Validation: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




# report = classification_report(b_test, b_predict, target_names = ['fall', 'rise'])
# print('=' * 80)
# print "Accuracy Report Table \n", report


# """
# Grid Search Section
# Exhausted search of predefined parameter
# """

# print('=' * 80)
# print "Grid Seach For Best Estimator"

# parameters = {'C':(0.2,0.5,1,2,3,4,5,10),
# 			  'gamma':(0.2,0.5,1,2,3,4,5,10)}

# C_range = 10. ** np.arange(-2, 9)
# gamma_range = 10. ** np.arange(-5, 4)

# param_grid = dict(gamma=gamma_range, C=C_range)


# gs_clf = GridSearchCV(svm.SVC(kernel='rbf'), param_grid=param_grid, n_jobs=-1)

# # Fit and train the train data
# gs_clf = gs_clf.fit(a_train,b_train)
# best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])

# # Print the score for each parameters
# for param_name in sorted(parameters.keys()):
# 	print("%s: %r" % (param_name, best_parameters[param_name]))

# print "Score is "
# print score


# print("The best classifier is: ", gs_clf.best_estimator_)


# # plot the scores of the grid
# # grid_scores_ contains parameter settings and scores
# score_dict = gs_clf.grid_scores_

# # We extract just the scores
# scores = [x[1] for x in score_dict]
# scores = np.array(scores).reshape(len(C_range), len(gamma_range))

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