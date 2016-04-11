# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm, cross_validation
from pprint import pprint
from datetime import *
import os
import re
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt
from sklearn.grid_search import GridSearchCV
from matplotlib import style
# import pylab as plt
style.use('ggplot')


def Remove_Digits(s):
	results = ''.join(i for i in s if not i.isdigit())
	return results



def load_data():
	cols = ['ItemID','Sentiment','SentimentSource','SentimentText']
	# df = pd.read_csv('5000_data.csv', skipinitialspace=True, usecols=fileds)
	df = pd.read_csv('10000_data.csv', sep=',', header=None, names=cols);
	print df.head()


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

# print data_content



print "Initialise feature_extraction function"

#n-gram

ngram_vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,10), min_df=1,max_df = 0.6,stop_words='english'
								   ,preprocessor=Remove_Digits,max_features = 4000)

counts= ngram_vectorizer.fit_transform(data_content)

pprint (ngram_vectorizer.get_feature_names()[:100])

# Initialise the feature_extraction func
#vectorizer = TfidfVectorizer(min_df=1,max_df = 0.6, stop_words='english',preprocessor=Remove_Digits, max_features = 4000)



# fit x train data
# data_idf_set = vectorizer.fit_transform(data_content)
# idf = vectorizer._tfidf.idf_


# print("n_samples: %d, n_features: %d" % data_idf_set.shape)




#pprint (dict(zip(vectorizer.get_feature_names()[:100], idf)))
# print vectorizer

a_train, a_test, b_train, b_test = train_test_split(counts, data_label, test_size=0.33, random_state=42)

# print("n_samples: %d, n_features: %d" % a_test.shape)


clf = svm.SVC(kernel = 'rbf', gamma=0.1, C=10)
print "svm configuration... \n", clf
clf.fit(a_train, b_train)


b_predict = clf.predict(a_test)

# print "predict_label", b_predict
# print "actual label", b_test




Accuracy(b_predict, b_test)






