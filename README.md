Group Project for COMPIG15@UCL Information Retrieval and Data Mining
===================

Project Info
-------------
### Sentiment Analysis with Twitter Data

Sentiment analysis aims to judge whether the emotional tend expressed in a text is positive or negative. Generally speaking, sentiment analysis is trying to determine the attitude of a speaker or a writer on a topic or the overall contextual polarity of a document. The basic task is classifying the polarity of the given text. In this project, it focused on the Tweeter, a popular microblog. The various models are built to classify the tweets into positive or negative sentiment. Before using the texts, there are several ways to pre-process the data. Moreover, then the feature extraction converts the twitter texts to a vector. There are three methods to obtain the feature: TI-IDF, TF, and FP. Moreover, the five supervised learning classifiers are implemented, including Naïve Bayes, Logistic regression, SVM, DecisionTrees, and KNN. Finally, the evaluation is used for these five models, and the SVM achieved the best performance.


Team Member
-------------
Shut Wang, MSc Web Science and Big Data
Yue Wang, MSc NCS
Xizhe Jiang, MSc NCS

How to use the code
-------------
> - Git clone repo, the program is within the **implementation** folder
> - Make sure you have already install scikit learn,  NLTK, pandas, matplotlib in advance
> - run command **python sentiment_analysis.py** in the **implementation** folder,  you now run the main program
> - by unblock and block the comment in the code, choose the feature extraction and classifier you want

Dataset
-------------


> **Note:**

> - **5000_random.csv** is the dataset with random chosen twitter **5000_seq.csv** is the dataset by sequential ordered by alphabet a-z. 
> - the 1,500,000 raw twitter downloaded from:  http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/


