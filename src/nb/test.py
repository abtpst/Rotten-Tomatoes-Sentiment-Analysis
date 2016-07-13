'''
Created on Sep 25, 2015

@author: atomar
'''
import pandas as pd
import numpy as np

test = pd.read_csv('../../data/test.tsv',sep='\t')
train = pd.read_csv('../../data/train.tsv',sep='\t')

from sklearn.feature_extraction.text import CountVectorizer

data = np.append(test[['Phrase']].values[:,0],train[['Phrase']].values[:,0])

outcome = np.array(map(int,train[['Sentiment']].values))

vectorizer = CountVectorizer(min_df=1)

X = vectorizer.fit(data)

from sklearn.naive_bayes import MultinomialNB

def fit_naivebayes(revVal,predTag):
   
   clf = MultinomialNB() 
   
   X_train=X.transform(revVal) 
   
   clf.fit(X_train,outcome)

mnbClassifier = fit_naivebayes(train[['Phrase']].values[:,0], outcome) 

'''
from sklearn.cross_validation import StratifiedKFold

def CrossValidation(outcome):
    score=[]
    kf = StratifiedKFold(outcome, n_folds=2)
    return kf

print(CrossValidation(outcome))
'''