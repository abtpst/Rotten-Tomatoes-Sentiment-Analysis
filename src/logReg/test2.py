'''
Created on Sep 25, 2015

@author: atomar
'''
'''
listA = [0]
listB = listA
l=listB
listB.append(1)
#print (l)

def reassign(list):
    
    list.append(1)
    list = [0, 9]

def append(list):
    list.append(1)

list = [0]
reassign(list)
#append(list)

print(list)
'''
import pandas as pd

train = pd.read_csv('../../data/train.tsv', header=0, delimiter="\t", quoting=3)

train0 = train.loc[(train.Sentiment==0)]

print(train0.info())
test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )

'''
import pandas as pd
df = pd.read_csv('../../data/train.tsv',header=0,delimiter='\t')


import numpy as np

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


X_train = df['Phrase']
y_train = df['Sentiment']

text_clf = Pipeline([('vect', TfidfVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LogisticRegression()),
                      ])

text_clf = text_clf.fit(X_train,y_train)

X_test = df.head()['Phrase']

predicted = text_clf.predict(X_test)

print (np.mean(predicted == df.head()['Sentiment']))

for phrase, sentiment in zip(X_test, predicted):
    print('%r => %s' % (phrase, sentiment))
    
test_df = pd.read_csv('../../data/test.tsv',header=0,delimiter='\t')
    
from numpy import savetxt
X_test = test_df['Phrase']
phraseIds = test_df['PhraseId']
predicted = text_clf.predict(X_test)
pred = [[index+156061,x] for index,x in enumerate(predicted)]
savetxt('../../submits/tfidfVecLogReg.csv',pred,delimiter=',',fmt='%d,%d',header='PhraseId,Sentiment',comments='')   
''' 