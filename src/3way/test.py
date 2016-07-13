'''
Created on Sep 28, 2015

@author: atomar
'''
import re
from bs4 import BeautifulSoup
import pickle
import pandas as pd


def Sentiment_to_wordlist(Sentiment):
    '''
    Meant for converting each of the IMDB Sentiments into a list of words.
    '''
    # First remove the HTML.
    Sentiment_text = BeautifulSoup(Sentiment).get_text()

    # Use regular expressions to only include words.
    Sentiment_text = re.sub("[^a-zA-Z]"," ", Sentiment_text)

    # Convert words to lower case and split them into separate words.
    words = Sentiment_text.lower().split()

    # Return a list of words
    return(words)




train = pd.read_csv('../../data/train.tsv', header=0,
                delimiter="\t", quoting=3)

y_train = train['Sentiment']

pickle.dump(y_train,open("../../classifier/logisticRegression/yTrain","wb"))                

test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t",
                quoting=3 )

traindata = []
for i in range(0,len(train['Phrase'])):
    traindata.append(" ".join(Sentiment_to_wordlist(train['Phrase'][i])))

testdata = []
for i in range(0,len(test['Phrase'])):
    testdata.append(" ".join(Sentiment_to_wordlist(test['Phrase'][i])))

from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

tfv = TFIV(min_df=3,  max_features=None, 
        strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
        ngram_range=(1, 2), use_idf=1,smooth_idf=1,sublinear_tf=1,
        stop_words = 'english')

X_all = traindata + testdata # Combine both to fit the TFIDF vectorization.
lentrain = len(traindata)

print("Fitting")
tfv.fit(X_all) # This is the slow part!
X_all = tfv.transform(X_all)
print("Fitting done")
X = X_all[:lentrain] # Separate back into training and test sets. 
X_test = X_all[lentrain:]

pickle.dump(X,open("../../classifier/logisticRegression/x","wb"))        

pickle.dump(X_all,open("../../classifier/logisticRegression/xAll","wb"))

pickle.dump(y_train,open("../../classifier/logisticRegression/yTrain","wb"))