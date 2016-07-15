'''
Created on Sep 29, 2015

@author: atomar
'''
import pandas as pd
import numpy as np
"""
We will create five sets of training data from our original training set. 
These will be fed to our binary classifiers. Each of these sets can be taken as training data for one value of the sentiment.
For example, lets say we mark all sentences with a 'Positive' sentiment as 1 and all other as 0 and we call this marked data as `train0`. 
Now, `train0` can be used to train a binary classifier to predict whether a given sentence falls under 'Positive' or not. We can do similar things for the other sentiment values.
"""

if __name__ == '__main__':
    # Read in traiining data 
    train = pd.read_csv('../../data/train.tsv', header=0, delimiter="\t", quoting=3)
    '''
    print(len(train[(train.Sentiment==0)])/len(train["PhraseId"]))
    print(len(train[(train.Sentiment==1)])/len(train["PhraseId"]))
    print(len(train[(train.Sentiment==2)])/len(train["PhraseId"]))
    print(len(train[(train.Sentiment==3)])/len(train["PhraseId"]))
    print(len(train[(train.Sentiment==4)])/len(train["PhraseId"]))
    '''
    # Convert prediction values from numbers to words
    train.loc[train.Sentiment==0, ['Sentiment']] = "Negative"
    train.loc[train.Sentiment==1, ['Sentiment']] = "SomewhatNegative"
    train.loc[train.Sentiment==2, ['Sentiment']] = "Neutral"
    train.loc[train.Sentiment==3, ['Sentiment']] = "SomewhatPositive"
    train.loc[train.Sentiment==4, ['Sentiment']] = "Positive"
    # Save the training data with the updated values for the sentiment column
    train.to_csv("../../data/pipeLine/trainSents.tsv", header=['PhraseId','SentenceId','Phrase','Sentiment'], sep ="\t", quoting=3, escapechar="\t", index=False)
    
    # List of all unique values in the sentiment column
    senList = train.Sentiment.unique()
    
    # For each sentiment, we are going to create a binary classifier
    for senti in senList:
        
        # Positive sentences, with respect to this sentiment, are the ones that have this sentiment
        trainP = train.loc[(train.Sentiment==senti)]
        # Rest are Negative sentences
        trainN = train.loc[(train.Sentiment!=senti)]
        
        # Create binary identifiers for positive and negative sentiment
        trainP['Sentiment']=1
        trainN['Sentiment']=0
        
        # Create training set for the current sentiment    
        trainFinal = pd.concat([trainP,trainN])
        
        trainFinal = trainFinal.reindex(np.random.permutation(trainFinal.index))
        # Save training set
        trainFinal.to_csv("../../data/pipeLine/trainingData/train"+senti+".tsv", header=['PhraseId','SentenceId','Phrase','Sentiment'], sep ="\t", quoting=3, escapechar="\t", index=False)