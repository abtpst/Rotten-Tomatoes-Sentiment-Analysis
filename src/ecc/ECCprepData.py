'''
Created on Sep 29, 2015

@author: atomar
'''
import pandas as pd
import numpy as np

train = pd.read_csv('../../data/train.tsv', header=0, delimiter="\t", quoting=3)
'''
print(len(train[(train.Sentiment==0)])/len(train["PhraseId"]))
print(len(train[(train.Sentiment==1)])/len(train["PhraseId"]))
print(len(train[(train.Sentiment==2)])/len(train["PhraseId"]))
print(len(train[(train.Sentiment==3)])/len(train["PhraseId"]))
print(len(train[(train.Sentiment==4)])/len(train["PhraseId"]))
'''
train.loc[train.Sentiment==0, ['Sentiment']] = "Negative"
train.loc[train.Sentiment==1, ['Sentiment']] = "SomewhatNegative"
train.loc[train.Sentiment==2, ['Sentiment']] = "Neutral"
train.loc[train.Sentiment==3, ['Sentiment']] = "SomewhatPositive"
train.loc[train.Sentiment==4, ['Sentiment']] = "Positive"

train.to_csv("../../data/pipeLine/ecc/trainSents.tsv", header=['PhraseId','SentenceId','Phrase','Sentiment'], sep ="\t", quoting=3, escapechar="\t", index=False)

senList = train.Sentiment.unique()

for suff in senList:
    
    trainP = train.loc[(train.Sentiment==suff)]
    trainN = train.loc[(train.Sentiment!=suff)]
    
    trainP['Sentiment']=1
    trainN['Sentiment']=0
        
    trainFinal = pd.concat([trainP,trainN])
    
    trainFinal = trainFinal.reindex(np.random.permutation(trainFinal.index))
    
    trainFinal.to_csv("../../data/pipeLine/ecc/trainingData/train"+suff+".tsv", header=['PhraseId','SentenceId','Phrase','Sentiment'], sep ="\t", quoting=3, escapechar="\t", index=False)