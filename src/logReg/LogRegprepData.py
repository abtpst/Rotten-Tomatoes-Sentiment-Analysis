'''
Created on Sep 29, 2015

@author: atomar
'''
import pandas as pd

train = pd.read_csv('../../data/train.tsv', header=0, delimiter="\t", quoting=3)

train0 = train.loc[(train.Sentiment==0)]

print(train0.info())
test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )