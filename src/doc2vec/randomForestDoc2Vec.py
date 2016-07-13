'''
Created on Sep 23, 2015

@author: atomar
'''
import logging
import pickle
import pandas as pd
import utilities.preProc as preProc
import utilities.classifierFuncs as cfun
from gensim.models import doc2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

train = pd.read_csv("../../data/train.tsv", header=0, delimiter="\t", quoting=3)

test = pd.read_csv("../../data/test.tsv", header=0, delimiter="\t", quoting=3)
'''
def myhash(obj):
    return hash(obj) % (2 ** 32)    

model = doc2vec.Doc2Vec(hashfxn=myhash)

model = doc2vec.Doc2Vec.load("../../classifier/doc2vec/Doc2VecRottenToms10Epochs")

print("Processing training data...")
cleaned_training_data = preProc.clean_data(train,"Phrase")
trainingDataFV = cfun.getAvgFeatureVecs(cleaned_training_data,model)
print("Processing test data...")
cleaned_test_data = preProc.clean_data(test,"Phrase")
testDataFV = cfun.getAvgFeatureVecs(cleaned_test_data,model)

pickle.dump(trainingDataFV,open("../../classifier/doc2vec/trainingFV.pickle","wb"))

pickle.dump(testDataFV,open("../../classifier/doc2vec/testFV.pickle","wb"))
'''

n_estimators = 100

trainingDataFV = pickle.load(open("../../classifier/doc2vec/trainingFV.pickle","rb"))

testDataFV = pickle.load(open("../../classifier/doc2vec/testFV.pickle","rb"))
print(testDataFV)
'''
result = cfun.rfClassifer(n_estimators, trainingDataFV, train["Sentiment"],testDataFV)

output = pd.DataFrame(data={"PhraseId": test["PhraseId"], "Sentiment": result})

output.to_csv("../../submits/Doc2VecTaggedDocsAvgVecPredict.csv", index=False, quoting=3)
'''