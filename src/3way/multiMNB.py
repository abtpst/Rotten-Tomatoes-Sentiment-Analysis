'''
Created on Sep 30, 2015

@author: atomar
'''
from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.cross_validation import cross_val_score
from utilities import helper as help

import numpy as np
import pickle
import pandas as pd


def makeMNB(trainAfterFit,predCol,testAfterFit,test,suff):
    
    model_NB = MNB()
    
    model_NB.fit(trainAfterFit,predCol)
    
    print ("ROC AUC 10 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(model_NB, trainAfterFit, predCol, cv=10, scoring='roc_auc')))
    print ("Precision 10 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(model_NB, trainAfterFit, predCol, cv=10, scoring='precision')))
    print ("Recall 10 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(model_NB, trainAfterFit, predCol, cv=10, scoring='recall')))
    print()
    # This will give us a 20-fold cross validation score that looks at ROC_AUC so we can compare with Logistic Regression. raise ValueError("{0} format is not supported".format(y_type))
    #ValueError: multiclass format is not supported
    
    MNB_result = model_NB.predict(testAfterFit)#[:,1]
    MNB_output = pd.DataFrame(data={"PhraseId":test["PhraseId"], "Sentiment":MNB_result})
    MNB_output.to_csv('../../submits/pipeLine/tags/TagpipelineMNB'+suff+'.csv', index = False, quoting = 3)

if __name__ == '__main__':
    
    #senList = ['0','1','2','3','4']
    
    senList = ["Negative","SomewhatNegative","Neutral","SomewhatPositive","Positive"]
    
    test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )

    for suff in senList:
        
        trainAfterFit = pickle.load(open("../../data/pipeLine/vectorData/trainAfterFit"+suff,"rb"))
        
        predCol = pickle.load(open("../../data/pipeLine/vectorData/predCol"+suff,"rb"))
        
        testAfterFit = pickle.load(open("../../data/pipeLine/vectorData/testAfterFit"+suff,"rb"))
        
        makeMNB(trainAfterFit,predCol,testAfterFit,test,suff)
    
    cDframe = help.combineRes(senList,test["PhraseId"],"MNB")
    
    outliers = pd.DataFrame(columns=['PhraseId','Sentiment'])
    
    outliers.PhraseId = list(set(test["PhraseId"]) - set(cDframe["PhraseId"]))
    
    outliers.Sentiment = "Neutral"
    
    finres = pd.concat([cDframe,outliers])
    
    finres["Sentiment"] = finres["Sentiment"].map({"Negative":int(0),"Positive":int(1),"SomewhatPositive":int(3),"SomewhatNegative":int(1), "Neutral":int(2)})
    finres["PhraseId"] = finres["PhraseId"].apply(lambda x : int(x))
    finres.to_csv('../../submits/pipeLine/tags/TagpipelineMNBCombined.csv', index = False, quoting = 3)