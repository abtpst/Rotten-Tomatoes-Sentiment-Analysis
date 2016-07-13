'''
Created on Oct 2, 2015

@author: atomar
'''
from utilities import helper as help
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.grid_search import GridSearchCV
import numpy as np
from sklearn.cross_validation import cross_val_score
import pickle
import pandas as pd

def makeSGD(trainAfterFit,predCol,testAfterFit,test,suff):
    
    sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]} # Constant that multiplies the regularization term. Defaults to 0.0001
    
    model_SGD = GridSearchCV(
                             SGD(
                                 random_state = 0, # The seed of the pseudo random number generator to use when shuffling the data.
                                 shuffle = True, # Whether or not the training data should be shuffled after each epoch. Defaults to True.
                                 loss = 'modified_huber'
                                 
                                 # The loss function to be used. Defaults to 'hinge', which gives a linear SVM. 
                                 # The 'log' loss gives logistic regression, a probabilistic classifier. 
                                 # 'modified_huber' is another smooth loss that brings tolerance to outliers as well as probability estimates. 
                                 # 'squared_hinge' is like hinge but is quadratically penalized. 'perceptron' is the linear loss used by the perceptron algorithm. 
                                 # The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
                                
                                 ), 
                             sgd_params,
                             scoring = 'roc_auc', # A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
                             cv = 20 # If an integer is passed, it is the number of folds.
                            ) 
    
    model_SGD.fit(trainAfterFit,predCol) # Fit the model.
    
    print(model_SGD.grid_scores_)
    
    #SGD_result = model_SGD.predict_proba(testAfterFit)[:,1]
    
    SGD_result = model_SGD.predict(testAfterFit)
    
    SGD_output = pd.DataFrame(data={"PhraseId":test["PhraseId"], "Sentiment":SGD_result})
    SGD_output.to_csv('../../submits//pipeLine/tags/TagpipelineSGD'+suff+'.csv', index = False, quoting = 3)
    
if __name__ == '__main__':
    
    #senList = ['0','1','2','3','4']
    
    senList = ["Negative","SomewhatNegative","Neutral","SomewhatPositive","Positive"]
    
    test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )

    for suff in senList:
        
        print(suff)
        
        trainAfterFit = pickle.load(open("../../data/pipeLine/vectorData/trainAfterFit"+suff,"rb"))
        
        predCol = pickle.load(open("../../data/pipeLine/vectorData/predCol"+suff,"rb"))
        
        testAfterFit = pickle.load(open("../../data/pipeLine/vectorData/testAfterFit"+suff,"rb"))
        
        makeSGD(trainAfterFit,predCol,testAfterFit,test,suff)
   
    cDframe = help.combineRes(senList,test["PhraseId"],"SGD")
    
    outliers = pd.DataFrame(columns=['PhraseId','Sentiment'])
    
    outliers.PhraseId = list(set(test["PhraseId"]) - set(cDframe["PhraseId"]))
    
    outliers.Sentiment = "Neutral"
    
    finres = pd.concat([cDframe,outliers])
    
    finres["Sentiment"] = finres["Sentiment"].map({"Negative":int(0),"Positive":int(1),"SomewhatPositive":int(3),"SomewhatNegative":int(1), "Neutral":int(2)})
    finres["PhraseId"] = finres["PhraseId"].apply(lambda x : int(x))
    finres.to_csv('../../submits/pipeLine/tags/TagpipelineSGDCombined.csv', index = False, quoting = 3)