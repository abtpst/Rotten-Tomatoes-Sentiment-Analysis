'''
Created on Oct 2, 2015

@author: atomar
'''
from utilities import helper as help
import numpy as np
from sklearn.cross_validation import cross_val_score
import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV

def makeLogReg(trainAfterFit,predCol,testAfterFit,test,suff):

    grid_values = {'C':[30]} # Decide which settings you want for the grid search. 
    
    # C: Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.
    
    #GridSearchCV implements a Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.fit" method and a Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.predict" method like any classifier except that the parameters of the classifier used to predict is optimized by cross-validation.
    
    modelLR = GridSearchCV(
                            LR
                                (
                                    penalty = 'l2', # Used to specify the norm used in the penalization. http://www.chioka.in/differences-between-l1-and-l2-as-loss-function-and-regularization/
                                    
                                    # One of the prime differences between Lasso and ridge regression is that in ridge regression, as the penalty is increased, all parameters are reduced while
                                    # still remaining non-zero, while in Lasso, increasing the penalty will cause more and more of the parameters to be driven to zero. This is an advantage of
                                    # Lasso over ridge regression, as driving parameters to zero deselects the features from the regression. Thus, Lasso automatically selects more relevant features 
                                    # and discards the others, whereas Ridge regression never fully discards any features. Some feature selection techniques are developed based on the LASSO including
                                    # Bolasso which bootstraps samples,[12] and FeaLect which analyzes the regression coefficients corresponding to different values of \alpha to score all the features.
                                                                    
                                    dual = True, # Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
                                    random_state = 0 # The seed of the pseudo random number generator to use when shuffling the data.
                                ), 
                            grid_values,
                            scoring = 'roc_auc', # A string (see model evaluation documentation) or a scorer callable object / function with signature scorer(estimator, X, y).
                            cv = 20 # If an integer is passed, it is the number of folds.
                           ) 
    # Try to set the scoring on what the contest is asking for. 
    # The contest says scoring is for area under the ROC curve, so use this.
    
    modelLR.fit(trainAfterFit,predCol) # Fit the model according to the given training data.
    
    print(modelLR.grid_scores_)
    
    '''
    print ("ROC AUC 10 Fold CV Score for Logistic Regression: ", np.mean(cross_val_score(modelLR, trainAfterFit, predCol, cv=10, scoring='roc_auc')))
    print ("Precision 10 Fold CV Score for Logistic Regression: ", np.mean(cross_val_score(modelLR, trainAfterFit, predCol, cv=10, scoring='precision')))
    print ("Recall 10 Fold CV Score for Logistic Regression: ", np.mean(cross_val_score(modelLR, trainAfterFit, predCol, cv=10, scoring='recall')))
    
    Contains scores for all parameter combinations in param_grid. Each entry corresponds to one parameter setting. Each named tuple has the attributes:
    parameters, a dict of parameter settings
    mean_validation_score, the mean score over the cross-validation folds
    cv_validation_scores, the list of scores for each fold
    '''
    
    #LR_result = modelLR.predict_proba(testAfterFit)[:,1] # Probability estimates. The returned estimates for all classes are ordered by the label of classes. 
    LR_result = modelLR.predict(testAfterFit)
    
    LR_output = pd.DataFrame(data={"PhraseId":test["PhraseId"], "Sentiment":LR_result}) # Create our dataframe that will be written.
    
    LR_output.to_csv('../../submits/pipeLine/tags/TagpipelineLogReg'+suff+'.csv', index=False, quoting=3)

if __name__ == '__main__':
    
    
    #senList = ['0','1','2','3','4']
    
    senList = ["Negative","SomewhatNegative","Neutral","SomewhatPositive","Positive"]
    
    test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )
    
    for suff in senList:
        
        print(suff)
        
        trainAfterFit = pickle.load(open("../../data/pipeLine/vectorData/trainAfterFit"+suff,"rb"))
        
        predCol = pickle.load(open("../../data/pipeLine/vectorData/predCol"+suff,"rb"))
        
        testAfterFit = pickle.load(open("../../data/pipeLine/vectorData/testAfterFit"+suff,"rb"))
        
        makeLogReg(trainAfterFit,predCol,testAfterFit,test,suff)
    
    cDframe = help.combineRes(senList,test["PhraseId"],"LogReg")
    
    outliers = pd.DataFrame(columns=['PhraseId','Sentiment'])
    
    outliers.PhraseId = list(set(test["PhraseId"]) - set(cDframe["PhraseId"]))
    
    outliers.Sentiment = "Neutral"
    
    finres = pd.concat([cDframe,outliers])
    
    finres["Sentiment"] = finres["Sentiment"].map({"Negative":int(0),"Positive":int(1),"SomewhatPositive":int(3),"SomewhatNegative":int(1), "Neutral":int(2)})
    finres["PhraseId"] = finres["PhraseId"].apply(lambda x : int(x))
    finres.to_csv('../../submits/pipeLine/tags/TagpipelineLogRegCombined.csv', index = False, quoting = 3)