'''
Created on Sep 28, 2015

@author: atomar
'''
from sklearn.linear_model import LogisticRegression as LR
from sklearn.grid_search import GridSearchCV
import pickle

X= pickle.load(open("../../classifier/logisticRegression/x","rb"))

y_train = pickle.load(open("../../classifier/logisticRegression/yTrain","rb"))


grid_values = {'C':[30]} # Decide which settings you want for the grid search. 

model_LR = GridSearchCV(LR(penalty = 'l2', dual = True, random_state = 0), 
                        grid_values, scoring = 'roc_auc', cv = 20) 
# Try to set the scoring on what the contest is asking for. 
# The contest says scoring is for area under the ROC curve, so use this.

from sklearn.preprocessing import label_binarize

y_train = label_binarize(y_train, classes=[0, 1, 2, 3, 4])

model_LR.fit(X,y_train) # Fit the model.raise ValueError("{0} format is not supported".format(y_type)) ValueError: multiclass format is not supported


print(model_LR.grid_scores_)

'''
from sklearn.naive_bayes import MultinomialNB as MNB


model_NB = MNB()
model_NB.fit(X, y_train)

from sklearn.cross_validation import cross_val_score
import numpy as np


print ("20 Fold CV Score for Multinomial Naive Bayes: ", np.mean(cross_val_score(model_NB, X, y_train, cv=20, scoring='roc_auc')))
# This will give us a 20-fold cross validation score that looks at ROC_AUC so we can compare with Logistic Regression. raise ValueError("{0} format is not supported".format(y_type))
#ValueError: multiclass format is not supported


from sklearn.linear_model import SGDClassifier as SGD


sgd_params = {'alpha': [0.00006, 0.00007, 0.00008, 0.0001, 0.0005]} # Regularization parameter

model_SGD = GridSearchCV(SGD(random_state = 0, shuffle = True, loss = 'modified_huber'), sgd_params, scoring = 'roc_auc', cv = 20) # Find out which regularization parameter works the best. 

model_SGD.fit(X, y_train) # Fit the model.
'''