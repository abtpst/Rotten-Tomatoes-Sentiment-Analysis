'''
Created on Sep 23, 2015

@author: atomar
'''
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier

def makeFeatureVec(review, model, num_features):
    """
    given a review, define the feature vector by averaging the feature vectors
    of all words that exist in the model vocabulary in the review
    :param review:
    :param model:
    :param num_features:
    :return:
    """

    featureVec = np.zeros(num_features, dtype=np.float32)
    nwords = 0

    # index2word is the list of the names of the words in the model's vocabulary.
    # convert it to set for speed
    vocabulary_set = set(model.index2word)
    print(review)
    # loop over each word in the review and add its feature vector to the total
    # if the word is in the model's vocabulary
    for word in review:
        if word in vocabulary_set:
            nwords = nwords + 1
            # add arguments element-wise
            # if x1.shape != x2.shape, they must be able to be casted
            # to a common shape
            featureVec = np.add(featureVec, model[word])
    
    if(nwords==0):
        nwords=0.00000000000000001
                
    featureVec = np.divide(featureVec,nwords)
    
    return featureVec

def getAvgFeatureVecs (reviewSet, model):

    # initialize variables
    counter = 0
    num_features = model.syn0.shape[1]
    reviewsetFV = np.zeros((len(reviewSet),num_features), dtype=np.float32)

    for review in reviewSet:
        reviewsetFV[counter] = makeFeatureVec(review, model, num_features)
        counter += 1
        
    return reviewsetFV

def rfClassifer(n_estimators, trainingSet, label, testSet):
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    forest = RandomForestClassifier(n_estimators)
    forest = forest.fit(trainingSet, label)
    result = forest.predict(testSet)

    return result

from sklearn.cluster import KMeans

def kmeans(num_clusters, dataSet):
    # n_clusters: number of centroids
    # n_jobs: number of jobs running in parallel
    
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
   
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    # Compute cluster centers and predict cluster index for each sample
    centroidIndx = kmeans_clustering.fit_predict(dataSet)

    return centroidIndx

def create_bag_of_centroids(reviewData,num_clusters,index_word_map):
        """
        assign each word in the review to a centroid
        this returns a numpy array with the dimension as num_clusters
        each will be served as one feature for classification
        :param reviewData:
        :return:
        """
        featureVector = np.zeros(num_clusters, dtype=np.float)
        for word in reviewData:
            if word in index_word_map:
                index = index_word_map[word]
                featureVector[index] += 1
        return featureVector