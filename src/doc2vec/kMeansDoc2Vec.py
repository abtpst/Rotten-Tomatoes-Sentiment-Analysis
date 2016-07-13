'''
Created on Sep 23, 2015

@author: atomar
'''
import time
import logging
import pickle
import pandas as pd
import numpy as np
import utilities.preProc as preProc
import utilities.classifierFuncs as cfun
from gensim.models import doc2vec

def myhash(obj):
        return hash(obj) % (2 ** 32)
    
def main():

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    model = doc2vec.Doc2Vec(hashfxn=myhash)

    model = doc2vec.Doc2Vec.load("../../classifier/doc2vec/Doc2VecRottenToms10Epochs")

    # model.init_sims(replace=True)

    word_vectors = model.syn0
      
    num_clusters = int(word_vectors.shape[0] / 5)
    
    print("number of clusters: {}".format(num_clusters))
    # input("Press enter to continue:")
    
    '''
    print("Clustering...")
    startTime = time.time()
    cluster_index = cfun.kmeans(num_clusters, word_vectors)
    endTime = time.time()

    print("Time taken for clustering: {} minutes".format((endTime - startTime)/60))
    
    clusterf = open("../../classifier/doc2vec/clusterIndex.pickle","wb") 
    
    pickle.dump(cluster_index,clusterf)
    '''
    cluster_index = pickle.load(open("../../classifier/doc2vec/clusterIndex.pickle","rb") )
    # create a word/index dictionary, mapping each vocabulary word to a cluster number
    # zip(): make an iterator that aggregates elements from each of the iterables
    index_word_map = dict(zip(model.index2word, cluster_index))
    
    train = pd.read_csv("../../data/train.tsv",
                    header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("../../data/test.tsv",
                   header=0, delimiter="\t", quoting=3)
    
    trainingDataFV = np.zeros((train["Phrase"].size, num_clusters), dtype=np.float)
    
    testDataFV = np.zeros((test["Phrase"].size, num_clusters), dtype=np.float)
    
    print("Processing training data...")
    counter = 0
    cleaned_training_data = preProc.clean_data(train,"Phrase")
    for review in cleaned_training_data:
        trainingDataFV[counter] = cfun.create_bag_of_centroids(review,num_clusters,index_word_map)
        counter += 1
        
    pickle.dump(trainingDataFV,open("../../classifier/doc2vec/kMeanstrainingFV.pickle","wb"))


    print("Processing test data...")
    counter = 0
    cleaned_test_data = preProc.clean_data(test,"Phrase")
    for review in cleaned_test_data:
        testDataFV[counter] = cfun.create_bag_of_centroids(review,num_clusters,index_word_map)
        counter += 1

    pickle.dump(testDataFV,open("../../classifier/doc2vec/kMeanstestFV.pickle","wb"))
    
    n_estimators = 100
    result = cfun.rfClassifer(n_estimators, trainingDataFV, train["Sentiment"],testDataFV)
    output = pd.DataFrame(data={"PhraseId": test["PhraseId"], "Sentiment": result})
    output.to_csv("../../submits/Doc2Vec_Clustering.csv", index=False, quoting=3)
    
if __name__ == '__main__':
    main()