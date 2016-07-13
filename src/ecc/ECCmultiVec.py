'''
Created on Sep 29, 2015

@author: atomar
'''
import pandas as pd
import utilities.preProc as preProc
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer as TFIV

def vectorify(train,test,suff):
    
    predCol = train['Sentiment']
    
    pickle.dump(predCol,open("../../data/pipeLine/ecc/vectorData/predCol"+suff,"wb"))                
    
    trainData = []
    
    numRevs = len(train['Phrase'])
    
    for i in range(0,numRevs):
        
        if( (i+1)%1000 == 0 ):
                
                print ("Train "+ suff +" Review %d of %d\n" % ( i+1, numRevs ))
                
        trainData.append(" ".join(preProc.Sentiment_to_wordlist(train['Phrase'][i])))
    
    testdata = []
    
    numRevs = len(test['Phrase'])
    
    for i in range(0,numRevs):
        
        if( (i+1)%1000 == 0 ):
                
                print ("Test " + suff + " Review %d of %d\n" % ( i+1, numRevs ))
                
        testdata.append(" ".join(preProc.Sentiment_to_wordlist(test['Phrase'][i])))
    
    print("Defining TFIDF Vectorizer")        
    
    tfIdfVec = TFIV(
                        min_df=3, # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
                        max_features=None, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
                        strip_accents='unicode', # Remove accents during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
                                                 # 'unicode' is a slightly slower method that works on any characters.
                        analyzer='word', # Whether the feature should be made of word or character n-grams.. Can be callable.
                        token_pattern=r'\w{1,}', # Regular expression denoting what constitutes a "token", only used if analyzer == 'word'. 
                        ngram_range=(1,2), # The lower and upper boundary of the range of n-values for different n-grams to be extracted.
                        use_idf=1, # Enable inverse-document-frequency reweighting.
                        smooth_idf=1, # Smooth idf weights by adding one to document frequencies.
                        sublinear_tf=1, # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
                        stop_words = 'english' # 'english' is currently the only supported string value.
                    )
    
    #pickle.dump(tfIdfVec,open("../../data/pipeLine/ecc/tfiv110gram","wb"))
    
    combineData = trainData + testdata # Combine both to fit the TFIDF vectorization.
    
    trainLen = len(trainData)
    
    print("Fitting")
    
    tfIdfVec.fit(combineData) # Learn vocabulary and idf from training set.
    
    print("Transforming")
    
    combineData = tfIdfVec.transform(combineData) # Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).
    
    print("Fitting and transforming done")
    
    trainAfterFit = combineData[:trainLen] # Separate back into training and test sets. 
    pickle.dump(trainAfterFit,open("../../data/pipeLine/ecc/vectorData/trainAfterFit"+suff,"wb"))    
    
    testAfterFit = combineData[trainLen:]
    pickle.dump(testAfterFit,open("../../data/pipeLine/ecc/vectorData/testAfterFit"+suff,"wb"))
    
'''       
if __name__ == '__main__':
    
    
    senList = ["Negative","SomewhatNegative","Neutral","SomewhatPositive","Positive"]
    
    test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )
    
    for suff in senList:
        
        #train = pickle.load(open("../../data/pipeLine/ecc/train"+suff,"rb"))
        
        train = pd.read_csv("../../data/pipeLine/ecc/trainingData/train"+suff+".tsv", header=0, delimiter="\t", quoting=3)
        
        vectorify(train,test,suff)
    '''