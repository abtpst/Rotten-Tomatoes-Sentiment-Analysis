'''
Created on Sep 29, 2015

@author: atomar
'''
import pandas as pd
import utilities.preProc as preProc
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer as TFIV
"""
This method cleans up the sentences, and then creates feature vectors using TfidfVectorizer
"""
def vectorify(train,test,senti):
    # Value to predict is stored in the Sentiment column of the training data
    predCol = train['Sentiment']
    # Save this column. This stpe is optional
    pickle.dump(predCol,open("../../data/pipeLine/vectorData/predCol"+senti,"wb"))                
    # Create array to hold cleaned up sentences
    trainData = []
    # Total number of reviews
    numRevs = len(train['Phrase'])
    # For each review
    for i in range(0,numRevs):
        # Print out progress update after each time 1000 reviews have been processed
        if( (i+1)%1000 == 0 ):
                
                print ("Train "+ senti +" Review %d of %d\n" % ( i+1, numRevs ))
        # Add the cleaned up review to the result        
        trainData.append(" ".join(preProc.Sentiment_to_wordlist(train['Phrase'][i])))
    # Perform similar cleanup on the test set
    testdata = []
    
    numRevs = len(test['Phrase'])
    
    for i in range(0,numRevs):
        
        if( (i+1)%1000 == 0 ):
                
                print ("Test " + senti + " Review %d of %d\n" % ( i+1, numRevs ))
                
        testdata.append(" ".join(preProc.Sentiment_to_wordlist(test['Phrase'][i])))
    
    # Initialize TfidfVectorizer
    print("Defining TFIDF Vectorizer")        
    
    tfIdfVec = TFIV(
                        min_df=3, # When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold
                        max_features=None, # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
                        strip_accents='unicode', # Remove accents during the preprocessing step. 'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
                                                 # 'unicode' is a slightly slower method that works on any characters.
                        analyzer='word', # Whether the feature should be made of word or character n-grams.. Can be callable.
                        token_pattern=r'\w{1,}', # Regular expression denoting what constitutes a "token", only used if analyzer == 'word'. 
                        ngram_range=(1,2), # The lower and upper boundary of the range of n-values for different n-grams to be extracted.
                        use_idf=1, # Enable inverse-document-frequency re-weighting.
                        smooth_idf=1, # Smooth idf weights by adding one to document frequencies.
                        sublinear_tf=1, # Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).
                        stop_words = 'english' # 'english' is currently the only supported string value.
                    )
    # Optionally, save TfidfVectorizer
    #pickle.dump(tfIdfVec,open("../../data/pipeLine/tfiv110gram","wb"))
    
    """ 
    Combine both to fit the TFIDF vectorization. Note that, for prediction, even the test data has to be vectorized.
    Now, this must be done separately for each sentiment value, as we are trying to create a binary classifier for each sentiment value.
    """
    combineData = trainData + testdata 
    
    # Marker for indicating where the training data ends in the combined structure
    trainLen = len(trainData)
    
    print("Fitting")
    
    tfIdfVec.fit(combineData) # Learn vocabulary and idf from training set.
    
    print("Transforming")
    
    combineData = tfIdfVec.transform(combineData) # Transform documents to document-term matrix. Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform).
    
    print("Fitting and transforming done")
    # Separate back into training and test sets and save.
    trainAfterFit = combineData[:trainLen]  
    pickle.dump(trainAfterFit,open("../../data/pipeLine/vectorData/trainAfterFit"+senti,"wb"))    
    
    testAfterFit = combineData[trainLen:]
    pickle.dump(testAfterFit,open("../../data/pipeLine/vectorData/testAfterFit"+senti,"wb"))
    
if __name__ == '__main__':
    
    # This is our list of sentiments. Remember that we converted the numeric value to words in prepData.py
    senList = ["Negative","SomewhatNegative","Neutral","SomewhatPositive","Positive"]
    # Load the test set
    test = pd.read_csv('../../data/test.tsv', header=0, delimiter="\t", quoting=3 )
    
    for senti in senList:
        # Load training data specific to this sentiment
        train = pd.read_csv("../../data/pipeLine/trainingData/train"+senti+".tsv", header=0, delimiter="\t", quoting=3)
        # Call module to create feature vectors for this training data
        vectorify(train,test,senti)