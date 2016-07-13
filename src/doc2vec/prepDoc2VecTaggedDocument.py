'''
Created on Sep 22, 2015

@author: atomar
'''
import pandas as pd
import json
import utilities.preProc as preProc
import nltk.data

if __name__ == '__main__':
    
    # quoting: int Controls whether quotes should be recognized
    # 0, 1, 2, and 3 for
    # QUOTE_MINIMAL, QUOTE_ALL, QUOTE_NONE, and QUOTE_NONNUMERIC, respectively
    
    train = pd.read_csv("../../data/train.tsv",
                    header=0, delimiter="\t", quoting=3)

    tt = train.head(5)

    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    labeled = []

    num_reviews = len(train["Phrase"])
     
    for i in range(0, num_reviews):
        
        if( (i+1)%1000 == 0 ):
            
            print ("Labeled Review %d of %d\n" % ( i+1, num_reviews ))
        
        labeled.append(preProc.review_to_sentences(train.Phrase[i], tokenizer,str(train.Sentiment[i])))
         
    json.dump(labeled,open("../../classifier/doc2vec/trainLabeled.json", "a"))
