'''
Created on Sep 23, 2015

@author: atomar
'''
import pickle
import json
from gensim.models.doc2vec import TaggedDocument


def labelizeReviews(reviewSet):
        """
        add label to each review
        :param reviewSet:
        :param label: the label to be put on the review
        :return:
        """
        labelized = []
        for review in reviewSet:
            
            sentiment = review.pop()
            labelized.append(TaggedDocument(words=review, tags=[sentiment]))
        
        return labelized
    # the input to doc2vec is an iterator of LabeledSentence objects
    # each consists a list of words and a list of labels

labeled = json.load(open("../../classifier/doc2vec/trainLabeled.json", "r"))

labeled = labelizeReviews(labeled)

pickle.dump(labeled,open("../../classifier/doc2vec/taggedDocs.pickle", "wb"))
