'''
Created on Sep 22, 2015

@author: atomar
'''
from gensim.models import doc2vec

def myhash(obj):
    return hash(obj) % (2 ** 32)    

model = doc2vec.Doc2Vec(hashfxn=myhash)

model = doc2vec.Doc2Vec.load("../../classifier/doc2vec/Doc2VecRottenToms10Epochs")

print(type(model.syn0))

# number of words, number of features
print(model.syn0.shape)

# doesnt_match function tries to deduce which word in a set is most
# dissimilar from the others
print(model.doesnt_match("man woman child kitchen".split()) + '\n')
# not a good result
#print(model.doesnt_match("paris berlin london austria".split()) + '\n')

# most_similar(): returns the score of the most similar words based on the criteria
# Find the top-N most similar words. Positive words contribute positively towards the
# similarity, negative words negatively.
print("most similar:")
print(model.most_similar(positive=['woman', 'boy'], negative=['man'], topn=10))
print()
print(model.most_similar("garbage"))

print(model.docvecs.most_similar("4"))