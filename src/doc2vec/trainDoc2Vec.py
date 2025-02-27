'''
Created on Sep 22, 2015

@author: atomar
'''
from gensim.models import doc2vec
import json
import pickle
import logging
from random import shuffle
import time

bagTaggedDocs = pickle.load(open("../../classifier/doc2vec/taggedDocs.pickle","rb"))
    
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# parameter values
num_features = 300

# minimum word count: any word that does not occur at least this many times
# across all documents is ignored
min_word_count = 10

# the paper (http://arxiv.org/pdf/1405.4053v2.pdf) suggests 10 is the optimal
context = 10

#  threshold for configuring which higher-frequency words are randomly downsampled;
# default is 0 (off), useful value is 1e-5
# set the same as word2vec
downsampling = 1e-3

num_workers = 4  # Number of threads to run in parallel

# if sentence is not supplied, the model is left uninitialized
# otherwise the model is trained automatically
# https://www.codatlas.com/github.com/piskvorky/gensim/develop/gensim/models/doc2vec.py?line=192

def myhash(obj):
       return hash(obj) % (2 ** 32)
   
   
model = doc2vec.Doc2Vec(size=num_features,
                        window=context, min_count=min_word_count,
                        sample=downsampling, workers=num_workers,hashfxn=myhash)

model.build_vocab(bagTaggedDocs)

# gensim documentation suggests training over data set for multiple times
# by either randomizing the order of the data set or adjusting learning rate
# see here for adjusting learn rate: http://rare-technologies.com/doc2vec-tutorial/
# iterate 10 times

for epoch in range(1,10):
    
    print("Starting Epoch ",epoch)
    
    start_time = time.time()
    
    shuffle(bagTaggedDocs)
    
    model.train(bagTaggedDocs)
    
    print("Epoch ",epoch," took %s minutes " % ((time.time() - start_time)/60))
    
# model.init_sims(replace=True)
print(model.most_similar(positive=['woman', 'king'], negative=['man'], topn=10))

model.save("../../classifier/doc2vec/Doc2VecRottenToms10Epochs")