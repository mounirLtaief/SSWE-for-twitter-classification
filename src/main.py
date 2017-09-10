#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:43:24 2017

@author: Mounir Ouled Ltaief
 
"""

import sys

sys.path.insert(0,'/root/Text_Mining_Project/src/feature_extractor')
from deepnl.extractors import *
from sswe_extraxtor import *
from deepnl.reader import TweetReader
from pprint import pprint

def main():
    ###################################################################################
    ######################################load dataset#################################
    train_set='/root/Text_Mining_Project/src/data/train.tsv' #load dataset
    # save model to file 
    model = 'sswe_model'  # model save file
    # vocab file
    sswe_vocabs='/root/Text_Mining_Project/src/words.txt' # vocab file
    # vectors file
    sswe_vectors='/root/Text_Mining_Project/src/vectors.txt' # vectors file model=model,train=train_set,vocab=vocab,vectors=vectors
    
    # sswe model 
    #sswe=sswe_model(train=train_set,vocab=sswe_vocabs,vectors=sswe_vectors)
    #train sswe model
    #sswe_trainer(sswe)
    # load tweets
    reader = TweetReader(text_field=2, label_field=1,ngrams=2)
    reader.read(train_set)
    # load vocab
    base_vocab = reader.load_vocabulary(sswe_vocabs)
    # load embedding
    #vocab_file=None, vectors=None, vocab=None,variant=None
    #emb = Embeddings()
    #print emb.load_vocabulary(sswe_vocabs)
    
    embeddings = Embeddings(vocab_file=sswe_vocabs,vectors=sswe_vectors,vocab=base_vocab,variant=None)
    
    # define converter
    converter = Converter()
    # add embeding to converter
    converter.add(embeddings)
    print reader.sentences
    # generate feature vectors for tweets
    converted_tweets = converter.generator(reader.sentences, cache=True)
    #for s in converted_tweets:
    #    print s.shape
    #pprint(converted_sentences)
    #for k in converted_tweets:
    #    print k.shape
    #sent= converter.generator([['I','am','happy']],cache=True)
    #for k in sent:
    #    print k
    return 0
    
if __name__ == '__main__':
    
    main()
