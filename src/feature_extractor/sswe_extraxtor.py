#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 15:43:24 2017

@author: Mounir Ouled Ltaief 
"""
import os
import logging
import numpy as np
from ConfigParser import ConfigParser
from itertools import chain

from deepnl import *
from deepnl.extractors import *
from deepnl.reader import TweetReader
from deepnl.network import Network
from deepnl.sentiwords import SentimentTrainer

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class sswe_model(object):
    def __init__(self, window=5,embeddings_size=50,epochs=10,learning_rate=0.1,
               eps=1e-8,ro=0.95,hidden=200, ngrams=2,textField=2,
               tagField=1,alpha=0.5,train=None,model=None,
               vocab=None,minOccurr=3, vocab_size=0,vectors= None,load=None, 
               threads= 5,variant= None,verbose=None,config_file=None):
        self.window=window
        self.embeddings_size =embeddings_size
        self.iterations=epochs
        self.learning_rate =learning_rate
        self.eps=eps
        self.ro=ro
        self.hidden=hidden
        self.ngrams=ngrams
        self.textField=textField
        self.tagField=tagField
        self.alpha=alpha
        self.train=train
        self.vocab=vocab
        self.minOccurr=minOccurr
        self.vocab_size =vocab_size
        self.vectors=vectors
        self.load=load
        self.variant=variant
        self.verbose=verbose
        self.model=model
        self.config_file =config_file
       
def sswe_trainer(model_parameters):
    
    # set the seed for replicability
    np.random.seed(42)
    #args = parser.parse_args()
    args=model_parameters
    log_format = '%(message)s'
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_level = logging.INFO
    logging.basicConfig(format=log_format, level=log_level)
    logger = logging.getLogger("Logger")
    
    config = ConfigParser()
    if args.config_file:
        config.read(args.config_file)
    # merge args with config
    reader = TweetReader(text_field=args.textField, label_field=args.tagField,ngrams=args.ngrams)
    reader.read(args.train)
    vocab, bigrams, trigrams = reader.create_vocabulary(reader.sentences,args.vocab_size,
                                                        min_occurrences=args.minOccurr)
    
    if args.variant == 'word2vec' and os.path.exists(args.vectors):
        embeddings = Embeddings(vectors=args.vectors, variant=args.variant)
        embeddings.merge(vocab)
        logger.info("Saving vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)
    elif os.path.exists(args.vocab):
        # start with the given vocabulary
        base_vocab = reader.load_vocabulary(args.vocab)
        if os.path.exists(args.vectors):
            # load embeddings
            embeddings = Embeddings(vectors=args.vectors, vocab=base_vocab,variant=args.variant)
        else:
        # create embeddings
            embeddings = Embeddings(args.embeddings_size, vocab=base_vocab,variant=args.variant)
            # add the ngrams from the corpus
            embeddings.merge(vocab)
            logger.info("Overriding vocabulary in %s" % args.vocab)
            embeddings.save_vocabulary(args.vocab)
    else:
        embeddings = Embeddings(args.embeddings_size, vocab=vocab,variant=args.variant)
        logger.info("Saving vocabulary in %s" % args.vocab)
        embeddings.save_vocabulary(args.vocab)

    # Assume bigrams are prefix of trigrams, or else we should put a terminator
    # on trie
    trie = {}
    for b in chain(bigrams, trigrams):
        tmp = trie
        for w in b:
            tmp = tmp.setdefault(embeddings.dict[w], {})

    converter = Converter()
    converter.add(embeddings)
    
    trainer = create_trainer(args, converter)

    report_intervals = max(args.iterations / 200, 1)
    report_intervals = 10000    # DEBUG

    logger.info("Starting training")

    # a generator expression (can be iterated several times)
    # It caches converted sentences, avoiding repeated conversions
    converted_sentences = converter.generator(reader.sentences, cache=True)
    print converted_sentences
    trainer.train(converted_sentences, reader.polarities, trie,
              args.iterations, report_intervals)

    logger.info("Overriding vectors to %s" % args.vectors)
    embeddings.save_vectors(args.vectors, args.variant)
    if args.model:
        logger.info("Saving trained model to %s" % args.model)
        trainer.save(args.model)
def create_trainer(args, converter):
    """
    Creates or loads a neural network according to the specified args.
    """

    logger = logging.getLogger("Logger")

    if args.load:
        logger.info("Loading provided network...")
        trainer = SentimentTrainer.load(args.load)
        # change learning rate
        trainer.learning_rate = args.learning_rate
    else:
        logger.info('Creating new network...')
        # sum the number of features in all extractors' tables 
        input_size = converter.size() * (args.window * 2 + 1)
        nn = Network(input_size, args.hidden, 2)
        options = {
                'learning_rate': args.learning_rate,
                'eps': args.eps,
                'ro': args.ro,
                'verbose': args.verbose,
                'left_context': args.window,
                'right_context': args.window,
                'ngram_size': args.ngrams,
                'alpha': args.alpha
                }
        trainer = SentimentTrainer(nn, converter, options)

    trainer.saver = saver(args.model, args.vectors)

    logger.info("... with the following parameters:")
    logger.info(trainer.nn.description())

    return trainer

def saver(model_file, vectors_file):
    """Function for saving model periodically"""
    def save(trainer):
        # save embeddings also separately
        if vectors_file:
            trainer.save_vectors(vectors_file)
        if model_file:
            trainer.save(model_file)
    return save
