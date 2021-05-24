# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 20:17:07 2020

@author: eilxaix
"""

import numpy as np
import re 
import nltk
porter = nltk.PorterStemmer()

class WordEmbeddings():

    def __init__(self, embeddings = None, vec_file="./embed_data/numberbatch-en-19.08.txt"):
        if embeddings:
            self.embeddings_index = embeddings
        else: 
            self.embeddings_index = self.load_embeddings(vec_file)

    def load_embeddings(self, file):
        f = open(file,'r', encoding = 'utf-8')
        embeddings_index = dict()
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print('Loaded %s word vectors.' % len(embeddings_index))
        return embeddings_index
    
    
    def embed_phrase(self, phrase):
        phrase = phrase.lower()
        if phrase in self.embeddings_index:
            return self.embeddings_index[phrase]
        elif re.sub(' ','_', phrase) in self.embeddings_index:
            return self.embeddings_index[re.sub(' ','_', phrase)]
        else:
            # average tokens embeddings
            phrase_tok = re.split("_| ", phrase)
            if len(phrase_tok) == 1:
                return np.zeros(shape=(300,))
            
            phrase_embed = np.zeros(shape=(300,))
            for i in range(1, len(phrase_tok)):
                substring = '_'.join(phrase_tok[i:])
                if substring in self.embeddings_index:
                    phrase_tok = phrase_tok[:i] + [substring]
            for tok in phrase_tok:
                if tok in self.embeddings_index:
                    phrase_embed += self.embeddings_index[tok]
            phrase_embed = phrase_embed/len(phrase_tok)
            return phrase_embed

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param tokenized_sents: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """
        sents_embed = [self.embed_phrase(tok) for tok in sents_tokened[0]]
        return np.array([sents_embed])

