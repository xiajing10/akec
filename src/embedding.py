# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:22:59 2020

@author: eilxaix
"""

# load the whole embedding into memory
import numpy as np
import re 
import nltk
from utils import lemmatize
porter = nltk.PorterStemmer()
from utils import lemmatize


def load_embeddings(file):
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

def embed_index(embeddings_index, phrase):
    if phrase in embeddings_index:
        return embeddings_index[phrase]
    elif re.sub(' ','_', phrase) in embeddings_index:
        return embeddings_index[re.sub(' ','_', phrase)]
    else:
        # average tokens embeddings
        phrase_tok = re.split("_| ", phrase)
        if len(phrase_tok) == 1:
            return np.zeros(shape=(300,))
        
        phrase_embed = np.zeros(shape=(300,))
        for i in range(1, len(phrase_tok)):
            substring = '_'.join(phrase_tok[i:])
            if substring in embeddings_index:
                phrase_tok = phrase_tok[:i] + [substring]
        for tok in phrase_tok:
            phrase_embed += embeddings_index[tok]
                
#             else KeyError:
#                 # longest prefix match
#                 for i in range(len(tok)-1, 0, -1):
#                     if tok[:i] in embeddings_index.keys():
#                         phrase_embed += embeddings_index[tok[:i]]
#                         break
        phrase_embed = phrase_embed/len(phrase_tok)
    return phrase_embed

def embed_phrase(phrase):
# =============================================================================
#     f = '../../numberbatch-en-19.08.txt'
#     conceptnet = load_embeddings(f)
#     embeddings_index = conceptnet
# =============================================================================
    phrase_embed = embed_index(embeddings_index, phrase)
    return phrase_embed
# =============================================================================
# cnmodel = gensim.models.KeyedVectors.load_word2vec_format('../numberbatch-en-19.08.txt', binary=False)
# =============================================================================
# =============================================================================
# import gensim
# temporary_filepath = '../conceptnet_w2v.model'
# cnmodel = new_model = gensim.models.KeyedVectors.load(temporary_filepath)
# =============================================================================
