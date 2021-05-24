# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 14:18:44 2020

@author: eilxaix
"""
import json
import re
import torch
import numpy as np

from sklearn import preprocessing
from constants import STOPWORDS
stop_words = STOPWORDS

def read_json(path):
    with open(path, 'r') as fin:
        return json.loads(fin.read())


# def cossim(vec_a, vec_b):
#     if vec_a.ndim == 1 and vec_b.ndim == 1:
#         return 1-cosine([vec_a],[vec_b])
#     elif vec_a.ndim == 2 and vec_b.ndim == 2:
#         return 1-cosine(vec_a,vec_b)
    
# normalize to [0,1]
def cos_sim(vector_a, vector_b):

    
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if(denom==0.0):
        return 0.0
    else:
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim
    
def l2_sim(vector_a, vector_b):
    return 1/(1+np.linalg.norm(vector_a-vector_b))
    
def get_dist_cosine(emb1, emb2, emb_method="elmo", elmo_layers_weight=[0.0,1.0,0.0]):
    sum = 0.0

    if(emb_method=="elmo"):
        
        assert emb1.shape == emb2.shape

        for i in range(0, 3):
            a = emb1[i]
            b = emb2[i]
            sum += cos_sim(a, b) * elmo_layers_weight[i]
        return sum

    elif(emb_method=="conceptnet"):
        sum=cos_sim(emb1,emb2)
        return sum

    elif (emb_method == "w2v"):
        sum = cos_sim(emb1, emb2)
        return sum
    
    return sum

def lemmatize(chunk):
    import re 
    from nltk.stem.wordnet import WordNetLemmatizer

    text = chunk
    
    #Remove punctuations
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # remove special characters and digits
    text=re.sub('\W+', ' ', text)
    
    text=re.sub('  ', ' ', text)

    ##Convert to list from string
    text = text.split()

    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = "_".join(text)
    return text

def text_normalize(chunk):
    text = chunk.lower()
    
    #Remove punctuations
    text = re.sub('[^a-zA-Z0-9]', ' ', text)
    
    text = re.sub('  ', ' ', text)

    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    return text

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

def embed_phrase(embeddings_index, phrase):
    phrase = phrase.lower()
    if phrase in embeddings_index:
        return embeddings_index[phrase]
    elif re.sub(' ','_', phrase) in embeddings_index:
        return embeddings_index[re.sub(' ','_', phrase)]
    else:
        # average tokens embeddings
        phrase_tok = [tok for tok in re.split("_| ", phrase)]
        if len(phrase_tok) == 1:
            return np.zeros(shape=(300,))
        
        phrase_embed = np.zeros(shape=(300,))
        for i in range(1, len(phrase_tok)):
            substring = '_'.join(phrase_tok[i:])
            if substring in embeddings_index:
                phrase_tok = phrase_tok[:i] + [substring]
        for tok in phrase_tok:
            if tok in embeddings_index:
                phrase_embed += embeddings_index[tok]
        phrase_embed = phrase_embed/len(phrase_tok)
    return phrase_embed

def minmax(score_dict):
    keys = []
    values = []
    for key in score_dict:
        keys.append(key)
        values.append(score_dict[key])
    values = preprocessing.minmax_scale(values)
    new_dict = {keys[i]:values[i] for i in range(len(keys))}
    return new_dict


