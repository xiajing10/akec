# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:24:08 2020

@author: eilxaix
"""



import math
import nltk
import numpy as np
import re

from collections import Counter

from utils import lemmatize, embed_phrase
from evaluation import get_ranked_kplist, get_ranked_kpidx

from scipy.spatial.distance import cosine
from sklearn.metrics import silhouette_score,davies_bouldin_score
from spherecluster import SphericalKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# ELMO = word_emb_elmo.WordEmbeddings(cuda_device=-1)
# conceptnet = word_emb_conceptnet.WordEmbeddings()
# 
# =============================================================================
'''
embed_method = 'conceptnet', embed_model = embeddings_index
embed_method = 'w2v', embed_model = w2vmodel
embed_method = 'elmo', embed_model = ELMO

'''

def select_kp(data, score_dict, abrv_corpus, topn=15, filter_thre = 0.2):
    


    kplist = get_ranked_kplist(score_dict)
    kpidx = get_ranked_kpidx(score_dict)
    kpdata = []
    for i in kplist:
        for kp in kplist[i]:
            toks_len = len(lemmatize(kp).split('_'))
            if kpidx[kp] <= topn:
                if kp in abrv_corpus and toks_len == 1:
                    kpdata.append(lemmatize(abrv_corpus[kp]))
                else:
                    kpdata.append(lemmatize(kp))
                    
                    
    all_text = [lemmatize(i) for i in nltk.wordpunct_tokenize(' '.join(data['title+abs']).lower())]
    

    frq = Counter(all_text)
    idf = {}
    N = len(all_text)
    
    for (term,term_frequency) in frq.items():
        idf[term] = math.log(float(N) / term_frequency)
        
    tfidf={}
    for term in kpdata:
        toks = term.split('_')
        try:
            tfidf[term] = np.mean([frq[tok]*idf[tok] for tok in toks])
        except:
            pass
    
    filtered_kpdata = [i[0] for i in sorted(tfidf.items(), key=lambda x:x[1], reverse =True)[:int((1-filter_thre)*len(tfidf))]]
    
    return filtered_kpdata

def find_most_similar(kp, data, clus, n='all'):
    
    kpid = data.kp2id[kp]
    num = clus.membership[kpid]
    distance = 1-data.similarity
    idx = data.kp2id[kp]
    if n == 'all':
        topn = [data.id2kp[i] for i in distance[idx].argsort()[::] if data.id2kp[i] in clus.class2word[num]]
    else:
        topn = [data.id2kp[i] for i in distance[idx].argsort()[::] if data.id2kp[i] in clus.class2word[num]][:n]
        
    return topn

class ClusData:
    
    def __init__(self, clus_data, all_candidates_embed=None, embed_model = None, embed_method = 'conceptnet'):
        self.keyphrases = clus_data
        if all_candidates_embed:
            self.embed = []
            for w in self.keyphrases:
                if w in all_candidates_embed:
                    self.embed.append(all_candidates_embed[w])
                else:
                    self.keyphrases.remove(w)
            self.embed = np.array(self.embed)
        elif all_candidates_embed==None:
            if embed_method == 'w2v':
                self.embed = np.array([embed_phrase(embed_model, k) for k in self.keyphrases])
            if embed_method == 'conceptnet':
                self.embed = np.array([embed_phrase(embed_model, k) for k in self.keyphrases])
            if embed_method == 'elmo':
                self.embed = np.array([np.mean(embed_model.get_tokenized_words_embeddings([re.split("_| ", k)])[0],axis=0)
                                   for k in self.keyphrases])
        
        self.similarity = np.round(cosine_similarity(self.embed),decimals = 5)  
        self.id2kp = {idx:phrase for idx,phrase in enumerate(self.keyphrases)}
        self.kp2id = {j: i for i, j in self.id2kp.items()}

    
class SynData:
    
    def __init__(self, label2word, labelsdict, embed_model = None, embed_method = 'conceptnet'):
        self.label2word = label2word
        self.labelsdict = labelsdict
        self.labels = []
        
        self.keyphrases = []
        for i in self.label2word:
            self.keyphrases += self.label2word[i]
            self.labels += [i for _ in range(len(label2word[i]))]
            
        if embed_method == 'w2v':
            self.embed = np.array([embed_phrase(embed_model, k) for k in self.keyphrases])
        if embed_method == 'conceptnet':
            self.embed = np.array([embed_phrase(embed_model, k) for k in self.keyphrases])
        if embed_method == 'elmo':
            self.embed = np.array([np.mean(embed_model.get_tokenized_words_embeddings([re.split("_| ", k)])[0],axis=0)
                               for k in self.keyphrases])
        
        self.similarity = np.round(cosine_similarity(self.embed),decimals = 5)      
        self.id2kp = {idx:phrase for idx,phrase in enumerate(self.keyphrases)}
        self.kp2id = {j: i for i, j in self.id2kp.items()}
    

class Clusterer:

    def __init__(self, data, n_cluster=None, distance_threshold = None, method = 'sp-kmeans',affinity = 'euclidean',linkage="average"):
        self.data = data
        self.method = method
        self.n_cluster = n_cluster
        self.distance_threshold = distance_threshold
        if method == 'sp-kmeans':
            self.clus = SphericalKMeans(self.n_cluster, init='k-means++', random_state = 0, n_init=50)
        elif method == 'agglo':
            self.clus=AgglomerativeClustering(n_clusters=n_cluster, distance_threshold=self.distance_threshold, affinity=affinity, linkage=linkage)
        # cluster id -> members
        self.membership = None  # a list contain the membership of the data points
        self.center_ids = None  # a list contain the ids of the cluster centers
        self.class2word = {}
        self.inertia_scores = None
        self.sil_score = None
        self.db_score = None
        
    def fit(self):
        self.clus.fit(self.data.embed)
        labels = self.clus.fit_predict(self.data.embed)
        for idx, label in enumerate(labels):
            if label not in self.class2word:
                self.class2word[label] = []
            self.class2word[label].append(self.data.id2kp[idx])
        self.membership = labels
        if self.n_cluster == None:
            self.n_cluster = max(self.class2word.keys())
        if self.method == 'ap':
            self.center_ids = self.gen_center_idx()
        elif self.method == 'sp-kmeans':
            self.center_ids = self.gen_center_idx()
            self.inertia_scores = self.clus.inertia_
            print('Clustering concentration score:', self.inertia_scores)

        self.sil_score = silhouette_score(self.data.embed, self.membership, metric = 'cosine')
        self.db_score = davies_bouldin_score(self.data.embed, self.membership)
        print('Clustering silhouette score:', self.sil_score)
        print('Clustering davies bouldin score:', self.db_score)

    # find the idx of each cluster center
    def gen_center_idx(self):
        ret = []
        for cluster_id in range(self.n_cluster):
            center_idx = self.find_center_idx_for_one_cluster(cluster_id)
            ret.append((cluster_id, center_idx))
        return ret


    def find_opt_k_sil(self, n_cluster):
        sil = []
        kmax = n_cluster
    
        for k in range(2, kmax+1):
            kmeans = SphericalKMeans(n_clusters = k, init='k-means++', random_state = 0, n_init=50, n_jobs = -2).fit(self.data.embed)
            labels = kmeans.labels_
            sil.append(silhouette_score(self.data.embed, labels, metric = 'cosine'))
        
        k_opt = sil.index(max(sil)) + 2
        return k_opt
    
    def find_opt_k_db(self, n_cluster):
        db = []
        kmax = n_cluster
    
        for k in range(2, kmax+1):
            kmeans = SphericalKMeans(n_clusters = k, init='k-means++', random_state = 0, n_init=50, n_jobs = -2).fit(self.data.embed)
            labels = kmeans.labels_
            db.append(davies_bouldin_score(self.data.embed, labels))
        
        k_opt = db.index(min(db)) + 2
        return k_opt
       
    def find_center_idx_for_one_cluster(self, cluster_id):
        query_vec = self.clus.cluster_centers_[cluster_id]
        members = self.class2word[cluster_id]
        best_similarity, ret = -1, -1
        for member in members:
            member_idx = self.data.kp2id[member]
            member_vec = self.data.embed[member_idx]
            cosine_sim = self.calc_cosine(query_vec, member_vec)
            if cosine_sim > best_similarity:
                best_similarity = cosine_sim
                ret = member_idx
        return ret

    def calc_cosine(self, vec_a, vec_b):
        return 1 - cosine(vec_a, vec_b)
    
    
