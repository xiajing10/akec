# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:57:40 2020

@author: eilxaix
"""

# Document Relevance
# -*- coding: utf-8 -*-
import re
import json
import time
import torch
import spacy
en_model = spacy.load("en")

import numpy as np
import pandas as pd

from tqdm import tqdm
from collections import defaultdict

from embeddings import sent_emb_sif, word_emb_elmo, word_emb_conceptnet, word_emb_bert
from model.method import SIFRank

import utils
from utils import load_embeddings, minmax,text_normalize, embed_phrase

from param import param
from extraction import acronym_extraction
from pmi_entropy import phrase_quality


'''
conceptnet_emb = param['conceptnet_emb']
embeddings_index = load_embeddings(conceptnet_emb)


options_file = "./embed_data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "./embed_data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"
ELMO = word_emb_elmo.WordEmbeddings(options_file, weight_file, cuda_device=-1)
conceptnet = word_emb_conceptnet.WordEmbeddings(embeddings=embeddings_index)
bert = word_emb_bert.WordEmbeddings()
'''

with open(param['terms_path'],'r', encoding = 'utf-8') as f:
    domain_list = [i.strip().lower() for i in f.readlines()]


'''
Document Relevance Score
'''

def SIFRank_score(text, embed_method, embed_model):
    if embed_method == 'elmo':
        SIF = sent_emb_sif.SentEmbeddings(embed_model, lamda=1.0, embeddings_type="elmo")
    if embed_method == 'bert':
        SIF = sent_emb_sif.SentEmbeddings(embed_model, lamda=1.0, embeddings_type="bert")
    elif embed_method == 'conceptnet':   
        SIF = sent_emb_sif.SentEmbeddings(embed_model, lamda=1.0, embeddings_type="conceptnet")

    score_dict = defaultdict()
    phrase_embed = defaultdict()
    
    for i, d in tqdm(enumerate(text), total = len(text)):
        score_dict[i], phrase_embed[i]  = SIFRank(d, SIF, en_model, N='all', \
                                                  sent_emb_method=embed_method)
    
    return score_dict, phrase_embed


def textrank_score(text, N=15):
    import pke
    textrank={}
    for i in tqdm(range(len(text)), total = len(text)):
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(input=text[i])
        extractor.candidate_selection()
        extractor.candidate_weighting()
        kp = extractor.get_n_best(n=N)
        extracted_kp = {i[0]:i[1] for i in kp}
        textrank[i]=extracted_kp
        
    return textrank

def document_relevance_score(text, titles, embed_model, rank_method = 'sifrank', embed_method='conceptnet', title_w = 1):
    '''
    text: list of input text, data['title+abs']
    titles: list of titles, data['titles']
    '''
    print("rank_method:", rank_method)
    print("embed_method:", embed_method)
    if rank_method == 'sifrank':
        score_dict, phrase_embed = SIFRank_score(text, embed_method, embed_model)
    elif rank_method == 'textrank':
        score_dict = textrank_score(text)
        phrase_embed = {}
        for i in score_dict:
            phrase_embed[i] = {}
            for w in score_dict[i]:
                if embed_method == 'elmo':
                    tokens_segmented = re.split("_| ", w)
                    phrase_embed[i][w], _ = embed_model.get_tokenized_words_embeddings(tokens_segmented)
                elif embed_method == 'bert':
                    tokens_segmented = re.split("_| ", w)
                    phrase_embed[i][w]= embed_model.get_tokenized_words_embeddings(tokens_segmented)                    
                else:
                    phrase_embed[i][w] = embed_model(w)
    
    # title weighted
    print("Adding title weights...")
    if title_w > 0:
        title_weighted_score = {}
        for i in score_dict:
            title_weighted_score[i] = {}
            for term in score_dict[i]:
                term_len = len(term.split())
                score = score_dict[i][term]
                t = titles[i].lower()
                if term_len > 1:
                    try:
                        if re.findall(r'\b{}\b'.format(term), t):
                            title_weighted_score[i][term] = title_w*term_len*score
                        else:
                            title_weighted_score[i][term] = score
                    except:
                        title_weighted_score[i][term] = score
                else:
                    title_weighted_score[i][term] = score
                    
            title_weighted_score[i] = minmax(title_weighted_score[i])
    
        return title_weighted_score, phrase_embed
    else:
        return score_dict, phrase_embed
    
    
'''
Domain Relevance Score
'''
    
def domain_relevance_table(all_candidates_embed, domain_list, embed_model, embed_method = 'conceptnet', N=0):
    N=int(0.5*len(domain_list))
    if embed_method == 'conceptnet':
        domain_embed = {w: embed_model.embed_phrase(w) for w in domain_list}
    elif embed_method == 'elmo':
        domain_embed = {}
        for term in domain_list:
            tokens_segmented = re.split("_| ", term)
            embed, _ = embed_model.get_tokenized_words_embeddings([tokens_segmented])
            domain_embed[term] = torch.mean(embed[0],dim=1)
    elif embed_method == 'bert':
        domain_embed = {}
        for term in domain_list:
            tokens_segmented = re.split("_| ", term)
            embed = embed_model.get_tokenized_words_embeddings([tokens_segmented])
            domain_embed[term] = np.mean(embed[0],axis=0)
    
    domain_score = defaultdict(float)
    cos_sim_table = defaultdict(dict)
    for w in tqdm(all_candidates_embed, total = len(all_candidates_embed)):
        if not cos_sim_table[w]:
            cos_sim_table[w] = defaultdict(dict)
        w_embed = all_candidates_embed[w]
        domain_score[w] = 0
        score_list = []
        for seed_embed in domain_embed.items():
            if cos_sim_table[w][seed_embed[0]]:
                score = cos_sim_table[w][seed_embed[0]]
            score = utils.get_dist_cosine(w_embed, seed_embed[1], embed_method)
            score_list.append(score)
            cos_sim_table[w][seed_embed[0]] = score
        domain_score[w] = np.mean(sorted(score_list, reverse = True)[:N])
    return domain_score, cos_sim_table


def domain_relevance_score(phrase, domain_embed, embed_model, embed_method, N=0):
    N=int(0.5*len(domain_list))
    phrase = utils.lemmatize(phrase)
    phrase_embd = embed_model.embed_phrase(phrase)

    score = 0
    score_list = []
    
    for seed_embed in domain_embed.items():
        score = get_dist_cosine(phrase_embd, seed_embed[1], embed_method)
        score_list.append(score)
    
    score = np.mean(sorted(score_list, reverse = True)[:N])
    
    return score

'''
Phrase Quality Score
'''

# load pre-calculated doc
def get_abrv(text):
    abrv_kp = acronym_extraction(text)
    abrv_corpus = {}
    for i in abrv_kp:
        abrv = []
        if len(abrv_kp[i]) != 0:
            for j in abrv_kp[i]:
                key = j[0].lower()
                key = re.sub('-',' ', key)
                ta = text_normalize(j[1].lower())
                abrv += [key,ta]
                abrv_corpus[key] = ta
            abrv_kp[i] = [w.lower() for w in abrv]
    return abrv_kp, abrv_corpus
            
# =============================================================================
# with open('../../dataset/ieee_xai/abrv.json', 'r') as f:
#     fi = json.loads(f.read())
#     abrv_kp = {}
# =============================================================================

                      


def score_table(data, domain_list, embed_model, \
                rank_method = 'sifrank', embed_method='conceptnet'):

    print("Calulating document relevance score...")
    if embed_method == 'elmo+conceptnet':
        document_score, _ = document_relevance_score(data['title+abs'], data['title'], embed_model, \
                                                     rank_method, embed_method = 'elmo', title_w = 1)
        _, candidates_embed = document_relevance_score(data['title+abs'], data['title'], embed_model, \
                                                       rank_method, embed_method = 'conceptnet', title_w = 1)
        embed_method = 'conceptnet'
    else:
        document_score, candidates_embed = document_relevance_score(data['title+abs'], data['title'], embed_model, \
                                                                    rank_method, embed_method, title_w = 1)
    all_candidate_embed = {}
    for i in candidates_embed:
        for w in candidates_embed[i]:
            embed = candidates_embed[i][w].squeeze()
            if w not in all_candidate_embed:
                all_candidate_embed[w] = embed
            else:
                all_candidate_embed[w] =  (all_candidate_embed[w] +embed)/2
    
    print("Calulating domain relevance score...")
    domain_score,cos_sim_table = domain_relevance_table(all_candidate_embed, domain_list, embed_model, embed_method)
    
    
    return document_score, domain_score, all_candidate_embed

def quality_score_table(text, document_score, normalized_quality, alpha, beta):
    
    quality_score = {}
    abrv_kp, abrv_corpus = get_abrv(text)
    
    print("Calulating phrase qualiy score...")
    for docid in document_score:
        
        c_list = [c[0] for c in sorted(document_score[docid].items(), key=lambda x:x[1],reverse=True)]

        quality_score[docid] = {}
        for t in c_list:
            
            toks_len = len(t.split())
            score = 0
            if not 2 <= toks_len <= 3: 
                score = - beta *(np.abs(toks_len-3))
            
            if t in normalized_quality:
                score += normalized_quality[t] 
            if str(docid) in abrv_kp and t in abrv_kp[str(docid)]:
                score += alpha
            
            quality_score[docid][t]=score

        quality_score[docid] = minmax(quality_score[docid])
    
    return quality_score

def domain_socre_table(document_score, domain_score):
    dom = {}

    for docid in document_score:
        
        c_list = [c[0] for c in sorted(document_score[docid].items(), key=lambda x:x[1],reverse=True)]

        dom[docid] = {}
        
        for w in c_list:
            dom[docid][w] = domain_score[w]
            
        dom[docid] = minmax(dom[docid])
        
    return dom
        
def weighted_ranking_score(document_score, doamin_score_table, quality_score_table, domain_w=0.1, quality_w=0.1):
    
    final_score = {}
    
    print("Calulating weighted ranking score...")
    for docid in document_score:
        
        c_list = [c[0] for c in sorted(document_score[docid].items(), key=lambda x:x[1],reverse=True)]

        final_score[docid] = {}
        for t in c_list:
            
#             document_w = 1-(domain_w+quality_w)
            document_w = 1

#             s = document_score[docid][t] + domain_score[t] + quality_score
            s = document_w*document_score[docid][t] + \
                domain_w*doamin_score_table[docid][t] + \
                quality_w*quality_score_table[docid][t]
            
            final_score[docid][t] = s 
        
       
    return final_score       
    
def rank(data, domain_list, pmi_en = None, \
         domain_w=0.1, quality_w=0.1, alpha = 1, beta = 0.5, \
         rank_method = 'sifrank', embed_method='conceptnet'):
    
    start = time.time()
    
    if pmi_en:       
        with open(pmi_en, 'r') as f:
            pmi_entropy = json.loads(f.read())
    else:
        pmi_entropy = phrase_quality(data['title+abs'])
        
    pmi_dict = pmi_entropy['pmi']
    entropy_dict = pmi_entropy['entropy']
    
    normalized_quality = minmax({i: entropy_dict[i] for i in entropy_dict if pmi_dict[i] > 2})

    if embed_method == 'conceptnet':
        conceptnet_emb = param['conceptnet_emb']
        embeddings_index = load_embeddings(conceptnet_emb)
        embed_model = word_emb_conceptnet.WordEmbeddings(embeddings=embeddings_index)
    elif embed_method == 'elmo':
        embed_model = word_emb_elmo.WordEmbeddings(param['elmo_options'], param['elmo_weight'], cuda_device=-1)
    elif embed_method == 'bert':
        embed_model = word_emb_bert.WordEmbeddings()
    else:
        print('Undefined embedding method, please choose from [conceptnet, elmo, bert]')
        raise

    document_score, domain_score, con_candidate_embed = score_table(data, domain_list, embed_model, rank_method = 'sifrank', embed_method=embed_method)
    domainrank = domain_socre_table(document_score, domain_score)
    qualityrank = quality_score_table(data['title+abs'], document_score, normalized_quality, alpha, beta)
    
    final_score =  weighted_ranking_score(document_score, domainrank, qualityrank, \
                                          domain_w=domain_w, quality_w=quality_w)  
    
    end = time.time()
    print("Time cost:", end-start) 
    
    return final_score

    


    


