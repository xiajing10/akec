# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:47:38 2020

@author: eilxaix
"""

import nltk
import re
from tqdm import tqdm
# =============================================================================
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words("english"))
# =============================================================================
from constants import STOPWORDS
import string
punct = string.punctuation
import spacy
# Load the spacy model that you have installed
nlp = spacy.load("en")


from collections import Counter, defaultdict
import numpy as np

def pos_tag_spacy(text, nlp, stop_words):

    doc = nlp(text)
    tokens = []
    tags = []
    tokens_tagged = []
    for tok in doc:
        tokens.append(tok.text)
        tokens_tagged.append((tok.text,tok.tag_))
        tags.append('<%s>' % tok.tag_ )
    for i, token in enumerate(tokens):
        if token.lower() in stop_words:
            tokens_tagged[i] = (token, "IN")
    return tokens, tags, tokens_tagged

def filtered_pos(words, words_tags_dict):
    pattern = re.compile('^(<JJ>)*(<NN>|<NNS>|<NNP>)+$')
    filtered = []
    for w in words:
        tags = words_tags_dict[w]
        match = True
        for tag in tags:
            if pattern.match(tag) is None:
                match = False
                break
        if match == True:
            filtered.append(w)
    return filtered


# finding the new words 

"""
To get n_gram word frequency dict
input: tokens list,int of n_gram 
output: word frequency dict

"""

def n_gram_words(tokens, tags, max_n):
    """
    To get n_gram word frequency dict
    input: tokens list,int of n_gram 
    output: word frequency dict

    """
    words = []
    words_tags = []
    words_tags_dict = defaultdict(list)
    for i in range(1, max_n+1):
        words += [' '.join(tokens[j:j+i]) for j in range(len(tokens)-i+1)]
        words_tags += [''.join(tags[j:j+i]) for j in range(len(tags)-i+1)]
    for w in range(len(words)):
        words_tags_dict[words[w]].append(words_tags[w])
    words_freq = dict(Counter(words))    
    words = [w for w in filtered_pos(words, words_tags_dict) if no_punct(w)]
    return words_freq, list(set(words))


def no_punct(word):
    for tok in word.split():
        if re.match('\W+', tok):
            return False
    return True  

def PMI_filter(word_freq_dic, words):
    """
    To get words witch  PMI  over the threshold
    input: word frequency dict , min threshold of PMI
    output: condinated word list

    """
    new_words = []
    pmi_dict = {}
    for word in words:
        toks = word.split()
        word_len = len(toks)
        if word_len == 1:
            pass
        else:
            p_x_y = min([word_freq_dic.get(' '.join(toks[:i]))* word_freq_dic.get(' '.join(toks[i:])) for i in range(1,word_len)])
            pmi = np.log2(p_x_y/word_freq_dic.get(word))
            pmi_dict[word] = pmi
            new_words.append(word)
    return new_words, pmi_dict


def calculate_entropy(char_list):
    """
    To calculate entropy for  list  of char
    input: char list 
    output: entropy of the list  of char
    """
    char_freq_dic =  dict(Counter(char_list)) 
    entropy = (-1)*sum([char_freq_dic.get(i)/len(char_list)*np.log2(char_freq_dic.get(i)/len(char_list)) for i in char_freq_dic])
    return entropy

def find_left_right_tok(word, text):
    match = re.finditer(word,text)
    span_list = [i.span() for i in match]
    left_right_toks = []
    for span in span_list:
        left_span = text[:span[0]].strip() 
        right_span = text[span[1]:].strip()
        left = left_span.split()[-1] if len(left_span) != 0 else ''
        right = right_span.split()[0] if len(right_span) != 0 else ''
        left_right_toks.append((left,right))
    return left_right_toks

def word_entropy(word, entropy_dict, text):
    '''
    return min (left,right) entroy of a word.
    '''
    if word in entropy_dict:
        return entropy_dict[word]
    else:
        left_right_char = find_left_right_tok(word, text)

        left_char = [i[0] for i in left_right_char] 
        left_entropy = calculate_entropy(left_char)

        right_char = [i[1] for i in left_right_char]
        right_entropy = calculate_entropy(right_char)

        return min(right_entropy,left_entropy)

def find_inner_entropy(word, entropy_dict,text):
    toks = word.split()

    if len(toks) == 2:
        return 0
    else:
        entropy = float("+inf")
        
        for i in range(1, len(toks)):
            left_tok, right_tok = toks[:i], toks[i:]
            
            entropy = min(word_entropy(' '.join(left_tok), entropy_dict,text), word_entropy(' '.join(right_tok), entropy_dict,text), entropy)
    
    return entropy


def entropy_left_right_filter(condinate_words, text, entropy_dict):
    """
    To filter the final new words from the condinated word list by entropy threshold
    input:  condinated word list ,min threshold of Entropy of left or right
    output: final word list
    """

    # inner_outer_score = {}
    
    for word in condinate_words:
        try:
            # inner_entropy = find_inner_entropy(word, entropy_dict, text)
            outer_entroy = word_entropy(word, entropy_dict, text)

            entropy_dict[word] = outer_entroy

            # inner_outer_score[word] = outer_entroy - inner_entropy

        except:
            pass

        
    return entropy_dict     
   # return entropy_dict,inner_outer_score



def pmi_entropy_score(text, max_n = 6, ):
    '''
    text: list of text
    '''
    text_tokenized = [i.lower() for i in text]
    text_corpus =' '.join(text_tokenized)
    text_tokens = [tok for tok in nltk.wordpunct_tokenize(text_corpus) if not re.match('\W+', tok)]
    cleaned_tokens = [tok for tok in text_tokens if tok not in STOPWORDS]
    cleaned_text = ' '.join(cleaned_tokens)
    tokens, tags, tokens_tagged = pos_tag_spacy(text_corpus, nlp, STOPWORDS)
    

    score_dict = {}  
    entropy_dict = {}
    
    print("Getting n-grams ...")
    score_dict['frq'], words = n_gram_words(tokens,tags, max_n)
    print(len(words))
    print()
    print("Calculating PMI ...")
    pmi_filtered, score_dict['pmi'] = PMI_filter(score_dict['frq'], words)
    print(len(pmi_filtered))
    print()
    
    print("Calculating entropy ...")
    # score_dict['entropy'], score_dict['inner-outer']  = \
        # Entropy_left_right_filter(pmi_filtered, cleaned_text, entropy_dict)
    score_dict['entropy'] = entropy_left_right_filter(pmi_filtered, cleaned_text, entropy_dict)
    
    return score_dict

def phrase_quality(text):
    score_dict = pmi_entropy_score(text)
    return score_dict
# =============================================================================
#     with open('./pmi_entropy_score.json' ,'w') as f:
#         f.write(json.dumps(score_dict))
# =============================================================================
