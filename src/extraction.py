# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 22:54:00 2020

@author: eilxaix
"""

import re
import nltk
import spacy
# Load the spacy model that you have installed
nlp = spacy.load("en")
scinlp = spacy.load("en_core_sci_sm")


from scispacy.abbreviation import AbbreviationDetector
abbreviation_pipe = AbbreviationDetector(nlp)
scinlp.add_pipe(abbreviation_pipe)


import string
punct = string.punctuation

# =============================================================================
# from nltk import word_tokenize
# stop_words = set(stopwords.words("english"))
# =============================================================================
from constants import STOPWORDS
# stop_words=nltk_stopwords
stop_words=STOPWORDS

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
lem = WordNetLemmatizer()
ps=PorterStemmer()
POS_BLACKLIST = ['INTJ', 'AUX', 'CCONJ', 'ADP', 'DET', 'NUM', 'PART','PRON', 'SCONJ', 'PUNCT','SYM', 'X']

from collections import Counter, defaultdict


def lemmatize(chunk):
    tokens = []
    if type(chunk) == spacy.tokens.span.Span:
        for token in [chunk[0], chunk[-1]]:
            if token.pos_ not in POS_BLACKLIST:
                tokens.append(token.text)
        text = ' '.join(tokens).lower()
    else:
        text = chunk
    
    #Remove punctuations
    text = re.sub('[^a-zA-Z0-9]', ' ', text)

    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    # remove special characters and digits
#     text=re.sub("(\\d|\\W)+"," ",text)

    ##Convert to list from string
    text = text.split()

    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)
    return text

def preprocess(chunk):
    tokens = []
    if type(chunk) == spacy.tokens.span.Span:
        for token in [chunk[0], chunk[-1]]:
            if token.pos_ not in POS_BLACKLIST:
                tokens.append(token.text)
        text = ' '.join(tokens).lower()
    else:
        text = chunk
    text = text.lower()
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    token = re.split(' ',text)
    text = [tok for tok in token if not tok in stop_words and not tok in punct] 
    text = " ".join(text)
    return text    

def np_extraction(text):
    nounphrases = []
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        candidate = preprocess(chunk.text)
        if candidate != '':
#             all_text[i] = all_text[i].replace(chunk.text, text)
            nounphrases.append(candidate)
    return nounphrases

def find_subset_np(nounphrases):
    subset = []
    for np in nounphrases:
        length = len(np.split())
        if length > 4:
            subnp = np_extraction(np)
            subset += [i for i in subnp if i not in nounphrases]
    return subset

        
# =============================================================================
# from stanfordcorenlp import StanfordCoreNLP
# en_model = StanfordCoreNLP(r'../stanford-corenlp-full-2018-10-05',quiet=True)
# =============================================================================

def pos_tag_standford(text, en_model,stop_words):
    tokens = en_model.word_tokenize(text)
    tokens_tagged = en_model.pos_tag(text)
    for i, token in enumerate(tokens):
        if token.lower() in stop_words:
            tokens_tagged[i] = (token, "IN")
    return tokens, tokens_tagged

def pos_tag_spacy(text, nlp, stop_words):

    doc = nlp(text)
    tokens = []
    tags = []
    tokens_tagged = []
    for tok in doc:
        tokens.append(tok.text)
        tokens_tagged.append((tok.text,tok.tag_))
        tags.append('<%s>' % tok.tag_ )
    return tokens, tags, tokens_tagged

    
def extract_candidates(tokens, tokens_tagged, no_subset=True):

    """
    Based on part of speech return a list of candidate phrases
    :param text_obj: Input text Representation see @InputTextObj
    :param no_subset: if true won't put a candidate which is the subset of an other candidate
    :return keyphrase_candidate: list of list of candidate phrases: [tuple(string,tuple(start_index,end_index))]
    """
    GRAMMAR1 = """  NP:
        {<NN.*|JJ|CD>*<NN.*>}   # Adjective(s)(optional) + Noun(s)"""
       
    GRAMMAR2 = """  NP:
        {<JJ|VBG>*<NN.*>{0,3}}  # Adjective(s)(optional) + Noun(s)"""
    
    GRAMMAR3 = """  NP:
        {<NN.*|JJ|VBG|VBN|CD>*<NN.*>}  # Adjective(s)(optional) + Noun(s)"""
     
    GRAMMAR = """
            NP:
              {<NN.*|JJ|CD>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns
            VBNP:
              {<VBG|VBN>*<NN.*>} # Above, connected with in/of/etc...
            """

    for i, token in enumerate(tokens):
        if token.lower() in stop_words:
            tokens_tagged[i] = (token, "IN")
    
    np_parser = nltk.RegexpParser(GRAMMAR1)  # Noun phrase parser
    keyphrase_candidate = {}
    np_pos_tag_tokens = np_parser.parse(tokens_tagged)
    count = 0
    for token in np_pos_tag_tokens:
        if (isinstance(token, nltk.tree.Tree) and token._label == "NP"):
            np = ' '.join(word for word, tag in token.leaves()).lower()
            length = len(token.leaves())
            if np in keyphrase_candidate:
                keyphrase_candidate[np].append((count, count + length)) 
            else:
                keyphrase_candidate[np] = [(count, count + length)]
            count += length

        else:
            count += 1

    return keyphrase_candidate


def find_acronyms(text):
    doc = scinlp(text)
    abrv_list = []
    altered_tok = [tok.text for tok in doc]
    for abrv in doc._.abbreviations:
        abrv_list.append((abrv.text, str(abrv._.long_form)))

    return list(set(abrv_list))

def replace_acronyms(text):
    doc = nlp(text)
    altered_tok = [tok.text for tok in doc]
    for abrv in doc._.abbreviations:
        altered_tok[abrv.start] = str(abrv._.long_form)

    return(" ".join(altered_tok))

def filter_unigram(nps, abrv):
    nps_validate = []
    for np in nps:
        length = len(np.split())
        if length > 1 or np in abrv:
            nps_validate.append(np)
    return nps_validate

def no_punct(word):
    for tok in word.split():
        if re.match('\W+', tok):
            return False
    return True 

def filtered_pos(words, words_tags_dict):
    pattern = re.compile('^(<JJ>|<NN>)*(<NN>|<NNS>|<NNP>)+$')
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
 
def n_gram_words(tokens, tags, n_gram = 8):
    """
    To get n_gram word frequency dict
    input: tokens list,int of n_gram 
    output: word frequency dict

    """
    words = []
    words_tags = []
    words_tags_dict = defaultdict(list)
    for i in range(1,n_gram+1):
        words += [' '.join(tokens[j:j+i]) for j in range(len(tokens)-i+1)]
        words_tags += [''.join(tags[j:j+i]) for j in range(len(tags)-i+1)]
    for w in range(len(words)):
        words_tags_dict[words[w]].append(words_tags[w])
# =============================================================================
#     words_freq = dict(Counter(words))    
# =============================================================================
    words = [w for w in filtered_pos(words, words_tags_dict) if no_punct(w)]
    return list(set(words)) 

def candidates_extraction(data):
    # input: [data] - a list of abstract
    # output: candidates:{'nps':[], 'sub_nps':[], 'abrv':[]}
    candidates={}
    abrv_corpus = {}
    for i,text in enumerate(data):
        # find_acronyms
        abrv_list = find_acronyms(text)  
        abrv_corpus[i]=abrv_list
        abrv = [i[0] for i in abrv_list]
        
        # pos tagging
        tokens, tags, tokens_tagged = pos_tag_spacy(text, nlp, stop_words)
        # ngram + pos matching '(<JJ>)*(<NN>|<NNS>|<NNP>)'
        ngrams = n_gram_words(tokens, tags, n_gram = 8)

        # np chunking
        # nps = np_extraction(text)
        nps = extract_candidates(tokens, tokens_tagged)

        candidates[i] = {'nps':nps, 'ngrams': ngrams, 'abrv':abrv}
    return candidates, abrv_corpus
    
def acronym_extraction(data):
    # input: [data] - a list of abstract
    # output: abrv：{'id':[]}, abrv_corpus: {'ai': ..., }
    abrv={}
    for i,text in enumerate(data):
        # find_acronyms
        abrv_list = find_acronyms(text)  
        abrv[i] = [[i[0],i[1]] for i in abrv_list]
    return abrv