#! /usr/bin/env python
# -*- coding: utf-8 -*-
# __author__ = "Sponge"
# Date: 2019/6/19
from allennlp.commands.elmo import ElmoEmbedder
from bert_serving.client import BertClient
import numpy as np



class WordEmbeddings():
    """
        ELMo
        https://allennlp.org/elmo

    """

    def __init__(self, cuda_device=-1):
        self.cuda_device=cuda_device
        self.bert = BertClient()

    def get_tokenized_words_embeddings(self, sents_tokened):
        """
        @see EmbeddingDistributor
        :param tokenized_sents: list of tokenized words string (sentences/phrases)
        :return: ndarray with shape (len(sents), dimension of embeddings)
        """

        bert_embedding = self.bert.encode(sents_tokened[0])
        
        return np.array([bert_embedding])


