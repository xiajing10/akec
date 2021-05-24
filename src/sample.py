# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:17:21 2020

@author: eilxaix
"""

import pandas as pd
import re

def remove_hashtag(t):
    t=re.sub('-',' ', t)
    t=' '.join(t.split())
    return t

def read_csv_data(df):
    title = [remove_hashtag(i) for i in df['Document Title']]
    abstract = [remove_hashtag(i) for i in df['Abstract']]
    doc = [title[i] + '. ' + abstract[i] for i in range(len(df))]
    inspec_controlled = [remove_hashtag(i) for i in df['INSPEC Controlled Terms']]
    inspec_uncontrolled = [remove_hashtag(i) for i in df['INSPEC Non-Controlled Terms']]
    for i in range(len(inspec_uncontrolled)):
        inspec_uncontrolled[i] = [k.lower() for k in inspec_uncontrolled[i].split(';')]
    for i in range(len(inspec_controlled)):
        inspec_controlled[i] = [k.lower() for k in inspec_controlled[i].split(';')]
    data = {'title': title, 'abstract': abstract, 'title+abs': doc, 'inspec_controlled': inspec_controlled,'inspec_uncontrolled':inspec_uncontrolled}      

    return data

# =============================================================================
# data = read_csv_data(pd.read_csv('../../dataset/ieee_xai/ieee_xai.csv'))
# =============================================================================
