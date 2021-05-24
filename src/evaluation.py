# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 21:22:09 2020

@author: eilxaix
"""

import nltk
import re
import numpy as np
from utils import lemmatize
porter = nltk.PorterStemmer()

def candidates_eval(data,candidates):
    porter = nltk.PorterStemmer()
    
    num_s = 0
    num_p = 0
    count=0
    
    for i in range(len(candidates)):
    
        labels = data['inspec_uncontrolled'][i]
        labels_stemmed = []
    
        for label in labels:
            tokens = re.split("_| ", lemmatize(label))
            labels_stemmed.append(' '.join(porter.stem(t) for t in tokens))
    
        dist_candidates = candidates[i]['nps']
# =============================================================================
#         dist_candidates = list(candidates[i].keys())
# =============================================================================
    
    
        for temp in dist_candidates:
            tokens = re.split("_| ", temp.lower())
            tt = ' '.join(porter.stem(t) for t in tokens)
            tt = re.sub('[^A-Za-z0-9]+', ' ', tt)
            tt = re.sub('  ', ' ', tt)
            if (tt in labels_stemmed or temp in labels):
                count += 1
    
        num_s += len(labels)
        num_p += len(dist_candidates)

    print(float(count) / float(num_s))
    print((float(count) / float(num_p)))
    
    
def get_ranked_kplist(score_dict):
    kp_list = {}
    for i in score_dict:
        sublist = [t[0] for t in sorted(score_dict[i].items(), key = lambda x:x[1], reverse = True)]
        kp_list[int(i)] = sublist
    return kp_list

def get_ranked_kpidx(score_dict):
    kp_list = {}
    for i in score_dict:
        sublist = [t[0] for t in sorted(score_dict[i].items(), key = lambda x:x[1], reverse = True)]
        kp_list[int(i)] = sublist
        
    kp_rankidx = {}
    for i in kp_list:
        for kp in kp_list[i]:        
            if kp in kp_rankidx:
                kp_rankidx[kp].append(kp_list[i].index(kp))
            else:
                kp_rankidx[kp] = [kp_list[i].index(kp)]
                
    for kp in kp_rankidx:
        kp_rankidx[kp] = np.mean(kp_rankidx[kp])
        
    return kp_rankidx

def evaluate(ranked_list, data):
    num_c_5 = num_c_10 = num_c_15 = num_c_20 = 0
    num_e_5 = num_e_10 = num_e_15 = num_e_20 = 0
    num_s = 0
    lamda = 0.0
    
    def get_PRF(num_c, num_e, num_s):
        P = R = F1 = 0.0
        P = float(num_c) / float(num_e)
        R = float(num_c) / float(num_s)
        if (P + R == 0.0):
            F1 = 0
        else:
            F1 = 2 * P * R / (P + R)
        return P, R, F1

    def print_PRF(P, R, F1, N):

        print("\nN=" + str(N), end="\n")
        print("P=" + str(P), end="\n")
        print("R=" + str(R), end="\n")
        print("F1=" + str(F1))
        return 0
    
    for i in range(len(ranked_list)):

        labels = data['inspec_uncontrolled'][i]
        labels_stemmed = []

        for label in labels:
            tokens = re.split("_| ", label.lower())
            labels_stemmed.append(' '.join(porter.stem(t) for t in tokens))

        dist_sorted = ranked_list[i]

        j = 0
        for temp in dist_sorted[0:20]:
            tokens = re.split("_| ", temp.lower())
            tt = ' '.join(porter.stem(t) for t in tokens)
            tt = re.sub('[^A-Za-z0-9]+', ' ', tt)
            tt = re.sub('  ', ' ', tt)
            if (tt in labels_stemmed or temp in labels):
                if (j < 5):
                    num_c_5 += 1
                    num_c_10 += 1
                    num_c_15 += 1
                    num_c_20 += 1

                elif (j < 10 and j >= 5):
                    num_c_10 += 1
                    num_c_15 += 1
                    num_c_20 += 1

                elif (j < 15 and j >= 10):
                    num_c_15 += 1
                    num_c_20 += 1

                elif (j < 20 and j >= 15):
                    num_c_20 += 1

            j += 1

        if (len(dist_sorted[0:5]) == 5):
            num_e_5 += 5
        else:
            num_e_5 += len(dist_sorted[0:5])

        if (len(dist_sorted[0:10]) == 10):
            num_e_10 += 10
        else:
            num_e_10 += len(dist_sorted[0:10])

        if (len(dist_sorted[0:15]) == 15):
            num_e_15 += 15
        else:
            num_e_15 += len(dist_sorted[0:15])

        if (len(dist_sorted[0:20]) == 20):
            num_e_20 += 20
        else:
            num_e_20 += len(dist_sorted[0:20])

        num_s += len(labels)
    
    results = {}
    p, r, f = get_PRF(num_c_5, num_e_5, num_s)
    results['top5'] = [p,r,f]
#     print_PRF(p, r, f, 5)
    p, r, f = get_PRF(num_c_10, num_e_10, num_s)
#     print_PRF(p, r, f, 10)
    results['top10'] = [p,r,f]
    p, r, f = get_PRF(num_c_15, num_e_15, num_s)
#     print_PRF(p, r, f, 15)
    results['top15'] = [p,r,f]
    p, r, f = get_PRF(num_c_20, num_e_20, num_s)
#     print_PRF(p, r, f, 20)
    results['top20'] = [p,r,f]
    
    return results
    
 
    
# =============================================================================
# pd.DataFrame(evaluate(get_ranked_kplist(nodomain), keys)).T
# =============================================================================
