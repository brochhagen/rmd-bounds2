##generation of lexica and prior
import numpy as np
from itertools import product,combinations,combinations_with_replacement

def get_lexica(s_amount,m_amount,mutual_exclusivity=True):
    columns = list(product([0.,1.],repeat=s_amount))
    columns.remove((0,0,0)) #remove message false of all states
    columns.remove((1,1,1)) #remove message true of all states
    if mutual_exclusivity:
        matrix = list(combinations(columns,r=m_amount)) #no concept assigned to more than one message
        out = []
        for mrx in matrix:
            lex = np.array([mrx[i] for i in xrange(s_amount)])
            lex = np.transpose(np.array([mrx[i] for i in xrange(s_amount)]))
            out.append(lex)
    else:
#        matrix = list(product(columns,repeat=m_amount)) #If we allow for symmetric lexica
        matrix = list(combinations_with_replacement(columns,m_amount)) 
        out = []
        for mrx in matrix:
            lex = np.array([mrx[i] for i in xrange(s_amount)])
            lex = np.transpose(np.array([mrx[i] for i in xrange(s_amount)]))
            out.append(lex)
    return out 

def get_prior(lexica_list):
    concepts = [[0,0,1],[0,1,0],[0,1,1],\
                    [1,0,0],[1,0,1],[1,1,0]]
    cost = [3,8,4,4,10,5] #cost of each concept in 'concepts'
    cost = np.array([float(max(cost) - c + 1) for c in cost])
    concept_prob = cost / np.sum(cost)
    
    out = []
    for lex_idx in xrange(len(lexica_list)):
        current_lex = np.transpose(lexica_list[lex_idx])
        lex_val = 1 #probability of current lexicon's concepts
        for concept_idx in xrange(len(current_lex)):
            lex_val *= concept_prob[concepts.index(list(current_lex[concept_idx]))]
        out.append(lex_val)
    out = out + out #double for two types of linguistic behavior
    return np.array(out) / np.sum(out)
