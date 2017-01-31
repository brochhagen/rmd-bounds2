## Functions for replicator-mutator dynamics with iterated learning as mutator dynamics

import numpy as np
np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product
from player import LiteralPlayer,GriceanPlayer
import sys 
import datetime
import csv

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def summarize_counts(lst,states,messages):
    """summarize counts for tuples of k-states and k-messages""" 
    counter = [0 for _ in xrange(states*messages)]
    for i in xrange(len(lst)):
        s,m = lst[i][0] *messages, lst[i][1]
        counter[s+m] += 1
    return counter

def get_obs(s_amount,m_amount,k,likelihoods,state_freq,sample_amount):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k"""
    s = list(xrange(s_amount))
    m = list(xrange(m_amount))
    atomic_observations = list(product(s,m))
   
    obs = [] #store all produced k-length (s,m) sequences 
    for t in xrange(len(likelihoods)):
        produced_obs = [] #store k-length (s,m) sequences of a type
        production_vector = likelihoods[t].flatten()
        doubled_state_freq = np.column_stack(tuple(state_freq for _ in xrange(m_amount))).flatten() # P(s)
        sample_vector = production_vector * doubled_state_freq #P(s) * P(m|s,t_i)
        for i in xrange(sample_amount):
            sample_t = [np.random.choice(len(atomic_observations),p=sample_vector) for _ in xrange(k)]
            sampled_obs = [atomic_observations[i] for i in sample_t]
            produced_obs.append(summarize_counts(sampled_obs,s_amount,m_amount))
        obs.append(produced_obs)
    return obs

def get_likelihood(obs,likelihoods):
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            flat_lhi = likelihoods[lhi].flatten()
            out[lhi,o] = np.prod([flat_lhi[x]**obs[o][x] for x in xrange(len(obs[o]))]) 
    return out

def get_mutation_matrix(s_amount,m_amount,state_freq, likelihoods,lexica_prior,learning_parameter,sample_amount,k):
    obs = get_obs(s_amount,m_amount,k,likelihoods,state_freq,sample_amount) #get production data from all types
    out = np.zeros([len(likelihoods),len(likelihoods)]) #matrix to store Q

    for parent_type in xrange(len(likelihoods)):
        type_obs = obs[parent_type] #Parent production data
        lhs = get_likelihood(type_obs,likelihoods) #P(parent data|t_i) for all types
        post = normalize(lexica_prior * np.transpose(lhs)) #P(t_j|parent data) for all types; P(d|t_j)P(t_j)
        parametrized_post = normalize(post**learning_parameter)

        out[parent_type] = np.dot(np.transpose(lhs[parent_type]),parametrized_post)

    return normalize(out)


def get_utils(typeList):
    out = np.zeros([len(typeList), len(typeList)])
    for i in range(len(typeList)):
        for j in range(len(typeList)):
            out[i,j] = (np.sum(typeList[i].sender_matrix * np.transpose(typeList[j].receiver_matrix)) /3. + \
                     np.sum(typeList[j].sender_matrix * np.transpose(typeList[i].receiver_matrix))/3. ) / 2
    return out

#print '#Computing Q, ', datetime.datetime.now()
#
#q = get_mutation_matrix(k)
#
#for i in q:
#    para = np.array([str(alpha), str(cost), str(lam), str(k), str(sample_amount), str(learning_parameter)])
#    j = np.append(para,i)
#    f_q.writerow(j)
#
#
####Multiple runs
#print '#Beginning multiple runs, ', datetime.datetime.now()
#
#p_sum = np.zeros(len(typeList)) #vector to store results from a run
#
#for i in xrange(runs):
#    p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
#    p_initial = p
#
#    for r in range(gens):
#        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
#        pPrime = pPrime / np.sum(pPrime)
#        p = np.dot(pPrime, q)
#        f.writerow([str(i),str(p_initial[0]), str(p_initial[1]), str(p_initial[2]), str(p_initial[3]), str(p_initial[4]), str(p_initial[5]), str(p_initial[6]), str(p_initial[7]), str(p_initial[8]), str(p_initial[9]), str(p_initial[10]), str(p_initial[11]), str(alpha), str(cost), str(lam), str(k), str(sample_amount), str(learning_parameter), str(gens), str(p[0]), str(p[1]),str(p[2]),str(p[3]),str(p[4]),str(p[5]),str(p[6]),str(p[7]),str(p[8]),str(p[9]),str(p[10]),str(p[11])])
#    
#    p_sum += p
#
#p_mean = p_sum / runs
#
#
#
#
#print '###Overview of results###', datetime.datetime.now()
#print 'Parameters: alpha = %d, c = %.2f, lambda = %d, k = %d, samples per type = %d, learning parameter = %.2f, generations = %d, runs = %d' % (alpha, cost, lam, k, sample_amount, learning_parameter, gens, runs)
#print 'Mean by type:'
#print p_mean
