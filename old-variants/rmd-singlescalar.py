#####
#RMD for unweighted and weighted (community) learning with Bayesian (iterated) learning as M. 
#1 pairs of scalar items, six lexica per scalar pair. 
#2 possible signaling behaviors: literal or gricean
#12 types (6 lexica * 2 signaling behaviors) per signaling pair
#6 literal types, 6 gricean types, 12 types in total
#####


import numpy as np
#np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product, permutations, combinations_with_replacement
from player import LiteralPlayer,GriceanPlayer
import sys 
import datetime
import csv


l1,l2,l3,l4,l5,l6 = np.array( [[0.,0.],[1.,1.]] ), np.array( [[1.,1.],[0.,0.]] ), np.array( [[1.,1.],[1.,1.]] ), np.array( [[0.,1.],[1.,0.]] ), np.array( [[0.,1.],[1.,1.]] ), np.array( [[1.,1.],[1.,0.]] )

alpha = 1# rate to control difference between semantic and pragmatic violations
cost = 0.9 # cost for LOT-concept with upper bound
lam = 30 # soft-max parameter
k = 20  # number of learning observations
sample_amount = 145 #amount of samples from OBS when k > 4

gens = 30 #number of generations per simulation run
runs = 100 #number of independent simulation runs

states = 2 #number of states
messages = 2 #number of messages


f_unwgh_mean = csv.writer(open('./results/singlescalar-unwgh-mean-a%d-c%f-l%d-k%d-g%d-r%d.csv' %(alpha,cost,lam,k,gens,runs),'wb')) #file to store each unweighted simulation run after n generations
f_wgh_mean = csv.writer(open('./results/singlescalar-wgh-mean-a%d-c%f-l%d-k%d-g%d-r%d.csv' %(alpha,cost,lam,k,gens,runs),'wb')) #file to store each weighted simulation run after n generations

f_q = csv.writer(open('./results/singlescalar-q-matrix-a%d-c%f-l%d-k%d-g%d-r%d.csv' %(alpha,cost,lam,k,gens,runs),'wb')) #file to store Q-matrix



print '#Starting, ', datetime.datetime.now()

t1,t2,t3,t4,t5,t6 = LiteralPlayer(alpha,l1), LiteralPlayer(alpha,l2), LiteralPlayer(alpha,l3), LiteralPlayer(alpha,l4), LiteralPlayer(alpha,l5), LiteralPlayer(alpha,l6)
t7,t8,t9,t10,t11,t12 =  GriceanPlayer(alpha,lam,l1), GriceanPlayer(alpha,lam,l2), GriceanPlayer(alpha,lam,l3), GriceanPlayer(alpha,lam,l4), GriceanPlayer(alpha,lam,l5), GriceanPlayer(alpha,lam,l6)

typeList = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12]

print '#Computing likelihood, ', datetime.datetime.now()
likelihoods = np.array([t.sender_matrix for t in typeList])

lexica_prior = np.array([2, 2- 2* cost, 2, 2 - cost , 2 , 2-cost, 2, 2- 2* cost, 2, 2 - cost , 2 , 2-cost])

lexica_prior = lexica_prior / sum(lexica_prior)

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def summarize_counts(lst,k):
    """summarize counts for tuples of k-states and k-messages""" #There are a couple of repeated observations, when summarized. For instance, if we have k = 2, then observation <<s_1,m_0>,<s_1,m_1>> is identical to <<s_1,m_1>,<s_1,m_0>>
    out = []
    for i in xrange(len(lst)):
        counter = [0 for _ in xrange(states**messages)]
        for j in xrange(k):
            s,m = lst[i][0][j] *2, lst[i][1][j]
            counter[s+m] += 1
        out.append(counter)
    print 'Number of observations:', len(out), datetime.datetime.now()
    return out


def get_obs(k):
    """Returns summarized counts of k-length <s_i,m_j> observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], sum[...] = k"""
    inputx = [x for x in list(product(range(states), repeat=k))] #k-tuple where the i-th observed state was state k_i 
    outputy = [y for y in list(product(range(messages),repeat=k))] #k-tuple where the i-th produced message was k_i 
    D = list(product(inputx,outputy)) #list of all possible state-message combinations

    if k < 4:
        D = list(product(inputx,outputy)) #list of all possible state-message combinations
    else: #producing the entire list crashed memmory for k>9. So we sample indices insead. This is still taxing for large k
        prod = len(inputx) * len(outputy)
        indices = sample(range(prod), sample_amount)
        D = [(inputx[idx % len(inputx)], outputy[idx // len(inputx)]) for idx in indices]
    return summarize_counts(D,k)

    

# likelihood function
def get_obs_likelihood(k):
    # idea: produce k messages for each state;
    obs = get_obs(k)
    out = np.ones([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            out[lhi,o] = likelihoods[lhi,0,0]**obs[o][0] * (likelihoods[lhi,0,1])**(obs[o][1]) *\
                         likelihoods[lhi,1,0]**obs[o][2] * (likelihoods[lhi,1,1])**(obs[o][3]) # first line is some, second is all
    return (normalize(out))



def get_mutation_matrix(k):
    # we want: Q_ij = \sum_d p(d|t_i) p(t_j|d); this is the dot-product of the likelihood matrix and the posterior
    lhs = get_obs_likelihood(k)
    post = normalize(lexica_prior * np.transpose(lhs))
    return np.dot(lhs, post)

def weighted_mutation(p,q):
    m = np.zeros([len(p),len(p)])
    for i in range(len(p)):
        for j in range(len(p)):
            m[i,j] = p[j] * q[i,j]
    return normalize(m)

def get_utils():
    out = np.zeros([len(typeList), len(typeList)])
    for i in range(len(typeList)):
        for j in range(len(typeList)):
            out[i,j] = (np.sum(typeList[i].sender_matrix * np.transpose(typeList[j].receiver_matrix)) + \
                     np.sum(typeList[j].sender_matrix * np.transpose(typeList[i].receiver_matrix))) / 4
    return out

print '#Computing Q, ', datetime.datetime.now()
q = get_mutation_matrix(k)

for i in q:
    f_q.writerow(i)

print '#Computing utilities, ', datetime.datetime.now()
u = get_utils()


###Multiple runs
print '#Beginning multiple runs, ', datetime.datetime.now()

f_unwgh_mean.writerow(["stype","l1","proportion"])
f_wgh_mean.writerow(["stype","l1","proportion"])

p_sum = np.zeros(len(typeList))
p_sum_wgh = np.zeros(len(typeList))



for i in xrange(runs):

    p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state

    for r in range(gens):
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, q)
    p_sum += p

    p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
    for r in range(gens):
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, weighted_mutation(p,q))
    p_sum_wgh += p


    print i, datetime.datetime.now()


    
p_mean = p_sum / runs
p_wgh_mean = p_sum_wgh / runs


def record_by_type(p_vec,csv_file):
    for i in range(len(p_vec)):
        if i < len(p_vec)/2:
            stype = 'literal'
        else: stype = 'gricean'
        if i < len(p_vec)/2:
            lexicon = i
        else:
            lexicon = i - len(p_vec)/2
        csv_file.writerow([stype,str(lexicon),str(p_vec[i])])

record_by_type(p_mean,f_unwgh_mean)
record_by_type(p_wgh_mean,f_wgh_mean)

def lexica_vec(p_vec): 
    p_by_lexica = np.zeros(6)
    for i in xrange(len(p_vec)):
        if i < len(p_vec)/2:
            p_by_lexica[i] += p_vec[i]
        else:
            r = i - len(p_vec)/2
            p_by_lexica[r] += p_vec[i]
    return p_by_lexica

print '### Quick overview of results###'
print 'Unweighted mean by lexica:', lexica_vec(p_mean)
print '###'
print 'Weighted mean by lexica:', lexica_vec(p_wgh_mean)
