#####
#RMD with parametrized iterated parental learning 
#1 pairs of scalar items, six lexica per scalar pair. 
#2 possible signaling behaviors: literal or gricean
#12 types (6 lexica * 2 signaling behaviors) per signaling pair
#6 literal types, 6 gricean types, 12 types in total
#####

import numpy as np
np.set_printoptions(threshold=np.nan)
from random import sample
from itertools import product
from player import LiteralPlayer,GriceanPlayer
import sys 
import datetime
import csv

#####
l1,l2,l3,l4,l5,l6 = np.array( [[0.,0.],[1.,1.]] ), np.array( [[1.,1.],[0.,0.]] ), np.array( [[1.,1.],[1.,1.]] ), np.array( [[0.,1.],[1.,0.]] ), np.array( [[0.,1.],[1.,1.]] ), np.array( [[1.,1.],[1.,0.]] )

alpha = 10 # rate to control difference between semantic and pragmatic violations
cost = 0 # cost for LOT-concept with upper bound
lam = 20 # soft-max parameter
k = 3  # length of observation sequences
sample_amount = 50 #amount of k-length samples for each production type
deltaE = 0.3 # probability of perceiving S-all, when true state is S-sbna
deltaA = 0.1 # probability of perceiving S-sbna, when true state is S-all

gens = 20 #number of generations per simulation run
runs = 50 #number of independent simulation runs

states = 2 #number of states
messages = 2 #number of messages

learning_parameter = 10 #prob-matching = 1, increments approach MAP
state_freq = np.ones(states) / float(states) #frequency of states s_1,...,s_n 


print '#Starting, ', datetime.datetime.now()

t1,t2,t3,t4,t5,t6 = LiteralPlayer(alpha,l1), LiteralPlayer(alpha,l2), LiteralPlayer(alpha,l3), LiteralPlayer(alpha,l4), LiteralPlayer(alpha,l5), LiteralPlayer(alpha,l6)
t7,t8,t9,t10,t11,t12 =  GriceanPlayer(alpha,lam,l1), GriceanPlayer(alpha,lam,l2), GriceanPlayer(alpha,lam,l3), GriceanPlayer(alpha,lam,l4), GriceanPlayer(alpha,lam,l5), GriceanPlayer(alpha,lam,l6)

typeList = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12]

print '#Computing likelihood, ', datetime.datetime.now()
likelihoods = np.array([t.sender_matrix for t in typeList])

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

## state confusability
state_confusion_matrix = np.array([[1-deltaE, deltaE],
                                   [deltaA, 1-deltaA]])
lh_perturbed = likelihoods
PosteriorState = normalize(np.array([[state_freq[sActual] * state_confusion_matrix[sActual, sPerceived] for sActual in xrange(states)] \
 for sPerceived in xrange(states)])) # probability of actual state given a perceived state
DoublePerception = np.array([[np.sum([ state_confusion_matrix[sActual, sTeacher] * PosteriorState[sLearner,sActual] \
 for sActual in xrange(states)]) for sTeacher in xrange(states) ] for sLearner in xrange(states)])# probability of teacher observing column, given that learner observes row

print PosteriorState
print DoublePerception

for t in xrange(len(likelihoods)):
   for sLearner in xrange(len(likelihoods[t])):
       for m in xrange(len(likelihoods[t][sLearner])):
           lh_perturbed[t,sLearner,m] = np.sum([ DoublePerception[sLearner,sTeacher] * likelihoods[t,sTeacher,m] for sTeacher in xrange(len(likelihoods[t]))])

print lh_perturbed

np.array([np.dot(state_confusion_matrix, t.sender_matrix) for t in typeList])


lexica_prior = np.array([2.0, 2.0- 2.0* cost, 2.0, 2.0 - cost , 2.0 , 2.0-cost, 2.0, 2.0- 2.0* cost, 2.0, 2.0 - cost , 2.0 , 2.0-cost])
lexica_prior = lexica_prior / sum(lexica_prior)


def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def summarize_counts(lst):
    """summarize counts for tuples of k-states and k-messages""" 
    counter = [0 for _ in xrange(states**messages)]
    for i in xrange(len(lst)):
        s,m = lst[i][0] *2, lst[i][1]
        counter[s+m] += 1
    return counter

def get_obs(k):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k"""
    s = list(xrange(states))
    m = list(xrange(messages))
    atomic_observations = list(product(s,m))
   
    obs = [] #store all produced k-length (s,m) sequences 
    for t in xrange(len(typeList)):
        produced_obs = [] #store k-length (s,m) sequences of a type
        production_vector = likelihoods[t].flatten()
        doubled_state_freq = np.column_stack((state_freq,state_freq)).flatten() # P(s)
        sample_vector = production_vector * doubled_state_freq #P(s) * P(m|s,t_i)
        for i in xrange(sample_amount):
            sample_t = [np.random.choice(len(atomic_observations),p=sample_vector) for _ in xrange(k)]
            sampled_obs = [atomic_observations[i] for i in sample_t]
            produced_obs.append(summarize_counts(sampled_obs))
        obs.append(produced_obs)
    return obs


def get_likelihood(obs, kind = "plain"):
    # allow three kinds of likelihood:
    ## 1. "plain" -> probability that speaker generates m when observing s
    ## 2. "production" -> probability that speaker generates m when true state is s
    ## 3. "observation" -> probability that speaker produces m when listener observes s
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    if kind == "plain":
        for lhi in range(len(likelihoods)):
            for o in range(len(obs)):
                out[lhi,o] = likelihoods[lhi,0,0]**obs[o][0] * (likelihoods[lhi,0,1])**(obs[o][1]) *\
                             likelihoods[lhi,1,0]**obs[o][2] * (likelihoods[lhi,1,1])**(obs[o][3]) # first line is some, second is all
    if kind == "perturbed":
        for lhi in range(len(likelihoods)):
            for o in range(len(obs)):
                out[lhi,o] = lh_perturbed[lhi,0,0]**obs[o][0] * (lh_perturbed[lhi,0,1])**(obs[o][1]) *\
                             lh_perturbed[lhi,1,0]**obs[o][2] * (lh_perturbed[lhi,1,1])**(obs[o][3]) # first line is some, second is all
    return out


def get_mutation_matrix(k):
    obs = get_obs(k) #get production data from all types
    out = np.zeros([len(likelihoods),len(likelihoods)]) #matrix to store Q

    for parent_type in xrange(len(likelihoods)):
        type_obs = obs[parent_type] #Parent production data
        lhs_perturbed = get_likelihood(type_obs, kind = "perturbed") #P(learner observes data|t_i) for all types;
        lhs = get_likelihood(type_obs, kind = "plain") #P(parent data|t_i) for all types; without all noise
        post = normalize(lexica_prior * np.transpose(lhs)) #P(t_j|parent data) for all types; P(d|t_j)P(t_j)
        parametrized_post = normalize(post**learning_parameter)

        out[parent_type] = np.dot(np.transpose(lhs_perturbed[parent_type]),parametrized_post)

    return normalize(out)

def get_utils():
    out = np.zeros([len(typeList), len(typeList)])
    for i in range(len(typeList)):
        for j in range(len(typeList)):
            out[i,j] = (np.sum(typeList[i].sender_matrix * np.transpose(typeList[j].receiver_matrix)) + \
                     np.sum(typeList[j].sender_matrix * np.transpose(typeList[i].receiver_matrix))) / 4
    return out


print '#Computing utilities, ', datetime.datetime.now()
u = get_utils()

print '#Computing Q, ', datetime.datetime.now()

q = get_mutation_matrix(k)

### single run

p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
p_initial = np.array([1,1,1,1,1,1,1,1,1,1,1,1.0]) / 12
p_initial = p

for r in range(gens):
    pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
    pPrime = pPrime / np.sum(pPrime)
    p = np.dot(pPrime, q)


print '###Overview of results###', datetime.datetime.now()
print 'Parameters: alpha = %d, c = %.2f, lambda = %d, k = %d, samples per type = %d, learning parameter = %.2f, gen = %d' % (alpha, cost, lam, k, sample_amount, learning_parameter, gens)
print 'end state:' 
print p
