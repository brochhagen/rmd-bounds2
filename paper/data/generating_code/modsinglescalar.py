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

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def summarize_counts(lst,states,messages):
    """summarize counts for tuples of k-states and k-messages""" 
    counter = [0 for _ in xrange(states**messages)]
    for i in xrange(len(lst)):
        s,m = lst[i][0] *2, lst[i][1]
        counter[s+m] += 1
    return counter

def get_obs(k,states,messages,likelihoods,state_freq,sample_amount):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k"""
    s = list(xrange(states))
    m = list(xrange(messages))
    atomic_observations = list(product(s,m))
   
    obs = [] #store all produced k-length (s,m) sequences 
    for t in xrange(12):
        produced_obs = [] #store k-length (s,m) sequences of a type
        production_vector = likelihoods[t].flatten()
        doubled_state_freq = np.column_stack((state_freq,state_freq)).flatten() # P(s)
        sample_vector = production_vector * doubled_state_freq #P(s) * P(m|s,t_i)
        for i in xrange(sample_amount):
            sample_t = [np.random.choice(len(atomic_observations),p=sample_vector) for _ in xrange(k)]
            sampled_obs = [atomic_observations[i] for i in sample_t]
            produced_obs.append(summarize_counts(sampled_obs,states,messages))
        obs.append(produced_obs)
    return obs


def get_likelihood(obs,likelihoods):
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            out[lhi,o] = likelihoods[lhi,0,0]**obs[o][0] * (likelihoods[lhi,0,1])**(obs[o][1]) *\
                         likelihoods[lhi,1,0]**obs[o][2] * (likelihoods[lhi,1,1])**(obs[o][3]) # first line is some, second is all
    return out


def get_mutation_matrix(k,states,messages,likelihoods,state_freq,sample_amount,lexica_prior,learning_parameter):
    obs = get_obs(k,states,messages,likelihoods,state_freq,sample_amount) #get production data from all types
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
            out[i,j] = (np.sum(typeList[i].sender_matrix * np.transpose(typeList[j].receiver_matrix)) + \
                     np.sum(typeList[j].sender_matrix * np.transpose(typeList[i].receiver_matrix))) / 4
    return out

def run(alpha,cost,lam,k,learning_parameter,gens,runs):
#####
    l1,l2,l3,l4,l5,l6 = np.array( [[0.,0.],[1.,1.]] ), np.array( [[1.,1.],[0.,0.]] ), np.array( [[1.,1.],[1.,1.]] ), np.array( [[0.,1.],[1.,0.]] ), np.array( [[0.,1.],[1.,1.]] ), np.array( [[1.,1.],[1.,0.]] )
    
    sample_amount = 10 #fixed value for fixed Q

    states = 2 #number of states
    messages = 2 #number of messages

    state_freq = np.ones(states) / float(states) #frequency of states s_1,...,s_n 


    f = csv.writer(open('./results/singlescalar-a%.2f-c%.2f-l%d-k%d-samples%d-learn%.2f-g%d-r%d.csv' %(alpha,cost,lam,k,sample_amount, learning_parameter, gens,runs),'wb')) #file to store mean results

    f.writerow(["run_ID", "t1_initial", "t2_initial","t3_initial","t4_initial","t5_initial","t6_initial","t7_initial","t8_initial","t9_initial","t10_initial","t11_initial","t12_initial","alpha", "prior_cost_c", "lambda", "k", "sample_amount", "learning_parameter", "generations", "t1_final", "t2_final","t3_final","t4_final","t5_final","t6_final","t7_final","t8_final","t9_final","t10_final","t11_final","t12_final"])

    f_q = csv.writer(open('./results/singlescalar-q-matrix-a%d-c%f-l%d-k%d-samples%d-learn%.2f.csv' %(alpha,cost,lam,k,sample_amount,learning_parameter),'wb')) #file to store Q-matrix
    
    f_q.writerow(["alpha", "prior_cost_c", "lambda", "k", "sample_amount", "learning_parameter","parent","t1_mut", "t2_mut", "t3_mut", "t4_mut", "t5_mut", "t6_mut", "t7_mut", "t8_mut", "t9_mut", "t10_mut", "t11_mut", "t12_mut"])
######



    print '#Starting, ', datetime.datetime.now()

    t1,t2,t3,t4,t5,t6 = LiteralPlayer(alpha,lam,l1), LiteralPlayer(alpha,lam,l2), LiteralPlayer(alpha,lam,l3), LiteralPlayer(alpha,lam,l4), LiteralPlayer(alpha,lam,l5), LiteralPlayer(alpha,lam,l6)
    t7,t8,t9,t10,t11,t12 =  GriceanPlayer(alpha,lam,l1), GriceanPlayer(alpha,lam,l2), GriceanPlayer(alpha,lam,l3), GriceanPlayer(alpha,lam,l4), GriceanPlayer(alpha,lam,l5), GriceanPlayer(alpha,lam,l6)

    typeList = [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12]

    print '#Computing likelihood, ', datetime.datetime.now()
    likelihoods = np.array([t.sender_matrix for t in typeList])

    lexica_prior = np.array([2, 2- 2* cost, 2, 2 - cost , 2 , 2-cost, 2, 2- 2* cost, 2, 2 - cost , 2 , 2-cost])
    lexica_prior = lexica_prior / sum(lexica_prior)


    print '#Computing utilities, ', datetime.datetime.now()
    u = get_utils(typeList)

    print '#Computing Q, ', datetime.datetime.now()
    
    q = get_mutation_matrix(k,states,messages,likelihoods,state_freq,sample_amount,lexica_prior,learning_parameter)
    
    for i in q:
        para = np.array([str(alpha), str(cost), str(lam), str(k), str(sample_amount), str(learning_parameter)])
        j = np.append(para,i)
        f_q.writerow(j)
    

###Multiple runs
    print '#Beginning multiple runs, ', datetime.datetime.now()

    p_sum = np.zeros(len(typeList)) #vector to store results from a run

    for i in xrange(runs):
        p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
        p_initial = p

        for r in range(gens):
            pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
            pPrime = pPrime / np.sum(pPrime)
            p = np.dot(pPrime, q)
        f.writerow([str(i),str(p_initial[0]), str(p_initial[1]), str(p_initial[2]), str(p_initial[3]), str(p_initial[4]), str(p_initial[5]), str(p_initial[6]), str(p_initial[7]), str(p_initial[8]), str(p_initial[9]), str(p_initial[10]), str(p_initial[11]), str(alpha), str(cost), str(lam), str(k), str(sample_amount), str(learning_parameter), str(gens), str(p[0]), str(p[1]),str(p[2]),str(p[3]),str(p[4]),str(p[5]),str(p[6]),str(p[7]),str(p[8]),str(p[9]),str(p[10]),str(p[11])])


        p_sum += p

    p_mean = p_sum / runs




    print '###Overview of results###', datetime.datetime.now()
    print 'Parameters: alpha = %.2f, c = %.2f, lambda = %d, k = %d, samples per type = %d, learning parameter = %.2f, generations = %d, runs = %d' % (alpha, cost, lam, k, sample_amount, learning_parameter, gens, runs)
    print 'Mean by type:'
    print p_mean
