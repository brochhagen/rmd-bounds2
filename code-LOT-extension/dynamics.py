##### Main file to run dynamics
#RMD with parametrized iterated parental learning 
#three states (none,sbna,all), three messages 
#2 possible signaling behaviors: literal or gricean
#Target type: index 24 in type/lexica/prior list
#####

import numpy as np
#np.set_printoptions(threshold=np.nan)
from random import sample
from lexica import get_lexica,get_prior
from player import LiteralPlayer,GriceanPlayer
from rmd import get_utils,get_mutation_matrix
import sys 
import datetime
import csv


##### Parameters & setup #####
alpha = 1 # rate to control difference between semantic and pragmatic violations
lam = 20 # soft-max parameter
k = 5  # length of observation sequences
sample_amount = 200 #amount of k-length samples for each production type

gens = 50 #number of generations per simulation run
runs = 50 #number of independent simulation runs

states = 3 #number of states
messages = 3 #number of messages

learning_parameter = 1 #prob-matching = 1, increments approach MAP
state_freq = np.ones(states) / float(states) #frequency of states s_1,...,s_n 

##### Auxiliary functions #####
def m_max(m): #aux function for convenience
    return np.unravel_index(m.argmax(), m.shape)

###############################
print '#Starting, ', datetime.datetime.now()

lexica = get_lexica(states,messages,mutual_exclusivity=True)
l_prior = get_prior(lexica)
typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(alpha,lam,lex) for lex in lexica]

print '#Computing likelihood, ', datetime.datetime.now()
likelihoods = np.array([t.sender_matrix for t in typeList])

print '#Computing utilities, ', datetime.datetime.now()
u = get_utils(typeList)

print '#Computing mutation matrix, ', datetime.datetime.now()
q = get_mutation_matrix(states,messages,state_freq, likelihoods,l_prior,learning_parameter,sample_amount,k)






p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state

for r in range(gens):
    pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
    pPrime = pPrime / np.sum(pPrime)
    p = np.dot(pPrime, q)
    print np.argmax(p), p[np.argmax(p)]






sys.exit()







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
