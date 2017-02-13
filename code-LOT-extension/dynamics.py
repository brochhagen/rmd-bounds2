##### Main file to run dynamics
from rmd import run_dynamics

##### Parameters & setup #####
a = [1] # rate to control difference between semantic and pragmatic violations
lamb = [10,30] # soft-max parameter
seq_length = [5,10,15,20]  # length of observation sequences
samples = [200] #amount of k-length samples for each production type
l = [1,5,10,15] #prob-matching = 1, increments approach MAP

g = [50] #number of generations per simulation run
r = [1000] #number of independent simulation runs

dynamics = ['r','m','rmd'] #kind is the type of dynamics, either 'rmd', 'm' or 'r'

states = 3 #number of states
messages = 3 #number of messages
me = [False] #mutual exclusivity

for alpha in a:
    for lam in lamb:
        for k in seq_length:
            for sample_amount in samples:
                for learning_parameter in l:
                    for gens in g:
                        for runs in r:
                            for kind in dynamics:
                                for m_excl in me:
                                    run_dynamics(alpha,lam,k,sample_amount,gens,runs,states,messages,learning_parameter,kind,m_excl) 

#### This is all that is needed to run either replication only, mutation only or both together #####




############################### Decomment for testing individual components ########################################
#import numpy as np
#from lexica import get_lexica,get_prior,get_lexica_bins
#from player import LiteralPlayer,GriceanPlayer
#from rmd import get_utils,get_mutation_matrix,run_dynamics
#import sys 
#import datetime
#import csv
#import os.path
#
#lam = 30
#alpha = 1
#mutual_exclusivity=False
#
#
#state_freq = np.ones(states) / float(states) #frequency of states s_1,...,s_n 
#
##### Auxiliary functions #####
#def m_max(m): #aux function for convenience
#    return np.unravel_index(m.argmax(), m.shape)
#
#print '#Starting, ', datetime.datetime.now()
#lexica = get_lexica(states,messages,mutual_exclusivity=False)
#l_prior = get_prior(lexica)
#typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(alpha,lam,lex) for lex in lexica]
#
#likelihoods = np.array([t.sender_matrix for t in typeList])
#
#u = get_utils(typeList,states,messages,lam,alpha,mutual_exclusivity)
#
##sys.exit()
##q = get_mutation_matrix(states, messages, state_freq, likelihoods,l_prior,learning_parameter,sample_amount,k,lam,alpha)
##
##
##
##print '#Beginning multiple runs, ', datetime.datetime.now()
##f = csv.writer(open('./results/%s-s%d-m%d-lam%d-a%d-k%d,samples%d-l%d-g%d.csv' %(kind,states,messages,lam,alpha,k,sample_amount,learning_parameter,gens),'wb'))
##f.writerow(['runID','kind']+['t_ini'+str(x) for x in xrange(len(typeList))] +\
##           ['lam', 'alpha','k','samples','l','gens'] + ['t_final'+str(x) for x in xrange(len(typeList))])
##
##if os.path.isfile('./results/00mean-%s-s%d-m%d-g%d-r%d.csv' %(kind,states,messages,gens,runs)):
##    f_mean = csv.writer(open('./results/00mean-%s-s%d-m%d-g%d-r%d.csv' %(kind,states,messages,gens,runs), 'a'))
##else: 
##    f_mean = csv.writer(open('./results/00mean-%s-s%d-m%d-g%d-r%d.csv' %(kind,states,messages,gens,runs), 'wb'))
##    f_mean.writerow(['kind','lam','alpha','k','samples','l','gens'] + ['t_mean'+str(x) for x in xrange(len(typeList))])
##   
##
##p_sum = np.zeros(len(typeList)) #vector to store mean across runs
##
##for i in xrange(runs):
##    p = np.random.dirichlet(np.ones(len(typeList))) # unbiased random starting state
##    p_initial = p
##    for r in range(gens):
##        if kind == 'rmd':
##            pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
##            pPrime = pPrime / np.sum(pPrime)
##            p = np.dot(pPrime, q)
##        elif kind == 'm':
##            p = np.dot(p,q)
##        elif kind == 'r':
##            pPrime = p * [np.sum(u[t,] * p)  for t in range(len(typeList))]
##            p = pPrime / np.sum(pPrime)
##
##    f.writerow([str(i),kind] + [str(p_initial[x]) for x in xrange(len(typeList))]+\
##               [str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens)] +\
##               [str(p[x]) for x in xrange(len(typeList))])
##    p_sum += p
##p_mean = p_sum / runs
##f_mean.writerow([kind,str(lam),str(alpha),str(k),str(sample_amount),str(learning_parameter),str(gens),str(runs)] +\
##                    [str(p_mean[x]) for x in xrange(len(typeList))])
