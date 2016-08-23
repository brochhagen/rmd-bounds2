#####
#RMD for unweighted and weighted (community) learning with Bayesian (iterated) learning as M. 
#3 pairs of scalar items, six lexica per scalar pair. 
#2 possible signaling behaviors: literal or gricean
#12 types (6 lexica * 2 signaling behaviors) per signaling pair
#6*6*6 = 216  literal types, 216 gricean types, 432 types in total
#Set of observations of length k:
##Sample <s,m> tuples from a type's production probabilities, P(a = <s,m> | t_i) = P(s) * P(m|s,t_i)
#
#####


import numpy as np
#np.set_printoptions(threshold=np.nan)
from random import sample,choice
from itertools import product, permutations, combinations_with_replacement
from player import LiteralPlayer,GriceanPlayer
import sys 
import datetime
import csv


alpha = 1 # rate to control difference between semantic and pragmatic violations
cost = 0.7 # cost for LOT-concept with upper bound
lam = 30 # soft-max parameter
k = 5  # number of learning observations
sample_amount = 50 #amount of samples per type production

gens = 30 #number of generations per simulation run
runs = 10 #number of independent simulation runs

lexical_pairs = 3 #number of 2x2 matrices
states = 2 #number of states
messages = 2 #number of messages

learning_parameter = 1 #prob-matching = 1, increments approach MAP

state_freq = np.ones(states*lexical_pairs) / float(states*lexical_pairs) #frequency of states. 
w_communal,w_parental = 0.5,0.5 #w_communal is the weight for communal learning, w_parental that of parental learning, w_communcal + w_parental = 1




f_unwgh_mean = csv.writer(open('./results/4multiscalar-unwgh-mean-a%d-c%f-l%d-k%d-g%d-r%d-samp%d.csv' %(alpha,cost,lam,k,gens,runs,learning_parameter),'wb')) #file to store each unweighted simulation run after n generations
f_wgh_mean = csv.writer(open('./results/4multiscalar-wgh-mean-a%d-c%f-l%d-k%d-g%d-r%d-samp%d.csv' %(alpha,cost,lam,k,gens,runs, learning_parameter),'wb')) #file to store each weighted simulation run after n generations

f_q = csv.writer(open('./results/4multiscalar-q-matrix-a%d-c%f-l%d-k%d-g%d-r%d-samp%d.csv' %(alpha,cost,lam,k,gens,runs, learning_parameter),'wb')) #file to store Q-matrix


print '#Starting, ', datetime.datetime.now()
lexica = [np.array( [[0.,0.],[1.,1.]] ), np.array( [[1.,1.],[0.,0.]] ), np.array( [[1.,1.],[1.,1.]] ), np.array( [[0.,1.],[1.,0.]] ), np.array( [[0.,1.],[1.,1.]] ), np.array( [[1.,1.],[1.,0.]] )]

def types(l):
    """outputs list of types from list of lexica input.""" 
    typeList = []
    for i in l:
        typeList.append(LiteralPlayer(alpha,i))
    for i in l:
        typeList.append(GriceanPlayer(alpha,lam,i))
    return typeList 

typeList = types(lexica)


def lex_prior(l):
    """outputs learning prior vector from list of lexica input. Assumes twice as many types than lexica. First half of the vector is literal, second half is gricean."""
    cost_v = np.zeros(len(lexica))
    for j in range(len(l)):
        lexicon_score = 0.
        for i in range(l[j].shape[1]):
            if (i == 0 or i == 1) and l[j][0,i] > 0 and l[j][1,i] == 0:
                lexicon_score += cost
        cost_v[j] = states - lexicon_score
    cost_v =  np.array([sum(p) for p in product(cost_v,repeat=lexical_pairs)]) #for all len(lexica)**(#states) "big" lexica 
    return np.append(cost_v,cost_v) #doubling for first half literal, second half gricean

lex_prior = lex_prior(lexica)
lexica_prior = lex_prior / sum(lex_prior)


print '#Computing likelihood, ', datetime.datetime.now()
likelihoods = np.array([t.sender_matrix for t in typeList])

def labels(s,m,lex):
    """Returns list of labels that specifies which lexica are assigned to what type."""
    a = [i for i in range(len(lexica))]
    b = [i for i in range(len(lexica), 2* len(lexica))]
    lab1 = [list(p) for p in product(a, repeat=lex)]
    lab2 = [list(p) for p in product(b, repeat=lex)]
    return lab1 + lab2

labels = labels(states,messages,lexical_pairs)


def normalize(m):
    """Matrix row-wise normalization"""
    return m / m.sum(axis=1)[:, np.newaxis]


def summarize_counts(lst,k):
    """summarize counts for tuples of k-states and k-messages""" 
    counter = [0 for _ in xrange(states**messages * lexical_pairs)]
    for i in xrange(len(lst)):
        s,m = lst[i][0] *2, lst[i][1]
        counter[s+m] += 1
    return counter

def get_obs(k):
    """Returns summarized counts of k-length <s_i,m_j> production observations as [#(<s_0,m_0>), #(<s_0,m_1), #(<s_1,m_0>, #(s_1,m_1)], ...]] = k"""
    s = list(xrange(states*lexical_pairs))
    m = list(xrange(messages))
    atomic_observations = list(product(s,m))
   
    obs = []
    for idx in xrange(len(labels)):
        l0,l1,l2 = labels[idx][0],labels[idx][1],labels[idx][2]
        production_vector = np.append(np.append(likelihoods[l0],likelihoods[l1]),likelihoods[l2]) #P(m|s,t_i)
        doubled_state_freq = np.column_stack((state_freq,state_freq)).flatten() # P(s)
        sample_vector = production_vector * doubled_state_freq #P(s) * P(m|s,t_i)
        for i in xrange(sample_amount):
            sample_idx = [np.random.choice(len(atomic_observations),p=sample_vector) for _ in xrange(k)]
            sampled_obs = [atomic_observations[i] for i in sample_idx]
            obs.append(summarize_counts(sampled_obs,k))
    return obs

def get_obs_likelihood(k):
    print 'Getting obs', datetime.datetime.now()
    obs = get_obs(k)

    print 'Computing likelihood', datetime.datetime.now()
    out = np.zeros([len(labels), len(obs)]) # matrix to store results in

    for lhi in xrange(len(labels)):
        l0,l1,l2 = labels[lhi][0],labels[lhi][1],labels[lhi][2]
        for o in xrange(len(obs)):
#            print obs[o], lhi, o
            out[lhi,o] = (likelihoods[l0,0,0]**obs[o][0] * (likelihoods[l0,0,1])**(obs[o][1]) *\
                           likelihoods[l0,1,0]**obs[o][2] * (likelihoods[l0,1,1])**(obs[o][3]) *\
                           likelihoods[l1,0,0]**obs[o][4] * (likelihoods[l1,0,1])**(obs[o][5]) *\
                           likelihoods[l1,1,0]**obs[o][6] * (likelihoods[l1,1,1])**(obs[o][7]) *\
                           likelihoods[l2,0,0]**obs[o][8] * (likelihoods[l2,0,1])**(obs[o][9]) *\
                           likelihoods[l2,1,0]**obs[o][10] * (likelihoods[l2,1,1])**(obs[o][11]))


    r,c = out.shape
    print 'Likelihood computed. Shape:', r, 'rows and',c ,'columns', datetime.datetime.now()
    return normalize(out)

likelihood_matrix = get_obs_likelihood(k)


def get_mutation_matrix(lhs,k):
    # we want: Q_ij = \sum_d p(d|t_i) p(t_j|d); this is the dot-product of the likelihood matrix and the posterior
    post = normalize(lexica_prior * np.transpose(lhs))
    return normalize(np.dot(lhs, post)**learning_parameter)

def population_learn_vector(lhs,p):
    #L_j = \sum_d P(d|\vec{p}) * P(t_j|d), where P(d|\vec{p}) = \sum_t_i P(d|t_i) * p_i
    r,c = lhs.shape
    out = np.zeros(r)
    post = normalize(lexica_prior * np.transpose(lhs))

    for i in xrange(r):
        for d in xrange(c):
            sum_t = 0
            for t in xrange(r):
                sum_t += lhs[t,d] * p[t]
            
            out[i] += sum_t * post[d,i]
    return out



def weighted_mutation(p,q,w1,w2):
    """Mutation by community learning instead of standard mutation"""
    m = np.zeros([len(p),len(p)])

    pop_learn_vector = population_learn_vector(likelihood_matrix,p)
    for i in range(len(p)):
        for j in range(len(p)):
            m[i,j] = (w1 * pop_learn_vector[j]) + (w2 * q[i,j])
    return normalize(m)

def get_utils():
    #Compute utilities for a single lexical pair first
    out = np.zeros([len(typeList), len(typeList)])
    for i in range(len(typeList)):
        for j in range(len(typeList)):
            out[i,j] = (np.sum(typeList[i].sender_matrix * np.transpose(typeList[j].receiver_matrix)) + \
                     np.sum(typeList[j].sender_matrix * np.transpose(typeList[i].receiver_matrix))) / 4

    #Combine utilities for 3 x lexical pairs
    macro_out = np.zeros([len(labels), len(labels)])
    for i in range(len(labels)):
        for j in range(len(labels)):
                macro_out[i,j] = (out[labels[i][0],labels[j][0]] + out[labels[i][1],labels[j][1]] + out[labels[i][2],labels[j][2]]) / 3.
    return macro_out


print '#Computing Q, ', datetime.datetime.now()

q = get_mutation_matrix(likelihood_matrix,k)

for i in q:
    f_q.writerow(i)

print '#Computing utilities, ', datetime.datetime.now()

u = get_utils()

###Multiple runs
print '#Beginning multiple runs, ', datetime.datetime.now()

f_unwgh_mean.writerow(["stype","l1","l2","l3","proportion"])
f_wgh_mean.writerow(["stype","l1","l2","l3","proportion"])

p_sum = np.zeros(len(labels))
p_sum_wgh = np.zeros(len(labels))

for i in xrange(runs):

    p = np.random.dirichlet(np.ones(len(labels))) # unbiased random starting state

    for r in range(gens): #unweighted mutation
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(labels))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, q)

    p_sum += p

    p = np.random.dirichlet(np.ones(len(labels))) # unbiased random starting state

    for r in range(gens):  #weighted (community) mutation
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(labels))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, weighted_mutation(p,q,w_communal,w_parental))

    p_sum_wgh += p

    print i, datetime.datetime.now()
    
    
p_mean = p_sum / runs
p_wgh_mean = p_sum_wgh / runs

def record_by_type(p_vec,csv_file):
    for i in range(len(p_vec)):
        if i < len(p_vec)/2:
            stype = 'literal'
        else: stype = 'gricean'
        l1,l2,l3 = labels[i][0],labels[i][1],labels[i][2]
        if l1 > len(lexica)-1: l1 = l1 - len(lexica)
        if l2 > len(lexica)-1: l2 = l2 - len(lexica)
        if l3 > len(lexica)-1: l3 = l3 - len(lexica)

        csv_file.writerow([stype,l1,l2,l3,str(p_vec[i])])

record_by_type(p_mean,f_unwgh_mean)
record_by_type(p_wgh_mean,f_wgh_mean)


def lexica_vec(p_vec): 
    p_by_lexica = np.zeros(len(lexica))
    for i in xrange(len(p_vec)):
        prop = 1/3. * p_vec[i]
        l1,l2,l3 = labels[i][0], labels[i][1],labels[i][2]
        if l1 > 5: l1 = l1 - 6
        if l2 > 5: l2 = l2 - 6
        if l3 > 5: l3 = l3 - 6
        p_by_lexica[l1] += prop
        p_by_lexica[l2] += prop
        p_by_lexica[l3] += prop
    return p_by_lexica

print '### Quick overview of results###'
print 'Unweighted mean by lexica:', lexica_vec(p_mean)
print '###'
print 'Weighted mean by lexica:', lexica_vec(p_wgh_mean)
sys.exit()


