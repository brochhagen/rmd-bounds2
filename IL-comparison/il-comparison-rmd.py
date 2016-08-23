import numpy as np
#np.set_printoptions(threshold=np.nan)
from itertools import product,permutations
from random import choice,sample
from player import Player
import sys 
import datetime
import csv

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]

def norm(m):
    m = m / m.sum(axis=1)[:, np.newaxis]
    m[np.isnan(m)] = 1. / np.shape(m)[1]
    return m

#lexicon = np.array([[1,0,1],[0,1,1]])
s = 4 #number of states
n = 10 #observation sequence length
generations = 10000 #generations per game
games = 1000 #amount of independent games
alpha = 0.5 #bias parameter
epsilon = 0.05
mc = 1000 #amount of samples


f_il_samp_mean = csv.writer(open('./results/comparison-il-samp-mean-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 
f_unwgh_samp_mean = csv.writer(open('./results/comparison-unwgh-samp-mean-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 
f_wgh_samp_mean = csv.writer(open('./results/comparison-wgh-samp-mean-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 

f_il_map_mean= csv.writer(open('./results/comparison-il-map-mean-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 
f_unwgh_map_mean= csv.writer(open('./results/comparison-unwgh-map-mean-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 
f_wgh_map_mean = csv.writer(open('./results/comparison-wgh-map-mean-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 

f_prior = csv.writer(open('./results/prior-a%d-e%f-k%d-g%d-r%d.csv' %(alpha,epsilon,n,generations,games),'wb')) 

print '#Starting, ', datetime.datetime.now()

def lexica(s):
    values = list(set(permutations((1.,0.,0.,0.), s )))
    return [np.array([[i[0],i[1], i[2], i[3]],[j[0],j[1],j[2], j[3]],[k[0],k[1],k[2], k[3]],[l[0],l[1],l[2], l[3]]]) for i in values for j in values for k in values for l in values]

hypotheses = lexica(s)


def compositional(H):
    compo = []
    for i in range(len(H)):
        ambcount = 0
        for j in np.transpose(H[i]):
            if np.sum(j) > 1:
                ambcount += 10
            else: 
                ambcount += 1
            if ambcount == 4:
                if np.argmax(H[i][0])+1 == np.argmax(H[i][1]) == np.argmax(H[i][2])-1 and np.argmax(H[i][1])+2 == np.argmax(H[i][3]):
                    compo.append(i)
                elif np.argmax(H[i][0])+2 == np.argmax(H[i][1]) == np.argmax(H[i][2])+1 and np.argmax(H[i][1])+1 == np.argmax(H[i][3]):
                    compo.append(i)
                elif np.argmax(H[i][0]) == np.argmax(H[i][1])+2 == np.argmax(H[i][2])+1 and np.argmax(H[i][1]) == np.argmax(H[i][3])+1:
                    compo.append(i)
                elif np.argmax(H[i][0]) == (np.argmax(H[i][1])+1) == np.argmax(H[i][2])+2 and np.argmax(H[i][1]) == np.argmax(H[i][3])+2:
                    compo.append(i)
    return compo

compIndices = compositional(hypotheses)
print '#Indices of compositional languages:', compIndices

##Adding 4 more holistic languages that are identical to the compositional ones (as Griffiths + Kalish do####
#for i in compIndices:
#    hypotheses.append(hypotheses[i])
###

def learning_prior(a,H,cIndices): #alpha, hypothesis space, indices of compositional languages
    prior = np.zeros(len(H))
    for i in range(len(H)):
        if i in cIndices:
            prior[i] = a / len(cIndices)
        else:
            prior[i] = (1 - a) / (len(H) - len(cIndices))
    return prior
prior = learning_prior(alpha,hypotheses,compIndices)

print '#Computing likelihood, ', datetime.datetime.now()
likelihoods = np.array([h.senderMatrix() for h in [Player(s,epsilon) for s in hypotheses]])



def get_obs(k):
    inputx = [x for x in list(product(range(4), repeat=k)) if sum(x) == k] #k-tuple where the i-th observed state was state k_i 
    outputy = [y for y in list(product(range(4),repeat=k)) if sum(y) == k] #k-tuple where the i-th produced message was k_i 
    if k > 4:
#	D = [(choice(inputx),choice(inputy)) for _ in xrange(mc)]
	prod = len(inputx) * len(outputy)
	indices = sample(xrange(prod), mc)
	D = [(inputx[idx % len(inputx)], outputy[idx // len(inputx)]) for idx in indices]
    else:
	D = list(product(inputx,outputy))

    
    out = []
    for i in range(len(D)):
        s0,lm0 = [r for r,x in enumerate(D[i][0]) if x == 0], np.zeros(4) # (i) indices, for each observation, in which state s_0 was witnessed, (ii) list to store the message uttered at that time for s_0
        s1,lm1 = [r for r,x in enumerate(D[i][0]) if x == 1], np.zeros(4)
        s2,lm2 = [r for r,x in enumerate(D[i][0]) if x == 2], np.zeros(4)
        s3,lm3 = [r for r,x in enumerate(D[i][0]) if x == 3], np.zeros(4)
        for j in range(k): 
            if j in s0: 
                lm0[D[i][1][j]] += 1 #whenever s_0 was uttered, check what message was uttered and store it, position-wise. E.g. [0,3,0,0] means that in this obervation m_2 was produced three times and no other message was produced.
            elif j in s1: lm1[D[i][1][j]] += 1
            elif j in s2: lm2[D[i][1][j]] += 1
            elif j in s3: lm3[D[i][1][j]] += 1
        out.append([lm0,lm1,lm2,lm3])
    return out #,D



def get_obs_likelihood(k):
    obs = get_obs(k)
    out = np.zeros([len(likelihoods),len(obs)])

    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            out[lhi,o] = (likelihoods[lhi,0,0]**obs[o][0][0] * likelihoods[lhi,0,1]**obs[o][0][1] *\
                          likelihoods[lhi,0,2]**obs[o][0][2] * likelihoods[lhi,0,3]**obs[o][0][3] *\
                          likelihoods[lhi,1,0]**obs[o][1][0] * likelihoods[lhi,1,1]**obs[o][1][1] *\
                          likelihoods[lhi,1,2]**obs[o][1][2] * likelihoods[lhi,1,3]**obs[o][1][3] *\
                          likelihoods[lhi,2,0]**obs[o][2][0] * likelihoods[lhi,2,1]**obs[o][2][1] *\
                          likelihoods[lhi,2,2]**obs[o][2][2] * likelihoods[lhi,2,3]**obs[o][2][3] *\
                          likelihoods[lhi,3,0]**obs[o][3][0] * likelihoods[lhi,3,1]**obs[o][3][1] *\
                          likelihoods[lhi,3,2]**obs[o][3][2] * likelihoods[lhi,3,3]**obs[o][3][3])
    return normalize(out)

#lhs = get_obs_likelihood(n)

def get_mutation_matrix(k):
    lhs = get_obs_likelihood(k)
    post = normalize(prior * np.transpose(lhs))
    return np.dot(lhs,post)

def get_mutation_matrix_map(k):
    lhs = get_obs_likelihood(k)
    post = normalize(prior * np.transpose(lhs))
    q = np.dot(lhs,post)
    r,c = np.shape(q)
    map_q = np.zeros((r,c))
    for i in xrange(r):
        map_q[i,np.argmax(q[i])] = 1
    return map_q

def weighted_mutation(p,q):
    """Mutation by community learning instead of standard mutation"""
    m = np.zeros([len(p),len(p)])
    for i in range(len(p)):
        for j in range(len(p)):
            m[i,j] = p[j] * q[i,j]
    return norm(m)


print '#Computing Q for sampling, ', datetime.datetime.now()
q = get_mutation_matrix(n)

print '#Computing Q for MAP, ', datetime.datetime.now()
q_map = get_mutation_matrix_map(n)


def get_utils():
    out = np.zeros([len(hypotheses), len(hypotheses)])
    receivers = np.array([np.transpose(h.receiverMatrix()) for h in [Player(s,epsilon) for s in hypotheses]])
    for i in range(len(hypotheses)):
        for j in range(len(hypotheses)):
            out[i,j] = (np.sum(likelihoods[i] * receivers[j]) + \
                        np.sum(likelihoods[j] * receivers[i])) /8.
    return out

print '#Computing utilities, ', datetime.datetime.now()

u = get_utils()

print '#Beginning multiple runs, ', datetime.datetime.now()

f_il_samp_mean.writerow(["lexicon","proportion"])
f_unwgh_samp_mean.writerow(["lexicon","proportion"])
f_wgh_samp_mean.writerow(["lexicon","proportion"])

f_il_map_mean.writerow(["lexicon","proportion"])
f_unwgh_map_mean.writerow(["lexicon","proportion"])
f_wgh_map_mean.writerow(["lexicon","proportion"])

f_prior.writerow(["lexicon","proportion"])


p_il_samp_sum = np.zeros(len(hypotheses))
p_unwgh_samp_sum = np.zeros(len(hypotheses))
p_wgh_samp_sum = np.zeros(len(hypotheses))

p_il_map_sum = np.zeros(len(hypotheses))
p_unwgh_map_sum = np.zeros(len(hypotheses))
p_wgh_map_sum = np.zeros(len(hypotheses))

p_prior = prior

###Multi games####

for i in range(games):
    print '#game', i, datetime.datetime.now()
    print '#game', i, 'IL-sample', datetime.datetime.now()


    p = np.random.dirichlet(np.ones(len(hypotheses))) # unbiased random starting state

    for r in range(generations):
        p = np.dot(p, q)

    p_il_samp_sum += p

    print '#game', i, 'unweighted-sample', datetime.datetime.now()

    p = np.random.dirichlet(np.ones(len(hypotheses))) # unbiased random starting state

    for r in range(generations):
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(hypotheses))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, q)

    p_unwgh_samp_sum += p

    print '#game', i, 'weighted-sample', datetime.datetime.now()

    p = np.random.dirichlet(np.ones(len(hypotheses))) # unbiased random starting state

    for r in range(generations):  #weighted (community) mutation
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(hypotheses))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, weighted_mutation(p,q))

    p_wgh_samp_sum += p
#MAP

    print '#game', i, 'IL-MAP', datetime.datetime.now()

    p = np.random.dirichlet(np.ones(len(hypotheses))) # unbiased random starting state

    for r in range(generations):
        p = np.dot(p, q_map)

    p_il_map_sum += p

    print '#game', i, 'unweighted-MAP', datetime.datetime.now()

    p = np.random.dirichlet(np.ones(len(hypotheses))) # unbiased random starting state

    for r in range(generations):
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(hypotheses))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, q_map)

    p_unwgh_map_sum += p

    print '#game', i, 'weighted-MAP', datetime.datetime.now()

    p = np.random.dirichlet(np.ones(len(hypotheses))) # unbiased random starting state

    for r in range(generations):  #weighted (community) mutation
        pPrime = p * [np.sum(u[t,] * p)  for t in range(len(hypotheses))]
        pPrime = pPrime / np.sum(pPrime)
        p = np.dot(pPrime, weighted_mutation(p,q_map))

    p_wgh_map_sum += p



p_il_samp_mean = p_il_samp_sum / games 
p_unwgh_samp_mean = p_unwgh_samp_sum / games
p_wgh_samp_mean = p_wgh_samp_sum / games

p_il_map_mean = p_il_map_sum / games 
p_unwgh_map_mean = p_unwgh_map_sum / games
p_wgh_map_mean = p_wgh_map_sum / games

for i in xrange(len(p_il_samp_mean)):
        f_il_samp_mean.writerow([str(i),str(p_il_samp_mean[i])])
        f_unwgh_samp_mean.writerow([str(i),str(p_unwgh_samp_mean[i])])
        f_wgh_samp_mean.writerow([str(i),str(p_wgh_samp_mean[i])])

        f_il_map_mean.writerow([str(i),str(p_il_map_mean[i])])
        f_unwgh_map_mean.writerow([str(i),str(p_unwgh_map_mean[i])])
        f_wgh_map_mean.writerow([str(i),str(p_wgh_map_mean[i])])

        f_prior.writerow([str(i),str(prior[i])])

