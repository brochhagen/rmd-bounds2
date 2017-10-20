import sys,os
lib_path = os.path.abspath(os.path.join('..')) #specifying path for player module
sys.path.append(lib_path) #specifying path for player module
from player import LiteralPlayer, GriceanPlayer
import numpy as np
import matplotlib.pyplot as plt
from itertools import product, combinations

lam = 20
type_indices = [0,2] #lb,ll,pb,pl
bias_para = 2
k = 5
post_para = 15


lexica = [np.array([[1.,0.],[0.,1.]]),np.array([[1.,0.],[1.,1.]])]
typeList = [LiteralPlayer(lam,lex) for lex in lexica] + [GriceanPlayer(1,lam,lex) for lex in lexica]
types = [typeList[x] for x in type_indices]

def normalize(m):
    return m / m.sum(axis=1)[:, np.newaxis]


def get_u(types):
    out = np.zeros([len(types), len(types)])
    for i in xrange(len(types)):
        for j in xrange(len(types)):
            out[i,j] = (np.sum(types[i].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5 +\
                       (np.sum(types[j].sender_matrix * np.transpose(types[j].receiver_matrix)) / 2.) * 0.5
    return out

U = get_u(types)

def fitness(x_prop,U):
    return np.sum((np.array([x_prop, 1-x_prop]) * U)[0])

def overall_fitness(x_prop,U):
    return (x_prop * np.sum((np.array([x_prop,1-x_prop]) * U)[0])) +\
           ((1.-x_prop) * np.sum((np.array([x_prop,1-x_prop])* U)[1]))

def replicator_step(x_prop):
    return (x_prop * fitness(x_prop,U)) / overall_fitness(x_prop,U)


def obs_counts(obs):
    out = []
    for i in xrange(len(obs)):
        out.append([obs[i].count(j) for j in xrange(4)])
    return out


def get_likelihood(obs,likelihoods):
    out = np.zeros([len(likelihoods), len(obs)]) # matrix to store results in
    for lhi in range(len(likelihoods)):
        for o in range(len(obs)):
            flat_lhi = likelihoods[lhi].flatten()
            out[lhi,o] = np.prod([flat_lhi[x]**obs[o][x] for x in xrange(len(obs[o]))]) 
    return out


def get_q(lexica_prior,learning_parameter,k):
    likelihoods = [i.sender_matrix for i in types]
    atomic_obs = [0,1,2,3] #0,1 upper-row; 2,3 are lower row
    D = list(product(atomic_obs,repeat=k))  
    D = obs_counts(D)
    
#    out = np.zeros([len(likelihoods),len(likelihoods)]) #matrix to store Q
    lhs = get_likelihood(D,likelihoods)
    post = normalize(lexica_prior * np.transpose(lhs))
    parametrized_post = normalize(post**learning_parameter)
    return normalize(np.dot(lhs,parametrized_post))


def learning_prior(types,c):
    out = np.zeros(len(types))
    for i in xrange(len(types)):
        lx = types[i].lexicon
        if np.sum(lx) == 2.: #lexicalized upper-bound
            out[i] = 1
        else:
            out[i] = 1*c
    return out / np.sum(out)


lexica_prior = learning_prior(types,bias_para)
q = get_q(lexica_prior,post_para,k)


def mutator_step(x_prop):
    xv = np.array([x_prop,1-x_prop])
    return np.dot(xv,q)[0]

print q
print mutator_step(0.9)



#Plots
X = np.linspace(0,1,1000) #x-coord for tail
rep = np.array([replicator_step(prop) for prop in X])
rep_diff = rep - X
line = np.zeros(len(X)) #no change in x

mut = np.array([mutator_step(prop) for prop in X])
mut_diff = mut - X

rmd = np.array([mutator_step(replicator_step(prop)) for prop in X])
rmd_diff = rmd - X

#Now plot direction of change and stationary points

def get_points_of_change_heuristic(d_diff,u):
    no_change_at = np.where(d_diff==0)[0] #stationary points
    attractor = []
    no_attractor = []
    for i in xrange(len(no_change_at)):
        if X[no_change_at[i]] == 0:
            if X[1] > u[1]:
                attractor.append(X[no_change_at[i]])
            else:
                no_attractor.append(X[no_change_at[i]])
        elif X[no_change_at[i]] == X[-1]:
            if X[-2] < u[-2]:
                attractor.append(X[no_change_at[i]])
            else:
                no_attractor.append(X[no_change_at[i]])
        else:
            if X[no_change_at[i]-1] < u[no_change_at[i]-1] and X[no_change_at[i]+1] < u[no_change_at[i]+1]:
                attractor.append(X[no_change_at[i]])
            else:
                no_attractor.append(X[no_change_at[i]])
    return [attractor,no_attractor]

def get_directionality_heuristic(d_diff,attractor,no_attractor):
    no_change_at = np.where(d_diff==0)[0] #stationary points
    to_compare = list(combinations([X[i] for i in no_change_at],2)) #pairwise groupings
    markers_left = []
    markers_right = []
    if not(len(to_compare) == 0):
         for i in to_compare:
            h = (max(i) - min(i)) / 2.
            hh = h/2.
            if max(i) in attractor and min(i) in no_attractor:
                markers_right.append(h-hh)
                markers_right.append(h+hh)
            elif min(i) in attractor and max(i) in no_attractor:
                markers_left.append(h-hh)
                markers_left.append(h+hh)
    else:
        steps = len(d_diff) * 1/8.
        for i in xrange(0,8+1):
            idx = int(i * steps-1)
            if d_diff[idx] > 0:
                markers_right.append(X[idx])
            else:
                markers_left.append(X[idx])
    return [markers_left,markers_right]


rep_stat = get_points_of_change_heuristic(rep_diff,rep)
rep_attr,rep_no_attr = rep_stat[0],rep_stat[1]
rep_dir = get_directionality_heuristic(rep_diff,rep_attr,rep_no_attr)
rep_left,rep_right = rep_dir[0],rep_dir[1]


mut_stat = get_points_of_change_heuristic(mut_diff,mut)
mut_attr,mut_no_attr = mut_stat[0],mut_stat[1]
mut_dir = get_directionality_heuristic(mut_diff,mut_attr,mut_no_attr)
mut_left,mut_right = mut_dir[0],mut_dir[1]

rmd_stat = get_points_of_change_heuristic(rmd_diff,rmd)
rmd_attr,rmd_no_attr = rmd_stat[0],rmd_stat[1]
rmd_dir = get_directionality_heuristic(rmd_diff,rmd_attr,rmd_no_attr)
rmd_left,rmd_right = rmd_dir[0],rmd_dir[1]



#plot replicator
plt.plot(X,np.ones(len(X)), color='black')
plt.plot(rep_left,np.ones(len(rep_left)),marker='3',color='black',markersize=10)
plt.plot(rep_right,np.ones(len(rep_right)), marker='4',color='black', markersize=10)
plt.scatter(rep_attr,np.ones(len(rep_attr)),s=80,color='black')
plt.scatter(rep_no_attr,np.ones(len(rep_no_attr)), s=80, color='white',edgecolor='black')

plt.annotate('RD', xy=(0.5,1.1), xycoords='data')


#plot mutator
plt.plot(X,np.zeros(len(X)), color='black')
plt.plot(mut_left,np.zeros(len(mut_left)),marker='3',color='black',markersize=10)
plt.plot(mut_right,np.zeros(len(mut_right)), marker='4',color='black', markersize=10)
plt.scatter(mut_attr,np.zeros(len(mut_attr)),s=80,color='black')
plt.scatter(mut_no_attr,np.zeros(len(mut_attr)),s=80,color='white',edgecolor='white')
plt.annotate('M', xy=(0.5,0.1), xycoords='data')

plt.plot(X,np.array([-1 for _ in xrange(len(X))]), color='black')
plt.plot(rmd_left,np.array([-1 for _ in xrange(len(rmd_left))]),marker='3',color='black',markersize=10)
plt.plot(rmd_right,np.array([-1 for _ in xrange(len(rmd_right))]), marker='4',color='black', markersize=10)
plt.scatter(rmd_attr,np.array([-1 for _ in xrange(len(rmd_attr))]),s=80,color='black')
plt.scatter(rmd_no_attr,np.array([-1 for _ in xrange(len(rmd_no_attr))]),s=80,color='white',edgecolor='white')
plt.annotate('RMD', xy=(0.5,-0.9), xycoords='data')

labels = []
for i in type_indices:
    if i == 0:
        b = 'lit.'
        l = 'bound'
    elif i == 1:
        b = 'lit.'
        l = 'lack'
    elif i == 2:
        b = 'prag.'
        l = 'bound'
    else:
        b = 'prag.'
        l = 'lack'
    labels.append(b)
    labels.append(l)

plt.suptitle(r'%s-$L_{%s}$ (x) vs. %s-$L_{%s}$ (1-x)' % (labels[0],labels[1],labels[2],labels[3]),fontsize=14)
plt.title(r'($\lambda$ = %d, bias = %d, l = %d, k = %d)' % (lam,bias_para,post_para,k), y=1.29)

plt.axis('off')
plt.tight_layout()
#plt.show()


#plt.plot(x,np.zeros(len(x)),color='black', marker='o',markevery=[int(attr) for attr in no_attractor],fillstyle='full',markerfacecolor='white',markeredgecolor='black',markersize=8)
#plt.plot(markers_left,np.zeros(len(markers_left)), marker='3',color='black',markersize=10)
#plt.plot(markers_right,np.zeros(len(markers_right)), marker='4',color='black',markersize=10)
#plt.scatter(attractor,np.zeros(len(attractor)),s=80,color='black')
##plt.scatter(no_attractor,np.zeros(len(no_attractor)), s=80, facecolors='white',edgecolors='black',color='white')
#plt.axis('off')
#plt.tight_layout()
#plt.show()



##############

#COLORED CMAP FOR QUIVERS ACCORDING TO INTENSITY
#import matplotlib
#norm = matplotlib.colors.Normalize()
##norm.autoscale(v/np.sum(v))
#cm = matplotlib.cm.cool #colormap name
#sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
#sm.set_array([])

#plt.quiver(x,y,u,u,angles='xy',scale_units='xy',scale=1,color=cm(norm(v))) #x-coord for tail, y-coord for tail, length of vector along x, y direction of head, anglex='xy' makes the arrow point from tail of the vector to its tip.

#plt.quiver(x,y,line,line,angles='xy',scale_units='xy',scale=1) #x-coord for tail, y-coord for tail, length of vector along x, y direction of head, anglex='xy' makes the arrow point from tail of the vector to its tip.

#plt.colorbar(sm)

##################

from scipy import optimize
def find_rep(x_prop):
    return abs(replicator_step(x_prop) - x_prop)

def find_mut(x_prop):
    return abs(mutator_step(x_prop) - x_prop)

def find_rmd(x_prop):
    return abs(mutator_step(replicator_step(x_prop)) - x_prop)

#
#
x0 = 0.2
##res1 = optimize.fmin_cg(replicator_step,x0)
#res1 = optimize.minimize(find_rep,x0,bounds=((0.,1.),))
#res2 = optimize.minimize(find_mut,x0,bounds=((0.,1.),))
#res3 = optimize.minimize(find_rmd,x0,bounds=((0.,1.),))

#minimizer_kwargs = dict(method='L-BFGS-B',bounds=((0.,1.),)) #, 
#
def print_fun(x,f,accepted):
    print "at minima %.4f accepted %d" % (f, int(accepted))

def mybounds(**kwargs):
    x = kwargs["x_new"]
    tmax = bool(np.all(x <= 1.0))
    tmin = bool(np.all(x >= 0.0))
    print x
    print tmin and tmax
    return tmax and tmin

res1 = optimize.basinhopping(find_mut,x0,accept_test=mybounds,callback=print_fun)
#print res1
#
#

