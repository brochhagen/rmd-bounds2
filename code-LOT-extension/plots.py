###
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_context(rc={'lines.markeredgewidth': 0.5})
import os
import glob
import sys
###
###Column headers:###
#####################


def get_prior_plot():
    from lexica import get_prior, get_lexica
    prior = get_prior(get_lexica(3,3,mutual_exclusivity=True))
    
    X = np.arange(40)
    Y_target = [0 for _ in xrange(24)] + [prior[24]] + [0 for _ in xrange(15)]
    print prior[24]
    prior[24] = 0
    Y_rest = prior
    print prior[24]
    
    fig, ax = plt.subplots()
    ax.bar(X,Y_target,width=1, color='green')
    ax.bar(X,Y_rest,width=1)
    plt.show()

#get_prior_plot()

def get_utility_heatmap(s_amount,m_amount,lam,alpha):
    print 'Loading U-matrix'
    df = pd.read_csv('./matrices/umatrix-s%d-m%d-lam%d-a%d.csv' %(s_amount,m_amount,lam,alpha))
    yticks = [0] + ["" for _ in xrange(9)] + [10] + ["" for _ in xrange(9)] +\
             [20] + ["" for _ in xrange(9)] + [30] + ["" for _ in xrange(9)]
    
    ax = sns.heatmap(df, yticklabels=yticks, xticklabels=yticks)#, annot=True) 
#    plt.yticks(rotation=0)
    from matplotlib.patches import Rectangle
    
    ax.add_patch(Rectangle((24,15), 1, 1, fill=False, edgecolor='blue', lw=3))
    ax.invert_yaxis()
    plt.show()

#get_utility_heatmap(3,3,10,1)


def get_mutation_heatmap(s_amount,m_amount,lam,alpha,k,samples,l):
    print 'Loading Q-matrix'
    df = pd.read_csv('./matrices/qmatrix-s%d-m%d-lam%d-a%d-k%d-samples%d-l%d.csv' %(s_amount,m_amount,lam,alpha,k,samples,l))
    yticks = [0] + ["" for _ in xrange(9)] + [10] + ["" for _ in xrange(9)] +\
             [20] + ["" for _ in xrange(9)] + [30] + ["" for _ in xrange(9)]
    
    ax = sns.heatmap(df, yticklabels=yticks, xticklabels=yticks, cmap="YlGnBu")#, annot=True) 
#    plt.yticks(rotation=0)
    from matplotlib.patches import Rectangle
    
    ax.add_patch(Rectangle((24,15), 1, 1, fill=False, edgecolor='black', lw=3))
    ax.invert_yaxis()
    plt.show()

#get_mutation_heatmap(3,3,30,1,5,200,1)
#get_mutation_heatmap(3,3,30,1,5,200,10)
#get_mutation_heatmap(3,3,30,1,15,200,1)
#get_mutation_heatmap(3,3,30,1,15,200,10)
#

group1 = [30,1,5,200,1,50,1000]
#def get_some_analysis(group1,group2,group3):
print 'Loading data'
df_r = pd.read_csv('./results/00mean-r-s3-m3-g50-r1000.csv')
df_m = pd.read_csv('./results/00mean-m-s3-m3-g50-r1000.csv')
df_rm = pd.read_csv('./results/00mean-rmd-s3-m3-g50-r1000.csv')

df_r = df_r.loc[df_r['lam'] == group1[0]]
df_r = df_r.loc[df_r['alpha'] == group1[1]]
df_r = df_r.loc[df_r['k'] == group1[2]]
df_r = df_r.loc[df_r['samples'] == group1[3]]
df_r = df_r.loc[df_r['l'] == group1[4]]
df_r = df_r.loc[df_r['gens'] == group1[5]]
df_r = df_r.loc[df_r['runs'] == group1[6]]
final_r = df_r.loc[:,'t_mean0':]

#target_lex = final_r['t_mean24']

r_array = map(list,final_r.values)
r_array = r_array[0]

target = r_array[24]
r_sorted = sorted(r_array,reverse=True)
r_top = r_sorted[:3]
idx = []
for i in r_top:
    index = r_array.index(i)
    idx.append(index)

if 24 not in idx:
    r_top.pop()
    idx.pop()
    r_top.append(target)
    idx.append(24)


X = np.arange(27)
empty = [0 for _ in xrange(3)]
Y_r = empty + r_top + empty * 7

fig, ax = plt.subplots()
ax.bar(X,Y_r,width=1, color='green')

r_labels = ['t'+str(x) for x in idx]

ax.set_xticks(xrange(27))
ax.set_xticklabels(['', '', '', '', r_labels[0], r_labels[1],r_labels[2]] + ['' for x in xrange(21)])
#ax.set_xticklabels([r_labels[0], r_labels[1],r_labels[2]])

#ax.bar(X,Y_rest,width=1)
plt.show()

#    df_m = df_m.loc[df['lam' = group1[0]]
#    df_m = df_m.loc[df['alpha' = group1[1]]
#    df_m = df_m.loc[df['k' = group1[2]]
#    df_m = df_m.loc[df['samples' = group1[3]]
#    df_m = df_m.loc[df['l' = group1[4]]
#    df_m = df_m.loc[df['gens' = group1[5]]
#    df_m = df_m.loc[df['runs' = group1[6]]
#    final_m = df_m.loc[:,'t_mean0':]
#
#
#    df_rm = df_rm.loc[df['lam' = group1[0]]
#    df_rm = df_rm.loc[df['alpha' = group1[1]]
#    df_rm = df_rm.loc[df['k' = group1[2]]
#    df_rm = df_rm.loc[df['samples' = group1[3]]
#    df_rm = df_rm.loc[df['l' = group1[4]]
#    df_rm = df_rm.loc[df['gens' = group1[5]]
#    df_rm = df_rm.loc[df['runs' = group1[6]]
#    final_rm = df_rm.loc[:,'t_mean0':]


sys.exit()
