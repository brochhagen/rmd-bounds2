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



def get_some_analysis(group1,group2,group3):
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
    
    df_m = df_m.loc[df_m['lam'] == group2[0]]
    df_m = df_m.loc[df_m['alpha'] == group2[1]]
    df_m = df_m.loc[df_m['k'] == group2[2]]
    df_m = df_m.loc[df_m['samples'] == group2[3]]
    df_m = df_m.loc[df_m['l'] == group2[4]]
    df_m = df_m.loc[df_m['gens'] == group2[5]]
    df_m = df_m.loc[df_m['runs'] == group2[6]]
    final_m = df_m.loc[:,'t_mean0':]
    
    df_rm = df_rm.loc[df_rm['lam'] == group3[0]]
    df_rm = df_rm.loc[df_rm['alpha'] == group3[1]]
    df_rm = df_rm.loc[df_rm['k'] == group3[2]]
    df_rm = df_rm.loc[df_rm['samples'] == group3[3]]
    df_rm = df_rm.loc[df_rm['l'] == group3[4]]
    df_rm = df_rm.loc[df_rm['gens'] == group3[5]]
    df_rm = df_rm.loc[df_rm['runs'] == group3[6]]
    final_rm = df_rm.loc[:,'t_mean0':]
    
    r_array = map(list,final_r.values)[0]
    r_sorted = sorted(r_array,reverse=True)
    r_top = r_sorted[:3]
    
    m_array = map(list,final_m.values)[0]
    m_sorted = sorted(m_array,reverse=True)
    m_top = m_sorted[:3]
    
    rm_array = map(list,final_rm.values)[0]
    rm_sorted = sorted(rm_array,reverse=True)
    rm_top = rm_sorted[:3]
    
    
    idx_r,idx_m,idx_rm = [], [], []
    
    for i in r_top:
        index = r_array.index(i)
        idx_r.append(index)
    
    for i in m_top:
        index = m_array.index(i)
        idx_m.append(index)
    
    for i in rm_top:
        index = rm_array.index(i)
        idx_rm.append(index)
    
    if 24 not in idx_r:
        r_top.pop()
        idx_r.pop()
        r_top.append(r_array[24])
        idx_r.append(24)
    
    if 24 not in idx_m:
        m_top.pop()
        idx_m.pop()
        m_top.append(m_array[24])
        idx_r.append(24)
    
    if 24 not in idx_rm:
        rm_top.pop()
        idx_rm.pop()
        rm_top.append(rm_array[24])
        idx_r.append(24)
    
    
    X = np.arange(21)
    empty = [0 for _ in xrange(3)]
    
    Y_r = empty + r_top + empty * 5
    Y_m = empty * 3 + m_top + empty * 3
    Y_rm = empty * 5 + rm_top + empty

    
    fig, ax = plt.subplots()
    ax.bar(X,Y_r,width=1, color='green')
    ax.bar(X,Y_m,width=1, color='blue')
    ax.bar(X,Y_rm,width=1, color='red')
    
    r_labels = ['t'+str(x) for x in idx_r]
    m_labels = ['t'+str(x) for x in idx_m]
    rm_labels = ['t'+str(x) for x in idx_rm]

    ax.set_xticks(xrange(21))
    ax.set_xticklabels(['', '', '', r_labels[0], r_labels[1],r_labels[2], '', '', '', m_labels[0], m_labels[1],m_labels[2], '', '', '',\
                        rm_labels[0], rm_labels[1],rm_labels[2], '', '', ''])
    
    plt.show()

#group1,group2,group3 = [30,1,5,200,1,50,1000], [30,1,5,200,1,50,1000],[30,1,5,200,1,50,1000]
#get_some_analysis(group1,group2,group3)
#
#group1,group2,group3 = [30,1,5,200,10,50,1000], [30,1,5,200,10,50,1000],[30,1,5,200,10,50,1000]
#get_some_analysis(group1,group2,group3)
#
#group1,group2,group3 = [30,1,15,200,1,50,1000], [30,1,15,200,1,50,1000],[30,1,15,200,1,50,1000]
#get_some_analysis(group1,group2,group3)
#
#group1,group2,group3 = [30,1,15,200,10,50,1000], [30,1,15,200,10,50,1000],[30,1,15,200,10,50,1000]
#get_some_analysis(group1,group2,group3)


sys.exit()
