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
#run_ID,t1_initial,t2_initial,t3_initial,t4_initial,t5_initial,t6_initial,t7_initial,t8_initial,t9_initial,t10_initial,t11_initial,t12_initial,
#alpha,prior_cost_c,lambda,k,sample_amount,learning_parameter,generations,
#t1_final,t2_final,t3_final,t4_final,t5_final,t6_final,t7_final,t8_final,t9_final,t10_final,t11_final,t12_final
#####################


def fitness_only_plot():
    print 'Loading data fitness'
    df = pd.read_csv('./mean-1000games-replication-only.csv')

    
    a = 1
    lam = 30
    
    final_groupA = df.loc[:,'alpha':]
    
    groupA = final_groupA.loc[final_groupA['alpha'] == a]
    
    t_finalA = groupA.groupby(['lambda'])
    t_finalA = t_finalA[['t1_final','t2_final','t3_final','t4_final','t5_final','t6_final','t7_final','t8_final','t9_final','t10_final','t11_final','t12_final']].agg(np.average) 
    
    d1 = t_finalA.loc[:,('t1_final','t2_final','t3_final', 't4_final', 't5_final', 't6_final')]
    d2 = t_finalA.loc[:,('t9_final','t10_final', 't11_final')]
    
    ax2 = plt.plot(d2)
    #ax1 = plt.plot(d1,marker='d',markevery=5)
    plt.title(r'($\alpha$ = %d)' %(a))
    plt.ylabel("Proportion in population")
    plt.xlabel(r'Rationality parameter $\lambda$')
    
    plt.legend(["prag. L-taut", "prag. L-bound","prag. L-lack"], loc='best')
    plt.xlim(0,48)
    plt.show() 


def learning_only_plot():
    print 'Loading data learn'
    dl = pd.read_csv('./mean-1000games-mutation-only.csv')
    
    a = 1
    lam = 30
    k = 5
    sample = 10
    learn = 1
    
    final_groupB = dl.loc[:,'alpha':]
    groupB = final_groupB.loc[final_groupB['alpha'] == a]
    groupB = groupB.loc[groupB['k'] == k]
    groupB = groupB.loc[groupB['lambda'] == lam]
    groupB = groupB.loc[groupB['sample_amount'] == sample]
    groupB = groupB.loc[groupB['learning_parameter'] == learn]
    
    t_finalB = groupB.groupby(['prior_cost_c'])
    t_finalB = t_finalB[['t1_final','t2_final','t3_final','t4_final','t5_final','t6_final','t7_final','t8_final','t9_final','t10_final','t11_final','t12_final']].agg(np.average) 
    
    d1 = t_finalB.loc[:,('t1_final','t2_final','t3_final', 't4_final', 't5_final', 't6_final')]
    d2 = t_finalB.loc[:,('t9_final','t10_final', 't11_final')]
    
    ax2 = plt.plot(d2)
    plt.title(r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
    plt.ylabel("Proportion in population")
    plt.xlabel('Learning bias c $\cdot 100$')
    
    plt.legend(["prag. L-taut", "prag. L-bound","prag. L-lack"], loc='best')
    plt.show() 

learning_only_plot()

def learning_only_fitness_only_side_by_side():
    df = pd.read_csv('./mean-1000games-replication-only.csv')
    dl = pd.read_csv('./mean-1000games-mutation-only.csv')

    a = 1
    k = 5
    lam = 30
    sample = 10
    learn = 1
    
    final_groupA = df.loc[:,'alpha':]
    final_groupB = dl.loc[:,'alpha':]
    
    groupA = final_groupA.loc[final_groupA['alpha'] == a]
    
    t_finalA = groupA.groupby(['lambda'])
    t_finalA = t_finalA[['t1_final','t2_final','t3_final','t4_final','t5_final','t6_final','t7_final','t8_final','t9_final','t10_final','t11_final','t12_final']].agg(np.average) 
    
    learn = 1
    
    groupB = final_groupB.loc[final_groupB['alpha'] == a]
    groupB = groupB.loc[groupB['k'] == k]
    groupB = groupB.loc[groupB['lambda'] == lam]
    groupB = groupB.loc[groupB['sample_amount'] == sample]
    groupB = groupB.loc[groupB['learning_parameter'] == learn]
    
    t_finalB = groupB.groupby(['prior_cost_c'])
    t_finalB = t_finalB[['t1_final','t2_final','t3_final','t4_final','t5_final','t6_final','t7_final','t8_final','t9_final','t10_final','t11_final','t12_final']].agg(np.average) 
    
    d1 = t_finalA.loc[:,('t1_final','t2_final','t3_final', 't4_final', 't5_final', 't6_final')]
    d2 = t_finalA.loc[:,('t9_final','t10_final', 't11_final')]
    d3 = t_finalB.loc[:,('t1_final','t2_final','t3_final', 't4_final', 't5_final', 't6_final')]
    d4 = t_finalB.loc[:,('t9_final','t10_final', 't11_final')]
    
    sns.set(font_scale=1.2) 
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1, xlabel=r'Rationality parameter $\lambda$', ylabel='Proportion in population')
    ax1.set_title("A", x=0.5, y=0.92, fontsize=20, fontweight='bold')
    
    ax2 = fig.add_subplot(1,2,2, sharey = ax1, xlabel='Prior parameter c')  #Share y-axes with subplot 1
    ax2.set_title("B", x=0.5, y=0.92, fontsize=20, fontweight='bold')
    
    plt.setp(ax2.get_yticklabels())#, visible=False)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
    
    #Plot data
    im1 = ax1.plot(d2)
    im2 = ax2.plot(d4)
    plt.legend(["prag. L-taut", "prag. L-bound","prag. L-lack"], loc='best', prop={'size':11})
    plt.show() 

learning_only_fitness_only_side_by_side()

def heatmap_cost_posterior():
    print 'Loading data'
    df = pd.read_csv('./mean-1000games-c-to-l.csv')
    
    a = 1
    k = 5
    lam = 30
    sample = 10
    
    final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t11_final')]
    
    group = final_group.loc[final_group['alpha'] == a]
    group = group.loc[group['k'] == k]
    group = group.loc[group['lambda'] == lam]
    group = group.loc[group['sample_amount'] == sample]
    
    t11_rel = group.loc[:,('prior_cost_c', 'learning_parameter','t11_final')] #ignore other columns, given that they are fixed
    t11_rec = t11_rel.pivot('prior_cost_c','learning_parameter','t11_final') #reshape to have a prior_cost_c by learning_parameter table
    
    sns.set(font_scale=1.2)
    
    vacio = ["" for _ in xrange(9)]
    yticks = [0] + vacio + [0.1] + vacio + [0.2] + vacio + [0.3] + vacio + [0.4] + vacio + [0.5] + vacio + [0.6] + vacio + [0.7] + vacio + [0.8] + vacio + [0.9] + ["" for _ in xrange(8)] + [0.99]
    ax = sns.heatmap(t11_rec,yticklabels=yticks)#, yticklabels=yticks) 
    ax.set(ylabel='Learning bias c',xlabel='Sampling to MAP parameter l', title=r'Pragmatic L-lack ($\alpha = %d, \lambda = %d$, samples = %d, k = %d)' %(a,lam,sample,k))
    plt.yticks(rotation=0)
    
    ax.invert_yaxis()
    plt.show()

heatmap_cost_posterior()

def dev_over_cost_l1():
    df = pd.read_csv('./mean-1000games-c-to-l.csv')
    a = 1
    k = 5
    lam = 30
    sample = 10
    learn = 1
    
    
    final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t9_final','t10_final','t11_final')]
    
    group = final_group.loc[final_group['alpha'] == a]
    group = group.loc[group['k'] == k]
    group = group.loc[group['lambda'] == lam]
    group = group.loc[group['sample_amount'] == sample]
    group = group.loc[group['learning_parameter'] == learn]
    
    t_final = group.groupby(['prior_cost_c'])
    t_final = t_final[['t9_final','t10_final','t11_final']].agg(np.average) 
    
    ax = t_final.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
    ax.set(ylabel="Proportion in population",xlabel='Learning bias c')
    plt.legend(["prag. L-taut","prag. L-bound","prag. L-lack"], loc='best')
    
    plt.show() 

dev_over_cost_l1()

def dev_over_cost_l3():
    df = pd.read_csv('./mean-1000games-c-to-l.csv')
    a = 1
    k = 5
    lam = 30
    sample = 10
    learn = 3
    
    final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t9_final','t10_final','t11_final')]
    group = final_group.loc[final_group['alpha'] == a]
    group = group.loc[group['k'] == k]
    group = group.loc[group['lambda'] == lam]
    group = group.loc[group['sample_amount'] == sample]
    group = group.loc[group['learning_parameter'] == learn]
    
    t_final = group.groupby(['prior_cost_c'])
    t_final = t_final[['t9_final','t10_final','t11_final']].agg(np.average) 
    
    ax = t_final.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
    ax.set(ylabel="Proportion in population",xlabel='Learning bias c')
    plt.legend(["prag. L-taut","prag. L-bound","prag. L-lack"], loc='best')
    
    plt.show() 

dev_over_cost_l3()

def dev_over_cost_side_by_side():
    print 'Loading data'
    df = pd.read_csv('./mean-1000games-c-to-l.csv')

    a = 1
    k = 5
    lam = 30
    sample = 10
    learn = 5
    
    final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t9_final','t10_final','t11_final')]
    
    group = final_group.loc[final_group['alpha'] == a]
    group = group.loc[group['k'] == k]
    group = group.loc[group['lambda'] == lam]
    group = group.loc[group['sample_amount'] == sample]
    group = group.loc[group['learning_parameter'] == learn]
    
    t_final = group.groupby(['prior_cost_c'])
    t_final = t_final[['t9_final','t10_final','t11_final']].agg(np.average) 

    
    ##
    learn = 1
    
    groupB = final_group.loc[final_group['alpha'] == a]
    groupB = groupB.loc[groupB['k'] == k]
    groupB = groupB.loc[groupB['lambda'] == lam]
    groupB = groupB.loc[groupB['sample_amount'] == sample]
    groupB = groupB.loc[groupB['learning_parameter'] == learn]
    
    t_finalB = groupB.groupby(['prior_cost_c'])
    t_finalB = t_finalB[['t9_final','t10_final','t11_final']].agg(np.average) 

    
    sns.set(font_scale=1.2) 
    fig = plt.figure()
    
    ax1 = fig.add_subplot(1,2,1, xlabel=r'Prior parameter c', ylabel='Proportion in population')
    ax1.set_title("A", x=0.5, y=0.92, fontsize=20, fontweight='bold')
    
    ax2 = fig.add_subplot(1,2,2, sharey = ax1, xlabel='Prior parameter c')  #Share y-axes with subplot 1
    ax2.set_title("B", x=0.5, y=0.92, fontsize=20, fontweight='bold')
    
    plt.setp(ax2.get_yticklabels())#, visible=False)
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
    
    #Plot data
    im1 = ax1.plot(t_finalB)
    im2 = ax2.plot(t_final)
    plt.legend(["prag. L-taut", "prag. L-bound","prag. L-lack"], loc='best', prop={'size':11})
    plt.show() 

dev_over_cost_side_by_side()

