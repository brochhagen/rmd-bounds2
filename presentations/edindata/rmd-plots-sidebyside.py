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


print 'Loading data'
path =r'./results1' 
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
df = pd.concat(pd.read_csv(f) for f in all_files)

def cust_mean(grp):
    grp['mean'] = grp['t11_final'].mean()
    return grp


### development over cost ###

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
plt.legend(["pragmatic L3", "pragmatic L4","pragmatic L5"], loc='best', prop={'size':11})


#ax2 = plt.plot(d2)
##ax1 = plt.plot(d1,marker='d',markevery=5)
#plt.title(r'($\alpha$ = %d)' %(a))
#plt.ylabel("Proportion in population")
#plt.xlabel(r'Rationality parameter $\lambda$')
#
#plt.legend(["pragmatic L-taut", "pragmatic L-bound","pragmatic L-lack"], loc='best')
#plt.xlim(0,48)
plt.show() 



plt.show()


