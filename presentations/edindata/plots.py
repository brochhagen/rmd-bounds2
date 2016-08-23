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


### Heatplot prior/posterior ####
a = 1
k = 5
lam = 30
sample = 10

final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t11_final')]

group = final_group.loc[final_group['alpha'] == a]
group = group.loc[group['k'] == k]
group = group.loc[group['lambda'] == lam]
group = group.loc[group['sample_amount'] == sample]


t11_final = group.groupby(['prior_cost_c', 'learning_parameter']).apply(cust_mean) #add new column with mean
t11_rel = t11_final.loc[:,('prior_cost_c', 'learning_parameter','mean')] #ignore other columns, given that they are fixed
t11_rel_uniq = t11_rel.drop_duplicates() #drop duplictes that arise from adding the mean column
t11_rec = t11_rel_uniq.pivot('prior_cost_c','learning_parameter','mean') #reshape to have a prior_cost_c by learning_parameter table

sns.set(font_scale=1.2)

#yticks = np.arange(min(t11_rec.index),max(t11_rec.index),0.15)
vacio = ["" for _ in xrange(9)]
yticks = [0] + vacio + [0.1] + vacio + [0.2] + vacio + [0.3] + vacio + [0.4] + vacio + [0.5] + vacio + [0.6] + vacio + [0.7] + vacio + [0.8] + vacio + [0.9] + ["" for _ in xrange(8)] + [0.99]
ax = sns.heatmap(t11_rec,yticklabels=yticks)#, yticklabels=yticks) 
ax.set(ylabel='Learning bias c',xlabel='Sampling to MAP parameter l', title=r'Pragmatic L-lack ($\alpha = %d, \lambda = %d$, samples = %d, k = %d)' %(a,lam,sample,k))
plt.yticks(rotation=0)

ax.invert_yaxis()
plt.show()


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

ax = t_final.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
ax.set(ylabel="Proportion in population",xlabel='Learning bias c')
plt.legend(["pragmatic L-taut","pragmatic L-bound","pragmatic L-lack"], loc='best')

plt.show() 

### 

learn = 1

groupB = final_group.loc[final_group['alpha'] == a]
groupB = groupB.loc[groupB['k'] == k]
groupB = groupB.loc[groupB['lambda'] == lam]
groupB = groupB.loc[groupB['sample_amount'] == sample]
groupB = groupB.loc[groupB['learning_parameter'] == learn]

t_finalB = groupB.groupby(['prior_cost_c'])
t_finalB = t_finalB[['t9_final','t10_final','t11_final']].agg(np.average) 

ax = t_finalB.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
ax.set(ylabel="Proportion in population",xlabel='Learning bias c')
plt.legend(["pragmatic L-taut","pragmatic L-bound","pragmatic L-lack"], loc='best')




plt.show()


