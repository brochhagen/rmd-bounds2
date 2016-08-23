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
path =r'./results2' 
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
df = pd.concat(pd.read_csv(f) for f in all_files)

def cust_mean(grp):
    grp['mean'] = grp['t11_final'].mean()
    return grp

### Heatplot seq-length/lam ####
a = 1
sample = 10
c = .4
sample_amount = 20
learn = 3
gens = 20
runs = 1000


final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t11_final')]

group = final_group.loc[final_group['alpha'] == a]
group = group.loc[group['prior_cost_c'] == c]
group = group.loc[group['sample_amount'] == sample]
group = group.loc[group['learning_parameter'] == learn]

t11_final = group.groupby(['lambda','k']).apply(cust_mean) #add new column with mean
t11_rel = t11_final.loc[:,('lambda','k','mean')] #ignore other columns, given that they are fixed
t11_rel_uniq = t11_rel.drop_duplicates() #drop duplictes that arise from adding the mean column
t11_rec = t11_rel_uniq.pivot('lambda','k','mean') #reshape to have a k by lambda table


sns.set(font_scale=1.2)
vacio = ["" for _ in xrange(4)]
yticks = [1] + ["" for _ in xrange(3)] + [5] + vacio + [10] + vacio + [15] + vacio + [20] + vacio + [25] + vacio + [30] + vacio + [35] + vacio + [40] + vacio + [45] + ["" for _ in xrange(4)] + [50]

ax = sns.heatmap(t11_rec,cmap='Blues',yticklabels=yticks)
ax.set(ylabel='Rationality parameter $\lambda$',xlabel='Sequence length k', title=r'Pragmatic $L5$ ($\alpha$ = %d, c = %.1f, samples = %d, l = %d)' %(a,c,sample,learn))
ax.invert_yaxis()
plt.yticks(rotation=0)

plt.show()



### Heatplot seq-length/lam T10 ####
def cust_mean_10(grp):
    grp['mean'] = grp['t10_final'].mean()
    return grp


a = 1
sample = 10
c = .4
sample_amount = 20
learn = 3
gens = 20
runs = 1000

#    seq_length = [x for x in xrange(1,20)]
#    lam = [1,10,20,30,40,50]


final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t10_final')]

group = final_group.loc[final_group['alpha'] == a]
group = group.loc[group['prior_cost_c'] == c]
group = group.loc[group['sample_amount'] == sample]
group = group.loc[group['learning_parameter'] == learn]



t11_final = group.groupby(['lambda','k']).apply(cust_mean_10) #add new column with mean
t11_rel = t11_final.loc[:,('lambda','k','mean')] #ignore other columns, given that they are fixed
t11_rel_uniq = t11_rel.drop_duplicates() #drop duplictes that arise from adding the mean column
t10_rec = t11_rel_uniq.pivot('lambda','k','mean') #reshape to have a k by lambda table


#yticks = np.arange(min(t11_rec.index),max(t11_rec.index),0.15)
sns.set(font_scale=1.2)
vacio = ["" for _ in xrange(4)]
yticks = [1] + ["" for _ in xrange(3)] + [5] + vacio + [10] + vacio + [15] + vacio + [20] + vacio + [25] + vacio + [30] + vacio + [35] + vacio + [40] + vacio + [45] + ["" for _ in xrange(4)] + [50]


ax = sns.heatmap(t10_rec,cmap='Blues',yticklabels=yticks)#, cmap="YlOrBr")#, yticklabels=yticks) 
ax.set(ylabel='Rationality parameter $\lambda$',xlabel='Sequence length k', title=r'Pragmatic $L4$ ($\alpha$ = %d, c = %.1f, samples = %d, l = %d)' %(a,c,sample,learn))
#plt.yticks(arange(5), (0,2,4,6,8), rotation=0)
ax.invert_yaxis()
plt.yticks(rotation=0)

plt.show()

sys.exit()

fig,axn = plt.subplots(1, 2, sharex=False, sharey=True)

axn.flat[0] = sns.heatmap(t10_rec)
axn.flat[1] = sns.heatmap(t11_rec)
#for i, ax in enumerate(axn.flat):
#    sns.heatmap(df, ax=ax,
#                cbar=i == 0,
#                vmin=0, vmax=1,
#                cbar_ax=None if i else cbar_ax)

fig.tight_layout(rect=[0, 0, .9, 1])
plt.show()


#### development over cost ###
#
#a = 1
#k = 5
#lam = 30
#sample = 10
#learn = 3
#
#
#final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t10_final','t11_final')]
#
#group = final_group.loc[final_group['alpha'] == a]
#group = group.loc[group['k'] == k]
#group = group.loc[group['lambda'] == lam]
#group = group.loc[group['sample_amount'] == sample]
#group = group.loc[group['learning_parameter'] == learn]
#
#t_final = group.groupby(['prior_cost_c'])
#t_final = t_final[['t10_final','t11_final']].agg(np.average) 
#
#ax = t_final.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
#ax.set(ylabel="Proportion in population",xlabel='Prior parameter c')
#plt.legend(["pragmatic $L4$","pragmatic $L5$"], loc='best')
#
#plt.show() 
#
#### 
#
#learn = 1
#
#groupB = final_group.loc[final_group['alpha'] == a]
#groupB = groupB.loc[groupB['k'] == k]
#groupB = groupB.loc[groupB['lambda'] == lam]
#groupB = groupB.loc[groupB['sample_amount'] == sample]
#groupB = groupB.loc[groupB['learning_parameter'] == learn]
#
#t_finalB = groupB.groupby(['prior_cost_c'])
#t_finalB = t_finalB[['t10_final','t11_final']].agg(np.average) 
#
#ax = t_finalB.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
#ax.set(ylabel="proportion",xlabel='prior parameter c')
#plt.legend(["pragmatic $L4$","pragmatic $L5$"], loc='best')
#
#
#
#plt.show()
#
#
