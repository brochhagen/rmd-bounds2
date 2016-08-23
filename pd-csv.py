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
path =r'./resultssubset' 
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
df = pd.concat(pd.read_csv(f) for f in all_files)


sys.exit()
###Plot 1###
a = 1
c = 0.2
k = 5
lam = 10
sample = 15
learn = 1

final_group = df.loc[:,'alpha':]
final_group = final_group.loc[final_group['alpha'] == a]
final_group = final_group.loc[final_group['prior_cost_c'] == c]
final_group = final_group.loc[final_group['k'] == k]
final_group = final_group.loc[final_group['lambda'] == lam]
final_group = final_group.loc[final_group['sample_amount'] == sample]
#final_group = final_group.loc[final_group['learning_parameter'] == learn]


all_finals = final_group.groupby(['learning_parameter'])
sub_finals = all_finals[['t10_final','t11_final','t12_final']].agg(np.average)

print sub_finals

ax = sub_finals.plot(title='alpha = %d, $\lambda = %d$, c = %.2f,k = %d, samples = %d' %(a,lam,c,k,sample))
ax.set(ylabel="proportion",xlabel='learning parameter')
plt.legend(["$pragmatic_{10}$","$pragmatic_{11}$","$pragmatic_{12}$"], loc='best')

plt.show()



###Plot 2###
a = 1
c = 0.5
k = 5
lam = 10
sample = 15
learn = 1

final_group = df.loc[:,'alpha':]
final_group = final_group.loc[final_group['alpha'] == a]
final_group = final_group.loc[final_group['prior_cost_c'] == c]
final_group = final_group.loc[final_group['k'] == k]
final_group = final_group.loc[final_group['lambda'] == lam]
final_group = final_group.loc[final_group['sample_amount'] == sample]
#final_group = final_group.loc[final_group['learning_parameter'] == learn]

all_finals = final_group.groupby(['learning_parameter'])
sub_finals = all_finals[['t10_final','t11_final','t12_final']].agg(np.average)

print sub_finals
ax = sub_finals.plot(title='alpha = %d, $\lambda = %d$, c = %.2f,k = %d, samples = %d' %(a,lam,c,k,sample))
ax.set(ylabel="proportion",xlabel='learning parameter')
plt.legend(["$pragmatic_{10}$","$pragmatic_{11}$","$pragmatic_{12}$"], loc='best')

plt.show()


####Plot 3###
a = 1
c = 0.3
k = 5
lam = 10
sample = 15
learn = 4

final_group = df.loc[:,'alpha':]
final_group = final_group.loc[final_group['alpha'] == a]
final_group = final_group.loc[final_group['prior_cost_c'] == c]
#final_group = final_group.loc[final_group['k'] == k]
final_group = final_group.loc[final_group['lambda'] == lam]
final_group = final_group.loc[final_group['sample_amount'] == sample]
final_group = final_group.loc[final_group['learning_parameter'] == learn]

all_finals = final_group.groupby(['k'])
sub_finals = all_finals[['t10_final','t11_final','t12_final']].agg(np.average)

ax = sub_finals.plot(title='alpha = %d, $\lambda = %d$, c = %.2f, learning = %d, samples = %d' %(a,lam,c,learn,sample))
ax.set(ylabel="proportion",xlabel='sequence length k')
plt.legend(["$pragmatic_{10}$","$pragmatic_{11}$","$pragmatic_{12}$"], loc='best')

plt.show()
#
#
##

####Plot 4###
a = 1
c = 0.3
k = 5
lam = 10
#sample = 15
learn = 4

final_group = df.loc[:,'alpha':]
final_group = final_group.loc[final_group['alpha'] == a]
final_group = final_group.loc[final_group['prior_cost_c'] == c]
final_group = final_group.loc[final_group['k'] == k]
final_group = final_group.loc[final_group['lambda'] == lam]
#final_group = final_group.loc[final_group['sample_amount'] == sample]
final_group = final_group.loc[final_group['learning_parameter'] == learn]

all_finals = final_group.groupby(['sample_amount'])
sub_finals = all_finals[['t10_final','t11_final','t12_final']].agg(np.average)

ax = sub_finals.plot(title='alpha = %d, $\lambda = %d$, c = %.2f, learning = %d, k = %d' %(a,lam,c,learn,k))
ax.set(ylabel="proportion",xlabel='# of samples')
plt.legend(["$pragmatic_{10}$","$pragmatic_{11}$","$pragmatic_{12}$"], loc='best')

plt.show()

####Plot 5###
a = 1
#c = 0.3
k = 5
lam = 10
sample = 15
learn = 4

final_group = df.loc[:,'alpha':]
final_group = final_group.loc[final_group['alpha'] == a]
#final_group = final_group.loc[final_group['prior_cost_c'] == c]
final_group = final_group.loc[final_group['k'] == k]
final_group = final_group.loc[final_group['lambda'] == lam]
final_group = final_group.loc[final_group['sample_amount'] == sample]
final_group = final_group.loc[final_group['learning_parameter'] == learn]

all_finals = final_group.groupby(['prior_cost_c'])
sub_finals = all_finals[['t10_final','t11_final','t12_final']].agg(np.average)

ax = sub_finals.plot(title='alpha = %d, $\lambda = %d$, samples = %.2f, learning = %d, k = %d' %(a,lam,sample,learn,k))
ax.set(ylabel="proportion",xlabel='prior parameter c')
plt.legend(["$pragmatic_{10}$","$pragmatic_{11}$","$pragmatic_{12}$"], loc='best')

plt.show()

###

####Plot 6###
a = 1
c = 0.3
k = 5
lam = 30
sample = 15
learn = 4

final_group = df.loc[:,'alpha':]
final_group = final_group.loc[final_group['alpha'] == a]
final_group = final_group.loc[final_group['prior_cost_c'] == c]
#final_group = final_group.loc[final_group['k'] == k]
final_group = final_group.loc[final_group['lambda'] == lam]
final_group = final_group.loc[final_group['sample_amount'] == sample]
final_group = final_group.loc[final_group['learning_parameter'] == learn]

all_finals = final_group.groupby(['k'])
sub_finals = all_finals[['t10_final','t11_final','t12_final']].agg(np.average)

ax = sub_finals.plot(title='alpha = %d, $\lambda = %d$, c = %.2f, learning = %d, samples = %d' %(a,lam,c,learn,sample))
ax.set(ylabel="proportion",xlabel='sequence length k')
plt.legend(["$pragmatic_{10}$","$pragmatic_{11}$","$pragmatic_{12}$"], loc='best')

plt.show()

###
a = 1
#c = 0.3
k = 5
lam = 30
sample = 15
#learn = 4

final_group = df.loc[:,('alpha','prior_cost_c','lambda','k','sample_amount','learning_parameter','t11_final')]

group = final_group.loc[final_group['alpha'] == a]
#group = group.loc[group['prior_cost_c'] == c]
group = group.loc[group['k'] == k]
group = group.loc[group['lambda'] == lam]
group = group.loc[group['sample_amount'] == sample]
#group = group.loc[group['learning_parameter'] == learn]

def cust_mean(grp):
    grp['mean'] = grp['t11_final'].mean()
    return grp

t11_final = group.groupby(['prior_cost_c', 'learning_parameter']).apply(cust_mean) #add new column with mean
t11_rel = t11_final.loc[:,('prior_cost_c', 'learning_parameter','mean')] #ignore other columns, given that they are fixed
t11_rel_uniq = t11_rel.drop_duplicates() #drop duplictes that arise from adding the mean column
t11_rec = t11_rel_uniq.pivot('prior_cost_c','learning_parameter','mean') #reshape to have a prior_cost_c by learning_parameter table

sns.set(font_scale=1.2)
ax = sns.heatmap(t11_rec) 
ax.set(ylabel='prior parameter c',xlabel='posterior parameter l', title=r'Pragmatic $L_5$ ($\alpha = %d, \lambda = %d, samples = %d, k = %d$)' %(a,lam,sample,k))
ax.invert_yaxis()
plt.show()


