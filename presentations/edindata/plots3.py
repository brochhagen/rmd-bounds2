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


print 'Loading data fitness'
path =r'./results-fit' 
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
df = pd.concat(pd.read_csv(f) for f in all_files)

print 'Loading data learn'
path =r'./results-learn' 
all_files = glob.glob(os.path.join(path, "*.csv"))     # advisable to use os.path.join as this makes concatenation OS independent
dl = pd.concat(pd.read_csv(f) for f in all_files)

def cust_mean(grp):
    grp['mean'] = grp['t11_final'].mean()
    return grp




### development over cost ###

a = 1
k = 5
lam = 30
sample = 10
learn = 1


final_groupA = df.loc[:,'alpha':]
final_groupB = dl.loc[:,'alpha':]

groupA = final_groupA.loc[final_groupA['alpha'] == a]
groupA = groupA.loc[groupA['k'] == k]
groupA = groupA.loc[groupA['sample_amount'] == sample]
groupA = groupA.loc[groupA['learning_parameter'] == learn]
groupA = groupA.loc[groupA['prior_cost_c'] == 0.1]

t_finalA = groupA.groupby(['lambda'])
t_finalA = t_finalA[['t1_final','t2_final','t3_final','t4_final','t5_final','t6_final','t7_final','t8_final','t9_final','t10_final','t11_final','t12_final']].agg(np.average) 

d1 = t_finalA.loc[:,('t1_final','t2_final','t3_final', 't4_final', 't5_final', 't6_final')]
d2 = t_finalA.loc[:,('t7_final','t8_final','t9_final', 't10_final', 't11_final', 't12_final')]

ax2 = plt.plot(d2)
#ax1 = plt.plot(d1,marker='d',markevery=5)
plt.title(r'($\alpha$ = %d)' %(a))
plt.ylabel("Proportion in population")
plt.xlabel('Rationality parameter')

plt.legend(["prag $L1$", "prag $L2$", "prag $L3$", "prag $L4$","prag $L5$", "prag $L6$"], loc='best')
plt.xlim(0,48)
plt.show() 


### 

learn = 1

groupB = final_groupB.loc[final_groupB['alpha'] == a]
groupB = groupB.loc[groupB['k'] == k]
groupB = groupB.loc[groupB['lambda'] == lam]
groupB = groupB.loc[groupB['sample_amount'] == sample]
groupB = groupB.loc[groupB['learning_parameter'] == learn]

t_finalB = groupB.groupby(['prior_cost_c'])
t_finalB = t_finalB[['t1_final','t2_final','t3_final','t4_final','t5_final','t6_final','t7_final','t8_final','t9_final','t10_final','t11_final','t12_final']].agg(np.average) 

#ax = t_finalB.plot(title=r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
#ax.set(ylabel="Proportion in population",xlabel='Prior parameter c')
#plt.legend(["$L1$", "$L2$", "$L3$", "$L4$","$L5$", "$L6$"], loc='best')

d1 = t_finalB.loc[:,('t1_final','t2_final','t3_final', 't4_final', 't5_final', 't6_final')]
d2 = t_finalB.loc[:,('t7_final','t8_final','t9_final', 't10_final', 't11_final', 't12_final')]

#d2 = t_finalB.loc[:,('t8_final','t9_final', 't10_final', 't11_final')]

ax2 = plt.plot(d2)
ax1 = plt.plot(d1,marker='d',markevery=5)
plt.title(r'($\alpha = %d, \lambda = %d, k = %d$, samples = %d, l =%d)' %(a,lam,k,sample,learn))
plt.ylabel("Proportion in population")
plt.xlabel('Prior Parameter c $\cdot 100$')

plt.legend(["prag $L2$", "prag $L3$", "prag $L4$","prag $L5$"], loc='best')
#plt.xlim(0,48)
plt.show() 



#
#
