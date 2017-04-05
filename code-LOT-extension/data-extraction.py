import pandas as pd
import numpy as np 
import csv

target_types = [231,236,291,306,326,336]
list1 = [1,5,20] #lambda
list2 = [1,5,15] #l
def get_data_to_csv(target_types,list1,list2):
    restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the majority type to come from other columns
    targets = ['t_final'+str(z) for z in target_types]
    non_targets = [x for x in restrict_to_final if x not in targets]
    seq_length = 5
 
    f = csv.writer(open('./data-plot4.csv','wb'))
    f.writerow(['runID','lambda','l','proportion largest target','proportion largest nontarget'])
  
    for x in xrange(len(list1)):
        for y in xrange(len(list2)):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % ('rmd',list1[y],seq_length,list2[x]))
            
            df['largest target'] = df[targets].max(axis=1)
            df['largest non target'] = df[non_targets].max(axis=1)
            for idx,row in df.iterrows():
                f.writerow([str(row['runID']),str(row['lam']),str(row['l']),str(row['largest target']),str(row['largest non target'])])


get_data_to_csv(target_types,list1,list2)

