###
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_context(rc={'lines.markeredgewidth': 0.5})
import os
import glob
import sys

def get_parameter_subfigs(type_list,list1,list2):
    kind = 'rmd'
    seq_length = 5
    num_of_runs = 10

    x_figs = len(list1)
    y_figs = len(list2)
    fig,axes = plt.subplots(nrows=y_figs,ncols=x_figs)
    types_to_plot = ['t_final'+str(x) for x in type_list]
#    factor = 1
    for x in xrange(y_figs):
        for y in xrange(x_figs):
            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[y],seq_length,list2[x]))
            df = df[types_to_plot]
            df = df.iloc[:num_of_runs]
            df.plot(ax=axes[x,y],kind='bar')#,sharey=True,ylim=(0,.8),logy=True)
    #Adding labels and modifying layout 
    list2.reverse()
    for idx in xrange(len(plt.gcf().axes)):
        ax = plt.gcf().axes[idx]

        #Only add top label to subplots that are on the first row
        if idx - x_figs < 0:
            x = idx % x_figs
            ax.set_xlabel(r'$\lambda$ = '+str(list1[x]),fontsize=20)
            ax.xaxis.set_label_position('top')

        #Only add label to subplots that are on the left-most column
        if idx % x_figs == 0:
            ax.set_ylabel('l = '+str(list2[-1]),fontsize=20)
            list2.pop()

        #Hide legend
        ax.legend().set_visible(False)

        #Start counting simulations with 1 and not 0 on the x-axis
        xlabels = [item.get_text() for item in ax.get_xticklabels()]
        xlabels = [str(int(x) + 1) for x in xlabels]
        ax.set_xticklabels(xlabels)
        
        #hide every n-th tick
        for label in ax.get_yticklabels()[::2]:
            label.set_visible(False)
    plt.tight_layout()
    plt.show()

types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
list1 = [1,5,20] #lambda
list2 = [1,5,15] #posterior parameter
get_parameter_subfigs(types,list1,list2)

