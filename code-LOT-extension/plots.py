###
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
#sns.set_context(rc={'lines.markeredgewidth': 0.5})
import os
import glob
import sys


def analysis(types,val1,val2,kind):
    from lexica import get_lexica,get_lexica_bins
    from rmd import get_type_bin
    seq_length = 5
    lex = get_lexica(3,3,False)
    bins = get_lexica_bins(lex)

    values_of_inc_in_bins = np.zeros(len(bins))
    numb_of_inc = np.zeros(len(bins))

    df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,val1,seq_length,val2))
    all_types = ['t_final'+str(z) for z in xrange(432)]
    df = df[all_types]
    for i in xrange(len(df)):
        type_of_inc = df.iloc[i].idxmax(axis=1)
        value = df.iloc[i][type_of_inc]
        stripped_type = int(type_of_inc[7:]) #convert name into idx
        bin_of_type = get_type_bin(stripped_type,bins)
        values_of_inc_in_bins[bin_of_type] += value #add value to store in vector
        numb_of_inc[bin_of_type] += 1 #add 1 to counter of times incumbent

    return [values_of_inc_in_bins / len(df), numb_of_inc]


def get_subfigs_incumbents(type_list,list1,list2,kind,prior=False):
    seq_length = 5
    num_of_runs = 5
    
    if not(kind == 'm'):
        x_figs = len(list1)
        y_figs = len(list2)
    else:
        x_figs = len(list2)
        y_figs = len(list1)
    fig,axes = plt.subplots(nrows=y_figs,ncols=x_figs)

    if prior == True: #To add prior subfigure
        fig,axes = plt.subplots(nrows=y_figs,ncols=x_figs+1)

    if y_figs == 1:
        axes = axes[np.newaxis] #add axis if 1d vector. Otherwise error

    types_to_plot = ['t_final'+str(x) for x in type_list] + ['incumbent']

    #Finding incumbent for each individual run per parameter configuration and checking whether it's already one of the target types
    #This is pretty roundabout, but explicit and relatively fast
    incumbent_list = [] #list of lists of incumbents. One list per parameter configuration, consisting of num_of_runs incumbents
    for x in xrange(y_figs):
        for y in xrange(x_figs):
            if not(kind == 'm'):
                df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[y],seq_length,list2[x]))
            else:
                df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[x],seq_length,list2[y]))

            all_types = ['t_final'+str(z) for z in xrange(432)]
            df = df[all_types]
            incumbents = []
            for run in xrange(num_of_runs):
                ds = df.iloc[run]
                incumbent = ds.idxmax()
                if incumbent in type_list: 
                    print '### Incumbent already a target_type ###'
                    print x,y
                incumbents.append(incumbent)
            incumbent_list.append(incumbents)

    for x in xrange(y_figs):
        for y in xrange(x_figs):
            if not(kind == 'm'):
                df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[y],seq_length,list2[x]))
            else:
                df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[x],seq_length,list2[y]))

            restrict_to_final = ['t_final'+str(z) for z in xrange(432)] #as to avoid the incumbent to come from other columns
            df = df[restrict_to_final]
            df['incumbent'] = df.max(axis=1)
            df = df[types_to_plot]
            df = df.iloc[:num_of_runs]
            df.plot(ax=axes[x,y],kind='bar',color=('grey','grey','grey','grey','grey','grey','white'))
    #Adding prior subfig
    if prior == True:
        from lexica import get_prior, get_lexica
    
        priors = get_prior(get_lexica(3,3,False))
        priors = list(priors)
        targets = [231,236,291,306,326,336]
        #reorder prior
        for t in targets:
            priors.insert(0,priors.pop(t))

        white_space = 10 #to separate from other priors

        X = np.arange(len(priors)+white_space)
        Y_target = [0 for _ in xrange(len(priors)+white_space)]
        for target in xrange(len(targets)):
            Y_target[target] = priors[target]
            priors[target] = 0
       
        Y_rest = priors[:len(targets)] + [0 for _ in xrange(white_space)] + priors[len(targets):]

        Y_target.reverse()
        Y_rest.reverse()
        
        ax = fig.add_subplot(133)
        ax.grid(False)

        ax.barh(X,Y_target,color='black')#, edgecolor='none')#,width=1, color='grey')#, orientation='horizontal')
        ax.barh(X,Y_rest,color='white')#,width=1, color='white' )#, orientation='horizontal')


        ylabels = ["" for x in xrange(len(X))]
        ylabels[9] = 'L-lack'
        ylabels[5] = 'Other types'
        ax.set_yticklabels(ylabels,rotation=90)

#        for label in ax.get_yticklabels()[::3]:
#            label.set_visible(False)

        for label in ax.get_xticklabels()[::2]:
            label.set_visible(False)

        ax.tick_params(axis='both', which='major', labelsize=17)
        
#        xlabels = [str(int(x) + 1) for x in xlabels]
#        ax.set_xticklabels(xlabels)


    #Adding labels and modifying layout 
    learning_labels = [x for x in list2]
    list2.reverse()

    for idx in xrange(len(plt.gcf().axes)):
        ax = plt.gcf().axes[idx]
        ax.tick_params(axis='both', which='major', labelsize=17)

        #Only add top label to subplots that are on the first row
        if idx - x_figs < 0 and not(kind == 'm'):
            x = idx % x_figs
            ax.set_xlabel(r'$\lambda$ = '+str(list1[x]),fontsize=20)
            ax.xaxis.set_label_position('top')

        elif idx - x_figs < 0:
            ax.set_xlabel('l = '+str(learning_labels[idx]),fontsize=25)
            ax.xaxis.set_label_position('top')
        elif idx > x_figs and kind == 'm': #If there are more, it means the last is the prior
                ax.set_xlabel('Prior',fontsize=25)
                ax.xaxis.set_label_position('top')

        #Only add label to subplots that are on the left-most column
        if idx % x_figs == 0:
            if kind == 'rmd':
                ax.set_ylabel('l = '+str(list2[-1]),fontsize=20)
                list2.pop()
            elif y_figs == 1 and prior == False:
                ax.set_ylabel('Proportion in population',fontsize=17)
                list2.pop()

        handles, labels = ax.get_legend_handles_labels()
        labels = ['prag. L-lack kind' for _ in xrange(len(types))] + ['Incumbent']
        display = (0,6)

        ax.legend([handle for i,handle in enumerate(handles) if i in display], \
                  [label for i,label in enumerate(labels) if i in display], loc='best')

#        #Alternatively: Hide legend
        if idx == 0 and kind == 'm' or (not(idx == 2) and kind == 'rmd'):
            ax.legend().set_visible(False)


        if idx == 1 and kind == 'm':
            ax.legend([handle for i,handle in enumerate(handles) if i in display], \
                  [label for i,label in enumerate(labels) if i in display], loc='best', prop={'size':20})


#        #Start counting simulations with 1 and not 0 on the x-axis
        if prior == False or idx < 1:
            xlabels = [item.get_text() for item in ax.get_xticklabels()]
            xlabels = [str(int(x) + 1) for x in xlabels]
            ax.set_xticklabels(xlabels)
#        
#        #hide every n-th tick
            for label in ax.get_yticklabels()[::2]:
                label.set_visible(False)
        #Hide everything for the prior plot. Otherwise there's overlap
        if prior == True and idx == 2:
            for label in ax.get_yticklabels()[::1]:
                label.set_visible(False)
            for label in ax.get_xticklabels()[::1]:
                label.set_visible(False)
    if not(kind == 'm'):
        fig.text(0.55, 0.015, 'Population', ha='center', va='center',fontsize=17)
    plt.tight_layout()
    plt.show()


def get_heatmap_diff_incumbents(type_list,list1,list2,kind):
    all_types = ['t_final'+str(z) for z in xrange(432)]
    target_types = ['t_final'+str(x) for x in type_list]
    other_types = [x for x in all_types if x not in target_types]
    seq_length = 5

    data = np.zeros((len(list1)+1, len(list2)+1))
    for i in xrange(len(list1)):
        for j in xrange(len(list2)):
            #To avoid errors when adding cols to slices of copies:
            dt = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))
            do = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[i],seq_length,list2[j]))

            data[0,i+1] = list1[i]
            data[j+1,0] = list2[j]
            
            dt = dt[target_types] 
            do = do[other_types]
            dt['incumbent_t'] = dt.max(axis=1) #incumbent amongst types in d_t
            do['incumbent_o'] = do.max(axis=1) #incumbent amongst types in d_o
            best_of_targets = dt['incumbent_t'].mean()
            best_of_others = do['incumbent_o'].mean()
            data[i+1,j+1] = best_of_targets - best_of_others
    dPrime = data[1:,1:]

#    axlabels=["" for _ in xrange(df.shape[1])]
#    axlabels[len(axlabels)/2] = 'types'
    sns.set(font_scale=2)
    ax = sns.heatmap(dPrime, annot=True, xticklabels=list1, yticklabels=list2, vmin=0,annot_kws={"size": 20})# xticklabels=axlabels)#, annot=True) 
    ax.set_xlabel('rationality parameter ('+r'$\lambda$'+')')#,fontsize=30)
    ax.set_ylabel('posterior parameter (l)')#,fontsize=30)

    ax.invert_yaxis()
    plt.show()
#    return dPrime


kind = 'rmd'
type_list = [231,236,291,306,326,336]
list1 = [1,5,20,30] # soft-max parameter
list2 = [1,5,10,15] #prob-matching = 1, increments approach MAP
a = get_heatmap_diff_incumbents(type_list,list1,list2,kind)





#Plot 1: 
#kind='r'
#types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
#list1 = [1,20] #lambda
#list2 = [5] #posterior parameter
#get_subfigs_incumbents(types,list1,list2,kind)

#Plot 2
#kind='m'
#types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
#list1 = [20] #lambda
#list2 = [1,15] #posterior parameter
#get_subfigs_incumbents(types,list1,list2,kind,prior=True)

#Plot 3
#kind='rmd'
#types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
#list1 = [1,5,20] #lambda
#list2 = [1,5,15] #posterior parameter
#get_subfigs_incumbents(types,list1,list2,kind)

####
#from lexica import get_lexica,get_lexica_bins,get_prior
#from rmd import get_type_bin
#
#lex = get_lexica(3,3,False)
#prior = get_prior(lex)
#bins = get_lexica_bins(lex)

#kind='r'
#types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
#val1 = 20 #lambda
#val2 = 5 #posterior parameter
#a = analysis(types,val1,val2,kind)

#kind='m'
#types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
#val1 = 20 #lambda
#val2 = 15 #posterior parameter
#a = analysis(types,val1,val2,kind)
#dr = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,20,5,1))
#df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,20,5,15))
#dx = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,1,5,1))
#dw = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,1,5,1))

###################
#def get_parameter_subfigs(type_list,list1,list2,kind):
#    seq_length = 5
#    num_of_runs = 10
#
#    x_figs = len(list1)
#    y_figs = len(list2)
#    fig,axes = plt.subplots(nrows=y_figs,ncols=x_figs)
#    types_to_plot = ['t_final'+str(x) for x in type_list]
#
#    for x in xrange(y_figs):
#        for y in xrange(x_figs):
#            df = pd.read_csv('./results/%s-s3-m3-lam%d-a1-k%d-samples250-l%d-g50-meFalse.csv' % (kind,list1[y],seq_length,list2[x]))
#            df = df[types_to_plot]
#            df = df.iloc[:num_of_runs]
#            df.plot(ax=axes[x,y],kind='bar')#,sharey=True,ylim=(0,.8),logy=True)
#
#    #Adding labels and modifying layout 
#    list2.reverse()
#    for idx in xrange(len(plt.gcf().axes)):
#        ax = plt.gcf().axes[idx]
#
#        #Only add top label to subplots that are on the first row
#        if idx - x_figs < 0:
#            x = idx % x_figs
#            ax.set_xlabel(r'$\lambda$ = '+str(list1[x]),fontsize=20)
#            ax.xaxis.set_label_position('top')
#
#        #Only add label to subplots that are on the left-most column
#        if idx % x_figs == 0:
#            ax.set_ylabel('l = '+str(list2[-1]),fontsize=20)
#            list2.pop()
#
#        #Hide legend
#        ax.legend().set_visible(False)
#
#        #Start counting simulations with 1 and not 0 on the x-axis
#        xlabels = [item.get_text() for item in ax.get_xticklabels()]
#        xlabels = [str(int(x) + 1) for x in xlabels]
#        ax.set_xticklabels(xlabels)
#        
#        #hide every n-th tick
#        for label in ax.get_yticklabels()[::2]:
#            label.set_visible(False)
#    plt.tight_layout()
#    plt.show()
#
##kind='rmd'
##types = [231,236,291,306,326,336]#[231,236,291,306,326,336]
##list1 = [1,5,20] #lambda
##list2 = [1,5,15] #posterior parameter
#
