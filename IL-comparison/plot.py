from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(rc={'lines.markeredgewidth': 0.5})
import csv
import numpy as np
import sys


f1 = csv.reader(open('./results/comparison-il-samp-mean-a0-e0.050000-k3-g100-r10.csv'))
f2 = csv.reader(open('./results/comparison-il-map-mean-a0-e0.050000-k3-g100-r10.csv'))
f3 = csv.reader(open('./results/comparison-unwgh-samp-mean-a0-e0.050000-k3-g100-r10.csv'))
f4 = csv.reader(open('./results/comparison-unwgh-map-mean-a0-e0.050000-k3-g100-r10.csv'))
f5 = csv.reader(open('./results/comparison-wgh-samp-mean-a0-e0.050000-k3-g100-r10.csv'))
f6 = csv.reader(open('./results/comparison-wgh-map-mean-a0-e0.050000-k3-g100-r10.csv'))
f7 = csv.reader(open('./results/prior-a0-e0.050000-k3-g100-r10.csv'))

compIndices = [45, 57, 108, 120]

def results_in_list(f):
    results = []
    firstline = True
    for i in f:
        if firstline:
            firstline = False
            continue
        results.append(float(i[1]))
    return results

il_samp = results_in_list(f1)
il_map = results_in_list(f2)
unwgh_samp = results_in_list(f3)
unwgh_map = results_in_list(f4)
wgh_samp = results_in_list(f5)
wgh_map = results_in_list(f6)
prior = results_in_list(f7)

def reorder_by_compositionality(lst):
    compo = [lst[compIndices[i]] for i in xrange(len(compIndices))]
    print compo
    for j in xrange(len(compIndices)):
        lst.pop(compIndices[j])
        lst.insert(j,compo[j])
    return lst

il_samp = reorder_by_compositionality(il_samp)
il_map = reorder_by_compositionality(il_map)
unwgh_samp = reorder_by_compositionality(unwgh_samp)
unwgh_map = reorder_by_compositionality(unwgh_map)
wgh_samp = reorder_by_compositionality(wgh_samp)
wgh_map = reorder_by_compositionality(wgh_map)
prior = reorder_by_compositionality(prior)


    
###Adding some white space to separate C from H:
def white_space(lst):
    for i in xrange(len(compIndices),len(compIndices)+10):
        lst.insert(i,0)
    return lst

il_samp = white_space(il_samp)
il_map = white_space(il_map)
unwgh_samp = white_space(unwgh_samp)
unwgh_map = white_space(unwgh_map)
wgh_samp = white_space(wgh_samp)
wgh_map = white_space(wgh_map)
prior = white_space(prior)

#Square trsf
def square_lst(lst):
    for i in xrange(len(lst)):
        lst[i] = sqrt(lst[i])
    return lst

il_samp = square_lst(il_samp)
il_map = square_lst(il_map)
unwgh_samp = square_lst(unwgh_samp)
unwgh_map = square_lst(unwgh_map)
wgh_samp = square_lst(wgh_samp)
wgh_map = square_lst(wgh_map)
prior = square_lst(prior)

        


X = np.arange(len(il_samp))

ymin,ymax = 0,1+0.01

fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,il_samp,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Iterated learning (post-sampling)',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()

fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,il_map,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Iterated learning (MAP)',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()


###
fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,unwgh_samp,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Functional pressure & learning (post-sampling, parental)',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()

fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,unwgh_map,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Functional pressure & learning (MAP, parental)',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()


###
fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,wgh_samp,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Functional pressure & learning (post-sampling, communal)',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()

fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,wgh_map,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Functional pressure & learning (MAP, communal)',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()


###
fig, ax = plt.subplots()
ax.set_xticks(np.array([len(compIndices)/2,(len(il_samp)-len(compIndices))/2]))
ax.set_xticklabels(('compositional', 'holistic'))
plt.ylabel('Mean in population')
ax.bar(X,prior,width=1)
ax.margins(0.025, 0.025)
ax.text(.5,.9,'Prior',
        horizontalalignment='center',
        transform=ax.transAxes)
ax.set_ylim([ymin,ymax])
plt.show()

