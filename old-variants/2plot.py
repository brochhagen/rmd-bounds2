#####
#Read in CSV, plot results.
#####
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context(rc={'lines.markeredgewidth': 0.5})
import csv
import sys

f1 = csv.reader(open('./results/2multiscalar-unwgh-mean-a1-c0.900000-l30-k9-g30-r100-s1.csv','rt'))
f2 = csv.reader(open('./results/2multiscalar-wgh-mean-a1-c0.900000-l30-k9-g30-r100-s1.csv','rt'))


def results_by_lexica_multi(f):
    p = np.zeros(6)
    firstline = True
    for i in f:
        if firstline:
            firstline = False
            continue
        p[int(i[1])] += float(i[-1]) /3.
        p[int(i[2])] += float(i[-1]) /3.
        p[int(i[3])] += float(i[-1]) /3.
    return p

def results_by_lexica_single(f):
    p = np.zeros(6)
    firstline = True
    for i in f:
        if firstline:
            firstline = False
            continue
        p[int(i[1])] += float(i[-1])
    return p

Y_single = results_by_lexica_single(f1)
Y_multi = results_by_lexica_multi(f2)

print Y_single
print Y_multi
###Plots##

X = np.arange(6) #number of lexica




width = 0.35
#ymin,ymax = 0,np.max(np.array([Ya[np.argmax(results3a)],Yb[np.argmax(results3b)],Yc[np.argmax(results3c)]]))+0.01
#ax.set_xticks(np.array([len(compIndices)/2,(len(hypotheses)-len(compIndices))/2]))

fig, ax = plt.subplots()
bar1 = ax.bar(X, Y_single, width,color='r')
bar2 = ax.bar(X+width,Y_multi,width,color='y')

plt.ylabel('Mean in population')
ax.set_xticks(X+width)
ax.set_xticklabels(('L1', 'L2', 'L3', 'L4','L5','L6'))
ax.legend((bar1[0], bar2[0]), ('unweighted', 'weighted'))
ax.margins(0.025, 0.025)
#ax.text(.5,.9,'$\alpha$ = 1, c = %r, $\lambda$ = %r, pairs = %r, k = %r',
#        horizontalalignment='center',
#        transform=ax.transAxes) % (alpha, cost, lam, lexical_pairs, k)
ax.set_title('3 scalars (alternative2). a = %r, c = %r, lam = %r, k = %r' % (1, .9, 30, 9))

#ax.set_ylim([ymin,ymax])
plt.show()

