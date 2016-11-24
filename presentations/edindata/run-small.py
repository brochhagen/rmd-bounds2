import modsinglescalar as rmd
import modsinglescalaronlyfitness as fit
import modsinglescalaronlylearning as learner
#parameter order for run(*) is run(alpha,cost,lam,k,sample_amount, learning_parameter,gens,runs)


def cost_to_learning(): #results 1
    a = 1
    lam = 30
    cost = [x/100. for x in xrange(100)]
    seq_length = 5
    sample_amount = 10
    learning_parameter = [x for x in xrange(1,11)]
    gens = 20
    runs = 1000
    
    for c in cost:
        for learn in learning_parameter:
            rmd.run(a,c,lam,seq_length,sample_amount,learn,gens,runs)

def ration_to_seq():
    a = 1
    lam = [x for x in xrange(1,50)]
    c = .4
    seq_length = [x for x in xrange(1,20)]
    sample_amount = 20
    learn = 3
    gens = 20
    runs = 1000
    
    for l in lam:
        for k in seq_length:
            rmd.run(a,c,l,k,sample_amount,learn,gens,runs)

def alpha_to_seq():
    a = [x/100. for x in xrange(1,101)]
    lam = 20
    c = .4
    seq_length = [x for x in xrange(1,20)]
    sample_amount = 20
    learn = 3
    gens = 20
    runs = 1000
    
    for al in a:
        for k in seq_length:
            rmd.run(al,c,l,k,sample_amount,learn,gens,runs)

def only_r_and_m():
    a = 1
    lam = 30
    cost = [x/100. for x in xrange(100)]
    learn = 1
    seq_length = 5
    sample_amount = 10
    gens = 20
    runs = 1000
    
    for c in cost:
        print 'learn-only'
        learner.run(a,c,lam,seq_length,sample_amount,learn,gens,runs)
        print 'fit-only'
        fit.run(a,c,lam,seq_length,sample_amount,learn,gens,runs)



#only_r_and_m()
