import modsinglescalar as rmd
#parameter order for run(*) is run(alpha,cost,lam,k, learning_parameter,gens,runs)

alpha = 1
lam = 30
cost = [x/100. for x in xrange(1,101)]
seq_length = [1,3,5,9,13,15,17,20]
learning_parameter = [1,2,4,6,8,10]
gens = 20
runs = 10000

for c in cost:
    for k in seq_length:
        for learn in learning_parameter:
                    rmd.run(alpha,c,lam,k,learn,gens,runs)
