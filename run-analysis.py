import modsinglescalar as rmd
#parameter order for run(*) is run(alpha,cost,lam,k,sample_amount, learning_parameter,gens,runs)

alpha = [0.25,0.5,0.7,1]
cost = [0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.8,0.9]
lam = [1,10,30,50]
seq_length = [1,3,5,9,13,15,17,20]
sample_amount = [1,3,5,7,9,12,15] + [20,40,60,80,100] #+ [200,300,400,500]
learning_parameter = [1,2,4,6,8,10]
gens = 20
runs = 1000

for a in alpha:
    for c in cost:
        for l in lam:
            for k in seq_length:
                for learn in learning_parameter:
                    rmd.run(a,c,l,k,learn,gens,runs)
