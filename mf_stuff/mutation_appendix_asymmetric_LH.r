library(tidyverse)

# 3 types, 3 data
# types t2 and t3 are more likely to produce data that is evidence for t1
e = 0.1
d = 0.2
lh = matrix(c(1 - 2*e, e, e, d, 1-(d+e), e, d, e, 1-(d+e)), nrow = 3, byrow = T) %>% prop.table(1)

# posterior over types for each datum:
l = 10

post = prop.table(t(lh),1)
post = prop.table(post^l,1)

q = matrix(0, nrow = 3, ncol = 3)

for (i in 1:3) {
  for (j in 1:3) {
    q[i,j] = sum(post[,j] * lh[i,])
  }
}
q %>% round(.,2) %>%  show()
eigen(t(q)) %>% show()
eigen(t(q))$vectors[,1] / sum(eigen(t(q))$vectors[,1])
