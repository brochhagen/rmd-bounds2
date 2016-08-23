require('ggplot2')
require('dplyr')
require('reshape2')

d = read.csv('merged_select.csv')
ds = select(d, prior_cost_c, k, learning_parameter, t11_initial, lambda, t11_final) %>% 
  rename(c = prior_cost_c, l = learning_parameter, target = t11_final)

mFull = lm(formula = target ~ c * k * l * lambda * t11_initial, data = ds)
m = lm(formula = target ~ c * k * l, data = ds)
mCKL = lm(formula = target ~ c + k + l + lambda, data = ds)
mCK = lm(formula = target ~ c + k, data = ds)
mCL = lm(formula = target ~ c + l, data = ds)
mKL = lm(formula = target ~ k + l, data = ds)
mC = lm(formula = target ~ c, data = ds)
mK = lm(formula = target ~ k, data = ds)
mL = lm(formula = target ~ l, data = ds)

AIC(mFull,m,mCKL,mCL,mCK,mC,mKL,mL,mK)
summary(mFull)

