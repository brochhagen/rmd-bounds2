library(tidyverse)
library(reshape2)
library(bootstrap)
# quantile(bootstrap::bootstrap(x = mean, nboot = 1000, theta = function(x){mean(x)})$thetastar, 0.025)

## only replication

d = read.csv('ordered-data-plot2.csv') 
e = melt(d, id.vars = c(1,2), variable.name = "strategy", value.name = "proportion") %>%
  mutate(strategy = as.character(strategy))


e$type = sapply(1:nrow(e), function(i) ifelse(strsplit(e$strategy[i], split = ".", fixed = T)[[1]][1] == "other", "other",
                                              strsplit(e$strategy[i], split = ".", fixed = T)[[1]][2]))
                       

f = e %>% filter(type != "other") %>% arrange(runID, lambda, type, - proportion) %>% group_by(runID, lambda, type) %>% mutate(rank = 1:n())

plot.data = f %>% group_by(lambda, type, rank) %>%
  summarize(mean = mean(proportion),
            bslo = quantile(bootstrap::bootstrap(x = proportion, nboot = 2000, theta = function(x){mean(x)})$thetastar, 0.025),
            bshi = quantile(bootstrap::bootstrap(x = proportion, nboot = 2000, theta = function(x){mean(x)})$thetastar, 0.975))
plot.data = plot.data %>% mutate(strategy = paste0(type, rank))

results.plot = ggplot(plot.data, aes(x = strategy, y = mean, fill = type)) + geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin=bslo, ymax=bshi), width=0.5, position = "dodge") + facet_grid(. ~ lambda) + coord_flip()
show(results.plot)
ggsave("plots/results_summary_replication.pdf", results.plot, width = 10, height = 10)


## only mutation

d = read.csv('ordered-data-plot3.csv') 
e = melt(d, id.vars = c(1,2,3), variable.name = "strategy", value.name = "proportion") %>%
  mutate(strategy = as.character(strategy))

e$type = sapply(1:nrow(e), function(i) ifelse(strsplit(e$strategy[i], split = ".", fixed = T)[[1]][1] == "other", "other",
                                              strsplit(e$strategy[i], split = ".", fixed = T)[[1]][2]))

f = e %>% filter(type != "other") %>% arrange(runID, lambda, l, type, - proportion) %>% 
  group_by(runID, lambda, l, type) %>% mutate(rank = 1:n())

plot.data = f %>% group_by(l, lambda, type, rank) %>%
  summarize(mean = mean(proportion),
            bslo = quantile(bootstrap::bootstrap(x = proportion, nboot = 2000, theta = function(x){mean(x)})$thetastar, 0.025),
            bshi = quantile(bootstrap::bootstrap(x = proportion, nboot = 2000, theta = function(x){mean(x)})$thetastar, 0.975))
plot.data = plot.data %>% mutate(strategy = paste0(type, rank))

results.plot = ggplot(plot.data, aes(x = strategy, y = mean, fill = type)) + geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin=bslo, ymax=bshi), width=0.5, position = "dodge") + facet_grid(l ~ lambda) + coord_flip()
show(results.plot)
ggsave("plots/results_summary_mutation.pdf", results.plot, width = 10, height = 10)

## mutation & replication

d = read.csv('ordered-data-plot4.csv') 
e = melt(d, id.vars = c(1,2,3), variable.name = "strategy", value.name = "proportion") %>%
  mutate(strategy = as.character(strategy))

e$type = sapply(1:nrow(e), function(i) ifelse(strsplit(e$strategy[i], split = ".", fixed = T)[[1]][1] == "other", "other",
                                              strsplit(e$strategy[i], split = ".", fixed = T)[[1]][2]))

f = e %>% filter(type != "other") %>% arrange(runID, lambda, l, type, - proportion) %>% 
  group_by(runID, lambda, l, type) %>% mutate(rank = 1:n())

plot.data = f %>% group_by(l, lambda, type, rank) %>%
  summarize(mean = mean(proportion),
            bslo = quantile(bootstrap::bootstrap(x = proportion, nboot = 2000, theta = function(x){mean(x)})$thetastar, 0.025),
            bshi = quantile(bootstrap::bootstrap(x = proportion, nboot = 2000, theta = function(x){mean(x)})$thetastar, 0.975))
plot.data = plot.data %>% mutate(strategy = paste0(type, rank))

results.plot = ggplot(plot.data, aes(x = strategy, y = mean, fill = type)) + geom_bar(stat = "identity") +
  geom_errorbar(aes(ymin=bslo, ymax=bshi), width=0.5, position = "dodge") + facet_grid(l ~ lambda) + coord_flip()
show(results.plot)
ggsave("plots/results_summary_combined.pdf", results.plot, width = 10, height = 10)