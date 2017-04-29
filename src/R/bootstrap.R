#!/usr/bin/env Rscript

library("boot")

temps <- read.table("../../output/critical_temps.dat")

sample_mean <- function(data,i) {
  return(mean(data[i,]))
}

b <- boot(temps, statistic=sample_mean, R=10000)
print(b)
print(boot.ci(b))
