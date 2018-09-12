#####################################
# Uses a feature file to run clustering
#
# Arguments: [1] input data
#            [2] Clustering Method (pam, heir, ji, sbm)
#            [3] k
#
# Written by: Christina Azodi
# Original: 9.10.18
# Modified: 
#####################################

#setwd('/Users/cazodi/Desktop/')
library(data.table)
library(ggplot2)
library(mpmi)


args <- commandArgs(TRUE)
file <- args[1]
clust_method <- args[2]
k <- args[3]

# Read in genotype data, convert to factor, and remove holdout
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL
print(g[1:5,1:5])

ho <- scan('holdout.txt', what='', sep='\n')
g_train <- subset(g, !(rownames(g) %in% ho))

# Transpose to cluster columns (i.e. features), not rows (i.e. instances)
g_train <- t(g_train)



library(cluster)
m <- pam(g_train, k, metric='euclidean')
df <- as.data.frame(m$clustering[m$id.med])
names(df) <- c('k')
df$mediod <- row.names(df)
row.names(df) <- df$k
df$k <- NULL
clusts <- as.data.frame(m$clustering)
names(clusts) <- c('k')

# For each k ID the features, calculate mean PCC and mean MI
pccs <- c()
mis <- c()

for(ik in 1:k){
  medio <- df[ik,'mediod']
  all <- row.names(subset(clusts, k==ik))
  cor <- cor(g[all])
  cor[lower.tri(cor, diag=T)] <- NA
  pccs <- c(pccs, summary(as.vector(abs(cor)))['Mean'])
  
  mi <- dmi(g[all])
  mi$bcmi[lower.tri(mi$bcmi, diag=T)] <- NA
  mis <- c(mis, summary(as.vector(abs(mi$bcmi)))['Mean'])
  
  all <- all[all != medio]
  df[ik,'group'] <- paste( unlist(all), collapse=',')
}

ns <- m$clusinfo[,'size']


# Output for each run
print('PCC:')
print(summary(pccs))
print('MI:')
print(summary(mis))
print('Clust size:')
print(summary(ns))

name <- paste('medio_', clust_method, '_', str(k), '.csv', sep = '')
write.table(df$mediod, name, quote=F, col.names=F, row.names=F)

      