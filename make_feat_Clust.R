#####################################
# Uses a feature file to run clustering
#
# Arguments: [1] input data
#            [2] Clustering Method (pam, hclus, ji, sbm)
#            [3] k
#
# Written by: Christina Azodi
# Original: 9.12.18
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
print(k)
# Read in genotype data, convert to factor, and remove holdout
g <- fread(file, sep=',', header=T)
#g <- fread('geno_noDups.csv', sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL
print(g[1:5,1:5])

ho <- scan('holdout.txt', what='', sep='\n')
g_train <- subset(g, !(rownames(g) %in% ho))

# Transpose to cluster columns (i.e. features), not rows (i.e. instances)
g_train <- t(g_train)




############
### pam ####
############

if(clust_method=='pam'){
  print('Clustering with pam...')
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
  print('Calculating cluster metrics...')
  pccs <- c()
  mis <- c()
  ns <- m$clusinfo[,'size']
  
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
}

####################################
### hclus: euclidean and Jaccard ###
####################################

if(clust_method=='hclus'){
  print('Hierarchical clustering using Euclidean distance...')
  d <- dist(g_train)
}

if(clust_method=='ji'){
  print('Hierarchical clustering using Jaccard Index for distance...')
  library(vegan)
  g_train[g_train == -1] <- FALSE
  g_train[g_train == 1] <- TRUE
  d <- vegdist(g_train, method='jaccard', na.rm=TRUE, binary=TRUE)
}


# Function to find medoids from the distance matrix
clust.medoid = function(i, distmat, clusters) {
  ind = (clusters == i)
  if(sum(ind)==1){
    names(clusters[ind]) # Returns the only item in cluster if cluster size = 1
  }else{
    names(which.min(rowSums(distmat[ind, ind] )))
  }
}


if(clust_method %in% c('hclus','ji')){
  d2 <- as.matrix(d)
  m <- cutree(hclust(d, method='ward.D'), k=k)
  
  # Put cluster assignments in clusts df
  clusts <- as.data.frame(m)
  names(clusts) <- c('k')
  
  # Pull out any clusters that are n=1:
  df <- as.data.frame(sapply(unique(m), clust.medoid, d2, m))
  names(df) <- c('mediod')

  # For each k ID the features, calculate mean PCC and mean MI
  print('Calculating cluster metrics...')
  ns <- summary(as.factor(m))
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
}



# Output for each run
print('PCC:')
print(summary(pccs))
print('MI:')
print(summary(mis))
print('Clust size:')
print(summary(ns))

sink('Clustering_Stats.csv')
paste('Clustering results for', clust_method, 'with k=', k, '(PCC,MI,Num)', sep=' ')
paste(summary(pccs), sep='')
paste(summary(mis), sep='')
paste(summary(ns), sep='')
sink()

name1 <- paste('mediod_', clust_method, '_', k, '.csv', sep = '')
write.table(df$mediod, name1, quote=F, col.names=F, row.names=F)
name2 <- paste('clustAssign_', clust_method, '_', k, '.csv', sep = '')
write.table(df, name2, sep=',', quote=F, col.names=F, row.names=F)


print('Done!')