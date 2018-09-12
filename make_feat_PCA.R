#####################################
# Uses a feature file to run PCA
#
# Arguments: [1] input data
#
# Written by: Christina Azodi
# Original: 9.10.18
# Modified: 
#####################################

# setwd('/Users/cazodi/Desktop/')
library(data.table)
library(ggplot2)

args = commandArgs(TRUE)
file = args[1]

# Read in genotype data and remove holdout set before making PCs
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL
print(g[1:5,1:5])

ho <- scan('holdout.txt', what='', sep='\n')

g_test <- subset(g, rownames(g) %in% ho)
g_train <- subset(g, !(rownames(g) %in% ho))


# Run PCA analysis on training set & calculate % var explained
print('Running PCA...')
pc_train <- prcomp(g_train, scale. = T)
prop_varex <- pc_train$sdev^2/sum(pc_train$sdev^2)

print('Generating variance explained figures...')
ggplot(as.data.frame(prop_varex), aes(y=prop_varex, x=seq(1, length(prop_varex)))) + geom_point() +
  theme_bw(10) + xlab("Principal Component") + ylab("% of Var Explained")
ggsave('plot_pca_VarExp.pdf', width = 4, height = 4, useDingbats=FALSE)

ggplot(as.data.frame(cumsum(prop_varex)), aes(y=cumsum(prop_varex), x=seq(1, length(cumsum(prop_varex))))) + geom_point() +
  theme_bw(10) + xlab("Principal Component") + ylab("Cumulative % of Var Explained")
ggsave('plot_pca_CumVarExp.pdf', width = 4, height = 4, useDingbats=FALSE)

print('Cumulartive % of Variance explained by the first 50, 500, and 1000 PCs')
print(sum(prop_varex[1:50]))
print(sum(prop_varex[1:500]))
print(sum(prop_varex[1:1000]))


pc_test <- as.data.frame(predict(pc_train, newdata=g_test))

pc_data <- rbind(pc_test, pc_train$x)

# Make sort into original order of geno.csv:
pc_data <- pc_data[match(rownames(g), rownames(pc_data)),]
write.csv(pc_data[,1:50], 'geno_pca_50.csv', quote=F, row.names = T)
write.csv(pc_data[,1:500], 'geno_pca_500.csv', quote=F, row.names = T)
write.csv(pc_data[,1:1000], 'geno_pca_1000.csv', quote=F, row.names = T)
write.csv(pc_train$rotation[,1:1000], 'pca_loadings.csv', quote=F, row.names=T)

print('Done!')
