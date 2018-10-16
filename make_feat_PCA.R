#####################################
# Uses a feature file to run PCA
#
# Arguments: [1] input data
#            [2] Number of dimension you want in output (if multiple, separate by comma)
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
n_s_input = args[2]
n_s <- unlist(strsplit(n_s_input, split=','))
n_sizes <- c()
for(ns in n_s){
  n_sizes <- c(n_sizes, as.numeric(ns))
}

# Read in genotype data and remove holdout set before making PCs
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL
print(g[1:5,1:5])

ho <- scan('holdout.txt', what='', sep='\n')

g_train <- subset(g, !(rownames(g) %in% ho))
g_test <- subset(g, rownames(g) %in% ho)

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

print('Cumulartive % of Variance explained by X PCs:')
for(ns in n_sizes){
  print(ns)
  print(sum(prop_varex[1:ns]))
}

pc_test <- as.data.frame(predict(pc_train, newdata=g_test))
pc_data <- rbind(pc_test, pc_train$x)
print('Dimensions of final PC dataframe:')
print(dim(pc_data))

# Make sort into original order of geno.csv:
pc_data <- pc_data[match(rownames(g), rownames(pc_data)),]
for(ns in n_sizes){
  write.csv(pc_data[,1:ns], paste('geno_pca_', ns,'.csv', sep=''), quote=F, row.names = T)
}

write.csv(pc_train$rotation[,1:max(n_sizes)], 'pca_loadings.csv', quote=F, row.names=T)


print('Done!')
