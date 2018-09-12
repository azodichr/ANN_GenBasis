#####################################
# Uses a feature file to run MCA
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
library(FactoMineR)
library(factoextra)

args = commandArgs(TRUE)
file = args[1]

# Read in genotype data, convert to factor, and remove holdout
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL
g[sapply(g, is.integer)] <- lapply(g[sapply(g, is.integer)], as.factor)
print(g[1:5,1:5])

ho <- scan('holdout.txt', what='', sep='\n')
ho_index <- match(ho, rownames(g))


# Run MCA analysis on training set & calculate % var explained
print('Running MCA...')
mc_train <- MCA(g, ncp=1000, graph = F, ind.sup=ho_index)

prop_varex <- get_eig(mc_train)[,'variance.percent']*0.01

print('Generating variance explained figures...')
ggplot(as.data.frame(prop_varex), aes(y=prop_varex, x=seq(1, length(prop_varex)))) + geom_point() +
  theme_bw(10) + xlab("Principal Component") + ylab("% of Var Explained")
ggsave('plot_mca_VarExp.pdf', width = 4, height = 4, useDingbats=FALSE)

ggplot(as.data.frame(cumsum(prop_varex)), aes(y=cumsum(prop_varex), x=seq(1, length(cumsum(prop_varex))))) + geom_point() +
  theme_bw(10) + xlab("Principal Component") + ylab("Cumulative % of Var Explained")
ggsave('plot_mca_CumVarExp.pdf', width = 4, height = 4, useDingbats=FALSE)

print('Cumulartive % of Variance explained by the first 50, 500, and 1000 PCs')
print(sum(prop_varex[1:50]))
print(sum(prop_varex[1:500]))
print(sum(prop_varex[1:1000]))


# Pull coordinates of training lines and predicted coordinates from test lines
mc_data <- rbind(mc_train$ind$coord, mc_train$ind.sup$coord)

# Sort into original order:
mc_data <- mc_data[match(rownames(g), rownames(mc_data)),]

write.csv(mc_data[,1:50], 'geno_mca_50.csv', quote=F, row.names = T)
write.csv(mc_data[,1:500], 'geno_mca_500.csv', quote=F, row.names = T)
write.csv(mc_data[,1:1000], 'geno_mca_1000.csv', quote=F, row.names = T)
write.csv(mc_train$var$contrib[,1:1000], 'mca_loadings.csv', quote=F, row.names=T)

print('Done!')
