#####################################
# Uses a feature file to run MCA
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
library(FactoMineR)
library(factoextra)

# Read in and format arguments
args = commandArgs(TRUE)
file = args[1]
n_s_input = args[2]
n_s <- unlist(strsplit(n_s_input, split=','))
n_sizes <- c()
for(ns in n_s){
  n_sizes <- c(n_sizes, as.numeric(ns))
}


# Read in genotype data, convert to factor, and remove holdout
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL

# Round to nearest int and convert to factor
g <- round(g, 0)
g[sapply(g, is.numeric)] <- lapply(g[sapply(g, is.numeric)], as.factor)
print(g[1:8,1:8])

ho <- scan('holdout.txt', what='', sep='\n')
ho_index <- match(ho, rownames(g))

# Remove duplicate columns AFTER dropping holdout set
keep <- subset(g, !(row.names(g) %in% ho))
keep2 <- keep[sapply(keep, function(x) length(levels(factor(x)))>1)]
g_keep <- g[,names(keep2)]

# Run MCA analysis on training set & calculate % var explained
print('Running MCA...')
keep_size <- as.numeric(max(n_sizes))

mc_train <- MCA(g_keep, graph=F, ncp=keep_size,ind.sup=ho_index)
prop_varex <- get_eig(mc_train)[,'variance.percent']*0.01



print('Generating variance explained figures...')
ggplot(as.data.frame(prop_varex), aes(y=prop_varex, x=seq(1, length(prop_varex)))) + geom_point() +
  theme_bw(10) + xlab("Principal Component") + ylab("% of Var Explained")
ggsave('plot_mca_VarExp.pdf', width = 4, height = 4, useDingbats=FALSE)

ggplot(as.data.frame(cumsum(prop_varex)), aes(y=cumsum(prop_varex), x=seq(1, length(cumsum(prop_varex))))) + geom_point() +
  theme_bw(10) + xlab("Principal Component") + ylab("Cumulative % of Var Explained")
ggsave('plot_mca_CumVarExp.pdf', width = 4, height = 4, useDingbats=FALSE)


print('Cumulartive % of Variance explained by X MCA axes:')
for(ns in n_sizes){
  print(ns)
  print(sum(prop_varex[1:ns]))
}

# Pull coordinates of training lines and predicted coordinates from test lines
mc_data <- rbind(mc_train$ind$coord, mc_train$ind.sup$coord)

# Sort into original order:
mc_data <- mc_data[match(rownames(g), rownames(mc_data)),]

print('Dimensions of final PC dataframe:')
print(dim(mc_data))

# Make sort into original order of geno.csv:
for(ns in n_sizes){
  write.csv(mc_data[,1:ns], paste('geno_mca_', ns,'.csv', sep=''), quote=F, row.names = T)
}

write.csv(mc_train$var$contrib[,1:max(n_sizes)], 'mca_loadings.csv', quote=F, row.names=T)

print('Done!')
