############################
# Script to remove duplicate features from the model
# A summary of what duplicates were found is output.
# The first feature listed in each line represents the 
# feature selected to keep.
############################

# setwd('/Users/cazodi/Desktop/')
library(data.table)
args = commandArgs(TRUE)

file = args[1]


# Read in genotype data
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL
all_snps <- names(g)
all_snps2 <- names(g)

# Make lists of sets of SNPs
drop_these <- c()
key <- c()
count <- 0

for(c_name in all_snps){
  all_snps2 <- all_snps2[all_snps2 != c_name]
  temp_list <- 'na'
  count = count + 1
  if(count %%10 == 0){
    print(paste('iteration', count,'complete\n'))
  } 
  for(c2_name in all_snps2){
    if(cor(g[c_name], g[c2_name])==1){
      if(c_name != c2_name){
        temp_list <- c(c_name)
        temp_list <- c(temp_list, c2_name)
        drop_these <- c(drop_these, c2_name)
        all_snps2 <- all_snps2[all_snps2 != c2_name]
        all_snps <- all_snps[all_snps != c2_name]
      }
    }
  }
  if(temp_list != 'na'){
    key <- c(key, list(temp_list))
  }
}


lapply(key, write, "duplicate_snp_info.csv", append=TRUE, ncolumns=10000)

# Remove duplicate SNPs
keep <- setdiff(all_snps, drop_these)
g2 <- g[keep]

write.table(g2, 'geno_noDups.csv', sep=',', quote=F)
print('Number of dupliate SNP pairs with PCC=1')
print(dim(g)[2] - dim(g2)[2])

print('Done!')



