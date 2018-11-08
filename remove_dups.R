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

# Caluclate correlation between features pairwise
cors <- cor(g)
diag(cors) <- NA

# Just look at features with PCC=1 
dups <- colnames(cors)[apply(cors, 2, function(value) any(value==1))]
dups <- na.omit(dups)

cors_sub <- cors[dups, dups]
cors_sub[lower.tri(cors_sub, diag=T)] <- 0


# Make lists of sets of SNPs
drop_these <- c()
key <- c()

for(r in 1:nrow(cors_sub)){
  row_name <- rownames(cors_sub)[r]
  
  # Checks if SNP is already in the list (will get duplicate sets without all matches if not)
  if(row_name %in% unlist(drop_these)){ 
  } else {
    col_name <- names(cors_sub[r,][cors_sub[r,] == 1])
    
    # If a match is found (which is should always be...) add to sets
    if(length(col_name)>0){
      drop_these <- c(drop_these, col_name)
      temp_list <- c(row_name, col_name)
      key <- c(key, list(temp_list))
    }
  }
}

lapply(key, write, "duplicate_snp_info.csv", append=F, ncolumns=10000)

# Remove duplicate SNPs
keep <- setdiff(all_snps, drop_these)
g2 <- g[keep]
print(g2[1:5,1:5])

write.table(g2, 'geno_noDups.csv', sep=',', quote=F)
print('Number of dupliate SNP pairs with PCC=1')
print(dim(g)[2] - dim(g2)[2])

print('Done!')



