#####################################
# Select best features using information gain (i.e. entropy)
#
# Arguments: [1] input features
#            [2] input traits
#            [3] what trait to predict
#            [4] high or low is good
#
# Written by: Christina Azodi
# Original: 11.06.18
# Modified: 
#####################################

#setwd('/Users/cazodi/Desktop/')
library(data.table)



args <- commandArgs(TRUE)
file <- args[1]
y_file <- args[2]
y_name <- args[3]
pos_dir <- args[4]

# Read in genotype data, convert to factor, and remove holdout
g <- fread(file, sep=',', header=T)
g <- as.data.frame(g)
row.names(g) <- g$ID
g$ID <- NULL

p <- read.table(y_file, sep=',', header=T, row.names = 'ID')
p$Y <- p[,y_name]
p$Ycat <- p$Y
if (pos_dir == 'high'){
  per <- quantile(p$Y, 0.8)
}
if (pos_dir == 'low'){
  per <- quantile(p$Y, 0.2)
}
p$Ycat[p$Y < per] <- "low"
p$Ycat[p$Y >= per] <- "high"

print('Threshold for classifying continuous DV as categorical:')
print(pos_dir)
print(per)

df <- merge(p['Ycat'], g, by='row.names')
row.names(df) <- df$Row.names
df$Row.names <- NULL

ho <- scan('holdout.txt', what='', sep='\n')
df_train <- subset(df, !(rownames(df) %in% ho))
print(df_train[1:5,1:5])

nh = unname(table(df_train$Ycat)['high'])
nl = unname(table(df_train$Ycat)['low'])
n = dim(df_train)[1]

entropy_1 <- - (nh/n)*log2(nh/n) - (nl/n)*log2(nl/n)
print('Entropy before split:')
print(entropy_1)

markers <- names(df_train)[2:dim(df_train)[2]]

result <- data.frame(marker = character(), n1=character(), n2=character(), ent1 = character(), ent2 = character(), info = character(), stringsAsFactors=FALSE)

i <- 0
for (mark in markers){
  i <- i + 1
  if(i %% 1000==0) {
    print(paste0("Running marker: ", i, "\n"))
  }
  
  left <- subset(df_train, df_train[mark] == 1)
  nleft <- dim(left)[1]
  nlefth <- unname(table(left$Ycat)['high'])
  nleftl <- unname(table(left$Ycat)['low'])
  entropy_left <- - (nlefth/nleft)*log2(nlefth/nleft) - (nleftl/nleft)*log2(nleftl/nleft)

  rig <- subset(df_train, df_train[mark] == -1)
  nrig <- dim(rig)[1]
  nrigh <- unname(table(rig$Ycat)['high'])
  nrigl <- unname(table(rig$Ycat)['low'])
  entropy_rig <- - (nrigh/nrig)*log2(nrigh/nrig) - (nrigl/nrig)*log2(nrigl/nrig)

  # If either right or left entropy = 0, replace NA with 0
  if (is.na(entropy_left)){
    entropy_left <- 0
  }
  if (is.na(entropy_rig)){
    entropy_rig <- 0
  }
  
  entropy_2 <- (nleft/n)*entropy_left + (nrig/n)*entropy_rig
  ig <- entropy_1 - entropy_2
  result <- rbind(result, list(marker = mark, n1 = nleft, n2 = nrig, ent1=entropy_1, ent2=entropy_2, info = ig), stringsAsFactors=F)
}


res_sort <- result[order(-result$info, result$marker),] 
print('Snapshot of Information Gain results:')
print(head(res_sort))

for (n in c(50,100,500,1000)){
  name <- paste('featsel', y_name, 'IG', n, sep='_')
  write.table(res_sort$marker[1:n], name, quote=F, col.names=F, row.names=F)
}

name2 <- paste('InfoGain_res_', y_name, '.csv', sep='')
write.table(res_sort, name2, quote=F, col.names=T, row.names=F)
print('Done!')
