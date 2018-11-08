setwd('/Users/cazodi/Desktop/')

library(data.table)
library(ggplot2)
########################################################
# Calcuate VIF (variance inflation factor) of each SNP #
########################################################
d <- fread('geno_noDups.csv', sep=',', header=T)
d <- as.data.frame(d)
row.names(d) <- d$ID
d$ID <- NULL
print(d[1:5,1:5])

ho <- scan('holdout.txt', what='', sep='\n')
dt <- subset(d, !(rownames(d) %in% ho))

# Test first 1k SNPs on Chromosome1 (all within 4.998 MB)
test <- dt[,1:1000]
r2 <- c()
for(i in 1:length(test)){
  y <- names(test)[i]
  x <- names(test)[-i]
  model<- lm(formula(paste(paste(y),"~",paste(x, collapse=" + "))),data=test) 
  r2 <- append(r2, summary(model)$r.squared)
}
summary(r2)
summary(1/(1-r2))


# Look for coliniarity within cluster medoids!
clust <- scan('mediod_hclus_1000.csv', what='', sep='\n')
dcl <- dt[clust]

r2_cl <- c()
for(i in 1:length(dcl)){
  y <- names(dcl)[i]
  x <- names(dcl)[-i]
  model<- lm(formula(paste(paste(y),"~",paste(x, collapse=" + "))),data=dcl) 
  r2_cl <- append(r2_cl, summary(model)$r.squared)
}

summary(r2_cl)
summary(1/(1-r2_cl))


#################################################
# Look for patterns in PCA and MCA eigenvectors #
#################################################
library(ComplexHeatmap)
library(circlize)
setwd('/Volumes/azodichr/05_Insight/01_FeatureEngineering/sp_rice/')

# PCA
pc <- fread('pca_loadings.csv', sep=',', header=T)
row.names(pc) <- pc$V1
pc$V1 <- NULL

pcmax <-apply(pc,1,max)
pcmin <-apply(pc,1,min)
pctemp <- data.frame(names=row.names(pc), min=pcmin, max=pcmax)
pc_keep <- subset(pctemp, min<quantile(pctemp$min,0.05) | max>quantile(pctemp$max,0.95) )
pctest <- subset(pc, row.names(pc) %in% pc_keep$names)
row.names(pctest) <- pc_keep$names

Heatmap(as.matrix(pctest), name='PCA Loadings', 
        cluster_columns = F, cluster_rows = F,
        show_row_names = F, show_column_names = F,
        col=colorRamp2(c(min(pctest),0,max(pctest)), c('#8c510a','white','#01665e')))

# MCA
mc <- read.csv('mca_loadings.csv', sep=',', header=T)
mc2 <- mc[grepl("_1$", mc$X),]
mc2$X <- as.character(mc2$X)
mc2$X = substr(mc2$X,1,nchar(mc2$X)-2)
row.names(mc2) <- mc2$X
mc2$X <- NULL

Heatmap(as.matrix(mc2), name='MCA Loadings', 
        cluster_columns = F, cluster_rows = F,
        show_row_names = F, show_column_names = F,
        col=colorRamp2(c(min(mc2),0.001,max(mc2)), c('white','#eff3ff','#084594')))



#############################################
# Plot coefficients from rrBLUP and B-LASSO #
#############################################

bl_ht <- read.csv('BL_coef_HT.txt')
bl_ft <- read.csv('BL_coef_FT.txt')
bl_yld <- read.csv('BL_coef_YLD.txt')

rb_ht <- read.csv('rrB_coef_HT.txt')
rb_ft <- read.csv('rrB_coef_FT.txt')
rb_yld <- read.csv('rrB_coef_YLD.txt')

rf_ht <- read.csv('featsel_HT_RF_RFScores.txt')
rf_ft <- read.csv('featsel_FT_RF_RFScores.txt')
rf_yld <- read.csv('featsel_YLD_RF_RFScores.txt')

ht <- data.frame(rb=rb_ht$coef, bl=bl_ht$coef, rf=rf_ht$imp)
row.names(ht) <- rf_ht$X
ht$trait <- 'HT'
ft <- data.frame(rb=rb_ft$coef, bl=bl_ft$coef, rf=rf_ft$imp)
row.names(ft) <- rf_ft$X
ft$trait <- 'FT'
yl <- data.frame(rb=rb_yld$coef, bl=bl_yld$coef, rf=rf_yld$imp)
row.names(yl) <- rf_yld$X
yl$trait <- 'YLD'
all <- rbind(ht, ft, yl)
allm <- melt(all, id='trait')
allm$value[allm$variable == rf] <- 3

ggplot(ht, aes(x=abs(rb), y=bl, color=rf)) + 
  geom_point(alpha=0.2, shape=20) + theme_bw(10) +
  scale_colour_gradientn(colours = rainbow(10))

ggplot(ft, aes(x=abs(rb), y=bl, color=rf)) + 
  geom_point(alpha=0.2, shape=20) + theme_bw(10) +
  scale_colour_gradientn(colours = rainbow(10))

ggplot(yl, aes(x=abs(rb), y=bl, color=rf)) + 
  geom_point(alpha=0.2, shape=20) + theme_bw(10) +
  scale_colour_gradientn(colours = rainbow(10))

cor.test(abs(ht$rb), ht$bl, method='pearson')
cor.test(abs(ht$rb), ht$rf, method='pearson')
cor.test(ht$bl, ht$rf, method='pearson')

cor.test(abs(ft$rb), ft$bl, method='pearson')
cor.test(abs(ft$rb), ft$rf, method='pearson')
cor.test(ft$bl, ft$rf, method='pearson')

cor.test(abs(yl$rb), yl$bl, method='pearson')
cor.test(abs(yl$rb), yl$rf, method='pearson')
cor.test(yl$bl, yl$rf, method='pearson')
