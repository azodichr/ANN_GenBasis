library(ggplot2)
setwd('/Volumes/azodichr/05_Insight/02_Modeling/sp_soy/')
setwd('/Users/cazodi/Desktop/')
df <- read.csv('accuracy_rice.txt', sep='\t')


df$fs_num <- as.factor(df$fs_num)
df$x_file <- NULL
df$feat_num <- NULL
df$feat_method <- NULL
d_rand <- subset(df, df$tag=='Random')
d_rand <- aggregate(d_rand, by=list(fs_num=d_rand$fs_num, y=d_rand$y, model=d_rand$model), FUN=mean)
d_rand$tag <- 'Random'
d_rand <- d_rand[,colSums(is.na(d_rand))<nrow(d_rand)]

d_tmp <- subset(df, df$tag != 'Random')

d <- rbind(d_tmp, d_rand)
d <- aggregate(d['PCC'], by=list(fs_num=d$fs_num, tag=d$tag, y=d$y,model=d$model), FUN=max)
d$r2 <- d$PCC^2
write.csv(d, file='rice_accuracy_r2.csv', quote=F)
reo_type <- function(x) { factor(x, levels = c('UN','I2','RL','RF','EN','BA','ji','hclus','pam','mca','pca','Random','all'))}
reo_mod <- function(x) { factor(x, levels = c('rrBLUP','BL','RF'))}


ht <- subset(d, y=='HT')
ggplot(ht, aes(x=fs_num, y=reo_type(tag))) + geom_tile(aes(fill=PCC), colour='white') +
  facet_grid(.~reo_mod(model)) + theme_bw(12) + geom_text(aes(label = round(PCC, 2))) +
  scale_fill_gradient2(limits=c(min(ht$PCC), max(ht$PCC)),
                       low='#ffffb2',mid='#fd8d3c', high='firebrick2', 
                       midpoint = mean(ht$PCC), guide = "colourbar")
ggsave('181018_HT_PredPower.pdf', width=6, height = 3, device='pdf')  # 1 column

ft <- subset(d, y=='FT')
ggplot(ft, aes(x=fs_num, y=reo_type(tag))) + geom_tile(aes(fill=PCC), colour='white') +
  facet_grid(.~reo_mod(model)) + theme_bw(12) + geom_text(aes(label = round(PCC, 2))) +
  scale_fill_gradient2(limits=c(min(ft$PCC), max(ft$PCC)),
                       low='#ffffb2',mid='#fd8d3c', high='firebrick2', 
                       midpoint = mean(ft$PCC), guide = "colourbar")
ggsave('181018_FT_PredPower.pdf', width=5, height = 3, device='pdf')  # 1 column

yl <- subset(d, y=='YLD')
ggplot(yl, aes(x=fs_num, y=reo_type(tag))) + geom_tile(aes(fill=PCC), colour='white') +
  facet_grid(.~reo_mod(model)) + theme_bw(12) + geom_text(aes(label = round(PCC, 2))) +
  scale_fill_gradient2(limits=c(min(yl$PCC), max(yl$PCC)),
                       low='#ffffb2',mid='#fd8d3c', high='firebrick2', 
                       midpoint = mean(yl$PCC), guide = "colourbar")
ggsave('181018_YLD_PredPower.pdf', width=5, height = 3, device='pdf')  # 1 column



#### Plot distribution of clustering metrics

m <- read.csv('../../01_FeatureEngineering/sp_soy/clustering_metrics.txt', header=T, sep='\t')

m$X..Features <- as.factor(m$X..Features)
ggplot(m, aes(x=X..Features, ymin=min, lower=qut1, middle=median, upper=qut3, ymax=max)) +
  geom_boxplot(stat='identity') + facet_grid(Method ~ Metric, scales = 'free') +
  theme_bw(10) + coord_flip()
ggsave('/Volumes/azodichr/05_Insight/00_Figs/180928_Clust_metrics.pdf', width=3.42, height = 2.42, device='pdf')  # 1 column


#### Plot stats of feature selection overlap
library(reshape2)
ol <- read.csv('FS_overlap_stats.txt', sep='\t')
olv <- melt(ol, id=c('trait', 'method'))
ggplot(olv, aes(x=variable, y=log10(value), fill=method)) + 
  geom_bar(position='dodge', stat='identity') +
  facet_grid(trait~.)

