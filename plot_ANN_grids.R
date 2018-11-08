library(ggplot2)
library(reshape2)
setwd('/Users/cazodi/Desktop/')
setwd('/Volumes/azodichr/05_Insight/02_Modeling/sp_rice/04_MLP/')

d <- read.csv('FT_pca250_GridSearch.txt', sep='\t', header=T)
d <- read.csv('FT_RL500_GridSearch.txt', sep='\t', header=T)
d <- read.csv('HT_UN1000_GridSearch.txt', sep='\t', header=T)
d <- read.csv('HT_RL1000_GridSearch.txt', sep='\t', header=T)
d <- read.csv('HT_HE1000_GridSearch.txt', sep='\t', header=T)
d <- read.csv('HT_pca250_GridSearch.txt', sep='\t', header=T)
d <- read.csv('YLD_EN500_GridSearch.txt', sep='\t', header=T)


d <- na.omit(d)
d$Tag <- paste(d$ActFun, d$Arch, d$L1, d$L2, d$LearnRate, sep='_')
dagg <- aggregate(d['Loss'], by=list(ActFun=d$ActFun, Arch=d$Arch, L1=d$L1, L2=d$L2, LearnRate=d$LearnRate), FUN=median)
dagg$Tag <- paste(dagg$ActFun, dagg$Arch, dagg$L1, dagg$L2, dagg$LearnRate, sep='_')
dm <- melt(dagg, id=c('Tag', 'Loss'), measure.vars = c('ActFun','Arch','L1','L2','LearnRate'))
dm$value <- as.factor(dm$value)

ggplot(d, aes(x=reorder(d$Tag, Loss, median), y=log10(Loss))) +
  geom_boxplot(outlier.shape = NA, fill='gray50') +
  theme_bw() #+ theme(panel.grid.major.x = element_blank() , panel.grid.major.y = element_line( size=.1, color="gray" ) )
ggsave('../00_Figures/181023_FT_pca250_GS_boxplot.pdf', width=8, height = 4, device='pdf')
ggsave('../00_Figures/181023_FT_RL500_GS_boxplot.pdf', width=8, height = 4, device='pdf')

ggsave('../00_Figures/181023_HT_UN1000_GS_boxplot.pdf', width=8, height = 4, device='pdf')
ggsave('../00_Figures/181023_HT_RL1000_GS_boxplot.pdf', width=8, height = 4, device='pdf')
ggsave('../00_Figures/181023_HT_HE1000_GS_boxplot.pdf', width=8, height = 4, device='pdf')
ggsave('../00_Figures/181023_HT_pca250_GS_boxplot.pdf', width=8, height = 4, device='pdf')

ggsave('../00_Figures/181023_YLD_EN500_GS_boxplot.pdf', width=8, height = 4, device='pdf')

ggplot(dm, aes(x=reorder(dm$Tag, Loss, median), y=variable)) + geom_tile(aes(fill=value), colour='white') +
  theme_bw() + theme(panel.grid.major.x = element_blank() )
ggsave('../00_Figures/181023_FT_pca250_GS_labels.pdf', width=8, height = 3, device='pdf')
ggsave('../00_Figures/181023_FT_RL500_GS_labels.pdf', width=8, height = 3, device='pdf')

ggsave('../00_Figures/181023_HT_UN1000_GS_labels.pdf', width=8, height = 3, device='pdf')
ggsave('../00_Figures/181023_HT_RL1000_GS_labels.pdf', width=8, height = 3, device='pdf')
ggsave('../00_Figures/181023_HT_HE1000_GS_labels.pdf', width=8, height = 3, device='pdf')
ggsave('../00_Figures/181023_HT_pca250_GS_labels.pdf', width=8, height = 3, device='pdf')

ggsave('../00_Figures/181023_YLD_EN500_GS_labels.pdf', width=8, height = 3, device='pdf')

keep <- dagg[order(dagg$Loss),][1:10,]
dkeep <- subset(d, d$Tag %in% keep$Tag)
ggplot(dkeep, aes(x=reorder(dkeep$Tag, Loss, median), y=log10(Loss))) +
  geom_boxplot(fill='gray50') +
  theme_bw() + theme(axis.text.x = element_text(angle = 90, hjust = 1))
ggsave('../00_Figures/181023_FT_pca250_GS_top10.pdf', width=4, height = 4, device='pdf')
ggsave('../00_Figures/181023_FT_RL500_GS_top10.pdf', width=4, height = 4, device='pdf')

ggsave('../00_Figures/181023_HT_UN1000_GS_top10.pdf', width=4, height = 4, device='pdf')
ggsave('../00_Figures/181023_HT_RL1000_GS_top10.pdf', width=4, height = 4, device='pdf')
ggsave('../00_Figures/181023_HT_HE1000_GS_top10.pdf', width=4, height = 4, device='pdf')
ggsave('../00_Figures/181023_HT_pca250_GS_top10.pdf', width=4, height = 4, device='pdf')

ggsave('../00_Figures/181023_YLD_EN500_GS_top10.pdf', width=4, height = 4, device='pdf')



### After grid search
r <- read.csv('RESULTS.txt', sep='\t', header=T)
r$pcc_all <- 0
r$pcc_all[r$Trait=='FT'] <- 0.67
r$pcc_all[r$Trait=='HT'] <- 0.50
r$pcc_all[r$Trait=='YLD'] <- 0.37

r$pcc_subset <- 0
r$pcc_subset[r$Trait=='FT'] <- 0.66
r$pcc_subset[r$Trait=='HT'] <- 0.55
r$pcc_subset[r$Trait=='YLD'] <- 0.38

ggplot(r, aes(x=Tag, y=HO_PCC)) +
  geom_boxplot(outlier.shape = NA, fill='gray50') +
  facet_grid(.~Trait, scales='free') +
  geom_hline(data=r, aes(yintercept=pcc_all)) +
  geom_hline(data=r, aes(yintercept=pcc_subset, color='red')) +
  theme_bw() + theme(panel.grid.major.x = element_blank(),
                     panel.grid.major.y = element_line(size=.1, color="gray" ),
                     axis.text.x = element_text(angle = 90, hjust = 1))
ggsave('../00_Figures/181024_PCC_afterGS.pdf', width=4, height = 4, device='pdf')


