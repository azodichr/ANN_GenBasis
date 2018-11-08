setwd('/Volumes/azodichr/05_Insight/02_Modeling/sp_rice/05_MLPbaseTF/')
library(ggplot2)
file <- 'test_losses.csv'
file <- 'HT_RL1000_losses.csv'
file <- 'HT_UN1000_losses.csv'

l <- read.csv(file)
lm <- melt(l, id='epoch')
ggplot(lm, aes(x=epoch, y=log10(value), color=variable)) +
  geom_line(alpha=0.6) +
  labs(y='log10(MSE)', title=file) + theme_bw()


r <- read.csv('RESULTS.txt', sep='\t')
r$FSmeth <- as.character(lapply(strsplit(as.character(r$FeatSel), split="/"),tail, n=1))
r$pcc_subset <- 0
r$pcc_subset[r$Trait=='FT'] <- 0.66
r$pcc_subset[r$Trait=='HT'] <- 0.55
r$pcc_subset[r$Trait=='YLD'] <- 0.38
r$pcc_all <- 0
r$pcc_all[r$Trait=='FT'] <- 0.67
r$pcc_all[r$Trait=='HT'] <- 0.50
r$pcc_all[r$Trait=='YLD'] <- 0.37

rs <- subset(r, r$lrate = 0.01)
ggplot(r, aes(x=FSmeth, y=PCC_ho)) +
  geom_boxplot() + facet_grid(.~Trait, scales='free') +
  theme_bw(10) + 
  geom_hline(data=r, aes(yintercept=pcc_all, color='red')) +
  geom_hline(data=r, aes(yintercept=pcc_subset, color='blue')) 
  
temp <- read.csv('/Volumes/ShiuLab/17_GP_SNP_Exp/maize/11_Geno2/geno.csv', sep=',', header=T)


ggplot(r, aes(x=Tag, y=HO_PCC)) +
  geom_boxplot(outlier.shape = NA, fill='gray50') +
  facet_grid(.~Trait, scales='free') +
  geom_hline(data=r, aes(yintercept=pcc_all)) +
  geom_hline(data=r, aes(yintercept=pcc_subset, color='red')) +
  theme_bw() + theme(panel.grid.major.x = element_blank(),
                     panel.grid.major.y = element_line(size=.1, color="gray" ),
                     axis.text.x = element_text(angle = 90, hjust = 1))
ggsave('../00_Figures/181024_PCC_afterGS.pdf', width=4, height = 4, device='pdf')


losses <- read.csv('HT_pca250_Long_losses.csv') 
lm <- melt(losses, id='epoch')
lm <- subset(lm, epoch <= 100000)
ggplot(lm, aes(x=epoch, y=log2(value), color=variable)) +
  geom_line(alpha=0.6) + 
  labs(y='log10(MSE)') + theme_bw()


htpca <- read.csv('RESULTS.txt', sep='\t')
plot(PCC_ho ~ Epochs, htpca)



