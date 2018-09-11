# ANN_GenBasis




# Goals




# Experiment


## 1. Get data

Maize:
ln -s /mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/geno.csv .

ln -s /mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/pheno.csv .

Soy: 




## 1. Remove Duplicate SNPs

'module swap GNU GNU/4.9

module load OpenMPI/1.10.0

module load R/3.3.2

Rscript ~/GitHub/ANN_GenBasis/remove_dups.R geno.csv'

Note: soy had 5 exact duplicates, maize had ??



## 2. Define holdout set

Soy: Hold out 10% = 

Maize: Hold out 20% = 78 instances

<code>python ~/GitHub/ML-Pipeline/holdout.py -df pheno.csv -sep ',' -type r -p 0.2<code/>