# ANN_GenBasis




# Goals




# Experiment

Goal: Better understand the genetic basis of a complex trait in X using artificial neural networks

Approach: 

## Notes on running scripts on HPCC at MSU

* To run on HPCC, need to load a more recent version of R so that freads will work:
<code>module swap GNU GNU/4.9

module load OpenMPI/1.10.0

module load R/3.3.2<code/>
* For some R scripts, additional packages are needed.
<code> export R_LIBS_USER=/mnt/home/azodichr/R/library <code/>


## 1. Get data

Maize:

/mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/geno.csv .

/mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/pheno.csv .

Soy: 




## 1. Remove Duplicate SNPs

<code>module swap GNU GNU/4.9

module load OpenMPI/1.10.0

module load R/3.3.2

Rscript ~/GitHub/ANN_GenBasis/remove_dups.R geno.csv<code/>
Note: soy had 5 exact duplicates, maize had ??



## 2. Define holdout set

Soy: Hold out 10% = 

Maize: Hold out 20% = 78 instances

<code>python ~/GitHub/ML-Pipeline/holdout.py -df pheno.csv -sep ',' -type r -p 0.2<code/>

## 3. Feature Engineering/Selection

### PCA
Rscript ~/GitHub/ANN_GenBasis/make_feat_PCA.R geno_noDups.csv

### MCA
Packages needed: FactoMineR & factoextra
<code>Rscript -e "install.packages('FactoMineR', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"
Rscript -e "install.packages('factoextra', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"<code/>
<code>Rscript ~/GitHub/ANN_GenBasis/make_feat_MCA.R geno_noDups.csv<code/>

### Clustering
Packages needed: mpmi
<code>Rscript -e "install.packages('mpmi', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"<code/>

#### PAM
Packages needed: cluster
<code>Rscript -e "install.packages('cluster', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))<code/>

<code>Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_noDups.csv pam 50
script ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_noDups.csv pam 500
script ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_noDups.csv pam 1000<code/>