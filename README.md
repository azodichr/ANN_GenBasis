# ANN_GenBasis




# Goals




# Experiment

Goal: Better understand the genetic basis of a complex trait in X using artificial neural networks

Approach: 

## Notes on running scripts on HPCC at MSU

To run on HPCC, need to load a more recent version of R so that freads will work:
```
module swap GNU GNU/4.9

module load OpenMPI/1.10.0

module load R/3.3.2
```

For some R scripts, additional packages are needed (all available on HPCC here:)
```export R_LIBS_USER=/mnt/home/azodichr/R/library```


## 1. Get data

Maize:

/mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/geno.csv .

/mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/pheno.csv .

Soy: 




## 1. Remove Duplicate SNPs

```
Rscript ~/GitHub/ANN_GenBasis/remove_dups.R geno.csv
```
Note: soy had 5 exact duplicates, maize had ??



## 2. Define holdout set

Soy: Hold out 10% = 

Maize: Hold out 20% = 78 instances

```python ~/GitHub/ML-Pipeline/holdout.py -df pheno.csv -sep ',' -type r -p 0.2```


## 3. Feature Engineering/Selection

### PCA
```Rscript ~/GitHub/ANN_GenBasis/make_feat_PCA.R geno_noDups.csv```

### MCA
Packages needed: FactoMineR & factoextra
```
Rscript -e "install.packages('FactoMineR', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"
Rscript -e "install.packages('factoextra', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"
```
```Rscript ~/GitHub/ANN_GenBasis/make_feat_MCA.R geno_noDups.csv```



### Clustering
Packages needed: mpmi, cluster (pam only), vegan (jaccard - ji only)
```
Rscript -e "install.packages('mpmi', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"
Rscript -e "install.packages('cluster', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))
Rscript -e "install.packages('vegan', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))
```

#### PAM
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_noDups.csv pam 50
```

#### Hierarchical clustering with Euclidean Distance
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_noDups.csv hclus 50
```

#### Hierarchical clustering with Jaccard Index
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_noDups.csv ji 50
```

* Example Submission:
```python ~shius/codes/qsub_hpc.py -f submit -u azodichr -c run_clust.sh -w 800 -m 20 -J clust_soy -wd /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/```



### Feature Selection
Utilizing Feature Selection tool built in the ML_Pipeline from the [Shiu Lab] (https://github.com/ShiuLab/ML-Pipeline)
Requirements: pandas, python3, scikit-learn
Running on HPCC at MSU:
```export PATH=/mnt/home/azodichr/miniconda/bin:$PATH```

#### Random Forest
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f RF -df geno_noDups.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -y_name YLD -n 50,500,1000```

#### Elastic Net
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f EN -p 0.5 -df geno_noDups.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -y_name YLD -n 50,500,1000```
* Note: using L1:L2 ratio = 0.5 there were 2995 non-zero features for soy

#### Relief
Based on the rebate approach. See [here] (https://github.com/EpistasisLab/scikit-rebate) for info.
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f relief -df geno_noDups.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -y_name YLD -n 50,500,1000```



