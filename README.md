

# Experiment Description

Goal: Better understand the genetic basis of a complex trait (YLD) in soy using artificial neural networks

Approach: [1] Feature engineering/selection to reduct p:n ratio. [2] Parameter sweep (including initialization approach) [3] Model selection [4] deep Connection Weights to decompose network.




## Notes on running scripts on HPCC at MSU

To run on HPCC, need to load a more recent version of R so that freads will work:
```
module swap GNU GNU/4.9

module load OpenMPI/1.10.0

module load R/3.3.2
```

For some R scripts, additional packages are needed (all available on HPCC here:)
```
export R_LIBS_USER=/mnt/home/azodichr/R/library
```


## 1. Get data

Maize:

/mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/geno.csv .

/mnt/research/ShiuLab/17_GP_SNP_Exp/maize/01_Data/pheno.csv .

Soy: 






## 1. Remove Duplicate SNPs

```
Rscript ~/GitHub/ANN_GenBasis/remove_dups.R geno.csv
```
Note: soy had 5 exact duplicates, rice had, and maize had ?? (script wouldn't finish!)

Did by hand in python for maize - need to go back and make a key for the ones that were dropped.
d2 = d.T.drop_duplicates().T
d2.to_csv('geno_noDups.csv', sep=',', index=False, header=True)
Went from 332178 to 242694 markers. 





## 2. Define holdout set

Soy: Hold out 10% = 501
Maize: Hold out 20% = 78 instances

Rice: Hold out 20% = 65
```python ~/GitHub/ML-Pipeline/holdout.py -df pheno.csv -sep ',' -type r -p 0.2```




## 3. Feature Engineering/Selection

Started using 50, 500, and 1000 features for each approach. After rrBLUP and BL modeling it became clear that there was a big jump in performance between 50 and 500, so I added 100. 18/9/25. 

### Dimension Reduction

#### PCA
```Rscript ~/GitHub/ANN_GenBasis/make_feat_PCA.R geno_noDups.csv```

#### MCA
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

Example Submission:

```
declare -a nclus=("50" "100" "500" "1000")

declare -a tclus=("pam" "hclus" "ji")

for nc in "${nclus[@]}"; do for tc in "${tclus[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/ANN_GenBasis/make_feat_Clust.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv $tc $nc >> run_clust.sh; done; done
python ~shius/codes/qsub_hpc.py -f submit -u azodichr -c run_clust.sh -w 2000 -m 60 -J clust_rice -wd /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/ 
```

Recount cluster size for hier and JI:

```
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_ji_50.csv > count_ji_50
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_ji_100.csv > count_ji_100
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_ji_500.csv > count_ji_500
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_ji_1000.csv > count_ji_1000
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_hclus_50.csv > count_hc_50
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_hclus_100.csv > count_hc_100
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_hclus_500.csv > count_hc_500
awk -F 'Gm' '{print NF-1, NR}'  clustAssign_hclus_1000.csv > count_hc_1000
```

### Feature Selection
Utilizing Feature Selection tool built in the ML_Pipeline from the [Shiu Lab](https://github.com/ShiuLab/ML-Pipeline)

Requirements: pandas, python3, scikit-learn

Running on HPCC at MSU:
```export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH```

#### Random Forest
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f RF -df geno_noDups.csv -df2 pheno.csv -sep ',' -type r -ho holdout.txt -scores t -y_name YLD -n 50,100,388, 500,1000,4000 -save featsel_YLD_RF```


#### Elastic Net
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f EN -p 0.5 -df geno_noDups.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -scores t -y_name YLD -n 50,100,388,500,1000,4000 -save featsel_YLD_EN```

* Note for soy: L1:L2 ratio = 0.5 there were 2995, 111, and 295 non-zero features for YLD, R8, and HT, respectively in soy. Ended up using YLD: 0.5 = 2995; R8: 0.08 = 1007; HT: 0.15 = 1049


#### Relief
Based on the rebase approach. See [here](https://github.com/EpistasisLab/scikit-rebate) for more info.

```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f relief -df geno_noDups.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -scores t -y_name YLD -n 50,100,500,1000 -save featsel_YLD_RL```


#### Bayes A
declare -a trait=("FT" "HT" "YLD")
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f ba -df geno_noDups.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -scores t -y_name YLD -n 50,100,388, 500,1000,4000 -save featsel_YLD_BA```


#### Get the union and intersect of features from the 4 methods tested:
```
python ~/GitHub/ANN_GenBasis/union_intersect.py featsel_YLD featsel_YLD_BA_100 featsel_YLD_EN_100 featsel_YLD_RF_100 featsel_YLD_RL_100
mv featsel_YLD_I2 featsel_YLD_I2_100
mv featsel_YLD_UN featsel_YLD_UN_100
```


##### Generate Venn Diagrams of how many features overlapped
Example:
```python ~/GitHub/Utilities/plot_venn.py -files featsel_YLD_BA_50,featsel_YLD_EN_50,featsel_YLD_RF_50,featsel_YLD_RL_50 -ids BA,EN,RF,RL -save plot_FS_YLD_50```
```bash run_FS_venn.sh```

# Redo with script that counts by cluster (i.e. considered overlapping if select SNP in same cluster)
bash run_FSclust_venn.sh





## 3. Modeling using different feature types/sets

## Soybean!!!
### rrBLUP
```
touch run_rrb.sh
declare -a trait=("HT" "R8" "YLD")
for i in mediod_*0.csv; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv $i $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt $i /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_rrb.sh; done; done

declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("RF" "BA" "EN" "RL")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv featsel_"$t"_"$tf"_"$nf" $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$tf"_$nf /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_rrb.sh; done; done; done

declare -a tfeat=("pca" "mca")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_"$tf"_"$nf".csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv all $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$tf"_"$nf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_rrb.sh; done; done; done

declare -a trait=("HT" "R8" "YLD")
declare -a ion=("union" "intersection")
declare -a nfeat=("50" "100" "500" "1000")
for nf in "${nfeat[@]}"; do for io in "${ion[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv featsel_"$t"_"$nf"_"$io" $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$io" /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_rrb.sh; done; done; done
```



### Bayesian LASSO

```
touch run_BL.sh
declare -a trait=("HT" "R8" "YLD")
for i in mediod_*0.csv; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv $i $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$i" /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_BL.sh; done; done

declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("RF" "BA" "EN" "RL")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv featsel_"$t"_"$tf"_"$nf" $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$tf"_$nf /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_BL.sh; done; done; done

declare -a tfeat=("pca" "mca")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_"$tf"_"$nf".csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv all $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$tf"_"$nf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_BL.sh; done; done; done

declare -a trait=("HT" "R8" "YLD")
declare -a ion=("union" "intersection")
declare -a nfeat=("50" "100" "500" "1000")
for nf in "${nfeat[@]}"; do for io in "${ion[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv featsel_"$t"_"$nf"_"$io" $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt "$io" /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/ >> run_BL.sh; done; done; done
```


### Random Forest
```
touch run_RF.sh
declare -a trait=("HT" "R8" "YLD")
for i in mediod_*0.csv; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv -y_name $t -feat $i -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$i" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/RF_"$i" >> run_RF.sh; done; done

declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("RF" "BA" "EN" "RL")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv -y_name $t -feat featsel_"$t"_"$tf"_"$nf" -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf"_"$nf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/RF_"$i" >> run_RF.sh; done; done; done

declare -a tfeat=("pca" "mca")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_"$tf"_"$nf".csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv -y_name $t -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/RF_"$tf"_"$nf" >> run_RF.sh; done; done; done

declare -a trait=("HT" "R8" "YLD")
declare -a ion=("union" "intersection")
declare -a nfeat=("50" "100" "500" "1000")
for nf in "${nfeat[@]}"; do for io in "${ion[@]}"; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv -y_name $t -feat featsel_"$t"_"$nf"_"$io" -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf"_"$nf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/RF_"$io"_"$nf" >> run_RF2.sh; done; done; done


python ~/GitHub/Utilities/qsub_hpc.py -f submit -u azodichr -c run_RF.sh -w 239 -m 80 -p 3 -A quantgen -J ml_soy -wd /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy
python ~/GitHub/Utilities/qsub_hpc.py -f submit -u azodichr -c run_RF_missed.sh -w 639 -m 90 -p 7 -A quantgen -J ml_soyPCA -wd /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy
```

##### Run all features (i.e. no feature selection or engineering #####
```
declare -a trait=("HT" "R8" "YLD")
for t in "${trait[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv all $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt all /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/; done

for t in "${trait[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv all $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt all /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/; done

for t in "${trait[@]}"; do python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/pheno.csv -y_name $t -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_soy/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag all -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_soy/RF_"$t"_all; done
```



## Rice!!!
### rrBLUP
```
touch run_rrb.sh
declare -a trait=("HT" "FT" "YLD")
declare -a nfeat=("50" "100" "500" "1000")
```
##### All
```
for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv all $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt all /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_rrb.sh; done
```
##### Random Subsets
```
for nf in "${nfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_Random_"$nf" $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt Random /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_rrb.sh; done; done
```

##### Factor Analysis
```
declare -a tfeat=("pca" "mca")
declare -a nfeat=("50" "100" "250")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_"$tf"_"$nf".csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv all $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_rrb.sh; done; done; done
```

##### Cluster Analysis
```
declare -a tfeat=("hclus" "ji" "pam")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv mediod_"$tf"_"$nf".csv $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_rrb.sh; done; done
```

##### Feature Selection
```
declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("BA" "RF" "EN" "RL" "UN" "I2")

for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do module load R";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_"$t"_"$tf"_"$nf" $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_rrb.sh; done; done; done
```





### Bayesian LASSO
```
touch run_BL.sh
declare -a trait=("HT" "FT" "YLD")
```

##### All
```
for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv all $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt all /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_BL.sh; done
```

##### Random Subsets
```
for nf in "${nfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_Random_"$nf" $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt Random /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_BL.sh; done; done
```

##### Factor Analysis
```
declare -a tfeat=("pca" "mca")
declare -a nfeat=("50" "100" "250")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_"$tf"_"$nf".csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv all $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_BL.sh; done; done; done
```

##### Cluster Analysis
```
declare -a tfeat=("hclus" "ji" "pam")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module swap GNU GNU/4.9";" module load OpenMPI/1.10.0";" module load R/3.3.2";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv mediod_"$tf"_"$nf".csv $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_BL.sh; done; done
```

##### Feature Selection
```
declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("RF" "BA" "EN")
 "RL" "UN" "IN" "I2")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module load R";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_"$t"_"$tf"_"$nf" $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_BL.sh; done; done; done


for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_"$t"_"$tf"_"$nf" $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/; done; done; done
```



### Random Forest
```
touch run_RF.sh
```
##### All
```for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag all -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_all >> run_RF2.sh; done```

##### Random
```for t in "${trait[@]}"; do for nf in "${nfeat[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_Random_"$nf" -sep ',' -cv 5 -n 100 -p 3 -tag random -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_rand_"$nf" >> run_RF.sh; done; done```

##### Factor Analysis
```
declare -a tfeat=("pca" "mca")
declare -a nfeat=("50" "100" "250")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_"$tf"_"$nf".csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_"$tf"_"$nf" >> run_RF2.sh; done; done; done
```



##### Clustering 
```
declare -a tfeat=("hclus" "ji" "pam")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -feat mediod_"$tf"_"$nf".csv -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_"$tf"_"$nf" >> run_RF3.sh; done; done; done
```

##### Feature Selection
```
declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("RF" "BA" "EN")
 "RL" "UN" "IN" "I2")
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -feat featsel_"$t"_"$tf"_"$nf" -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf"_"$nf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_"$tf"_"$nf" >> run_RF3.sh; done; done; done

for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -feat featsel_"$t"_"$tf"_"$nf" -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -sep ',' -cv 5 -n 100 -p 3 -tag "$tf"_"$nf" -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_"$tf"_"$nf"; done; done; done
```
