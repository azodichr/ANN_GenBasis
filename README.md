

# Experiment Description

Goal: Better understand the genetic basis of a complex trait (YLD) in soy using artificial neural networks

Approach: [1] Feature engineering/selection to reduct p:n ratio. [2] Parameter sweep (including initialization approach) [3] Model selection [4] deep Connection Weights to decompose network.




## Notes on running scripts on HPCC at MSU

For R scripts on HPCC run:
```
module load R
export R_LIBS_USER=/mnt/home/azodichr/R/library
Rscript -e "install.packages('XXX', lib='~/R/library', contriburl=contrib.url('http://cran.r-project.org/'))"
```

For python scripts on HPCC, may need to run:
```
export PATH=/mnt/home/azodichr/miniconda3/bin:$PATH
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


Rice: also did by hand (need to get key of ones that were dropped...) First I only dropped duplicates from geno.csv. Then realized I needed to first, round any estimated SNP values to the nearest value (-1 or 1) and remove the holdout validation set, and THEN remove duplicates! 
- Starting size: 327 x 73,148
- After dropping duplicates: 327 x 57,543
- After dropping duplicates after rounding and removing holdout: 327 x 37,284



## 2. Define holdout set

Soy: Hold out 10% = 501
Maize: Hold out 20% = 78 instances
Rice: Hold out 20% = 65
```python ~/GitHub/ML-Pipeline/holdout.py -df pheno.csv -sep ',' -type r -p 0.2```




## 3. Feature Engineering/Selection

Started using 50, 500, and 1000 features for each approach. After rrBLUP and BL modeling it became clear that there was a big jump in performance between 50 and 500, so I added 100. 18/9/25. 

### Dimension Reduction

#### PCA
```Rscript ~/GitHub/ANN_GenBasis/make_feat_PCA.R geno_use.csv 100,250```

#### MCA
Packages needed: FactoMineR & factoextra
```Rscript ~/GitHub/ANN_GenBasis/make_feat_MCA.R geno_use.csv 100,250```



### Clustering
Packages needed: mpmi, cluster (pam), vegan (jaccard/ji), cba (rock)

#### PAM
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_use.csv pam 100
```

#### Hierarchical clustering with Euclidean Distance
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_use.csv hclus 100
```

#### Hierarchical clustering with Jaccard Index
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_use.csv ji 100
```

#### Clustering with RObust Clustering using linKs (ROCK)
```
Rscript ~/GitHub/ANN_GenBasis/make_feat_Clust.R geno_use.csv rock 100
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


#### Random
working directory: 01_Random
```for i in $(seq 1 100); do python ~/GitHub/ML-Pipeline/Feature_Selection.py -f random -df ../geno_use.csv -df2 ../pheno.csv -y_name YLD -n 100,500,1000 -sep ',' -save fs_RAN_"$i"; done```
Run time < 10 seconds


#### Random Forest
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f RF -df geno_use.csv -df2 pheno.csv -sep ',' -type r -ho holdout.txt -scores t -y_name YLD -n 100,500,1000 -save fs_RF_YLD```
Run time < 30 seconds


#### Elastic Net
```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f EN -p 0.5 -df geno_use.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -scores t -y_name YLD -n 100,500,100 -save featsel_EN_YLD```
Run time < 30 seconds

* Note for soy: L1:L2 ratio = 0.5 there were 2995, 111, and 295 non-zero features for YLD, R8, and HT, respectively in soy. Ended up using YLD: 0.5 = 2995; R8: 0.08 = 1007; HT: 0.15 = 1049
* Rice: L1:L2 = 0.25 for HT (1299 >0), L1:L2 = 0.5 for YLD (19092 >0), L1:L2 = 0.15 for FT (1142 >0)


#### Relief
Based on the rebase approach. See [here](https://github.com/EpistasisLab/scikit-rebate) for more info.

```python ~/GitHub/ML-Pipeline/Feature_Selection.py -f relief -df geno_use.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -scores t -y_name YLD -n 100,500,1000 -save featsel_YLD_RL```


#### Bayes A
```module load R; export R_LIBS_USER=/mnt/home/azodichr/R/library; python ~/GitHub/ML-Pipeline/Feature_Selection.py -f ba -df geno_use.csv -df2 pheno.csv  -sep ',' -ho holdout.txt -scores t -y_name YLD -n 100,500,100 -save featsel_YLD_BA```
Run time ~ 10 minutes

#### Information Gain
Convert trait into categorical variable then calucate information gain using base R.
Formulas taken from: https://stackoverflow.com/questions/1859554/what-is-entropy-and-information-gain
```Rscript ~/GitHub/ANN_GenBasis/feat_sel_IG.R geno_use.csv pheno.csv YLD```
```python ~/GitHub/Utilities/qsub_slurm.py -f submit -u azodichr -c run_IG.txt -A quantgen -w 2000 -m 50 -J riIG -wd /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/```


```
python ~/GitHub/Utilities/qsub_slurm.py -f submit -u azodichr -A quantgen -c run_clust.sh -w 329 -m 60 -J ri_FS -wd /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/ 
```

#### Get the union and intersect of features from the 4 methods tested:
```
python ~/GitHub/ANN_GenBasis/union_intersect.py featsel_YLD featsel_YLD_BA_100 featsel_YLD_EN_100 featsel_YLD_RF_100 featsel_YLD_RL_100
mv featsel_YLD_I2 featsel_YLD_I2_100
mv featsel_YLD_UN featsel_YLD_UN_100
```


#### Generate Venn Diagrams of how many features overlapped
Example:
```python ~/GitHub/Utilities/plot_venn.py -files featsel_YLD_BA_50,featsel_YLD_EN_50,featsel_YLD_RF_50,featsel_YLD_RL_50 -ids BA,EN,RF,RL -save plot_FS_YLD_50```
```bash run_FS_venn.sh```

Redo with script that counts by cluster (i.e. considered overlapping if select SNP in same cluster)
bash run_FSclust_venn.sh





## 3. Modeling using different feature types/sets

### rrBLUP

Packages Needed:
-rrBLUP
-data.table
-AICcmodavg
-psych

```
touch run_rrb.sh
declare -a trait=("HT" "FT" "YLD")
declare -a nfeat=("100" "500" "1000")
```
##### All
```
for t in "${trait[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_use.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv all $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt all /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/; done
```
##### Random Subsets
```
cd 01_Random
declare -a rep=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
for nf in "${nfeat[@]}"; do for t in "${trait[@]}";  do for r in "${rep[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_Random_"$r"_"$nf" $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt Random /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ ; done; done; done
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
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv mediod_"$tf"_"$nf".csv $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ done; done; done

##### Feature Selection
```
declare -a nfeat=("50" "100" "500" "1000")
declare -a tfeat=("BA" "RF" "EN" "RL" "UN" "I2")

for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do module load R";" export R_LIBS_USER=~/R/library";"Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_rrBLUP.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_"$t"_"$tf"_"$nf" $t /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_rrb.sh; done; done; done
```





### Bayesian LASSO
Packages Needed:
-BGLR
-data.table
-AICcmodavg

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
cd 01_Random
declare -a rep=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
for nf in "${nfeat[@]}"; do for t in "${trait[@]}";  do for r in "${rep[@]}"; do Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv featsel_Random_"$r"_"$nf" $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt Random /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ ; done; done; done
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
for nf in "${nfeat[@]}"; do for tf in "${tfeat[@]}"; do for t in "${trait[@]}"; do echo module load R";" export R_LIBS_USER=~/R/library";" Rscript /mnt/home/azodichr/GitHub/GenomicPrediction_2018/scripts/Feature_Selection/predict_FS_BGLR.R /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv mediod_"$tf"_"$nf".csv $t BL /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt "$tf" /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/ >> run_BL.sh; done; done
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
```
declare -a rep=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
for t in "${trait[@]}"; do for nf in "${nfeat[@]}"; do for r in "${rep[@]}"; do echo export PATH=/mnt/home/azodichr/miniconda3/bin:\$PATH\;python /mnt/home/azodichr/GitHub/ML-Pipeline/ML_regression.py -df /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -df2 /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name $t -alg RF -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/01_Random/featsel_Random_"$r"_"$nf" -sep ',' -cv 5 -n 100 -p 3 -tag random -save /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/RF_"$t"_rand_"$nf"_"$r" >> run_RF.sh; done; done; done
```

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






## 4. Building ANNs using DNNRegressor API

### Lit Review Notes

#### Rice Info
* In contrast to maize, GWAS and QTL studies in rice have found large effect QTLs for agronomic traits, with yield being the most complex and FT and HT having more large effect SNPs. (Spindel 2015)
* Report of the r2 (percent var explained by haplotypes) showed that the highest r2 for FT = 0.34, HT = 0.12, and YLD = 0.067  (Sup Tab4-6 Begum, Spindel, et al 2015)


### Test with random from normal distribution script (i.e. mlp_DNNReg.py)
declare -a larch=("10" "100" "500" "10,5" "100,50" "500" "10,5,5" "100,50,50")
declare -a ll1=("0.0" "0.1" "0.5")
declare -a ll2=("0.0" "0.1" "0.5")
declare -a llr=("0.001" "0.01" "0.1")
declare -a reps=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")


#### Height
for arch in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for r in "${reps[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/03_GenomicSelection/06_otherTraits/mlp/mlp_DNNReg_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep ',' -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_HT_RL_1000 -save HT_RL1000 -tag HT_RL1000 -arch "$arch" -l1 "$l1" -l2 "$l2" -lrate "$lr" >> run_gsHT.txt; done; done; done; done; done

python ~/GitHub/Utilities/qsub_slurm.py -f submit -u azodichr -c run_gsHT.txt; -w 15 -m 100 -J HT_mlp -wd /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/04_MLP/

python ~/03_GenomicSelection/06_otherTraits/mlp/mlp_DNNReg.py -x ~/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y ~/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -ho ~/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -y_name HT -sep ',' -feat ~/05_Insight/01_FeatureEngineering/sp_rice/featsel_HT_EN_1000 -actfun sigmoid -arch 500 -l1 0.0 -l2 0.5 -lrate 0.1 -tag test -save test -gs F


#### Flowering Time
for arch in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for r in "${reps[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/03_GenomicSelection/06_otherTraits/mlp/mlp_DNNReg_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_pca_250.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep ',' -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -save HT_pca250 -tag FT_pca250 -arch "$arch" -l1 "$l1" -l2 "$l2" -lrate "$lr" >> run_gsFT.txt; done; done; done; done; done

python ~/GitHub/Utilities/qsub_slurm.py -f submit -u azodichr -c run_gsFT.txt; -w 15 -m 100 -J FT_mlp -wd /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/04_MLP/

python ~/03_GenomicSelection/06_otherTraits/mlp/mlp_DNNReg.py -x ~/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y ~/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -ho ~/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -y_name HT -sep ',' -feat ~/05_Insight/01_FeatureEngineering/sp_rice/featsel_HT_EN_1000 -actfun sigmoid -arch 500 -l1 0.0 -l2 0.5 -lrate 0.1 -tag test -save test -gs F


#### Yield
for arch in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for r in "${reps[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/03_GenomicSelection/06_otherTraits/mlp/mlp_DNNReg_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep ',' -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_YLD_EN_1000 -save YLD_EN1000 -tag YLD_EN1000 -arch "$arch" -l1 "$l1" -l2 "$l2" -lrate "$lr" >> run_gsYLD.txt; done; done; done; done; done

for i in run_gs*.txt; do python ~/GitHub/Utilities/qsub_slurm.py -f queue -n 900 -u azodichr -c $i -w 15 -m 100 -J ri_mlp -wd /mnt/home/azodichr/05_Insight/02_Modeling/sp_rice/04_MLP/; done

python ~/03_GenomicSelection/06_otherTraits/mlp/mlp_DNNReg.py -x ~/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y ~/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -ho ~/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -y_name HT -sep ',' -feat ~/05_Insight/01_FeatureEngineering/sp_rice/featsel_HT_EN_1000 -actfun sigmoid -arch 500 -l1 0.0 -l2 0.5 -lrate 0.1 -tag test -save test -gs F





## 4. Building ANNs using base TensorFlow
Modifications to pipeline include:
- Addition of 3rd activation function: elu
- Moving from hard epoch cutoff for training to a tolerance method:
	- Once epochs > 500 count number of times the % change is < threshold (default =  0.0001). Stop once count = 10

I want to focus on height and yield because one SNP on chromosome 3 explains 40% of variation in FT, while HT and YLD seem to be more complex, with YLD being the most complex.

declare -a lactfun=("sigmoid" "relu" "elu")
declare -a larch=("10" "50" "100" "500" "10,5" "50,25" "100,50" "10,5,5" "50,20,5")
declare -a ll1=("0.0" "0.1" "0.5")
declare -a ll2=("0.0" "0.1" "0.5")
declare -a llr=("0.001" "0.01" "0.1")

### Height
#### PCA
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_pca_250.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -save HT_pca250 -tag HT_pca250 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_HTpca250.txt; done; done; done; done; done

python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_b.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_pca_250.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -save HT_pca250 -tag HT_pca250 -params HT_pca250_GridSearch.txt


#### Heuc
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/mediod_hclus_1000.csv -save HT_HE1000 -tag HT_HE1000 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_HThe1000.txt; done; done; done; done; done

python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_b.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_pca_250.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -save HT_HE1000 -tag HT_HE1000 -params HT_HE1000_GridSearch.txt

#### Feature Selection
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_HT_RL_1000 -save HT_RL1000 -tag HT_RL1000 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_gs.txt; done; done; done; done; done
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name HT -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_HT_UN_1000 -save HT_UN1000 -tag HT_UN1000 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_gs.txt; done; done; done; done; done

### Yield
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_pca_250.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name YLD -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -save YLD_pca100 -tag YLD_pca100 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_gs.txt; done; done; done; done; done
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name YLD -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/mediod_hclus_500.csv -save YLD_HE500 -tag YLD_HE500 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_gs.txt; done; done; done; done; done
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name YLD -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_YLD_EN_500 -save YLD_EN500 -tag YLD_EN500 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_gs.txt; done; done; done; done; done
for ar in "${larch[@]}"; do for l1 in "${ll1[@]}"; do for l2 in "${ll2[@]}"; do for lr in "${llr[@]}"; do for af in "${lactfun[@]}"; do echo source /mnt/home/azodichr/python3-tfcpu/bin/activate\; python /mnt/home/azodichr/GitHub/TF-GenomicSelection/mlp_baseTF_a.py -x /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/geno_noDups.csv -y /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/pheno.csv -y_name YLD -sep , -ho /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/holdout.txt -feat /mnt/home/azodichr/05_Insight/01_FeatureEngineering/sp_rice/featsel_YLD_I2_1000 -save YLD_I21000 -tag YLD_I21000 -actfun $af -arch $ar -l1 $l1 -l2 $l2 -lrate $lr >> run_gs.txt; done; done; done; done; done
