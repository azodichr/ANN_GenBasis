"""
PURPOSE:
Fully connected MLP Regressor implemented in 
TensorFlow using tf.contrib.learn.DNNRegressor

INPUTS:

    OPTIONAL:
    -arch   MLP architecture: 10,5 = 2 hidden layers with 10 & 5 nodes
            Default = 10
    -norm   T/F if features should be normalized

 

Enter TensorFlow environment on MSU's HPCC:
ssh dev-intel16-k80
module purge
module load singularity
module load centos/7.4
/opt/software/CentOS/7.4--singularity/bin/centos7.4
module load CUDA/8.0
module load cuDNN/6.0
source ~/python3-tf/bin/activate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import timeit
start_time = timeit.default_timer()
from random import randint


### Set default values ####
norm = 'f'
y_name = 'pass'
FEAT = 'pass'
SEP = '\t'
val_perc = 0.2
feature_cols = 'temp'
SAVE = ''
rand_walk_level = 1 # i.e. all combinations

# Practice run Grid Search space
'''
search_actfun = {'sigmoid':tf.nn.sigmoid}
search_arch = ['10','100,10']
search_l1 = [0.5]
search_l2 = [0.0]
search_learnrate = [0.01]
search_epochs = [500]

'''
# Grid Search space
search_actfun = {'sigmoid':tf.nn.sigmoid, 'relu':tf.nn.relu}
search_arch = ['10','100','1000','10,5','100,50','1000,500','10,5,5','100,50,50','1000,500,50']
search_l1 = [0.0, 0.1, 0.5] 
search_l2 = [0.0, 0.1, 0.5]
search_learnrate = [0.01]
search_epochs = [500]


for i in range (1,len(sys.argv),2):
    if sys.argv[i].lower() == "-norm":
        norm = sys.argv[i+1].lower()
    if sys.argv[i].lower() == "-rand":
        rand_walk_level = float(sys.argv[i+1])
    if sys.argv[i].lower() == "-x":
        X_file = sys.argv[i+1]
    if sys.argv[i].lower() == "-y":
        Y_file = sys.argv[i+1]
    if sys.argv[i].lower() == "-y_name":
        y_name = sys.argv[i+1]
    if sys.argv[i].lower() == "-sep":
        SEP = sys.argv[i+1]
    if sys.argv[i].lower() == "-feat":
        FEAT = sys.argv[i+1]
    if sys.argv[i].lower() == "-val_perc":
        val_perc = float(sys.argv[i+1])
    if sys.argv[i].lower() == '-ho':
        ho = sys.argv[i+1]
    if sys.argv[i].lower() == '-save':
        SAVE = sys.argv[i+1]
    if sys.argv[i].lower() == '-tag':
        TAG = sys.argv[i+1]

################
### Format Save Name
################
if SAVE == "":
	temp = X_file.strip('.csv')
	temp = temp.strip('.txt')
	SAVE = temp + "_" + y_name + "_MLP"


################
### Features: read in file, keep only those in FEAT if given, and define feature_cols for DNNReg.
################
x = pd.read_csv(X_file, sep=SEP, index_col = 0)
if FEAT != 'pass':
    with open(FEAT) as f:
        features = f.read().strip().splitlines()
    x = x.loc[:,features]
feat_list = list(x.columns)
feature_cols = [tf.contrib.layers.real_valued_column(k) for k in feat_list]

print("\n\nTotal number of instances: %s" % (str(x.shape[0])))
print("\nNumber of features used: %s" % (str(x.shape[1])))

################
### Y: read in file, keep only column to predict, normalize if needed, and merge with features
################
y = pd.read_csv(Y_file, sep=SEP, index_col = 0)
if y_name != 'pass':
    print('Building model to predict: %s' % str(y_name))
    y = y[[y_name]]
if norm == 't':
    mean = y.mean(axis=0)
    std = y.std(axis=0)
    y = (y - mean) / std
df = pd.merge(y, x, left_index=True, right_index=True)
yhat = df[y_name]

print('\nSnapshot of data being used:')
print(df.head())

################
### Holdout: Read in holdout set and remove from data for grid search
################
print('Removing holdout instances to apply model on later...')
df_all = df.copy()
with open(ho) as ho_file:
    ho_instances = ho_file.read().splitlines()
    num_ho = len(ho_instances)
try:
    ho_df = df.loc[ho_instances, :]
    df = df.drop(ho_instances)
except:
    ho_instances = [int(x) for x in ho_instances]
    ho_df = df.loc[ho_instances, :]
    df = df.drop(ho_instances)


################
### Split train/valid: Split non-holdout data into training and validatin for grid search
################
val_set_index = np.random.rand(len(df)) < val_perc
train = df[~val_set_index]
valid = df[val_set_index]

train.reset_index(drop = True, inplace =True)
valid.reset_index(drop = True, inplace =True)

# Input generator: Defines feature columns and the y_name label
def input_fn(data_set, pred=False):
    if pred == False:
        feature_cols = {k: tf.constant(data_set[k].values) for k in feat_list}
        labels = tf.constant(data_set[y_name].values)
        return feature_cols, labels
    if pred == True:
        feature_cols = {k: tf.constant(data_set[k].values) for k in feat_list}
        return feature_cols

################
### Grid Search: Using train:validate splits 
################
print('\n\n\n###################\nStarting grid search...\n###################\n')
gs_results = pd.DataFrame()
count = 0
for actfun_name in search_actfun:
    actfun = search_actfun[actfun_name]
    print(actfun_name)
    for arch in search_arch:
        hidden_units = arch.strip().split(',')
        hidden_units = list(map(int, hidden_units))
        print(arch)
        for l1 in search_l1:
            for l2 in search_l2:
                for lrate in search_learnrate:
                    for epoch in search_epochs:
                        rand = random.uniform(0, 1)
                        if rand <= rand_walk_level:
                            rand_save = randint(1000000, 9999999)
                            count += 1
                            regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, 
                                        activation_fn = actfun, hidden_units=hidden_units,
                                        optimizer = tf.train.ProximalAdagradOptimizer(
                                            learning_rate= lrate,
                                            l1_regularization_strength=l1,
                                            l2_regularization_strength=l2),
                                        model_dir="/mnt/scratch/azodichr/tmp/"+str(rand_save))
                            regressor.train(input_fn=lambda: input_fn(train), steps=epoch)
                            sweep_loss = regressor.evaluate(input_fn=lambda: input_fn(valid), steps=10)['loss']
                            gs_results = gs_results.append({'ActFun':actfun_name, 'Arch':arch, 'L1':l1, 'L2':l2, 'LearnRate':lrate,'Epochs':epoch, 'Loss':sweep_loss}, ignore_index=True)

# Assess grid search results to find best set of parameters...
gs_results_ave = gs_results.groupby(['ActFun','L1','L2','Arch','LearnRate','Epochs']).agg({'Loss': 'mean'}).reset_index()
gs_results_ave.columns = ['ActFun','L1','L2','Arch','LearnRate','Epochs', 'Loss_mean']
results_sorted = gs_results_ave.sort_values(by='Loss_mean', ascending=True)
results_sorted.to_csv(SAVE + "_GridSearch.txt")
print('\nSnapshot of grid search results:')
print(results_sorted.head())

################
### Run final model: Training on df (both train and valid from GS) and applying to holdout
################

low_GS_loss = float(results_sorted['Loss_mean'].iloc[0])
actfun_name_use = results_sorted['ActFun'].iloc[0]
actfun_use = search_actfun[actfun_name_use]
l1_use = float(results_sorted['L1'].iloc[0])
l2_use = float(results_sorted['L2'].iloc[0])
lrate_use = float(results_sorted['LearnRate'].iloc[0])
epochs_use = int(results_sorted['Epochs'].iloc[0])
arch_use = results_sorted['Arch'].iloc[0].strip().split(',')
layer_number = len(arch_use)
hidden_units_use = list(map(int, arch_use))


print('\n\n###################\nParameters To Use\n###################\n')
print('Activation Fun: %s\nArchitecture: %s\nLearning Rate: %s\nL1 Weight: %s\nL2 Weight: %s\nNum Epochs: %i\n'% (
    actfun_name_use, hidden_units_use, lrate_use, l1_use, l2_use, epochs_use))
print('\nNumber of parameter combinations tested: %i' % count)

rand_save = randint(1000000, 9999999)
final_regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols, 
                        activation_fn = actfun_use, hidden_units=hidden_units_use,
                        optimizer = tf.train.ProximalAdagradOptimizer(
                            learning_rate= lrate_use,
                            l1_regularization_strength=l1_use,
                            l2_regularization_strength=l2_use),
                        model_dir="/mnt/scratch/azodichr/tmp/"+str(rand_save))

final_regressor.train(input_fn=lambda: input_fn(df), steps=2000)

# Evaluate fitted model on holdout set
loss = final_regressor.evaluate(input_fn=lambda: input_fn(ho_df), steps=1)['loss']
y = final_regressor.predict(input_fn=lambda: input_fn(ho_df))
predictions = list(itertools.islice(y, ho_df.shape[0]))
preds = []
for i in predictions:
    preds.append(i['predictions'][0])
cor = np.corrcoef(ho_df[y_name], preds)


# Get predictions for all lines using the model (wanted for yhat output)
y_all = final_regressor.predict(input_fn=lambda: input_fn(df_all))
predictions_all = list(itertools.islice(y_all, df_all.shape[0]))
preds_all = []
for i in predictions_all:
    preds_all.append(i['predictions'][0])


print('\n\n###################\nResults on Holdout Set\n###################\n')
print('Testing loss: %.5f' % loss)
print('Accuracy: %.5f' % cor[0,1])


################
### Save results and predicted scores
################

# Merge predicted scores into dataframe to save
df_all[SAVE] = preds_all
yhat_all = pd.merge(pd.DataFrame(yhat), pd.DataFrame(df_all[SAVE]), how='left', left_index=True, right_index=True)
yhat_all = yhat_all.transpose()
if not os.path.isfile(TAG + "_yhat.txt"):
    yhat_all.to_csv(TAG + "_yhat.txt", header='column_names', sep='\t')
else: 
    yhat_all = yhat_all.drop(yhat_all.index[[0]])
    yhat_all.to_csv(TAG + "_yhat.txt", mode='a', header=False, sep='\t')



if not os.path.isfile('RESULTS.txt'):
    out2 = open('RESULTS.txt', 'a')
    out2.write('RunTime\tDFs\tDFy\tTrait\tFeatFile\tNumFeat\tHoldoutSet\tNumHidLay\tArchit\tActFun\tEpochs\tL1\tL2\tLearnRate\tGS_MinLoss\tHO_Loss\tHO_PCC\n')
run_time = timeit.default_timer() - start_time
out2 = open('RESULTS.txt', 'a')
out2.write('%0.5f\t%s\t%s\t%s\t%s\t%i\t%s\t%i\t%s\t%s\t%i\t%.2f\t%.2f\t%.2f\t%0.5f\t%0.5f\t%0.5f\n' % (
    run_time, X_file, Y_file, y_name, FEAT, x.shape[1], ho, layer_number, arch_use, actfun_name_use, epochs_use, l1_use, l2_use, lrate_use, low_GS_loss, loss, cor[0,1]))

print('finished!')
