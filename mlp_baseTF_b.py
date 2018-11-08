"""
PURPOSE:
Fully connected MLP Neural Network Regression implemented in TensorFlow (TF)

INPUTS:
    REQUIRED:
    -x      File with genotype information
    -y      File with values you want to predict 
    -y_name Name of column in y with the value you want to predict (Default = Y)
    -ho     File with holdout set
    -save   Name to include in RESULTS file (i.e. what dataset are you running)

    

    OPTIONAL:
    -arc    Desired NN architecture as comma separated layer sizes (i.e. 100,50 or 200,200,50)
    -actfun What activation function to use (sigmoid (default), relu, elu)
    -epochs Number of epochs to train on (default = 1000)
    -lrate     Learning rate (default = 0.01)
    -l1     Parameter for L1 (i.e. drop out) regularization (default = 0, i.e. no dropout)
    -l2     Parameter for L2 (i.e. shrinkage) regularization (default = 0, i.e. no shrinkage)

OUTPUTS:
    -RESULTS  Summary of results from the run located in the dir where the script was called.
                    Results will be appended to this file as they complete. Use -save to give
                    a run a unique identifier.     

EXAMPLE ON HPCC:
Log on to development node with GPUs:
$ ssh dev-intel16-k80   
Load linuxbrew, modules required by TF, & activate the TF python environment
$ source /opt/software/tensorflow/1.1.0/load_tf
Run example MLP (files in /mnt/home/azodichr/GitHub/TF-GenomicSelection/):
$ python TF_MLP_GridSearch.py -x geno.csv -y pheno.csv -label Yld_Env1 -cv CVFs.csv -save wheat -arc 100,50,20

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import timeit
from sklearn.model_selection import KFold

#CUDA_VISIBLE_DEVICES=""
start_time = timeit.default_timer()
timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# FUNCTIONS
def multilayer_perceptron(x, weights, biases, layer_number, activation_function, l1, keep_prob):
    layer = x
    for l in range(1,layer_number+1):
        weight_name = 'h' + str(l)
        bias_name = 'b' + str(l)
        layer = tf.add(tf.matmul(layer, weights[weight_name]), biases[bias_name])
        if activation_function.lower() == 'sigmoid':
            layer = tf.nn.sigmoid(layer)
        elif activation_function.lower() == 'relu':
            layer = tf.nn.relu(layer)
        elif activation_function.lower() == 'elu':
            layer = tf.nn.elu(layer)
        else:
            print("Given activation function is not supported")
            quit()
        if l1 != 0:
            keep_prob = 1.0 - l1
            drop_out = tf.nn.dropout(layer, keep_prob)
            print("Applying dropout regularization")
    out_layer = tf.matmul(layer, weights['out']) + biases['out']

    return out_layer



#### Set default values #####
actfun = 'sigmoid'
train_epochs = 1000
arc = '10,5'
lrate = 0.01
l1 = l2 = 0.0
SAVE = 'test'
TAG = FEAT = norm = params = ''
y_name = 'Y'
max_epochs = 50000
epoch_thresh = 0.0001

for i in range (1,len(sys.argv),2):
  if sys.argv[i].lower() == "-x":
    X_file = sys.argv[i+1]
  if sys.argv[i].lower() == "-y":
    Y_file = sys.argv[i+1]
  if sys.argv[i].lower() == '-ho':
    ho = sys.argv[i+1]
  if sys.argv[i].lower() == '-sep':
    SEP = sys.argv[i+1]
  if sys.argv[i].lower() == "-feat":
    FEAT = sys.argv[i+1]
  if sys.argv[i].lower() == "-tag":
    TAG = sys.argv[i+1]
  if sys.argv[i].lower() == "-y_name":
    y_name = sys.argv[i+1]
  if sys.argv[i].lower() == "-save":
    SAVE = sys.argv[i+1]
  if sys.argv[i].lower() == "-actfun":
    actfun = sys.argv[i+1] 
  if sys.argv[i].lower() == "-epochs":
    train_epochs = int(sys.argv[i+1])
  if sys.argv[i].lower() == "-epoch_thresh":
    epoch_thresh = float(sys.argv[i+1])
  if sys.argv[i].lower() == "-epoch_max":
    epoch_thresh = int(sys.argv[i+1])
  if sys.argv[i].lower() == "-lrate":
    lrate = float(sys.argv[i+1])
  if sys.argv[i].lower() == "-l2":
    l2 = float(sys.argv[i+1])
  if sys.argv[i].lower() == "-l1":
    l1 = float(sys.argv[i+1])
  if sys.argv[i].lower() == "-arch":     # Desired layer sizes comma separated (i.e. 100,50,20)
    arc = sys.argv[i+1]
  if sys.argv[i].lower() == "-params":     # Desired layer sizes comma separated (i.e. 100,50,20)
    params = sys.argv[i+1]


################
### Features: read in file, keep only those in FEAT if given, and define feature_cols for DNNReg.
################
x = pd.read_csv(X_file, sep=SEP, index_col = 0)
if FEAT != '':
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
### Holdout: Set test as holdout set and training as the remaining
################
print('Removing holdout instances to apply model on later...')

with open(ho) as ho_file:
    ho_instances = ho_file.read().splitlines()
    num_ho = len(ho_instances)
try:
    test = df.loc[ho_instances, :]
    train = df.drop(ho_instances)
except:
    ho_instances = [int(x) for x in ho_instances]
    test = df.loc[ho_instances, :]
    train = df.drop(ho_instances)

X_train = train.drop(y_name, axis=1).values
X_test = test.drop(y_name, axis=1).values
Y_train = train.loc[:, y_name].values
Y_test = test.loc[:, y_name].values


n_input = X_train.shape[1]
n_samples = X_train.shape[0]
n_classes = 1


################
### Pull parameters to use from grid search results file
################

if params != '':
	gs_res = pd.read_csv(params, sep='\t')
	gs_ave = gs_res.groupby(['ActFun','L1','L2','Arch','LearnRate']).agg({
		'Valid_Loss': 'median', 'Train_Loss': 'median'}).reset_index()
	gs_ave.columns = ['ActFun','L1','L2','Arch','LearnRate', 'VLoss_median', 'TLoss_median']
	gs_ave['Loss_ratio'] = gs_ave['VLoss_median']/gs_ave['TLoss_median']
	results_sorted = gs_ave.sort_values(by='Loss_ratio', ascending=True)
	print('\nSnapshot of grid search results:')
	print(results_sorted.head())

	actfun = results_sorted['ActFun'].iloc[0]
	l1 = float(results_sorted['L1'].iloc[0])
	l2 = float(results_sorted['L2'].iloc[0])
	lrate = float(results_sorted['LearnRate'].iloc[0])
	arc = results_sorted['Arch'].iloc[0]


# Architecture
hidden_units = arc.strip().split(',')
archit = list(map(int, hidden_units))
layer_number = len(archit)

print("\n\n##########\nBuilding MLP with the following parameters:\n")
print('Architecture: %s' % archit)
print('Regularization: L1 = %f  L2 = %f' % (l1, l2))
print('Learning rate: %f' % lrate)

# Set up results holder
yhat = np.zeros(shape = Y_test.shape)

## Build ANN ##
# TF Graph Placeholders 
nn_x = tf.placeholder(tf.float32, [None, n_input])
nn_y = tf.placeholder(tf.float32, [None, 1])
keep_prob = tf.placeholder(tf.float32) # For dropout, allows it to be turned on during training and off during testing

# Store layers weight & bias (default: mean=0, sd = 1)
weights = {}
biases = {}
weights['h1'] = tf.Variable(tf.random_normal([n_input, archit[0]]))
biases['b1'] = tf.Variable(tf.random_normal([archit[0]]))
for l in range(1,layer_number):
    w_name = 'h' + str(l+1)
    b_name = 'b' + str(l+1)
    weights[w_name] = tf.Variable(tf.random_normal([archit[l-1], archit[l]]))
    biases[b_name] = tf.Variable(tf.random_normal([archit[l]]))
weights['out'] = tf.Variable(tf.random_normal([archit[-1], n_classes]))
biases['out'] = tf.Variable(tf.random_normal([n_classes]))

# Construct model
pred = multilayer_perceptron(nn_x, weights, biases, layer_number, actfun, l1, keep_prob)

# Define loss and optimizer
loss = tf.reduce_mean(tf.square(pred - nn_y))   # Mean squared error
if l2 != 0.0:
    print('Applying L2 (weights) regularization')
    try:
        regularizer = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2'])
    except:
        regularizer = tf.nn.l2_loss(weights['h1'])

    loss = tf.reduce_mean(loss + l2 * regularizer)
else:
    loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=lrate).minimize(loss)
correct_prediction = tf.equal(tf.argmax(pred, axis=1), tf.argmax(nn_y, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Launch the graph
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

## Train the ANN model ##
losses = []
epoch_count = 0
train='yes'
stop_counts = 0
old_c = 1

while train == 'yes':
    epoch_count += 1
    _, c, train_acc = sess.run([optimizer, loss, accuracy],feed_dict = {nn_x:X_train, nn_y:pd.DataFrame(Y_train), keep_prob:l1})
    test_c = sess.run(loss, feed_dict={nn_x: X_test, nn_y:pd.DataFrame(Y_test), keep_prob:1.0})
    losses.append([epoch_count, c, test_c])

    pchange = (old_c-c)/old_c
    if epoch_count > 500:
        if abs(pchange) < epoch_thresh:
            stop_counts += 1
            print(epoch_count)
            if stop_counts >= 10:
                train='no'

    if (epoch_count) % 100 == 0:
        print("Epoch:", '%i' % (epoch_count), "; Training MSE=", "{:.3f}".format(c), "; Testing MSE=", "{:.3f}".format(test_c), '; Percent change=', str(pchange))

    old_c = c
    if epoch_count == max_epochs or train=='no':
        print('Final MSE after %i epochs for training: %.5f and testing: %.5f' % (epoch_count, c, test_c))

losses_df = pd.DataFrame(losses, columns=['epoch', 'train_loss', 'test_loss'])        
losses_df.to_csv(SAVE+'_losses.csv', index=False)

# Predict test set and add to yhat output
y_pred = sess.run(pred, feed_dict={nn_x: X_test, keep_prob:1.0})
print(y_pred[:,0])
ho_cor = np.corrcoef(Y_test, y_pred[:,0])
print('Holdout Accuracy (correlation coef): %.5f' % ho_cor[0,1])


run_time = timeit.default_timer() - start_time


if not os.path.isfile('RESULTS.txt'):
    out2 = open('RESULTS.txt', 'a')
    out2.write('DateTime\tRunTime\tTag\tDFs\tDFy\tTrait\tFeatSel\tNumFeat\tHoldout\tNumHidLay\tArchit\tActFun\tEpochs\tL1\tL2\tLearnRate\tMSE_Train\tMSE_Test\tPCC_ho\n')

out2 = open('RESULTS.txt', 'a')
out2.write('%s\t%0.5f\t%s\t%s\t%s\t%s\t%s\t%i\t%s\t%i\t%s\t%s\t%i\t%f\t%f\t%f\t%0.5f\t%0.5f\t%0.5f\n' % (
    timestamp, run_time, TAG, X_file, Y_file, y_name, FEAT, x.shape[1], ho, layer_number, str(arc), actfun, epoch_count, l1, l2, lrate, c, test_c, ho_cor[0,1]))

print('finished!')