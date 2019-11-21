import os
import glob
import pickle
import random
import numpy as np
import scipy.io as sio
import scipy.stats as scst
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.keras import models, layers, losses, optimizers, utils
from tensorflow.python.keras import backend as K

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import som_utils as sutils
from densenet_utils import *

################################
## Parameters
code_path = os.getcwd()
trial_value_set = np.arange(1,4,dtype=int)
seed_value_set = np.array([1,4,6])

window_size = 250
input_shape = [window_size,3]

free_liv_train_size_set = np.arange(10,61,10,dtype=int)
print(free_liv_train_size_set)
free_liv_main_test_size = 4
block_sizes = [4,4,4,4]

################################
## Paths
data_path = code_path + '/Window-size-' + str(window_size)
temp = sio.loadmat(code_path + '/Successful_Failed_cases_Window-size-' + str(window_size) + '.mat')
data_subject_class = temp['success_cases']
test_set_size = free_liv_main_test_size

save_models_var = 'DenseNet-Supervised_TrainUSCHAD_TestUSCHAD'

accuracy_summary = np.zeros((3,3,6))
precision_summary = np.zeros((3,3,3,6))
recall_summary = np.zeros((3,3,3,6))
f1score_summary = np.zeros((3,3,3,6))

confusion_matrix_summary = np.zeros((3,3,6,12,12))

trial_indices = np.arange(0,3,dtype=int)
seed_indices = np.arange(0,3,dtype=int)
amount_of_data_indices = np.arange(0,6,dtype=int)

if os.path.exists(code_path+'/Supervised_Learning_Summary_Window-'+str(window_size)+'.pckl'):
    f = open(code_path+'/Supervised_Learning_Summary_Window-'+str(window_size)+'.pckl','rb')
    temp = pickle.load(f)
    f.close()
    
    accuracy_summary = temp[0]
    precision_summary = temp[1]
    recall_summary = temp[2]
    f1score_summary = temp[3]
    confusion_matrix_summary = temp[4]

for free_liv_train_size_index in amount_of_data_indices:
    free_liv_train_size = (free_liv_train_size_set[free_liv_train_size_index])/100
    print(free_liv_train_size)
    classifier_models_path  = code_path + '/' + save_models_var + '_WindowSize-' + str(window_size) + '/TrainSize-' + str(int(free_liv_train_size*100)) + '_TestSize-' + str(int(free_liv_main_test_size*100))
    
    for seed_value_index in seed_indices:
        seed_value = seed_value_set[seed_value_index]
        x, y, x_test, y_test =  sutils.load_classifier_data_uschad(free_liv_main_test_size, data_subject_class, data_path,seed_value)

        x = x.reshape((-1,window_size,3))
        x_test = x_test.reshape((-1,window_size,3))
        
        train_set_size = int((free_liv_train_size)*(x.shape[0]+x_test.shape[0]))
        random.seed(seed_value)
        train_indices = random.sample(range(x.shape[0]),train_set_size)
        x_train = x[train_indices,:,:]
        y_train = y[train_indices,:]
        
        #print('Train data shape: ',x_train.shape)
        #print('Test data shape: ',x_test.shape)
        #print('Train label shape: ',y_train.shape)
        #print('Test label shape: ',y_test.shape)
        
        num_outputs = y_train.shape[1]
        
        for trial_index in trial_indices:
            trial = trial_value_set[trial_index]
            #Model
            K.clear_session()

            classifier_model = timeseries_classifier_model_02(input_shape,num_outputs,block_sizes)
            #classifier_model.summary()
            classifier_model.load_weights(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-250-epochs.h5')
            y_test_predict = classifier_model.predict(x_test)
            
            y_test_array = np.argmax(y_test,1)
            y_test_predict_array = np.argmax(y_test_predict,1)
            
            ##
            accuracy_summary[seed_value_index,trial_index,free_liv_train_size_index] = accuracy_score(y_test_array,y_test_predict_array)
            
            ##
            precision_summary[0,seed_value_index,trial_index,free_liv_train_size_index] = precision_score(y_test_array,y_test_predict_array,average='micro')
            
            precision_summary[1,seed_value_index,trial_index,free_liv_train_size_index] = precision_score(y_test_array,y_test_predict_array,average='macro')
            
            precision_summary[2,seed_value_index,trial_index,free_liv_train_size_index] = precision_score(y_test_array,y_test_predict_array,average='weighted')
            
            ##
            recall_summary[0,seed_value_index,trial_index,free_liv_train_size_index] = recall_score(y_test_array,y_test_predict_array,average='micro')
            
            recall_summary[1,seed_value_index,trial_index,free_liv_train_size_index] = recall_score(y_test_array,y_test_predict_array,average='macro')
            
            recall_summary[2,seed_value_index,trial_index,free_liv_train_size_index] = recall_score(y_test_array,y_test_predict_array,average='weighted')
            
            ##
            f1score_summary[0,seed_value_index,trial_index,free_liv_train_size_index] = f1_score(y_test_array,y_test_predict_array,average='micro')
            
            f1score_summary[1,seed_value_index,trial_index,free_liv_train_size_index] = f1_score(y_test_array,y_test_predict_array,average='macro')
            
            f1score_summary[2,seed_value_index,trial_index,free_liv_train_size_index] = f1_score(y_test_array,y_test_predict_array,average='weighted')
            
            confusion_matrix_summary[seed_value_index,trial_index,free_liv_train_size_index,:,:] = confusion_matrix(y_test_array,y_test_predict_array)
            
            print('Train-Size: ',str(int(free_liv_train_size*100)),' | Seed: ',str(seed_value),' | Trial: ',str(trial),'\nAccuracy: ',accuracy_summary[seed_value_index,trial_index,free_liv_train_size_index],' | Precision: ',precision_summary[2,seed_value_index,trial_index,free_liv_train_size_index],' | Recall: ',recall_summary[2,seed_value_index,trial_index,free_liv_train_size_index],' | F1: ',f1score_summary[2,seed_value_index,trial_index,free_liv_train_size_index],'\n')

            
            f = open(code_path+'/Supervised_Learning_Summary_Window-'+str(window_size)+'.pckl','wb')
            pickle.dump([accuracy_summary,precision_summary,recall_summary,f1score_summary,confusion_matrix_summary],f)
            f.close()
            
            
            
            
        
