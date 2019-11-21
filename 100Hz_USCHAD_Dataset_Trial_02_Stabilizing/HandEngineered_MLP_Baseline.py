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


from sklearn.metrics import confusion_matrix, f1_score
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

free_liv_train_size_set = np.arange(0.10,0.61,0.10)
free_liv_main_test_size = 4
block_sizes = [4,4,4,4]

################################
## Paths
data_path = code_path + '/Window-size-' + str(window_size)
temp = sio.loadmat(code_path + '/Successful_Failed_cases_Window-size-' + str(window_size) + '.mat')
data_subject_class = temp['success_cases']
test_set_size = free_liv_main_test_size


save_models_var = 'HandEngineered-MLP-Baseline_TrainUSCHAD_TestUSCHAD'

for free_liv_train_size in free_liv_train_size_set:
    classifier_models_path  = code_path + '/' + save_models_var + '_WindowSize-' + str(window_size) + '/TrainSize-' + str(int(free_liv_train_size*100)) + '_TestSize-' + str(int(free_liv_main_test_size*100))
    
    for seed_value in seed_value_set:
        if not os.path.exists(classifier_models_path + '/Seed-'+str(seed_value)):
            os.makedirs(classifier_models_path + '/Seed-'+str(seed_value))
        
        x, y, x_test, y_test =  sutils.load_classifier_data_uschad(free_liv_main_test_size, data_subject_class, data_path,seed_value)

        x = x.reshape((-1,window_size,3))
        x_test = x_test.reshape((-1,window_size,3))
        
        train_set_size = int(free_liv_train_size*(x.shape[0]+x_test.shape[0]))
        random.seed(seed_value)
        train_indices = random.sample(range(x.shape[0]),train_set_size)
        x_train = x[train_indices,:,:]
        y_train = y[train_indices,:]
        
        x_train_feat = sutils.time_series_feat_baseline(x_train)
        x_test_feat = sutils.time_series_feat_baseline(x_test)
        
        print('Train data shape: ',x_train_feat.shape)
        print('Test data shape: ',x_test_feat.shape)
        print('Train label shape: ',y_train.shape)
        print('Test label shape: ',y_test.shape)
        
        sc = StandardScaler()
        sc.fit(x_train_feat)
        x_train_feat = sc.transform(x_train_feat)
        x_test_feat = sc.transform(x_test_feat)
        
        for trial in trial_value_set:
            #Model
            K.clear_session()
            
            model = models.Sequential()
            
            model.add(layers.Dense(64, activation='relu', input_dim=x_train_feat.shape[1],kernel_regularizer=regularizers.l2(0.01)))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
            model.add(layers.Dropout(0.2))
            #model.add(layers.Dense(256, activation='relu'))
            #model.add(layers.Dropout(0.2))
            model.add(layers.Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
            model.add(layers.Dropout(0.2))
            model.add(layers.Dense(y_train.shape[1], activation='softmax'))

            opt1 = optimizers.Adam(lr=1e-2, beta_1=0.99, epsilon=1e-1)
            opt2 = optimizers.Adam(lr=1e-3, beta_1=0.99, epsilon=1e-1)
            opt3 = optimizers.Adam(lr=1e-4, beta_1=0.99, epsilon=1e-1)

            model.compile(loss='categorical_crossentropy',
                          optimizer=opt1,
                          metrics=['accuracy'])
            history1 = model.fit(x_train_feat, y_train,
                      epochs=50,
                      batch_size=4, validation_data=(x_test_feat,y_test),verbose=1)
            model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-50-epochs.h5')


            model.compile(loss='categorical_crossentropy',
                          optimizer=opt2,
                          metrics=['accuracy'])
            history2 = model.fit(x_train_feat, y_train,
                      epochs=100,
                      batch_size=4, validation_data=(x_test_feat,y_test),verbose=1)
            model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-150-epochs.h5')

            model.compile(loss='categorical_crossentropy',
                          optimizer=opt3,
                          metrics=['accuracy'])
            history3 = model.fit(x_train_feat, y_train,
                      epochs=100,
                      batch_size=4, validation_data=(x_test_feat,y_test),verbose=1)
            model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-250-epochs.h5')

            f = open(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_History.pckl','wb')
            pickle.dump([history1.history,history2.history,history3.history],f)
            f.close()
        
        