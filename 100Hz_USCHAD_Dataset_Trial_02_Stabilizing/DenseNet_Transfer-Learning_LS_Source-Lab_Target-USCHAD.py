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
seed_value_set = np.array([1,2,4])

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

save_models_var = 'DenseNet-Transfer-Learning-Latent-Representations_Source-Lab_Target-USCHAD' 

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
        
        ## Pre-trained model
        pretrained_model_path = code_path + '/DenseNet-Supervised_TrainLab_TestLab' + '_WindowSize-' + str(window_size) + '/TrainSize-90_TestSize-10/Seed-'+str(2)+'/Trial-1_Model-250-epochs.h5'
        
        for trial in trial_value_set:
            if not os.path.exists(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_History.pckl'):
            
                #Model
                K.clear_session()
                
                pretrained_model = timeseries_classifier_model_02(input_shape, 8, [4,4,4,4])
                pretrained_model.load_weights(pretrained_model_path)
                #pretrained_model.summary()
                pretrained_ls_model = models.Model(pretrained_model.input,pretrained_model.get_layer('activation_20').output)
                
                x_train_ls = pretrained_ls_model.predict(x_train)
                x_test_ls = pretrained_ls_model.predict(x_test)

                print('Train data shape: ',x_train_ls.shape)
                print('Test data shape: ',x_test_ls.shape)
                print('Train label shape: ',y_train.shape)
                print('Test label shape: ',y_test.shape)
        
                classifier_input_shape = [x_train_ls.shape[1],x_train_ls.shape[2]]
                classifier_num_outputs = y_train.shape[1]
                
                
                classifier_model = timeseries_classifier_model_02(classifier_input_shape,classifier_num_outputs,[4,4,4,4])
                classifier_model.summary()
                
                # 1-300 Epochs
                opt1 = optimizers.Adam(lr=1e-3, beta_1=0.99, epsilon=1e-1)
                classifier_model.compile(optimizer=opt1, loss='categorical_crossentropy',metrics=['accuracy'])
                history1 = classifier_model.fit(x_train_ls,y_train,batch_size=4,epochs=250,verbose=2,validation_data=(x_test_ls,y_test))
                classifier_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-250-epochs.h5')

                # 300-600 Epochs
                opt2 = optimizers.Adam(lr=1e-4, beta_1=0.99, epsilon=1e-1)
                classifier_model.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['accuracy'])
                history2 = classifier_model.fit(x_train_ls,y_train,batch_size=4,epochs=100,verbose=2,validation_data=(x_test_ls,y_test))
                classifier_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-350-epochs.h5')

                # 600-1000 Epochs
                opt3 = optimizers.Adam(lr=1e-5, beta_1=0.99, epsilon=1e-1)
                classifier_model.compile(optimizer=opt3, loss='categorical_crossentropy', metrics=['accuracy'])
                history3 = classifier_model.fit(x_train_ls,y_train,batch_size=4,epochs=100,verbose=2,validation_data=(x_test_ls,y_test))
                classifier_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-450-epochs.h5')

                f = open(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_History.pckl','wb')
                pickle.dump([history1.history,history2.history,history3.history],f)
                f.close()
            else:
                print('Trial-' + str(trial) + ' done!')
        