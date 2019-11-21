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

T = 5
sample_cases_split = 2
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

save_models_var = 'DenseNet-KA-Dual_T-'+str(T)+'_Source-Lab_Target-USCHAD_Split-Case-' + str(sample_cases_split)

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
        if sample_cases_split==2:
            teacher_model_path = code_path + '/DenseNet-Supervised_TrainRandom_TestRandom_Split-Case-' + str(sample_cases_split) + '_WindowSize-' + str(window_size) + '/TrainSize-90_TestSize-10/Seed-'+str(1)+'/Trial-1_Model-250-epochs.h5'
        elif sample_cases_split==3:
            teacher_model_path = code_path + '/DenseNet-Supervised_TrainRandom_TestRandom_Split-Case-' + str(sample_cases_split) + '_WindowSize-' + str(window_size) + '/TrainSize-90_TestSize-10/Seed-'+str(1)+'/Trial-2_Model-250-epochs.h5'
            
        
        for trial in trial_value_set:
            if not os.path.exists(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_History.pckl'):
            
                #Model
                K.clear_session()
                
                teacher_model = timeseries_classifier_model_02(input_shape, 8, [4,4,4,4])
                teacher_model.load_weights(teacher_model_path)
                #teacher_model.summary()
                
                teacher_model_activation_output = models.Model(teacher_model.input,teacher_model.get_layer('dense').output)
                
                y_train_activation = teacher_model_activation_output.predict(x_train)
                y_test_activation = teacher_model_activation_output.predict(x_test)
                
                ## Softening logits
                

                y_train_softmax = np.exp(y_train_activation/T)
                y_train_sum = np.sum(y_train_softmax,1)
                y_train_softmax = y_train_softmax/y_train_sum[:,None]

                y_test_softmax = np.exp(y_test_activation/T)
                y_test_sum = np.sum(y_test_softmax,1)
                y_test_softmax = y_test_softmax/y_test_sum[:,None]
                
                #########################################
                ## Create student model
                student_model = tl_dualclassifier_model_knowledge_adaptation(input_shape,[8,12],[4,4,4,4])
                student_model.summary()
                
                # 1-50 Epochs
                opt0 = optimizers.Adam(lr=1e-2, beta_1=0.99, epsilon=1e-1)
                student_model.compile(optimizer=opt0, loss=['kullback_leibler_divergence','categorical_crossentropy'],metrics=['accuracy'])
                
                history0 = student_model.fit(x_train,[y_train_softmax,y_train],batch_size=4,epochs=50,verbose=2,validation_data=(x_test,[y_test_softmax,y_test]))
                student_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/KA-Trial-' + str(trial) + '_Model-50-epochs.h5')
                
                
                # 1-50 Epochs
                opt1 = optimizers.Adam(lr=1e-3, beta_1=0.99, epsilon=1e-1)
                student_model.compile(optimizer=opt1, loss=['kullback_leibler_divergence','categorical_crossentropy'],metrics=['accuracy'])
                
                history1 = student_model.fit(x_train,[y_train_softmax,y_train],batch_size=4,epochs=100,verbose=2,validation_data=(x_test,[y_test_softmax,y_test]))
                student_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/KA-Trial-' + str(trial) + '_Model-150-epochs.h5')


                # 50-250 Epochs
                opt2 = optimizers.Adam(lr=1e-4, beta_1=0.99, epsilon=1e-1)
                student_model.compile(optimizer=opt2, loss=['kullback_leibler_divergence','categorical_crossentropy'], metrics=['accuracy'])
                history2 = student_model.fit(x_train,[y_train_softmax,y_train],batch_size=4,epochs=100,verbose=2,validation_data=(x_test,[y_test_softmax,y_test]))
                student_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/KA-Trial-' + str(trial) + '_Model-250-epochs.h5')

                
                f = open(classifier_models_path + '/Seed-'+str(seed_value) + '/KA-Trial-' + str(trial) + '_History.pckl','wb')
                pickle.dump([history0.history,history1.history,history2.history],f)
                f.close()
                
                
                layer_name = 'activation_84'
                classifier_model = models.Model(inputs=student_model.input, outputs=student_model.get_layer(layer_name).output)
                
                

                
                classifier_model.summary()
                # 1-300 Epochs
                opt1 = optimizers.Adam(lr=1e-3, beta_1=0.99, epsilon=1e-1)
                classifier_model.compile(optimizer=opt1, loss='categorical_crossentropy',metrics=['accuracy'])
                history1 = classifier_model.fit(x_train,y_train,batch_size=4,epochs=10,verbose=2,validation_data=(x_test,y_test))
                classifier_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-50-epochs.h5')

                # 300-600 Epochs
                opt2 = optimizers.Adam(lr=1e-4, beta_1=0.99, epsilon=1e-1)
                classifier_model.compile(optimizer=opt2, loss='categorical_crossentropy', metrics=['accuracy'])
                history2 = classifier_model.fit(x_train,y_train,batch_size=4,epochs=100,verbose=2,validation_data=(x_test,y_test))
                classifier_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-150-epochs.h5')

                # 600-1000 Epochs
                opt3 = optimizers.Adam(lr=1e-5, beta_1=0.99, epsilon=1e-1)
                classifier_model.compile(optimizer=opt3, loss='categorical_crossentropy', metrics=['accuracy'])
                history3 = classifier_model.fit(x_train,y_train,batch_size=4,epochs=100,verbose=2,validation_data=(x_test,y_test))
                classifier_model.save(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_Model-250-epochs.h5')

                f = open(classifier_models_path + '/Seed-'+str(seed_value) + '/Trial-' + str(trial) + '_History.pckl','wb')
                pickle.dump([history1.history,history2.history,history3.history],f)
                f.close()
            else:
                print('Trial-' + str(trial) + ' done!')
        
                
