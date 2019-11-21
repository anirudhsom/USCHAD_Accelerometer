import os
import time
import glob
import pickle
import random
import numpy as np
import scipy.io as sio
import scipy.stats as scst
from sklearn.preprocessing import StandardScaler, MinMaxScaler

################################
def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        #print( "Elapsed time: %f seconds.\n" %tempTimeInterval )
        return np.round(tempTimeInterval,4)

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)
    
################################
class DoDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

################################
# Creating train-test splits based on classes specified 
def load_train_test_sets_honeybee(train_cases, test_cases, data_subject_class, data_path):
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # Loop for class
    for i in range(len(train_cases)):
    #i = 0 
        current_class = train_cases[i]
        indices = data_subject_class[:,1]==current_class

        subject_indices = data_subject_class[indices,:]

        # Loop for subjects with class i
        for j in range(subject_indices.shape[0]):
        #j = 0

            file_name = 'ID-' + str(subject_indices[j,0]) + '_CLASS-' + format(subject_indices[j,1], '02d')
            dir_list = glob.glob(data_path + '/' + file_name + '*')
            dir_list.sort()

            for k in range(len(dir_list)):
                temp = sio.loadmat(dir_list[k])
                x_train.append(temp['xw'].reshape((-1)))
                y_train.append(current_class)

    # Loop for class
    for i in range(len(test_cases)):
    #i = 0 
        current_class = test_cases[i]
        indices = data_subject_class[:,1]==current_class

        subject_indices = data_subject_class[indices,:]

        # Loop for subjects with class i
        for j in range(subject_indices.shape[0]):
        #j = 0

            file_name = 'ID-' + str(subject_indices[j,0]) + '_CLASS-' + format(subject_indices[j,1], '02d')
            dir_list = glob.glob(data_path + '/' + file_name + '*')
            dir_list.sort()

            for k in range(len(dir_list)):
                temp = sio.loadmat(dir_list[k])
                x_test.append(temp['xw'].reshape((-1)))
                y_test.append(current_class)

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Normalizing w.r.t. train set
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, y_train, x_test, y_test


################################
# Creating train-test splits into individual labels specified by sample_cases
def load_classifier_data_honeybee(sample_cases, test_set_size, data_subject_class, data_path,seed_value):
    x = []
    y = []
    subj_numbers = []

    # Loop for class
    for i in range(len(sample_cases)):

    #i = 0 
        current_class = sample_cases[i]

        indices = data_subject_class[:,1]==current_class

        subject_indices = data_subject_class[indices,:]

        # Loop for subjects with class i
        for j in range(subject_indices.shape[0]):
        #j = 0

            file_name = 'ID-' + str(subject_indices[j,0]) + '_CLASS-' + format(subject_indices[j,1], '02d')
            dir_list = glob.glob(data_path + '/' + file_name + '*')
            dir_list.sort()

           # print(len(dir_list))
            for k in range(len(dir_list)):
                temp = sio.loadmat(dir_list[k])
                x.append(temp['xw'].reshape((-1)))
                y_temp = np.zeros((len(sample_cases)))
                y_temp[i] = 1
                y.append(y_temp)
                subj_numbers.append([subject_indices[j,0],subject_indices[j,1]])

    x = np.vstack(x)
    y = np.vstack(y)
    subj_numbers = np.array(subj_numbers)

    unique_subjects = np.sort(np.array(list(set(subj_numbers[:,0]))))
    total_unique_subjects = len(unique_subjects)

    total_test_subjects = int(np.floor(total_unique_subjects*test_set_size))
    seed_value = seed_value

    while True:

        #print(seed_value)
        random.seed(seed_value)
        test_subjects = unique_subjects[random.sample(range(total_unique_subjects), total_test_subjects)]
        
        test_subs = []
        train_subs = []

        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for ii in range(x.shape[0]):
            test_flag = True
            for tt in range(total_test_subjects):
                if subj_numbers[ii,0]==test_subjects[tt]:
                    test_subs.append(test_subjects[tt])
                    
                    x_test.append(x[ii,:])
                    y_test.append(y[ii,:])
                    test_flag = False
            if test_flag:
                train_subs.append(subj_numbers[ii,0])
                x_train.append(x[ii,:])
                y_train.append(y[ii,:])
                
        #print('Train: ',sorted(set(train_subs)))
        #print('Test: ',sorted(set(test_subs)))

        if len(set(np.argmax(y_test,1)))==len(set(np.argmax(y,1))):
            break
        else:
            seed_value = seed_value + 1

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    # Normalizing w.r.t. train set
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, y_train, x_test, y_test


################################
# Creating train-test splits into individual labels specified by sample_cases
def load_classifier_data_uschad(test_set_size, data_subject_class, data_path,seed_value):
    dir_list =  glob.glob(data_path+'/*.mat')
    
    x = []
    y = []
    y_info = []
    for i in range(len(dir_list)):
        y_info.append(np.array([int(dir_list[i][-27:-25]),int(dir_list[i][-18:-16])-1]))
        temp = sio.loadmat(dir_list[i])
        x.append(temp['xw'].reshape((-1)))

        y_temp = np.zeros((len(np.unique(data_subject_class[:,1]))))
        y_temp[int(dir_list[i][-18:-16])-1] = 1
        y.append(y_temp)
    x = np.array(x)
    y = np.array(y)
    y_info = np.array(y_info)
    
    unique_subjects =np.sort(np.array(list(set(data_subject_class[:,0]))))
    total_unique_subjects = len(unique_subjects)

    random.seed(seed_value)
    test_subjects = list(unique_subjects[random.sample(range(total_unique_subjects), test_set_size)])
    train_subjects = list(unique_subjects)

    for i in range(len(test_subjects)):
        temp = test_subjects[i]
        train_subjects.remove(temp)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(len(train_subjects)):
    #i = 0    
        indices = y_info[:,0] == train_subjects[i]
        x_train.append(x[indices,:])
        y_train.append(y[indices,:])

    for i in range(len(test_subjects)):
    #i = 0    
        indices = y_info[:,0] == test_subjects[i]
        x_test.append(x[indices,:])
        y_test.append(y[indices,:])

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    # Normalizing w.r.t. train set
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    
    return x_train, y_train, x_test, y_test

################################
# Creating train-test splits with energy-expenditure labels
def load_traintest_sets_energy_expenditure_classes(sample_cases, ee_labels, test_set_size, data_subject_class, data_path):
    x = []
    y = []
    subj_numbers = []

    # Loop for class
    for i in range(len(sample_cases)):

    #i = 0 
        current_class = sample_cases[i]

        indices = data_subject_class[:,1]==current_class

        subject_indices = data_subject_class[indices,:]

        # Loop for subjects with class i
        for j in range(subject_indices.shape[0]):
        #j = 0

            file_name = 'ID-' + str(subject_indices[j,0]) + '_CLASS-' + format(subject_indices[j,1], '02d')
            dir_list = glob.glob(data_path + '/' + file_name + '*')
            dir_list.sort()

            for k in range(len(dir_list)):
                temp = sio.loadmat(dir_list[k])
                x.append(temp['xw'].reshape((-1)))
                y_temp = np.zeros((len(set(ee_labels))))
                
                if i == 0 or i == 1:
                    y_temp[0] = 1
                elif i>1 and i<=4:
                    y_temp[1] = 1
                elif i>4 and i<=7:
                    y_temp[2] = 1
                    
                y.append(y_temp)
                subj_numbers.append([subject_indices[j,0],subject_indices[j,1]])

    x = np.vstack(x)
    y = np.vstack(y)
    subj_numbers = np.array(subj_numbers)

    unique_subjects = np.sort(np.array(list(set(subj_numbers[:,0]))))
    total_unique_subjects = len(unique_subjects)

    total_test_subjects = int(np.floor(total_unique_subjects*test_set_size))
    seed_value = 0

    while True:
        random.seed(seed_value)
        test_subjects = unique_subjects[random.sample(range(total_unique_subjects), total_test_subjects)]

        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for ii in range(x.shape[0]):
            test_flag = True
            for tt in range(total_test_subjects):
                if subj_numbers[ii,0]==test_subjects[tt]:
                    x_test.append(x[ii,:])
                    y_test.append(y[ii,:])
                    test_flag = False
            if test_flag:
                x_train.append(x[ii,:])
                y_train.append(y[ii,:])

        if len(set(np.argmax(y_test,1)))==len(set(np.argmax(y,1))):
            break
        else:
            seed_value = seed_value + 1

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    # Normalizing w.r.t. train set
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, y_train, x_test, y_test



################################
# Creating train-test splits into individual labels specified by sample_cases for encoder state classification obtained from pretrained autoencoder
def load_enc_state_classifier_data_honeybee(x_data, sample_cases, test_set_size, data_subject_class, data_path):
    x = []
    y = []
    subj_numbers = []

    # Loop for class
    for i in range(len(sample_cases)):
    #i = 0 
        current_class = sample_cases[i]

        indices = data_subject_class[:,1]==current_class

        subject_indices = data_subject_class[indices,:]

        # Loop for subjects with class i
        for j in range(subject_indices.shape[0]):
        #j = 0

            file_name = 'ID-' + str(subject_indices[j,0]) + '_CLASS-' + format(subject_indices[j,1], '02d')
            dir_list = glob.glob(data_path + '/' + file_name + '*')
            dir_list.sort()

            for k in range(len(dir_list)):
                y_temp = np.zeros((len(sample_cases)))
                y_temp[i] = 1
                y.append(y_temp)
                subj_numbers.append([subject_indices[j,0],subject_indices[j,1]])

    x = x_data
    y = np.vstack(y)


    subj_numbers = np.array(subj_numbers)

    unique_subjects = np.sort(np.array(list(set(subj_numbers[:,0]))))
    total_unique_subjects = len(unique_subjects)

    total_test_subjects = int(np.floor(total_unique_subjects*test_set_size))
    seed_value = 0

    while True:
        random.seed(seed_value)
        test_subjects = unique_subjects[random.sample(range(total_unique_subjects), total_test_subjects)]

        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for ii in range(x.shape[0]):
            test_flag = True
            for tt in range(total_test_subjects):
                if subj_numbers[ii,0]==test_subjects[tt]:
                    x_test.append(x[ii,:])
                    y_test.append(y[ii,:])
                    test_flag = False
            if test_flag:
                x_train.append(x[ii,:])
                y_train.append(y[ii,:])

        if len(set(np.argmax(y_test,1)))==len(set(np.argmax(y,1))):
            break
        else:
            seed_value = seed_value + 1

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    # Normalizing w.r.t. train set
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, y_train, x_test, y_test


################################
# Creating train-test splits into energy expendditure labels specified by sample_cases for encoder state classification obtained from pretrained autoencoder
def load_enc_state_classifier_energy_expenditure_classes(x_data, sample_cases, test_set_size, data_subject_class, data_path):
    x = []
    y = []
    subj_numbers = []

    # Loop for class
    for i in range(len(sample_cases)):
    #i = 0 
        current_class = sample_cases[i]

        indices = data_subject_class[:,1]==current_class

        subject_indices = data_subject_class[indices,:]

        # Loop for subjects with class i
        for j in range(subject_indices.shape[0]):
        #j = 0

            file_name = 'ID-' + str(subject_indices[j,0]) + '_CLASS-' + format(subject_indices[j,1], '02d')
            dir_list = glob.glob(data_path + '/' + file_name + '*')
            dir_list.sort()

            for k in range(len(dir_list)):
                y_temp = np.zeros((3))
                
                #if saved_models_var == 'Saved_encoder_state_EE_classifier_models_Lab':
                if i == 0 or i == 1:
                    y_temp[0] = 1
                elif i>1 and i<=4:
                    y_temp[1] = 1
                elif i>4 and i<=7:
                    y_temp[2] = 1
                #else:
                    
                    
                y.append(y_temp)
                subj_numbers.append([subject_indices[j,0],subject_indices[j,1]])

    x = x_data
    y = np.vstack(y)


    subj_numbers = np.array(subj_numbers)

    unique_subjects = np.sort(np.array(list(set(subj_numbers[:,0]))))
    total_unique_subjects = len(unique_subjects)

    total_test_subjects = int(np.floor(total_unique_subjects*test_set_size))
    seed_value = 0

    while True:
        random.seed(seed_value)
        test_subjects = unique_subjects[random.sample(range(total_unique_subjects), total_test_subjects)]

        x_train = []
        x_test = []
        y_train = []
        y_test = []
        for ii in range(x.shape[0]):
            test_flag = True
            for tt in range(total_test_subjects):
                if subj_numbers[ii,0]==test_subjects[tt]:
                    x_test.append(x[ii,:])
                    y_test.append(y[ii,:])
                    test_flag = False
            if test_flag:
                x_train.append(x[ii,:])
                y_train.append(y[ii,:])

        if len(set(np.argmax(y_test,1)))==len(set(np.argmax(y,1))):
            break
        else:
            seed_value = seed_value + 1

    x_train = np.vstack(x_train)
    x_test = np.vstack(x_test)
    y_train = np.vstack(y_train)
    y_test = np.vstack(y_test)

    # Normalizing w.r.t. train set
    sc = StandardScaler()
    sc.fit(x_train)
    x_train = sc.transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, y_train, x_test, y_test


##################################################################
def time_series_feat_baseline(x):
    
    train_mean = np.mean(x,axis=1)
    train_var = np.var(x,axis=1)
    train_rms = np.sqrt(np.mean(np.square(x),axis=1))

    train_corr = []
    for i in range(len(x)):
        temp = x[i]
        corr_val = []
        for j in range(3):
            for k in range(j+1,3):
                haha = scst.pearsonr(temp[:,j],temp[:,k])
                corr_val.append(haha[0])
        train_corr.append(np.array(corr_val))
    train_corr = np.array(train_corr)


    train_min = np.min(x,axis=1)
    train_max = np.max(x,axis=1)
    train_diff = train_max-train_min

    train_dxy = np.sqrt(np.square(train_diff[:,0])+np.square(train_diff[:,1])).reshape((len(x),1))
    train_dxz = np.sqrt(np.square(train_diff[:,0])+np.square(train_diff[:,2])).reshape((len(x),1))
    train_dyz = np.sqrt(np.square(train_diff[:,1])+np.square(train_diff[:,2])).reshape((len(x),1))
    train_dxyz = np.sqrt(np.square(train_diff[:,0])+np.square(train_diff[:,1])+np.square(train_diff[:,2])).reshape((len(x),1))
    
    x_feat = np.concatenate((train_mean,train_var,train_rms,train_corr,train_diff,train_dxy,train_dxz,train_dyz,train_dxyz),axis=1)
    
    return x_feat