import os
import glob
import pickle
import random
import numpy as np
import scipy.io as sio
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import matplotlib.pyplot as plt

from tensorflow.python.keras import models, layers, losses, optimizers, utils, regularizers
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Lambda

###############################################
def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

###############################################
def soft_dtw_custom_loss(t1,t2):
    gamma=1
    return my_func(soft_dtw(t1,t2))

###############################################
def H_l(x, k, bottleneck_size, kernel_width, dropout_rate):
    use_bottleneck = bottleneck_size > 0
    num_bottleneck_output_filters = k * bottleneck_size

    if use_bottleneck:
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Conv1D(
            num_bottleneck_output_filters,
            1,
            strides=1,
            padding="same",
            dilation_rate=1,kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(dropout_rate)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(
        k,
        kernel_width,
        strides=1,
        padding="same",
        dilation_rate=1,kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.Dropout(dropout_rate)(x)
    return x
    
###############################################
def dense_block(x, k, num_layers, kernel_width, bottleneck_size,dropout_rate):
    layers_to_concat = [x]
    for _ in range(num_layers):
        x = H_l(x, k, bottleneck_size, kernel_width, dropout_rate)
        layers_to_concat.append(x)
        x = layers.Concatenate(axis=-1)(layers_to_concat)
    return x

###############################################
def transition_block(x, pool_size=2, stride=2, theta=0.5): 
    assert theta > 0 and theta <= 1
    
    num_transition_output_filters = int(int(x.shape[2]) * float(theta))
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv1D(
        num_transition_output_filters,
        1,
        strides=1,
        padding="same",
        dilation_rate=1,kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.AveragePooling1D(
        pool_size=pool_size,
        strides=stride,
        padding="same")(x)
    return x

###############################################
def autoencoder_model(input_shape,block_sizes):
    # Parameters
    ### Initial
    initial_conv_width=7
    initial_stride=1
    initial_filters=32
    initial_pool_width=3
    initial_pool_stride=1
    dropout_rate = 0.2

    ### Dense
    k=16
    conv_kernel_width=3
    bottleneck_size=4

    ### Transition
    transition_pool_size=2
    transition_pool_stride=1
    theta=0.5
    
    
    # Encoder 
    encoder_input = layers.Input(shape=input_shape)
    x_en = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same",kernel_regularizer=regularizers.l2(0.01))(encoder_input)
    x_en = layers.BatchNormalization()(x_en)
    x_en = layers.Activation("relu")(x_en)
    x_en = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x_en)

    for block_size in block_sizes[:-1]:
        x_en = dense_block(x_en,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        x_en = transition_block(x_en,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)

    final_block_size = block_sizes[-1]
    x_en = dense_block(x_en,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x_en = layers.BatchNormalization()(x_en)
    x_en = layers.Activation("relu")(x_en)

    # Latent Space

    ls = layers.Conv1D(
            initial_filters*2,
            initial_conv_width,
            initial_stride,
            padding='same',kernel_regularizer=regularizers.l2(0.01))(x_en)
    ls = layers.BatchNormalization()(ls)
    ls = layers.Activation("relu")(ls)

    # Decoder
    dec_input = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same",kernel_regularizer=regularizers.l2(0.01))(ls)
    x_dec = layers.BatchNormalization()(dec_input)
    x_dec = layers.Activation("relu")(x_dec)
    x_dec = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x_dec)

    for block_size in block_sizes[:-1]:
        x_dec = dense_block(x_dec,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        x_dec = transition_block(x_dec,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)

    final_block_size = block_sizes[-1]
    x_dec = dense_block(x_dec,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Activation("relu")(x_dec)

    x_dec = layers.Conv1D(
            initial_filters*2,
            initial_conv_width,
            initial_stride,
            padding='same',kernel_regularizer=regularizers.l2(0.01))(x_dec)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Activation("relu")(x_dec)

    x_dec = layers.Conv1D(
            initial_filters,
            initial_conv_width,
            initial_stride,
            padding='same',kernel_regularizer=regularizers.l2(0.01))(x_dec)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Activation("relu")(x_dec)

    x_dec = layers.Conv1D(
            initial_filters//2,
            initial_conv_width,
            initial_stride,
            padding='same',kernel_regularizer=regularizers.l2(0.01))(x_dec)
    x_dec = layers.BatchNormalization()(x_dec)
    x_dec = layers.Activation("relu")(x_dec)

    x_dec = layers.Conv1D(
            3,
            1,
            initial_stride,
            padding='same',kernel_regularizer=regularizers.l2(0.01))(x_dec)
    dec_output = layers.BatchNormalization()(x_dec)

    net = models.Model(encoder_input, dec_output)
    
    return net

###############################################


def ls_classifier_model(input_shape,num_outputs,block_sizes):
    k=16
    conv_kernel_width=3
    bottleneck_size=4
    transition_pool_size=2
    transition_pool_stride=2
    theta=0.5
    initial_conv_width=7
    initial_stride=2
    initial_filters=32
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2
    
    model_input = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    for block_size in block_sizes[:-1]:
        x = dense_block(x,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        x = transition_block(x,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)

    final_block_size = block_sizes[-1]
    x = dense_block(x,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    model_output = layers.Dense(num_outputs, activation="softmax")(x)

    net = models.Model(model_input, model_output)
    
    return net


def timeseries_classifier_model_02(input_shape,num_outputs,block_sizes):
    k=16
    conv_kernel_width=3
    bottleneck_size=4
    transition_pool_size=2
    transition_pool_stride=2
    theta=0.5
    initial_conv_width=7
    initial_stride=2
    initial_filters=32
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2
    
    model_input = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    for block_size in block_sizes[:-1]:
        x = dense_block(x,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        x = transition_block(x,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)

    final_block_size = block_sizes[-1]
    x = dense_block(x,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    model_output = layers.Dense(num_outputs, activation="softmax")(x)

    net = models.Model(model_input, model_output)
    
    return net




def tl_classifier_model(input_shape,num_outputs,block_sizes):
    k=16
    conv_kernel_width=3
    bottleneck_size=4
    
    use_global_pooling = True
    dropout_rate = 0.2
    
    model_input = layers.Input(shape=input_shape)

    

    final_block_size = block_sizes[-1]
    x = dense_block(model_input,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    model_output = layers.Dense(num_outputs, activation="softmax")(x)

    net = models.Model(model_input, model_output)
    
    return net


def timeseries_classifier_model_knowledge_adaptation(input_shape,num_outputs,block_sizes):
    k=32
    conv_kernel_width=3
    bottleneck_size=4
    transition_pool_size=2
    transition_pool_stride=2
    theta=0.5
    initial_conv_width=7
    initial_stride=2
    initial_filters=64
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2
    
    model_input = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    for block_size in block_sizes[:-1]:
        x = dense_block(x,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        x = transition_block(x,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)

    final_block_size = block_sizes[-1]
    x = dense_block(x,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(num_outputs, activation="relu")(x)
    model_output = layers.Activation('softmax')(x)

    net = models.Model(model_input, model_output)
    
    return net

def timeseries_classifier_model_smooth_logits_knowledge_adaptation(input_shape,T,num_outputs,block_sizes):
    k=32
    conv_kernel_width=3
    bottleneck_size=4
    transition_pool_size=2
    transition_pool_stride=2
    theta=0.5
    initial_conv_width=7
    initial_stride=2
    initial_filters=64
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2
    
    
    model_input = layers.Input(shape=input_shape)

    x = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(x)

    for block_size in block_sizes[:-1]:
        x = dense_block(x,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        x = transition_block(x,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)

    final_block_size = block_sizes[-1]
    x = dense_block(x,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(num_outputs, activation="relu")(x)
    
    x = Lambda(lambda x: x * (1/T))(x)
    
    model_output = layers.Activation('softmax')(x)

    net = models.Model(model_input, model_output)
    
    return net


def tl_classifier_model_knowledge_adaptation(input_shape,num_outputs,block_sizes):
    k=32
    conv_kernel_width=3
    bottleneck_size=4
    
    use_global_pooling = True
    dropout_rate = 0.2
    
    model_input = layers.Input(shape=input_shape)

    

    final_block_size = block_sizes[-1]
    x = dense_block(model_input,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(num_outputs, activation="relu")(x)
    model_output = layers.Activation('softmax')(x)

    net = models.Model(model_input, model_output)
    
    return net


def tl_dualclassifier_model_knowledge_adaptation(input_shape,num_outputs,block_sizes):
    
    
    #########################################################
    ##############################################################
    k=16
    conv_kernel_width=3
    bottleneck_size=4
    transition_pool_size=2
    transition_pool_stride=2
    theta=0.5
    initial_conv_width=7
    initial_stride=2
    initial_filters=32
    initial_pool_width=3
    initial_pool_stride=2
    use_global_pooling = True
    dropout_rate = 0.2
    
    model_input = layers.Input(shape=input_shape)

    z = layers.Conv1D(
        initial_filters,
        initial_conv_width,
        strides=initial_stride,
        padding="same")(model_input)
    z = layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)
    z = layers.MaxPooling1D(
        pool_size=initial_pool_width,
        strides=initial_pool_stride,
        padding="same")(z)

    for block_size in block_sizes[:-1]:
        z = dense_block(z,
            k,
            block_size,
            conv_kernel_width,
            bottleneck_size,dropout_rate)
        z = transition_block(z,
            pool_size=transition_pool_size,
            stride=transition_pool_stride,
            theta=theta)
    
    ############################################################
    #########################################################
    
    ## Classifier 01
    final_block_size = block_sizes[-1]
    x = dense_block(z,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if use_global_pooling:
        x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(num_outputs[0], activation="relu")(x)
    model_output1 = layers.Activation('softmax')(x)
    
    ## Classifier 02
    final_block_size = block_sizes[-1]
    y = dense_block(z,
        k,
        final_block_size,
        conv_kernel_width,
        bottleneck_size,dropout_rate)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    if use_global_pooling:
        y = layers.GlobalAveragePooling1D()(y)

    y = layers.Dense(num_outputs[1], activation="relu")(y)
    model_output2 = layers.Activation('softmax')(y)
    
    
    

    net = models.Model(model_input, [model_output1,model_output2])
    
    return net