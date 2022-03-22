import sys, getopt

import numpy as np
import re
from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
from decimal import Decimal
import ipynb
import json
import pickle
import csv

import pandas as pd
import numpy as np
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold, GroupKFold,StratifiedKFold

from tensorflow.keras import losses


from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)

from sklearn.metrics import mean_squared_error

import tensorflow.keras as keras
import pandas as pd

import gc
import matplotlib.pyplot as plt
import os


from tqdm import tqdm
tqdm.pandas()

LOSS_WGTS = [0.3, 0.3, 0.3, 0.05, 0.05] #column weights, need to sum up to 1


DIST_NEW = True
DIST_NEW2 = True

BBP = True
BBP1 = True
BBP2 = True
BBP3 = True
BBP4 = True

BBP_TOTAL = BBP+BBP1+BBP2+BBP3+BBP4*4

# This will tell us the columns we are predicting
pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C', 'deg_pH10', 'deg_50C']



def convert_structure_to_bps(secstruct):

    bps = []

    left_delimiters = ['(','{','[']
    right_delimiters = [')','}',']']

    for (left_delim, right_delim) in list(zip(left_delimiters, right_delimiters)):

        left_list = []
        for i, char in enumerate(secstruct):
            if char == left_delim:
                left_list.append(i)

            elif char == right_delim:
                bps.append([left_list[-1],i])
                left_list = left_list[:-1]

        assert len(left_list)==0

    return bps

def secstruct_to_partner(secstruct):
    '''Convert secondary structure string to partner array.
    I.E. ((.)) -> [4,3,-1,1,0]
    '''
    bps = convert_structure_to_bps(secstruct)
    partner_vec = -1*np.ones([len(secstruct)]) 

    for (i,j) in bps:
        partner_vec[i] = j
        partner_vec[j] = i

    return partner_vec

def write_bprna_string(dbn_string):
    '''Input: dot-parenthesis string
    Output: bpRNA-style loop type assignments
    Author: H Wayment-Steele, 2021
    '''
    
    pair_partners = secstruct_to_partner(dbn_string)
    
    #print(pair_partners)
    bprna_string=['u']*len(dbn_string)

    # assign stems
    for s_ind, s in enumerate(dbn_string):
        if s != '.':
            bprna_string[s_ind] = 'S'
                
    # get loop regions
    
    while 'u' in ''.join(bprna_string):
        #print(''.join(bprna_string))

        obj = re.search(r"uu*", ''.join(bprna_string))
        start_ind, end_ind = obj.start(), obj.end()
        
        n_open_hps = dbn_string[:start_ind].count(')') - dbn_string[:start_ind].count('(')
        
        if n_open_hps == 0:
            bprna_string[start_ind:end_ind] = 'E'*(end_ind-start_ind)

        else:

            last_stem_pairing = int(pair_partners[start_ind - 1])
            next_stem_pairing = int(pair_partners[end_ind ])
            
            if last_stem_pairing == end_ind:
                bprna_string[start_ind:end_ind] = 'H'*(end_ind-start_ind)

            elif (last_stem_pairing - 1 == next_stem_pairing):
                bprna_string[start_ind:end_ind] = 'B'*(end_ind-start_ind)
                
            elif dbn_string[start_ind-1]==')' and dbn_string[end_ind]=='(':
                bprna_string[start_ind:end_ind] = 'M'*(end_ind-start_ind)
                
            else:
                if dbn_string[next_stem_pairing+1:last_stem_pairing] == '.'*(last_stem_pairing - next_stem_pairing-1):
                    bprna_string[start_ind:end_ind] = 'I'*(end_ind-start_ind)
                    bprna_string[next_stem_pairing+1:last_stem_pairing] = 'I'*(last_stem_pairing - next_stem_pairing-1)

                else:
                    bprna_string[start_ind:end_ind] = 'M'*(end_ind - start_ind)
    return ''.join(bprna_string)

def encode_input(sequence, bprna_string, window_size=1, pad=0):
    '''Creat input/output for regression model for predicting structure probing data.
    H Wayment-Steele, 2020
    Inputs:
    
    dataframe (in EternaBench RDAT format)
    window_size: size of window (in one direction). so window_size=1 is a total window size of 3
    pad: number of nucleotides at start to not include
    seq (bool): include sequence encoding
    struct (bool): include bpRNA structure encoding
    
    Outputs:
    Input array (n_samples x n_features): array of windowed input features
    feature_names (list, length = kernel x window): feature names, i.e. `S_-12`
    
    '''    
    inpts = []

    feature_kernel=['A','U','G','C','H','E','I','M','B','S', 'X']

    length = len(sequence)
    arr = np.zeros([length,len(feature_kernel)])
        
    for index in range(length):
        ctr=0
        for char in ['A','U','G','C']:
            if sequence[index]==char:
                arr[index,ctr]+=1
            ctr+=1

        for char in ['H','E','I','M','B','S', 'X']:
            if bprna_string[index]==char:
                arr[index,ctr]+=1
            ctr+=1

        # add zero padding to the side

    padded_arr = np.vstack([np.zeros([window_size,len(feature_kernel)]), arr, np.zeros([window_size,len(feature_kernel)])])

    for index in range(length):
        new_index = index+window_size-pad
        tmp = padded_arr[new_index-window_size:new_index+window_size+1]
        inpts.append(tmp.flatten())
            
    return np.array(inpts)

def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )


def calc_neighbor(d, dim, n):
    lst_x,lst_y = np.where(d==n)
    for c, x in enumerate(lst_x):
        y = lst_y[c]    
        if x+1<dim:
            d[x+1,y] = min(d[x+1,y], n+1)
        if y+1<dim:
            d[x,y+1] = min(d[x,y+1], n+1)
        if x-1>=0:
            d[x-1,y] = min(d[x-1,y], n+1)
        if y-1>=0:
            d[x,y-1] = min(d[x,y-1], n+1)
    return d
            

def get_distance_matrix_2d(Ss):
    Ds = []
    n = Ss.shape[0]
    dim = Ss.shape[1]
    for i in range(n):
        s = Ss[i,:,:,0]
        d = 10+np.zeros_like(s)
        d[s==1] = 1
        for i in range(dim):
            d[i,i] = 0
        for x in range(0, 9):
            d = calc_neighbor(d, dim, x)
        Ds.append(d)
    Ds =  np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[:, :,:, None]
    
    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    return Ds[:,:,:,:,0]


# loss functions
def MCRMSE(y_true, y_pred):
    colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=(1))
    return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)



class MSE(losses.MeanSquaredError):
    def __init__(self, *args, **kwargs):
        losses.MeanSquaredError.__init__(self, *args, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)
        
        temp = losses.MeanSquaredError.__call__(self, y_true[:, :, 0], y_pred[:, :, 0], sample_weight=None)
        temp = tf.sqrt(temp+1e-12)
        temp = tf.tensordot(temp,sample_weight,1)/tf.reduce_sum(sample_weight)
        s = temp*LOSS_WGTS[0]
#         s = tf.sqrt(temp)*LOSS_WGTS[0]
        for i in range(1,5):
            temp = losses.MeanSquaredError.__call__(self, y_true[:, :, i], y_pred[:, :, i], sample_weight=None)
            temp = tf.sqrt(temp+1e-12)
            temp = tf.tensordot(temp,sample_weight,1)/tf.reduce_sum(sample_weight)
#             s += tf.sqrt(temp)*LOSS_WGTS[i]
            s += (temp)*LOSS_WGTS[i]
            
        return s
    


def mean_squared_error1(y_true, y_pred, sample_weight):
    return np.sum((np.sqrt(np.mean((y_true-y_pred)**2, axis=1)))*sample_weight)/np.sum(sample_weight)

def MCRMSE_NAN_sample_wgt(y_true, y_pred, sample_weight=None, loss_cap=None):
    if loss_cap is not None:
        y_true_adj = np.minimum(np.maximum(y_true, y_pred-loss_cap), y_pred+loss_cap)
        return MCRMSE_NAN_sample_wgt(y_true_adj, y_pred, sample_weight=sample_weight, loss_cap=None)
    
    y_wgt = tf.ones_like(y_true)
    y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)

    s = (mean_squared_error1(y_true[:, :, 0], y_pred[:, :, 0], sample_weight=sample_weight)/(tf.reduce_mean(y_wgt[:,:, 0])))*LOSS_WGTS[0]
    for i in range(1,5):
        s += (mean_squared_error1(y_true[:, :, i], y_pred[:, :, i], sample_weight=sample_weight)/(tf.reduce_mean(y_wgt[:,:, i])))*LOSS_WGTS[i]
    return s

def MCRMSE_NAN_sample_wgt_single(y_true, y_pred, sample_weight=None, loss_cap=None):
    if loss_cap is not None:
        y_true_adj = np.minimum(np.maximum(y_true, y_pred-loss_cap), y_pred+loss_cap)
        return MCRMSE_NAN_sample_wgt_single(y_true_adj, y_pred, sample_weight=sample_weight, loss_cap=None)
        
    y_wgt = tf.ones_like(y_true)
    y_true = tf.where(tf.math.is_nan(y_true), y_pred, y_true)

    s = (mean_squared_error1(y_true[:, :], y_pred[:, :], sample_weight=sample_weight)/(tf.reduce_mean(y_wgt[:,:])))
    return s


def MCRMSE_NAN(y_true, y_pred, wgt=LOSS_WGTS, loss_cap=None):
    return MCRMSE_NAN_sample_wgt(y_true, y_pred, sample_weight=tf.ones_like(y_true[:,0,0]), loss_cap=loss_cap)

# reverse inputs
def reverse_input(train_input):
    reverse = train_input[:, ::-1, :]
    return reverse

def reverse_BBP_3D(mat):
    return mat[:, ::-1, ::-1,:]

# from https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model

def gru_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.GRU(hidden_dim, dropout=dropout, return_sequences=True, kernel_initializer='orthogonal')
    )


def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(
              L.LSTM(hidden_dim,dropout=dropout, return_sequences=True,kernel_initializer = 'orthogonal'))

# from https://www.kaggle.com/ragnar123/wavenet-gru-baseline

def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2 ** i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters = filters, 
                               kernel_size = 1,
                               padding = 'same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(filters = filters,
                          kernel_size = kernel_size,
                          padding = 'same', 
                          activation = 'tanh', 
                          dilation_rate = dilation_rate)(x)
        sigm_out = tf.keras.layers.Conv1D(filters = filters,
                          kernel_size = kernel_size,
                          padding = 'same',
                          activation = 'sigmoid', 
                          dilation_rate = dilation_rate)(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters = filters,
                   kernel_size = 1,
                   padding = 'same')(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x

# main model edited from https://www.kaggle.com/mrkmakr/covid-ae-pretrain-gnn-attn-cnn

def attention(x_inner, x_outer, n_factor, dropout):
    x_Q =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_inner)
    x_K =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_V =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att

def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

def res(x, unit, kernel = 3, rate = 0.1):
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])

def forward(x, unit, kernel = 3, rate = 0.1):
    h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    h = L.LayerNormalization()(h)
    h = L.Dropout(rate)(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate)
    return h

def adj_attn(x, adj, unit, n = 2, rate = 0.1):
    x_a = x
    x_as = []
    for i in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a)
        x_as.append(x_a)
    if n == 1:
        x_a = x_as[0]
    else:
        x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a


def get_base(config, X_node, As, dim=None):
    node = tf.keras.Input(shape = (dim, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (dim, dim, As.shape[3]), name = "adj")
    
    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis = 3)([adj, adj_learned])
        
    xs = []
    xs.append(node)
    x1 = forward(node, 128*2, kernel = 3, rate = 0.1)
    x2 = forward(x1, 64*2, kernel = 6, rate = 0.1)
    x3 = forward(x2, 32*2, kernel = 15, rate = 0.1)
    x4 = forward(x3, 16*2, kernel = 30, rate = 0.1)
    x = L.Concatenate()([x1, x2, x3, x4])
    
    for unit in [64*2, 32*2]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate = 0)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel = 30)
        
        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)
        
    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    return model


def get_ae_model(base, config, dim=None):
    node = tf.keras.Input(shape = (dim, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (dim, dim, As.shape[3]), name = "adj")

    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64*2, rate = 0.2)
    p = L.Dense(X_node.shape[2], "sigmoid")(x)
    

    node_1 = tf.where((node>1-1e-8), node, tf.zeros_like(node))
    node_0 = tf.where((node<1e-8), node, tf.ones_like(node))
    node_float = tf.where((node<=1-1e-8)&(node>=1e-8), node, p) 
    
    loss = - tf.reduce_mean(20 * node_1 * tf.math.log(p + 1e-4) + (1 - node_0) * tf.math.log(1 - p + 1e-4) - 5*(node_float-p)**2)
    
    model = tf.keras.Model(inputs = [node, adj], outputs = [loss])
    
    opt = get_optimizer()
    model.compile(optimizer = opt, loss = lambda t, y : y)
    return model


def get_model(base, config, Diversity_type, X_node, As, dim=None):
    node = tf.keras.Input(shape = (dim, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (dim, dim, As.shape[3]), name = "adj")
    
    x = base([node, adj])
    if not Diversity_type in ['forward']:
        x = forward(x, 128*2, rate = 0.2)
    
    if Diversity_type == 'gru':
        x = gru_layer(128*2, dropout=0.2)(x)
    elif Diversity_type == 'lstm':
        x = lstm_layer(128*2, dropout=0.2)(x)
    elif Diversity_type == 'forward':
        x = forward(x, 128*4, kernel=5, rate = 0.1)
        x = forward(x, 128*4, kernel=3, rate = 0.1)
        x = forward(x, 128*4, kernel=1, rate = 0.1)
    elif Diversity_type == 'wave':
        dropout = 0.1
        x = wave_block(x, 16*2, 3, 12)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        x = wave_block(x, 32*2, 3, 8)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        x = wave_block(x, 64*2, 3, 4)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)

        x = wave_block(x, 128*2, 3, 1)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    
    
    x = x[:, 1:-1,:]
    x = L.Dense(5)(x)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    
    opt = get_optimizer()
    model.compile(optimizer = opt, loss = MSE(reduction=tf.keras.losses.Reduction.NONE))#mcrmse_loss)
    return model

def get_optimizer():
#     sgd = tf.keras.optimizers.SGD(0.05, momentum = 0.9, nesterov=True)
    adam = tf.optimizers.Adam()
#     radam = tfa.optimizers.RectifiedAdam()
#     lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)
#     swa = tfa.optimizers.SWA(adam)
    return adam

## sequence
def return_ohe(n, i):
    tmp = [0] * n
    tmp[i] = 1
    return tmp

def get_input(train):
    
    len_app = 28
    seq_app = 'AGCUAGCUAGCUAGCUAGCUAGCUAGCU'
    loop_app = 'SSSSMMMMIIIIBBBBHHHHEEEEXXXX'
    stru_app = '.'*len_app
    
    train = train.copy()
    train['sequence'] = train['sequence'].apply(lambda x: x+seq_app)
    train['bpRNA_string'] = train['bpRNA_string'].apply(lambda x: x+loop_app)
    train['structure'] = train['structure'].apply(lambda x: x+stru_app)
    
    mapping = {}
    vocab = ["A", "G", "C", "U", "s", "e"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], ['s']+list(x)+['e']))))

    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train["bpRNA_string"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    X_loop = np.concatenate([np.zeros((X_loop.shape[0], 1, X_loop.shape[2])), X_loop, np.zeros((X_loop.shape[0], 1, X_loop.shape[2]))], axis=1)
    

    mapping = {}
    vocab = [".", "(", ")"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    X_structure = np.concatenate([np.zeros((X_structure.shape[0], 1, X_structure.shape[2])), X_structure, np.zeros((X_structure.shape[0], 1, X_structure.shape[2]))], axis=1)
    
    
    X_node = np.concatenate([X_node, X_loop], axis = 2)
    
    ## interaction
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)
    vocab = sorted(set(a.flatten()))
    #print(vocab)
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis = 2)
    X_node = np.concatenate([X_node, ohes], axis = 2).astype(np.float32)
    
    X_node = np.concatenate([X_node[:, :(-len_app-1), :], X_node[:, -1, :][:, None,:]], axis=1)
    #print(X_node.shape)
    return X_node

# copy and edited from https://www.kaggle.com/xhlulu/openvaccine-simple-gru-model

token2int = {x:i for i, x in enumerate('().ACGUBEHIMSXse')}

def pandas_list_to_array(df):
    """
    Input: dataframe of shape (x, y), containing list of length l
    Return: np.array of shape (x, l, y)
    """
    
    return np.transpose(
        np.array(df.values.tolist()),
        (0, 2, 1)
    )

def get_pair_idx(arr, sft=0):
    n = len(arr)
    out = np.zeros((n))
    l = []
    for c, i in enumerate(arr):
        if i == '.':
            out[c] = c
        elif i == '(':
            l.append(c)
        else:
            temp = l.pop()
            if sft == 0:
                out[c] = temp
                out[temp] = c
            elif sft >= 1:
                out[c] = min(temp+sft, n-1)
                out[temp] = max(c-sft, 0)
            elif sft <= -1:
                out[c] = max(temp-sft, 0)
                out[temp] = min(c+sft, n-1)
    return out

def calc_dist_to_pair(struct):
    n = len(struct)
    out = np.zeros((n))+10000
    curr_dist = 10000
    for c,i in enumerate(struct):
        curr_dist += 1
        if i in ['(', ')']:
            out[c] = 1
            curr_dist = 0
        else:
            out[c] = min(out[c], curr_dist)
    curr_dist = 10000
    for c,i in enumerate(struct[::-1]):
        curr_dist += 1
        if i in ['(', ')']:
            out[n-1-c] = 0
            curr_dist = 0
        else:
            out[n-1-c] = min(out[n-1-c], curr_dist)
    return out


def calc_dist_to_single(struct):
    n = len(struct)
    out = np.zeros((n))+10000
    curr_dist = 10000
    for c,i in enumerate(struct):
        curr_dist += 1
        if i == '.':
            out[c] = 1
            curr_dist = 0
        else:
            out[c] = min(out[c], curr_dist)
    curr_dist = 10000
    for c,i in enumerate(struct[::-1]):
        curr_dist += 1
        if i == '.':
            out[n-1-c] = 0
            curr_dist = 0
        else:
            out[n-1-c] = min(out[n-1-c], curr_dist)
    return out


def preprocess_inputs1(df, token2int, cols=['sequence', 'structure', 'bpRNA_string']):
    return pandas_list_to_array(
        df[cols].applymap(lambda seq: [token2int[x] for x in 's'+seq+'e'])
    )
def preprocess_inputs(df, token2int, bp_matrix):
    dict_row_idx = {}

    train_inputs = preprocess_inputs1(df, token2int)
    new = np.zeros((train_inputs.shape[0], train_inputs.shape[1], len(token2int)))
    for layer in range(3):
        for i in range(len(token2int)):
            new[train_inputs[:, :, layer]==i, i]=1

    if BBP_TOTAL>=1:
        bbp =[]
        bbp1 =[]
        bbp2 = []
        bbp3 = []
        bbp4_0 = []
        bbp4_1 = []
        bbp4_2 = []
        bbp4_3 = []

        ids = df.id.values
        for c, i in enumerate(ids):

            probability = bp_matrix
            if BBP:
                bbp.append(probability.max(-1).tolist())
            if BBP1:
                bbp1.append((1-probability.sum(axis=1)).tolist())
            if BBP2:
                srt = np.sort(probability)
                bbp2.append((srt[:,-1] - srt[:, -2]).tolist())
            if BBP3:
                m_lst = probability.max(axis=0)
                argmax_lst = m_lst[np.argmax(probability, axis=0)]
                bbp3.append((argmax_lst-m_lst).tolist())
            if BBP4:
                pair_idx = get_pair_idx(df.structure.values[c]).astype(int)
                pij = probability[np.arange(len(pair_idx)),pair_idx]
                bbp4_0.append(pij.tolist())
                m_lst = probability.max(axis=0)
                bbp4_1.append((m_lst-pij).tolist())
                bbp4_2.append((m_lst[pair_idx]-pij).tolist())
                s_lst = probability.sum(axis=0)
                bbp4_3.append((s_lst[pair_idx]-pij).tolist())

        temp = np.zeros((train_inputs.shape[0], train_inputs.shape[1]))
        if BBP:
            temp[:, 1:-1] = np.array(bbp)
            dict_row_idx['BBP'] = new.shape[2]
            new = np.concatenate([new, temp[:, :,None]], axis=2)
        if BBP1:
            temp[:, 1:-1] = np.array(bbp1)
            dict_row_idx['BBP1'] = new.shape[2]
            new = np.concatenate([new, temp[:, :,None]], axis=2)
        if BBP2:
            temp[:, 1:-1] = np.array(bbp2)
            dict_row_idx['BBP2'] = new.shape[2]
            new = np.concatenate([new, temp[:, :,None]], axis=2)
        if BBP3:
            temp[:, 1:-1] = np.array(bbp3)
            dict_row_idx['BBP3'] = new.shape[2]
            new = np.concatenate([new, temp[:, :,None]], axis=2)
        if BBP4: 
            for cnt, b in enumerate([bbp4_0, bbp4_1, bbp4_2, bbp4_3]):
                dict_row_idx['BBP4_%s'%cnt] = new.shape[2]
                temp[:, 1:-1] = np.array(b)
                new = np.concatenate([new, temp[:, :,None]], axis=2)
            dict_row_idx['BBP4_ed'] = new.shape[2]

            
    if DIST_NEW:
        lst_dist = []
        lst_dist_sqrt = []
        ids = df.id.values
        for c, i in enumerate(ids):
            temp_dist = calc_dist_to_pair(df['structure'].values[c])+1
            lst_dist.append((1/temp_dist).tolist())
            lst_dist_sqrt.append((np.sqrt(1/temp_dist)).tolist())
        temp = np.zeros((train_inputs.shape[0], train_inputs.shape[1]))
        temp[:, 1:-1] = np.array(lst_dist)
        new = np.concatenate([new, temp[:, :,None]], axis=2)
        temp = np.zeros((train_inputs.shape[0], train_inputs.shape[1]))
        temp[:, 1:-1] = np.array(lst_dist_sqrt)
        new = np.concatenate([new, temp[:, :,None]], axis=2)
        
    if DIST_NEW2:
        lst_dist = []
        lst_dist_sqrt = []
        ids = df.id.values
        for c, i in enumerate(ids):
            temp_dist = calc_dist_to_single(df['structure'].values[c])+1
            lst_dist.append((1/temp_dist).tolist())
            lst_dist_sqrt.append((np.sqrt(1/temp_dist)).tolist())
        temp = np.zeros((train_inputs.shape[0], train_inputs.shape[1]))
        temp[:, 1:-1] = np.array(lst_dist)
        new = np.concatenate([new, temp[:, :,None]], axis=2)
        temp = np.zeros((train_inputs.shape[0], train_inputs.shape[1]))
        temp[:, 1:-1] = np.array(lst_dist_sqrt)
        new = np.concatenate([new, temp[:, :,None]], axis=2)
    

    return new[:,:,len(token2int):]

def get_structure_adj(train):
    Ss = []
    for i in (range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    new = np.zeros((Ss.shape[0], Ss.shape[1]+2, Ss.shape[2]+2, Ss.shape[3]))
    new[:, 1:-1, 1:-1, :] = Ss
    return new

def get_distance_matrix(As):
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    return Ds


def padding_2D(Ss):
    new = np.zeros((Ss.shape[0], Ss.shape[1]+2, Ss.shape[2]+2))
    new[:, 1:-1, 1:-1] = Ss
    return new

def get_inputs(df_temp, bp_matrix):
    

    X_node = get_input(df_temp).astype(np.float32)
    X_node_new = preprocess_inputs(df_temp, token2int, bp_matrix).astype(np.float32)
    X_node = np.concatenate([X_node, X_node_new], axis=2)
    del X_node_new


    As = [bp_matrix]
    
    As = np.array(As)
    As = padding_2D(As)
    Ss = get_structure_adj(df_temp).astype(np.float32)
    Ds = get_distance_matrix(As)
    DDs = get_distance_matrix_2d(Ss)
    As = np.concatenate([As[:,:,:,None],Ss, Ds, DDs], axis = 3).astype(np.float32)
    del Ss, Ds, DDs
    return X_node, As


def dict_maker(df_0, bp_matrix):
    
    dict_X = {}
    dict_A = {}
    for i in df_0.id.values:
        df_temp = df_0.loc[df_0.id == i]
        dict_X[i], dict_A[i] = get_inputs(df_temp, bp_matrix)
        
        
    return dict_X, dict_A


def nn_preds(df_0, bp_matrix, Diversity_type, wgts_dir):

    config = {}
    Diversity_type = Diversity_type #'lstm'
    wgts_dir = wgts_dir #'../../model_files/ov-v40032-wgts/'
    dict_X, dict_A = dict_maker(df_0, bp_matrix)
    X_node, As = dict_X[0], dict_A[0]
    
    base = get_base(config, X_node, As)
    model = get_model(base, config, Diversity_type, X_node, As)
    
    all_pred_dfs=[]
    
    for m in range(5):
        model.load_weights(wgts_dir+'model_%s.h5'%m)
        preds_ls = []
        for uid in df_0.id.values:
            X_node, As = dict_X[uid], dict_A[uid]
            out1 = model.predict([X_node, As])
            out2 = model.predict([reverse_input(X_node), reverse_BBP_3D(As)])[:,::-1,:]
            out = (out1+out2)/2
        
            single_pred = out[0]
            single_df = pd.DataFrame(single_pred, columns=pred_cols)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            preds_ls.append(single_df)
            del out1, out2, out, single_pred, single_df
        
        preds_df = pd.concat(preds_ls).set_index('id_seqpos')
        #preds_df.to_csv("sub_%s_%s.csv"%(Diversity_type, m))
        all_pred_dfs.append(preds_df)

        del preds_df, preds_ls
        gc.collect()
        gc.collect()


    del base, model
    gc.collect()

    K.clear_session()

    return all_pred_dfs
    
    
def get_preds_df(all_pred_dfs):

    preds_df_agg2 = all_pred_dfs[0]
    for df in all_pred_dfs[1:]:
        df[df<-0.5] = -0.5
        df[df>6] = 6
        preds_df_agg2 += df
    preds_df_agg2 = preds_df_agg2/len(all_pred_dfs)
    preds_df_agg2 = preds_df_agg2.reset_index()
    #return ','.join(map(str, preds_df_agg2['reactivity'].values))
    return preds_df_agg2 #['reactivity'].values
    ######
    
    # lst_pred = os.listdir()
    # lst_pred = sorted([x for x in lst_pred if x.startswith('sub_')])

    # preds_df_agg = pd.read_csv(lst_pred[0], index_col=0)
    # for n in lst_pred[1:]:
    #     pred_temp = pd.read_csv(n, index_col=0)
    #     pred_temp[pred_temp<-0.5] = -0.5
    #     pred_temp[pred_temp>6] = 6
    #     preds_df_agg += pred_temp
    # preds_df_agg = preds_df_agg/len(lst_pred)
    # preds_df_agg = preds_df_agg.reset_index()

    # for fil in lst_pred:
    #     os.remove(fil)
    
    # return preds_df_agg

def get_preds_string(preds_):
    
    
    predictions = [str(pred) for pred in preds_]
    
    predictions = ", ".join(predictions)
    
    return predictions

def _make_pred(sequence, output_feature,mfe_structure=None, bp_matrix=None):
 #encoding = feature_generation(sequence)
    if mfe_structure is None:
        mfe_structure = mfe(sequence, package='eternafold')
    #mfe_structure=mfe(sequence, package='contrafold',param_file='/Users/hwayment/das/github/EternaFold/parameters/EternaFoldParams.v1')

    bprna_string = write_bprna_string(mfe_structure)
    if bp_matrix is None:
        bp_matrix = bpps(sequence, package='eternafold')
    #bp_matrix=bpps(sequence, package='contrafold',param_file='/Users/hwayment/das/github/EternaFold/parameters/EternaFoldParams.v1')
    df = pd.DataFrame(data = [{'id': 0, 'sequence': sequence, 'bpRNA_string': bprna_string, 'structure': mfe_structure, 'seq_length': len(sequence)}])
    #df.sort_values(by='seq_length')
    print(df)

    all_dfs = []

    all_dfs.extend(nn_preds(df, bp_matrix, 'lstm', os.environ['KOV_PATH']+'/model_files/ov-v40032-wgts/'))
    all_dfs.extend(nn_preds(df, bp_matrix, 'gru', os.environ['KOV_PATH']+'/model_files/ov-v40131-wgts/'))
    all_dfs.extend(nn_preds(df, bp_matrix, 'forward', os.environ['KOV_PATH']+'/model_files/ov-v40237-wgts/'))
    all_dfs.extend(nn_preds(df, bp_matrix, 'wave', os.environ['KOV_PATH']+'/model_files/ov-v40334-wgts/'))


    preds_df = get_preds_df(all_dfs)
    predictions = preds_df[output_feature].values
    #predictions = get_preds_string(predictions)

    return predictions
        #predictions = bprna_string #get_predictions(encoding)



def make_preds(Lines, output_feature):

    all_preds = []
    
    for sequence in Lines:

        predictions = _make_pred(sequence)
        all_preds.append(predictions)
        
    return all_preds
        
def main(argv):
    inputfile = 'input.txt'
    outputfile = 'preds.txt'
    output_feature = 'deg_Mg_pH10'
   
    if len(sys.argv)==1:
        print('python nullrecurrent_inference.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    try:
        opts, args = getopt.getopt(argv,"hi:o:d:",["ifile=","ofile=", "deg="])
    except getopt.GetoptError:
        print('python nullrecurrent_inference.py -i <inputfile> -o <outputfile> --deg <deg_type, default Mg pH 10>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python nullrecurrent_inference.py -i <inputfile> -o <outputfile> --deg <deg_type, default Mg pH 10>')
            sys.exit()
        elif opt in ("-d", "--deg"):
            output_feature = arg
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            
            file1 = open(inputfile, 'r')

            with open(inputfile) as f:
                Lines = [line.rstrip() for line in f]
            
            print(Lines)
            df = pd.DataFrame({'sequence': Lines})
            #all_preds = make_preds(Lines, output_feature)
            all_preds = df.progress_apply(lambda row: _make_pred(row['sequence'], output_feature), axis=1)

        elif opt in ("-o", "--ofile"):
            outputfile = arg
            
            with open(outputfile, 'w') as f:
                for item in all_preds:
                    f.write("%s\n" % item)
                    
        
    print('Input file is', inputfile)
    print('Output file is', outputfile)
    print('Output feature is', output_feature)

if __name__ == "__main__":
    main(sys.argv[1:])
