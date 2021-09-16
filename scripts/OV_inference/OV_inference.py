import sys, getopt

arnie_path = '/home/tunguz/arnie'
home_path = '/home/tunguz/'

sys.path.append(arnie_path)
sys.path.append(home_path)

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
import plotly.express as px
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
import tensorflow as tf
from sklearn.model_selection import train_test_split,KFold, GroupKFold,StratifiedKFold

import tensorflow.keras as keras
import pandas as pd

import gc
import matplotlib.pyplot as plt
import os


from tqdm import tqdm

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
    Output: bpRNA-style loop type assignments'''
    
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


def make_preds(Lines):

    all_preds = []
    
    for sequence in Lines:
        #encoding = feature_generation(sequence)
        predictions = sequence #get_predictions(encoding)
        
        all_preds.append(predictions)
        
    return all_preds
        
        
def main(argv):
    inputfile = 'input.txt'
    outputfile = 'preds.txt'
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('python BT_inference.py -i <inputfile> -o <outputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: python BT_inference.py -i <inputfile> -o <outputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            
            file1 = open(inputfile, 'r')

            with open(inputfile) as f:
                Lines = [line.rstrip() for line in f]
            
            print(Lines)
            
            all_preds = make_preds(Lines)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
            
            with open(outputfile, 'w') as f:
                for item in all_preds:
                    f.write("%s\n" % item)
    print('Input file is', inputfile)
    print('Output file is', outputfile)
    
    

if __name__ == "__main__":
    main(sys.argv[1:])