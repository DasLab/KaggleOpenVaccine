import sys, getopt, os

import numpy as np
import re
from arnie.pfunc import pfunc
from arnie.free_energy import free_energy
from arnie.bpps import bpps
from arnie.mfe import mfe
import arnie.utils as utils
from decimal import Decimal
import ipynb
from xgboost import XGBRegressor
import xgboost as xgb
print(xgb.__version__)

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
    '''
    H Wayment Steele 2021.

    Input: dot-parenthesis string
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

def encode_input(sequence, bprna_string, window_size=1, pad=0):
    '''Creat input/output for regression model for predicting structure probing data.
    H Wayment-Steele 2020.

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

def feature_generation(sequence):
    
    mfe_structure = mfe(sequence)
    
    bprna_string = write_bprna_string(mfe_structure)
    
    encoding = encode_input(sequence, bprna_string, 20, 0)
    
    return(encoding)

def get_predictions(encoding):
    
    reg = XGBRegressor(n_estimators=8200, tree_method='hist', learning_rate=0.005, max_depth=7, subsample=0.8, colsample_bytree=0.9, reg_alpha=0.005)
    reg.load_model(os.environ['KOV_PATH']+'/model_files/bt_xgb/bt_xgb.model')
    
    predictions = reg.predict(encoding)
    
    predictions = [str(pred) for pred in predictions]
    
    predictions = ", ".join(predictions)
    
    return predictions


def make_preds(Lines):

    all_preds = []
    
    for sequence in Lines:
        encoding = feature_generation(sequence)
        predictions = get_predictions(encoding)
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
            print('Usage: python degscore_xgboost_inference.py -i <inputfile> -o <outputfile>')
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