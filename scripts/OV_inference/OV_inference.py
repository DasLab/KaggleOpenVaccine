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