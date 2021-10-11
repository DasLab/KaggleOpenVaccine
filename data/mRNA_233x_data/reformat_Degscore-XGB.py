import numpy as np
import pandas as pd
import gzip

df = pd.read_csv('../233x_sequences_degdata_081120.csv')
fil='predictions/Degscore_XGB_233_preds.csv'

output_array = np.ones([233,1588])*np.nan

for i,x in enumerate(open(fil,'r').readlines()):
	dat = [float(k) for k in x.strip().split(',')]
	output_array[i,:len(dat)] = dat

np.savetxt('formatted_predictions/Degscore-XGB_FULL_233x.csv',output_array, delimiter=',')

for i, row in df.iterrows():
    if not np.isnan(row['startpos']):
        output_array[i, :int(row['startpos'])] = np.NaN
        output_array[i, int(row['endpos']):] = np.NaN
        
np.savetxt('formatted_predictions/Degscore-XGB_PCR_233x.csv',output_array, delimiter=',')
