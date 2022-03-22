import numpy as np
import pandas as pd
import gzip

df = pd.read_csv('233x_sequences_degdata_081120.csv')
fil='predictions/nullrecurrent_233x_output_14Oct2021.csv'

output_array = np.ones([233,1588])*np.nan

for i,x in enumerate(open(fil,'r').readlines()):
	dat = [float(k) for k in x.strip().split(',')]
	output_array[i,:len(dat)] = dat

np.savetxt('formatted_predictions/nullrecurrent_posthoc_hkws_FULL_233x.csv',output_array, delimiter=',')

for i, row in df.iterrows():
    if not np.isnan(row['RT_PCR_start_pos']):
        output_array[i, :int(row['RT_PCR_start_pos'])] = np.NaN
        output_array[i, int(row['RT_PCR_end_pos']):] = np.NaN
        
np.savetxt('formatted_predictions/nullrecurrent_posthoc_hkws_PCR_233x.csv',output_array, delimiter=',')
