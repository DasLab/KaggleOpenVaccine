import numpy as np
import pandas as pd
import gzip

input_file = 'predictions/2nd-place-233-seq.csv'
nickname='kazuki2'

df = pd.read_csv('../233x_sequences_degdata_081120.csv')
df1 = pd.read_csv(input_file)

df1['ID'] = [int(x.split('_')[0]) for x in df1['id_seqpos']]
df1['seqpos'] = [int(x.split('_')[1]) for x in df1['id_seqpos']]

df1['startpos'] = df['startpos']
df1['endpos'] = df['endpos']

for data_type in ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C','deg_50C']:
    output_array = np.ones([233,1588])*np.NaN

    for _, row in df1.iterrows():
        output_array[row['ID'],row['seqpos']] = row[data_type]

    np.savetxt('formatted_predictions/%s_%s_flat_FULL_233x.csv' % (nickname, data_type),output_array, delimiter=',')

    for i, row in df1.iterrows():
        if not np.isnan(row['startpos']):
            output_array[i, :int(row['startpos'])] = np.NaN
            output_array[i, int(row['endpos']):] = np.NaN
            
    np.savetxt('formatted_predictions/%s_%s_flat_PCR_233x.csv' % (nickname, data_type),output_array, delimiter=',')
