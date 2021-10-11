import numpy as np
import pandas as pd
import gzip
import sys
from DegScore import DegScore

df = pd.read_csv('../233x_sequences_degdata_081120.csv')

print(df.keys())

output_array = np.ones([len(df),max([len(x) for x in df['RNA_sequence']])])*np.NaN
output_array_pcr = np.ones([len(df),max([len(x) for x in df['RNA_sequence']])])*np.NaN

for i,row in df.iterrows():
	print("wrote", i)
	mdl = DegScore(row['RNA_sequence'])
	output_array[i,:len(row['RNA_sequence'])] = mdl.degscore_by_position
	output_array_pcr[i,row['startpos']:row['endpos']] = mdl.degscore_by_position[row['startpos']:row['endpos']]

np.savetxt('formatted_predictions/Degscore2.1_flat_FULL_233x.csv', output_array,delimiter=',')
np.savetxt('formatted_predictions/Degscore2.1_flat_PCR_233x.csv', output_array_pcr,delimiter=',')

