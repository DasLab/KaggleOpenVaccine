import numpy as np
import pandas as pd
import gzip

df = pd.read_csv('../degradation_rates_081120_233x_stability_WITH_ERRORS.csv')

print(df.keys())
output_array = np.ones([len(df),max([len(x) for x in df['RNA_sequence']])])*np.NaN

for pkg in ['vienna','eternafold']:
	output_array = np.ones([len(df),max([len(x) for x in df['RNA_sequence']])])*np.NaN
	output_array_pcr = np.ones([len(df),max([len(x) for x in df['RNA_sequence']])])*np.NaN

	for i,row in df.iterrows():

		# from cached bp matrices
	    # f = gzip.GzipFile('../bpps_%s_233x_hkws/%s.npy.gz' % (pkg, row['Barcode']), "r")
	    # tmp = np.load(f)
	    
	    bp_matrix = bpps(row['RNA_sequence'], package=pkg)
	    p_unp = 1 - np.sum(bp_matrix, axis=0)
	    output_array[i,:len(p_unp)] = p_unp
	    output_array_pcr[i,row['startpos']:row['endpos']] = p_unp[row['startpos']:row['endpos']]

	np.savetxt('P_UNP_%s_flat_FULL_233x.csv' % pkg, output_array,delimiter=',')
	np.savetxt('P_UNP_%s_flat_PCR_233x.csv' % pkg, output_array_pcr,delimiter=',')

