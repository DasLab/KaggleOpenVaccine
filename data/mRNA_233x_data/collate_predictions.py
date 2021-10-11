import numpy as np
import pandas as pd

df = pd.read_csv('degradation_rates_081120_233x_stability_WITH_ERRORS.csv')

print(df.keys())

df['RT_PCR_length'] = df['RT_PCR_end_pos']-df['RT_PCR_start_pos']

df['k_deg_normalize'] = df['k_deg_per_hour']/df['RT_PCR_length']
df['k_deg_err_normalize'] = df['k_deg_err_per_hour']/df['RT_PCR_length']

df['length'] = [len(x) for x in df['RNA_sequence']]

df['SUP DegScore2.1 FULL'] = np.nansum(np.loadtxt('Degscore_2.1_flat_FULL_233x.csv',delimiter=','),axis=1)
df['SUP DegScore2.1 PCR'] = np.nansum(np.loadtxt('Degscore_2.1_flat_PCR_233x.csv',delimiter=','),axis=1)

df['SUP Vienna FULL'] = np.nansum(np.loadtxt('P_UNP_vienna_flat_FULL_233x.csv',delimiter=','),axis=1)
df['SUP Vienna PCR'] = np.nansum(np.loadtxt('P_UNP_vienna_flat_PCR_233x.csv',delimiter=','),axis=1)

df['SUP EternaFold FULL'] = np.nansum(np.loadtxt('P_UNP_eternafold_flat_FULL_233x.csv',delimiter=','),axis=1)
df['SUP EternaFold PCR'] = np.nansum(np.loadtxt('P_UNP_eternafold_flat_PCR_233x.csv',delimiter=','),axis=1)

df['SUP nullrecurrent FULL'] = np.nansum(np.loadtxt('nullrecurrent_FULL_233x.csv',delimiter=','),axis=1)
df['SUP nullrecurrent PCR'] = np.nansum(np.loadtxt('nullrecurrent_PCR_233x.csv',delimiter=','),axis=1)

df['SUP kazuki2 FULL'] = np.nansum(np.loadtxt('kazuki2_deg_Mg_pH10_flat_FULL_233x.csv',delimiter=','),axis=1)
df['SUP kazuki2 PCR'] = np.nansum( np.loadtxt('kazuki2_deg_Mg_pH10_flat_PCR_233x.csv',delimiter=','),axis=1)

df['SUP Degscore-XGB FULL'] = np.nansum(np.loadtxt('Degscore-XGB_FULL_233x.csv',delimiter=','),axis=1)
df['SUP Degscore-XGB PCR'] = np.nansum(np.loadtxt('Degscore-XGB_PCR_233x.csv',delimiter=','),axis=1)

df['SUP nr_k2_ensembled FULL'] = df.apply(lambda row: 0.5*(row['SUP nullrecurrent FULL']+row['SUP kazuki2 FULL']), axis=1)
df['SUP nr_k2_ensembled PCR'] = df.apply(lambda row: 0.5*(row['SUP nullrecurrent PCR']+row['SUP kazuki2 PCR']), axis=1)

predictor_list = ['Vienna', 'EternaFold', 'DegScore2.1', 'Tunguz','nullrecurrent','kazuki2', 'nr_k2_ensembled']

for pred in predictor_list:
	df['AUP %s PCR'% pred] = df['SUP %s PCR'%pred]/df['RT_PCR_length']
	df['AUP %s FULL'%pred] = df['SUP %s FULL'%pred]/df['length']

df = df.loc[df['single_exp_fit_ok']==1]
df = df.loc[df['k_deg_per_hour']>0]

for typ in ['SUP','AUP']:
	for typ2 in ['FULL','PCR']:
		lst = ['k_deg_per_hour', 'k_deg_normalize']
		for pred in predictor_list:
			lst.append("%s %s %s" % (typ, pred, typ2))
		print(df[lst].corr())
		#print(df[lst+['Expt type']].groupby('Expt type').corr())

df.to_csv('all_collated_predictions_233x.csv',index=False)

