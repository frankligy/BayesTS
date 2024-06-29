#!/gpfs/data/yarmarkovichlab/Frank/BayesTS/logit_gate_env/bin/python3.7

import os,sys
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import pickle
import pyro.distributions as dist
import torch

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'




'''mutation category'''

df = pd.read_csv('Cosmic_MutantCensus_v99_GRCh38.tsv',sep='\t')
col0 = [] # genes
col1 = []  # number of samples
col2 = []  # number of unambiguous mutation id
col3 = []  # number of intron_variant
col4 = []  # number of synonymous_variant
col5 = []  # number of missense_variant
col6 = []  # number of stop_gained
col7 = []  # number of 3_prime_UTR_variant
col8 = []  # 5_prime_UTR_variant
col9 = [] # splice_region_variant
col10 = []  # splice_donor_variant
col11 = []  # splice_acceptor_variant
col12 = []  # inframe_deletion
col13 = []  # inframe_insertion
col14 = []  # stop_lost
for g,sub_df in df.groupby('GENE_SYMBOL'):
    col0.append(g)
    col1.append(len(sub_df['COSMIC_SAMPLE_ID'].unique()))
    sub_df2 = sub_df.dropna(subset=['MUTATION_DESCRIPTION'])
    col2.append(len(sub_df2['MUTATION_ID'].unique()))
    sub_df3 = sub_df2.drop_duplicates(subset='MUTATION_ID')
    vc = sub_df3['MUTATION_DESCRIPTION'].value_counts()
    col3.append(vc.loc[vc.index.to_series().str.contains('intron_variant')].sum())
    col4.append(vc.loc[vc.index.to_series().str.contains('synonymous_variant')].sum())
    col5.append(vc.loc[vc.index.to_series().str.contains('missense_variant')].sum())
    col6.append(vc.loc[vc.index.to_series().str.contains('stop_gained')].sum())
    col7.append(vc.loc[vc.index.to_series().str.contains('3_prime_UTR_variant')].sum())
    col8.append(vc.loc[vc.index.to_series().str.contains('5_prime_UTR_variant')].sum())
    col9.append(vc.loc[vc.index.to_series().str.contains('splice_region_variant')].sum())
    col10.append(vc.loc[vc.index.to_series().str.contains('splice_donor_variant')].sum())
    col11.append(vc.loc[vc.index.to_series().str.contains('splice_acceptor_variant')].sum())
    col12.append(vc.loc[vc.index.to_series().str.contains('inframe_deletion')].sum())
    col13.append(vc.loc[vc.index.to_series().str.contains('inframe_insertion')].sum())
    col14.append(vc.loc[vc.index.to_series().str.contains('stop_lost')].sum())
combined_df = pd.DataFrame(data={
    'number_of_samples':col1,
    'number_of_mutations':col2,
    'intron_variant':col3,
    'synonymous_variant':col4,
    'missense_variant':col5,
    'stop_gained':col6,
    '3_prime_UTR_variant':col7,
    '5_prime_UTR_variant':col8,
    'splice_region_variant':col9,
    'splice_donor_variant':col10,
    'splice_acceptor_variant':col11,
    'inframe_deletion':col12,
    'inframe_insertion':col13,
    'stop_lost':col14
},index=col0)



outdir = 'output_xyz'
ts = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
gene_lfc = pd.read_csv('gene_lfc.txt',sep='\t',index_col=0)
ensg2biotype = gene_lfc['biotype'].to_dict()
ensg2symbol = gene_lfc['gene_symbol'].to_dict()
ts['gene'] = ts.index.map(ensg2symbol)
ts = ts.loc[ts['gene'].notna(),:]

ts.set_index(keys='gene',inplace=True)
dic = ts['mean_sigma'].to_dict()
combined_df['BayesTS'] = combined_df.index.map(dic).values
combined_df_valid = combined_df.dropna(subset=['BayesTS'])

combined_df_valid.to_csv('correlation/mutation_df.txt',sep='\t')

# get relative version
tmp_df1 = combined_df_valid.iloc[:,:2]
tmp_df2 = combined_df_valid.iloc[:,2:-1]
tmp_df3 = combined_df_valid.iloc[:,-1:]
tmp_df2 = (tmp_df2.T / tmp_df2.sum(axis=1).values).T
combined_df_valid_rel = pd.concat([tmp_df1,tmp_df2,tmp_df3],axis=1)


# particular inspection
data = []
for y in combined_df_valid_rel.columns[2:-1]:
    s,p = pearsonr(combined_df_valid_rel['BayesTS'].values,combined_df_valid_rel[y].values)
    data.append((y,s,p))
stat = pd.DataFrame.from_records(data=data,columns=['variable','correlation','p-value'])
stat.to_csv('correlation/mutation_stat.txt',sep='\t',index=None)