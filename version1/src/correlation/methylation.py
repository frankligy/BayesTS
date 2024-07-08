#!/data/salomonis2/LabFiles/Frank-Li/neoantigen/revision/ts/bayesian/pytorch_pyro_mamba_env/bin/python3.7

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr


df = pd.read_csv('Cosmic_CompleteDifferentialMethylation_Tsv_v98_GRCh38/Cosmic_CompleteDifferentialMethylation_v98_GRCh38.tsv',sep='\t')
df_valid = df.dropna(subset=['GENE_SYMBOL'])
df_valid = df_valid.loc[(df_valid['GENE_SYMBOL']!='Unclassified_Cell_type_specific')&(df_valid['GENE_SYMBOL']!='Unclassified')&(df_valid['GENE_SYMBOL']!='Promoter_Associated'),:]
df_valid['uid'] = [','.join([str(item1),str(item2),str(item3)]) for item1,item2,item3 in zip(df_valid['CHROMOSOME'],df_valid['POSITION'],df_valid['STRAND'])]
df_valid = df_valid.drop_duplicates(subset='uid')

col0 = []
col1 = []
col2 = []
for g,sub_df in df_valid.groupby(by='GENE_SYMBOL'):
    col0.append(g)
    vc_dic = sub_df['METHYLATION'].value_counts().to_dict()
    col1.append(vc_dic.get('H',0))
    col2.append(vc_dic.get('L',0))
combined_df = pd.DataFrame(data={'hypermethylation':col1,'hypomethylation':col2},index=col0)
combined_df = (combined_df.T / combined_df.sum(axis=1).values).T

ts = pd.read_csv('full_results_XYZ.txt',sep='\t',index_col=0)
ts.set_index(keys='gene',inplace=True)
dic = ts['mean_sigma'].to_dict()
combined_df['BayesTS'] = combined_df.index.map(dic).values
combined_df_valid = combined_df.dropna(subset=['BayesTS'])

for y in combined_df_valid.columns[:-1]:
    sns.jointplot(data=combined_df_valid,x='BayesTS',y=y,kind='reg',ylim=None)
    s,p = pearsonr(combined_df_valid['BayesTS'].values,combined_df_valid[y].values)
    print(y,s,p)




