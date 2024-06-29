#!/gpfs/data/yarmarkovichlab/Frank/BayesTS/revision/enrich/gseapy_env/bin/python3.12

import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

outdir = 'output_xyz'
result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
gene_lfc = pd.read_csv('gene_lfc.txt',sep='\t',index_col=0)
ensg2biotype = gene_lfc['biotype'].to_dict()
ensg2symbol = gene_lfc['gene_symbol'].to_dict()

result['biotype'] = result.index.map(ensg2biotype).values
result['symbol'] = result.index.map(ensg2symbol).values

result = result.loc[result['biotype']=='protein_coding',:]
result = result.loc[result['symbol'].notna(),:]

outdir = 'enrich'
result.to_csv(os.path.join('enrich','result_xyz.txt'),sep='\t')

import gseapy as gp
# names = gp.get_library_name()
# with open('supported_libraries.txt','w') as f:
#     for name in names:
#         f.write('{}\n'.format(name))

dbs = ['ChEA_2022','KEGG_2021_Human','GO_Biological_Process_2023','GO_Cellular_Component_2023','GO_Molecular_Function_2023','HDSigDB_Human_2021','CellMarker_2024']
df = result.sort_values(by='mean_sigma',ascending=True)
df = pd.DataFrame({0:df['symbol'].tolist(),1:np.negative(df['mean_sigma'].values)}).set_index(keys=0)
pre_res = gp.prerank(rnk=df, 
                     gene_sets=dbs,
                     permutation_num=100,
                     outdir='enrich',
                     min_size=5,
                     max_size=1000,
                     seed=6,
                     no_plot=True,
                     verbose=True)   
final = pd.read_csv(os.path.join(outdir,'gseapy.gene_set.prerank.report.csv'),sep=',')
for db in dbs:
    final_sub = final.loc[final['Term'].str.startswith(db),:]
    final_sub.sort_values(by='NES',ascending=False,inplace=True)
    terms = final_sub['Term'].iloc[:3].tolist() + final_sub['Term'].iloc[-3:].tolist()
    pre_res.plot(terms=terms,show_ranking=True,ofname=os.path.join(outdir,'plot_{}.pdf'.format(db)))



