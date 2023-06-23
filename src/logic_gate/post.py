#!/data/salomonis2/LabFiles/Frank-Li/neoantigen/revision/ts/bayesian/pytorch_pyro_mamba_env/bin/python3.7

import anndata as ad  # need to install from -c bioconda, not -c conda-forge, but still fail (install 0.6.2 instead), finally use pip to solve (0.8.0)
import numpy as np
import pandas as pd
import os,sys
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import torch
import pyro
import pickle
import pyro.poutine as poutine
import pyro.distributions as dist
import pyro.distributions.constraints as constraints
from pyro.infer import SVI,Trace_ELBO
from pyro.optim import Adam,ClippedAdam
from kneed import KneeLocator
from scipy.sparse import csr_matrix
from pyro.poutine import scale
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr, spearmanr
import numpy.ma as ma
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import argparse

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# read in 54 CAR-T targets with all information
target = pd.read_csv('targets_full.txt',sep='\t',index_col=0)
ensg2symbol = target['gene'].to_dict()
ensgs = target.index.tolist()

# try to build gene-pair input
adata = ad.read_h5ad('../coding.h5ad')
protein = pd.read_csv('../normal_tissue.tsv',sep='\t')
valid_level = set(['High','Medium','Low','Not detected'])
protein = protein.loc[protein['Level'].isin(valid_level),:]
from itertools import combinations
pairs = list(combinations(ensgs,2))

# read results
result = pd.read_csv('full_results.txt',sep='\t',index_col=0)
col = []
for item in result.index:
    no1,no2 = item.split(',')
    gene1 = ensg2symbol[no1]
    gene2 = ensg2symbol[no2]
    col.append(','.join([gene1,gene2]))
result['gene_symbols'] = col

# compare to their respective single arm
db = pd.read_csv('../full_results_XYZ.txt',sep='\t',index_col=0)
gene2sigma = db['mean_sigma'].to_dict()
col1 = []
col2 = []
col3 = []
col4 = []
for item,sigma in zip(result.index.values,result['mean_sigma']):
    t1,t2 = item.split(',')
    sigma1 = gene2sigma[t1]
    sigma2 = gene2sigma[t2]
    delta1 = sigma - sigma1
    delta2 = sigma - sigma2
    col1.append(sigma1)
    col2.append(sigma2)
    col3.append(delta1)
    col4.append(delta2)
result['sigma1'] = col1
result['sigma2'] = col2
result['delta1'] = col3
result['delta2'] = col4
# result.to_csv('full_results_post.txt',sep='\t')

# visualize it
from gtex_viewer import *
adata_single = ad.read_h5ad('../coding.h5ad')
adata_double = ad.read_h5ad('adata_final.h5ad')
def visualize_pair(uids,outdir):
    ylim = (0,30)
    figsize = (6.4,1.5)
    norm = True
    uid1,uid2 = uids.split(',')
    gtex_viewer_configuration(adata_single)
    gtex_visual_combine_barplot(uid=uid1,ylim=ylim,facecolor='#D99066',figsize=figsize,norm=norm,outdir=outdir)
    gtex_visual_combine_barplot(uid=uid2,ylim=ylim,facecolor='#50BF80',figsize=figsize,norm=norm,outdir=outdir)
    gtex_viewer_configuration(adata_double)
    gtex_visual_combine_barplot(uid=uids,ylim=ylim,facecolor='#795D8C',figsize=figsize,norm=norm,outdir=outdir)

# visualize_pair(uids='ENSG00000240505,ENSG00000196083',outdir='inspection/ENSG00000240505,ENSG00000196083')
# valid_result = result.loc[(result['mean_sigma']<0.1) & (result['delta1']<-0.2) & (result['delta2']<-0.2),:]
# for item in valid_result.index:
#     visualize_pair(uids=item,outdir='inspection/{}'.format(item))

# heatmap
col1 = [item.split(',')[0] for item in result['gene_symbols']]
col2 = [item.split(',')[1] for item in result['gene_symbols']]
result['gene1'] = col1
result['gene2'] = col2
result_new = result.loc[:,['mean_sigma','gene1','gene2']]

# ERBB3 is always on the gene2 slot, so add it back to gene 1 as well
result_new_miss = result_new.loc[result_new['gene2']=='ERBB3',:]
result_new_miss_new = result_new_miss.copy()
result_new_miss_new['gene1'] = result_new_miss['gene2']
result_new_miss_new['gene2'] =  result_new_miss['gene1']

result_new = pd.concat([result_new,result_new_miss_new],axis=0)

df = result_new.groupby(by=['gene1','gene2'])['mean_sigma'].mean().unstack(fill_value=0)

u_indices = np.triu_indices(df.shape[0],k=1)
u_values = df.values[u_indices]
reshaped_u_indices = [(i,j) for i,j in zip(*u_indices)] 
new_reshaped_u_indices = []
for u_i,u_v in zip(reshaped_u_indices,u_values):
    if u_v > 0:
        new_reshaped_u_indices.append(u_i)
new_u_indices = list(zip(*new_reshaped_u_indices))
new_u_indices = tuple([np.array(item) for item in new_u_indices])
tmp = df.values.T
tmp[new_u_indices] = df.values[new_u_indices]
new = tmp.T
new_df = pd.DataFrame(data=new,index=df.index,columns=df.columns)

tmp = []
for ensg,sigma in gene2sigma.items():
    try:
        gs = ensg2symbol[ensg]
    except KeyError:
        continue
    tmp.append((gs,sigma))
tmp.sort(key=lambda x:x[1])
sorted_gene = list(list(zip(*tmp))[0])
new_df = new_df.loc[sorted_gene,sorted_gene]

fig,ax = plt.subplots()
sns.heatmap(new_df,mask=np.triu(new_df.values,k=0),cmap='viridis',xticklabels=1,yticklabels=1,ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=1)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=1)
plt.savefig('heatmap_lower.pdf',bbox_inches='tight')
plt.close()




