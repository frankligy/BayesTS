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

# result = pd.read_csv('pair_full_results.txt',sep='\t',index_col=0)
# sanity_check = pd.read_csv('membrane_to_cell_membrane.txt',sep='\t')
# final_target = sanity_check.loc[sanity_check['in_db'],:].reset_index(drop=True)  
# from itertools import combinations
# singles = final_target['ensg'].tolist()
# pairs = list(combinations(singles,2))  # 3570
# ensg2symbol = pd.Series(index=final_target['ensg'].tolist(),data=final_target['target'].tolist()).to_dict()

# col = []
# for item in result.index:
#     no1,no2 = item.split(',')
#     gene1 = ensg2symbol[no1]
#     gene2 = ensg2symbol[no2]
#     col.append(','.join([gene1,gene2]))
# result['gene_symbols'] = col

# db = pd.read_csv('full_results_XY.txt',sep='\t',index_col=0)
# gene2sigma = db['mean_sigma'].to_dict()
# col1 = []
# col2 = []
# col3 = []
# col4 = []
# for item,sigma in zip(result.index.values,result['mean_sigma']):
#     t1,t2 = item.split(',')
#     sigma1 = gene2sigma[t1]
#     sigma2 = gene2sigma[t2]
#     delta1 = sigma - sigma1
#     delta2 = sigma - sigma2
#     col1.append(sigma1)
#     col2.append(sigma2)
#     col3.append(delta1)
#     col4.append(delta2)
# result['sigma1'] = col1
# result['sigma2'] = col2
# result['delta1'] = col3
# result['delta2'] = col4
# result_sort = result.sort_values(by='mean_sigma')
# result_sort['percentile'] = np.arange(result_sort.shape[0]) / result_sort.shape[0] * 100
# result_sort.to_csv('pair_full_results_post_sort.txt',sep='\t')

# # subset
# result = pd.read_csv('pair_full_results_post_sort.txt',sep='\t',index_col=0)
# result_subset = result.loc[(result['prior_sigma']<=0.2) & (result['A_mean']>0) & (result['B_mean']<0),:]
# result_subset.to_csv('pair_full_results_post_sort_subset.txt',sep='\t')


# visualize
uids = 'ENSG00000132432,ENSG00000157227'
outdir = 'inspection/{}'.format(uids.replace(',','_'))

### 1. specificity
from gtex_viewer import *
adata_double = ad.read_h5ad('adata_final.h5ad')
adata_single = ad.read_h5ad('coding.h5ad')

def make_order_same(x):
    lis = x.split(',')
    lis.sort()
    new_x = ','.join(lis)
    return new_x

def visualize_pair(uids,reverse,outdir):
    if reverse:
        tmp = uids.split(',')
        uids = ','.join([tmp[1],tmp[0]])
    ylim = (0,500)
    figsize = (6.4,1.5)
    norm = True
    uid1,uid2 = uids.split(',')
    gtex_viewer_configuration(adata_single)
    gtex_visual_combine_barplot(uid=uid1,ylim=ylim,facecolor='#D99066',figsize=figsize,norm=norm,outdir=outdir)
    gtex_visual_combine_barplot(uid=uid2,ylim=ylim,facecolor='#50BF80',figsize=figsize,norm=norm,outdir=outdir)
    gtex_viewer_configuration(adata_double)
    gtex_visual_combine_barplot(uid=uids,ylim=ylim,facecolor='#795D8C',figsize=figsize,norm=norm,outdir=outdir)
visualize_pair(uids=uids,reverse=False,outdir=outdir)

### 2. high expression in tumor, currently handled by GEPIA

### 3. dependency plot, currently handled by Depmap https://depmap.org/portal/gene/GPM6A?tab=overview

### 4. heterogeneity plot, handled by the single cell script

