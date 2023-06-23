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
adata = ad.read_h5ad('coding.h5ad')
protein = pd.read_csv('normal_tissue.tsv',sep='\t')
valid_level = set(['High','Medium','Low','Not detected'])
protein = protein.loc[protein['Level'].isin(valid_level),:]
from itertools import combinations
pairs = list(combinations(ensgs,2))


## build pair version of adata input
adata = adata[ensgs,:]
adata_list = []
for pair in tqdm(pairs,total=len(pairs)):
    no1, no2 = pair
    adata1 = adata[no1,:]
    adata2 = adata[no2,:]
    x1 = adata1.X.toarray()
    x2 = adata2.X.toarray()
    x12 = np.concatenate([x1,x2],axis=0).min(axis=0)[np.newaxis,:]
    mean_ = x12.mean()
    symbol_ = ','.join([ensg2symbol[no1],ensg2symbol[no2]])
    adata12 = ad.AnnData(X=x12,var=adata.var,obs=pd.DataFrame(index=[','.join([no1,no2])],data={'mean':[mean_],'symbol':[symbol_]}))
    adata_list.append(adata12)
adata_final = ad.concat(adata_list,axis=0,join='inner',merge='first')
adata_final.X = csr_matrix(adata_final.X)
adata_final.write('adata_final.h5ad')

## build pair version of protein input
dic = {}
ensgs_set = set(ensgs)
for gene,sub_df in protein.groupby(by='Gene'):
    if gene in ensgs_set:
        dic[gene] = sub_df


correspondance = {'High':0,
                  'Medium':1,
                  'Low':2,
                  'Not detected':3}
reverse_correspondance = {v:k for k,v in correspondance.items()}
mapping_vec = np.vectorize(correspondance.get)

pair_df_list = []
for pair in tqdm(pairs,total=len(pairs)):
    no1, no2 = pair
    no1_df = dic[no1]
    no2_df = dic[no2]
    com_df = pd.concat([no1_df,no2_df],axis=0)
    data = []
    for t,sub_df in com_df.groupby(by='Tissue'):
        tmp = list(sub_df.groupby(by='Gene'))   # [(ensg,sub_df),(ensg,sub_df)]
        if len(tmp) == 1:
            continue
        dfs = [item.reset_index() for item in list(zip(*tmp))[1]]
        tmps = dfs[0].join(other=dfs[1],how='inner',lsuffix='_1',rsuffix='_2')
        try:
            v = mapping_vec(tmps.loc[:,['Level_1','Level_2']].values).max(axis=1)
        except:
            print(tmps.loc[:,['Level_1','Level_2']].values)
            raise Exception
        for item in v:
            add = (','.join([no1,no2]),','.join([ensg2symbol[no1],ensg2symbol[no2]]),t,reverse_correspondance[item])
            data.append(add)
    pair_df = pd.DataFrame.from_records(data,columns=['Gene','Gene name','Tissue','Level'])
    pair_df_list.append(pair_df)
final_protein = pd.concat(pair_df_list,axis=0).reset_index()
final_protein.to_csv('final_protein.txt',sep='\t',index=None)









