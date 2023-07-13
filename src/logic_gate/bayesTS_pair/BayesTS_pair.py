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



# generate BayesTS score for each pair in GBM

### 1. first re-run BayesTS to get the XY version

### 2. add gene symbol to XY a
def ensemblgene_to_symbol(query,species):  # paste from the scTriangulate package
    # assume query is a list, will also return a list
    import mygene
    mg = mygene.MyGeneInfo()
    out = mg.querymany(query,scopes='ensemblgene',fileds='symbol',species=species,returnall=True,as_dataframe=True,df_index=True)
    result = out['out']['symbol'].fillna('unknown_gene').tolist()
    try:
        assert len(query) == len(result)
    except AssertionError:    # have duplicate results
        df = out['out']
        df_unique = df.loc[~df.index.duplicated(),:]
        result = df_unique['symbol'].fillna('unknown_gene').tolist()
    return result

# xy = pd.read_csv('rerun_bayesTS_XY/full_results.txt',sep='\t',index_col=0)
# symbols = ensemblgene_to_symbol(query=xy.index.tolist(),species='human')
# xy['symbols'] = symbols
# xy.to_csv('full_results_XY.txt',sep='\t')


### 3. finalize the gene pair in GBM
from ast import literal_eval
output = pd.read_csv('output.csv',index_col=0)
output.index = [literal_eval (item) for item in output.index]
target = set()
for item in output.index:
    target1,target2 = item
    for t in [target1,target2]:
        if not t.startswith('MT-'):
            target.add(t)
target = list(target)   # to gprofiler to change to ensembl, remove three NPIPB5 ambiguity
lut = pd.read_csv('180_target_gprofile_convert.csv',index_col=0)['converted_alias'].to_dict()
target_ensg = [lut[item] for item in target]
human_membrane = set(pd.read_csv('human_membrane_proteins_acc2ens.txt',sep='\t',index_col=0)['Ens'].tolist())
col = [True if item in human_membrane else False for item in target_ensg]
sanity_check = pd.DataFrame(data={'target':target,'ensg':target_ensg,'in_db':col})
# sanity_check.to_csv('membrane_to_cell_membrane.txt',sep='\t',index=None)
final_target = sanity_check.loc[sanity_check['in_db'],:].reset_index(drop=True)  # validate in GEPIA, 85 in total, over 5TPM in GBM tumor
from itertools import combinations
singles = final_target['ensg'].tolist()
pairs = list(combinations(singles,2))  # 3570
ensg2symbol = pd.Series(index=final_target['ensg'].tolist(),data=final_target['target'].tolist()).to_dict()

# ### 4. get BayesTS posterior beta for those
# adata = ad.read_h5ad('coding.h5ad')
# adata = adata[singles,:]  # 85 × 3644
# adata_list = []
# for pair in tqdm(pairs,total=len(pairs)):
#     no1, no2 = pair
#     adata1 = adata[no1,:]
#     adata2 = adata[no2,:]
#     x1 = adata1.X.toarray()
#     x2 = adata2.X.toarray()
#     x12 = np.concatenate([x1,x2],axis=0).min(axis=0)[np.newaxis,:]
#     mean_ = x12.mean()
#     symbol_ = ','.join([ensg2symbol[no1],ensg2symbol[no2]])
#     adata12 = ad.AnnData(X=x12,var=adata.var,obs=pd.DataFrame(index=[','.join([no1,no2])],data={'mean':[mean_],'symbol':[symbol_]}))
#     adata_list.append(adata12)
# adata_final = ad.concat(adata_list,axis=0,join='inner',merge='first')
# adata_final.X = csr_matrix(adata_final.X)
# adata_final.write('adata_final.h5ad')  # 3570 × 3644

# ### 5. add gene symbol to the full_result
# result = pd.read_csv('run_bayesTS_pairs/full_results.txt',sep='\t',index_col=0)
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
# result.to_csv('full_results_XY_pair_post.txt',sep='\t')

# generate samples for other three metrics



# build and train the model



# visualizing and post-inspection
