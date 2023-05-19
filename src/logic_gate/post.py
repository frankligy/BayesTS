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

# read results
result = pd.read_csv('full_results.txt',sep='\t',index_col=0)
col = []
for item in result.index:
    no1,no2 = item.split(',')
    gene1 = ensg2symbol[no1]
    gene2 = ensg2symbol[no2]
    col.append(','.join([gene1,gene2]))
result['gene_symbols'] = col
result.to_csv('full_results_post.txt',sep='\t')
