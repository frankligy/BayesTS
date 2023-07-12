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

xy = pd.read_csv('rerun_bayesTS_XY/full_results.txt',sep='\t',index_col=0)
symbols = ensemblgene_to_symbol(query=xy.index.tolist(),species='human')
xy['symbols'] = symbols
xy.to_csv('full_results_XY.txt',sep='\t')
sys.exit('stop')


# generate samples for other three metrics



# build and train the model



# visualizing and post-inspection
