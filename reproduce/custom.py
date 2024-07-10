#!/gpfs/data/yarmarkovichlab/Frank/BayesTS/logit_gate_env/bin/python3.7

import anndata as ad  
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
from scipy.sparse import csr_matrix
from pyro.poutine import scale
from scipy.stats import pearsonr, spearmanr
import numpy.ma as ma
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

global N,subsample_size

N = None   # used when it is just point estimate not a data distribution
subsample_size = 10   # do not exceed the total number of observations

def generate_and_configure(uids):

    tumor_erv = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/pan_cancer/atlas/DLBC/tumor_erv.txt',sep='\t',index_col=0)
    tumor_erv_aux = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/pan_cancer/atlas/DLBC/tumor_erv_aux_df.txt',sep='\t',index_col=0)
    erv = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/pan_cancer/atlas/DLBC/ERV.txt',sep='\t',index_col=0)

    total = 48
    f_cutoff = 0.5
    valid_erv = erv.loc[erv['n_sample']>total*f_cutoff,:].index
    tumor_erv = tumor_erv.loc[valid_erv,:]

    erv_data = tumor_erv.values / tumor_erv_aux['total_count'].values.reshape(1,-1) * 1e6

    CUSTOM_df = pd.DataFrame(data=erv_data,index=tumor_erv.index,columns=tumor_erv.columns)

    '''
    The CUSTOM_df should be a dataframe where index are ensg/feature id, and columns are observations (any property) of this gene in a set of samples.
    If incorporating a point estimate such as log fold change in this case, we randomly sample N data points as observations using normal distribution.
    If incorporating a set of observations, then it is naturally making sense.

                lfc_obs_1      lfc_obs_2        lfc_obs_3      ...     lfc_obs_n
    ensg1
    ensg2
    ...
    ensg_n
    '''

    common = list(set(CUSTOM_df.index).intersection(set(uids)))
    order = [uids.index(item) for item in tqdm(common)]
    CUSTOM_df = CUSTOM_df.loc[common,:]
    CUSTOM = CUSTOM_df.values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CUSTOM = torch.tensor(CUSTOM.T,device=device)   # n_obs * n_common_feature

    return CUSTOM, common, order, device

def model_custom(CUSTOM,device):

    '''
    Users need to define the data generation process to match the specific prior knowledge frmo sigma
    '''
    n = CUSTOM.shape[1]
    N = CUSTOM.shape[0]

    a = torch.tensor(2.0, device=device)
    b = torch.tensor(2.0, device=device)
    sigma = pyro.sample('sigma', dist.Beta(a, b).expand([n]).to_event(1))

    a = torch.tensor(2.0, device=device)
    b = torch.tensor(1.0, device=device)
    beta_custom = pyro.sample('beta_custom', dist.Gamma(a, b))

    with pyro.poutine.scale(scale=1), pyro.plate('data_CUSTOM', N, subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        custom = pyro.sample('custom',
                          dist.Normal(-torch.log10(sigma) * beta_custom, 0.5).expand([subsample_size, n]).to_event(1),
                          obs=CUSTOM.index_select(0, ind))

    return {'custom': custom}

def guide_custom(CUSTOM,device):

    n = CUSTOM.shape[1]

    alpha = pyro.param('alpha', lambda: torch.tensor(2.0, device=device), constraint=constraints.positive)
    beta = pyro.param('beta', lambda: torch.tensor(2.0, device=device), constraint=constraints.positive)
    sigma = pyro.sample('sigma', dist.Beta(alpha, beta).expand([n]).to_event(1))

    a = torch.tensor(2.0, device=device)
    b = torch.tensor(1.0, device=device)
    beta_custom = pyro.sample('beta_custom', dist.Gamma(a, b))

    return {'sigma': sigma}

def model_X_Y_custom(X,Y,CUSTOM,weights,ebayes_beta_y,t,s,device,w_x,w_y,w_custom,prior_alpha,prior_beta):

    '''CUSTOM'''
    n = CUSTOM.shape[1]


    a = torch.tensor(2.0, device=device)
    b = torch.tensor(1.0, device=device)
    beta_custom = pyro.sample('beta_custom', dist.Gamma(a, b))

    '''copy from model_X_Y'''
    # now X is counts referenced to 25, but here we need proportion
    constant = torch.tensor(25.,device=device)
    X = X / constant
    # now continue
    subsample_size = 10
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(ebayes_beta_y,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    a = torch.tensor(50.,device=device)
    total = pyro.sample('total',dist.Binomial(a,weights).expand([t]).to_event(1))
    scaled_X = torch.round(X * total.unsqueeze(-1))


    with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
    with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))

    '''CUSTOM'''
    N = CUSTOM.shape[0]
    with pyro.poutine.scale(scale=w_custom), pyro.plate('data_CUSTOM', N, subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        custom = pyro.sample('custom',
                          dist.Normal(-torch.log10(sigma) * beta_custom, 0.5).expand([subsample_size, n]).to_event(1),
                          obs=CUSTOM.index_select(0, ind))

    return {'c':c,'nc':nc,'custom':custom}



def guide_X_Y_custom(X,Y,CUSTOM,weights,ebayes_beta_y,t,s,device,w_x,w_y,w_custom,prior_alpha,prior_beta):

    '''CUSTOM'''
    n = CUSTOM.shape[1]

    a = torch.tensor(2.0, device=device)
    b = torch.tensor(1.0, device=device)
    beta_custom = pyro.sample('beta_custom', dist.Gamma(a, b))

    '''copy from model_X_Y'''
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,prior_alpha),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,prior_beta),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(ebayes_beta_y,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma}



