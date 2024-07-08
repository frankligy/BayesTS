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

N = 100
subsample_size = 10

def generate_and_configure(uids):
    deg = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/BayesTS/revision/deg.txt',sep='\t',index_col='Gene ID')
    ensg2lfc = deg['Log2(Fold Change)'].to_dict()
    CUSTOM = np.empty((len(ensg2lfc),N),dtype=np.float32)
    from scipy.stats import norm
    for i,lfc in enumerate(ensg2lfc.values()):
        CUSTOM[i,:] = norm.rvs(loc=lfc,scale=0.5,size=N)
    CUSTOM_df = pd.DataFrame(data=CUSTOM,index=list(ensg2lfc.keys()),columns=['sample_{}'.format(i+1) for i in range(N)])

    '''
    The CUSTOM_df should be a dataframe where index are ensg id, and columns are observations (any property) of this gene in a set of samples.
    In the context of incorporating the log fold change value, it will be lfc values, you can use normal expression to generate such samples when
    only singular value is available or you'd like to expand the sample size.

                lfc_obs_1      lfc_obs_2        lfc_obs_3      ...     lfc_obs_n
    ensg1
    ensg2
    ...
    ensg_n
    '''

    common = list(set(CUSTOM_df.index).intersection(set(uids)))
    order = [uids.index(item) for item in common]
    CUSTOM_df = CUSTOM_df.loc[common,:]
    CUSTOM = CUSTOM_df.values

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CUSTOM = torch.tensor(CUSTOM.T,device=device)   # 100 * 6266

    return CUSTOM, common, order, device

def model_custom(CUSTOM,device):

    n = CUSTOM.shape[1]

    a = torch.tensor(2.0, device=device)
    b = torch.tensor(2.0, device=device)
    sigma = pyro.sample('sigma', dist.Beta(a, b).expand([n]).to_event(1))

    a = torch.tensor(2.0, device=device)
    b = torch.tensor(1.0, device=device)
    beta_custom = pyro.sample('beta_custom', dist.Gamma(a, b))

    with pyro.poutine.scale(scale=1), pyro.plate('data_CUSTOM', N, subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        lfc = pyro.sample('lfc',
                          dist.Normal(-torch.log10(sigma) * beta_custom, 0.5).expand([subsample_size, n]).to_event(1),
                          obs=CUSTOM.index_select(0, ind))

    return {'lfc': lfc}

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
    with pyro.poutine.scale(scale=w_custom), pyro.plate('data_CUSTOM', N, subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        lfc = pyro.sample('lfc',
                          dist.Normal(-torch.log10(sigma) * beta_custom, 0.5).expand([subsample_size, n]).to_event(1),
                          obs=CUSTOM.index_select(0, ind))

    return {'c':c,'nc':nc,'lfc':lfc}



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



