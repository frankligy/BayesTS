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




def model_A(A):
    # A must be n_sample * n_pairs
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))

    a = torch.tensor(4.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_a = pyro.sample('lambda_a',dist.Gamma(a,b))

    with pyro.poutine.scale(scale=1), pyro.plate('data_A',na,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        lfc = pyro.sample('lfc',dist.Normal(-sigma*lambda_a,1).expand([subsample_size,n]).to_event(1),obs=A.index_select(0,ind))

    return {'lfc':lfc}

def guide_A(A):
    alpha = pyro.param('alpha',lambda:torch.tensor(prior_alpha,device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(prior_beta,device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))

    a = torch.tensor(4.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_a = pyro.sample('lambda_a',dist.Gamma(a,b))

    return {'sigma':sigma}


def model_B(B):
    pass

def guide_B(B):
    pass

def model_C(C):
    pass

def guide_C(C):
    pass

def model(A,B,C,wa,wb,wc):
    pass

def guide(A,B,C,wa,wb,wc):
    pass


def train_single(model,guide,name,*args):
    adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
    clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
    elbo = Trace_ELBO()

    n_steps = 1000
    pyro.clear_param_store()
    svi = SVI(model,guide,adam,loss=Trace_ELBO())
    losses = []
    for step in tqdm(range(n_steps),total=n_steps):  
        loss = svi.step(*args)
        losses.append(loss)
    plt.figure(figsize=(5, 2))
    plt.plot(losses)
    plt.xlabel("SVI step")
    plt.ylabel("ELBO loss")
    plt.savefig('elbo_loss_train_single_{}.pdf'.format(name),bbox_inches='tight')
    plt.close()

    largest10 = np.sort(losses)[-10:]
    # return the scale of this modality
    return np.median(largest10)


##################################################
global prior_alpha,prior_beta,subsample_size,device,na,nb,nc,n

n = 3321
prior_alpha = np.full(3321,0)
prior_beta = np.full(3321,0)
subsample_size = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
na = 100
nb = 49
nc = 100

prior = pd.read_csv('full_results_XY_pair_post.txt',sep='\t',index_col=0)
A = pd.read_csv('A_input.txt',sep='\t',index_col=0)
B = pd.read_csv('B_input.txt',sep='\t',index_col=0)
C = pd.read_csv('C_input.txt',sep='\t',index_col=0)

prior_eff = prior.loc[C.index,:]
A_eff = A.loc[C.index,:]
B_eff = B.loc[C.index,:]
C_eff = C

prior_alpha = prior_eff['alpha'].values
prior_beta = prior_eff['beta'].values
A_eff = torch.tensor(A_eff.T.values,device=device)
B_eff = torch.tensor(B_eff.T.values,device=device)
C_eff = torch.tensor(C_eff.T.values,device=device)

sa = train_single(model_A,guide_A,'A',A_eff)

