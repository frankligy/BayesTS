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
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))

    a = torch.tensor(2.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_b = pyro.sample('lambda_b',dist.Gamma(a,b))

    with pyro.poutine.scale(scale=1), pyro.plate('data_B',nb,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        dep = pyro.sample('dep',dist.Normal(torch.log10(sigma)*lambda_b,1).expand([subsample_size,n]).to_event(1),obs=B.index_select(0,ind))

    return {'dep':dep}


def guide_B(B):
    alpha = pyro.param('alpha',lambda:torch.tensor(prior_alpha,device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(prior_beta,device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))

    a = torch.tensor(2.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_b = pyro.sample('lambda_b',dist.Gamma(a,b))

    return {'sigma':sigma}

def model_C(C):
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))

    a = torch.tensor(3.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_c = pyro.sample('lambda_c',dist.Gamma(a,b))

    with pyro.poutine.scale(scale=1), pyro.plate('data_C',nc,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        cv = pyro.sample('cv',dist.Normal(sigma*lambda_c,1).expand([subsample_size,n]).to_event(1),obs=C.index_select(0,ind))

    return {'cv':cv}

def guide_C(C):
    alpha = pyro.param('alpha',lambda:torch.tensor(prior_alpha,device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(prior_beta,device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))

    a = torch.tensor(3.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_c = pyro.sample('lambda_c',dist.Gamma(a,b))

    return {'sigma':sigma}

def model(A,B,C,wa,wb,wc):
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))

    a = torch.tensor(4.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_a = pyro.sample('lambda_a',dist.Gamma(a,b))

    a = torch.tensor(2.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_b = pyro.sample('lambda_b',dist.Gamma(a,b))

    a = torch.tensor(3.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_c = pyro.sample('lambda_c',dist.Gamma(a,b))

    with pyro.poutine.scale(scale=wa), pyro.plate('data_A',na,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        lfc = pyro.sample('lfc',dist.Normal(-sigma*lambda_a,1).expand([subsample_size,n]).to_event(1),obs=A.index_select(0,ind))

    with pyro.poutine.scale(scale=wb), pyro.plate('data_B',nb,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        dep = pyro.sample('dep',dist.Normal(torch.log10(sigma)*lambda_b,1).expand([subsample_size,n]).to_event(1),obs=B.index_select(0,ind))

    with pyro.poutine.scale(scale=wc), pyro.plate('data_C',nc,subsample_size=subsample_size) as ind:
        ind = ind.to(device=device)
        cv = pyro.sample('cv',dist.Normal(sigma*lambda_c,1).expand([subsample_size,n]).to_event(1),obs=C.index_select(0,ind))

    return {'lfc':lfc,'dep':dep,'cv':cv}


def guide(A,B,C,wa,wb,wc):
    alpha = pyro.param('alpha',lambda:torch.tensor(prior_alpha,device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(prior_beta,device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))

    a = torch.tensor(4.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_a = pyro.sample('lambda_a',dist.Gamma(a,b))

    a = torch.tensor(2.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_b = pyro.sample('lambda_b',dist.Gamma(a,b))

    a = torch.tensor(3.0,device=device)
    b = torch.tensor(1.0,device=device)
    lambda_c = pyro.sample('lambda_c',dist.Gamma(a,b))

    return {'sigma':sigma}




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

def train_and_infer(model,guide,passA,passB,passC,*args):
    adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
    clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
    elbo = Trace_ELBO()

    # generate prior
    with pyro.plate('samples',1000,dim=-1):
        samples = guide(*args)
    svi_sigma = samples['sigma']
    prior_sigma = svi_sigma.data.cpu().numpy().mean(axis=0)

    # train
    n_steps = 5000
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
    plt.savefig('elbo_loss_train_and_infer.pdf',bbox_inches='tight')
    plt.close()

    # obtain results
    with pyro.plate('samples',1000,dim=-1):
        samples = guide(*args)
    svi_sigma = samples['sigma']  # torch.Size([1000, n])
    sigma = svi_sigma.data.cpu().numpy().mean(axis=0)
    alpha = pyro.param('alpha').data.cpu().numpy()
    beta = pyro.param('beta').data.cpu().numpy()
    df = pd.DataFrame(index=uids,data={'mean_sigma':sigma,'alpha':alpha,'beta':beta,'prior_sigma':prior_sigma})
    A_mean = passA.mean(axis=0)
    B_mean = passB.mean(axis=0)
    C_mean = passC.mean(axis=0)
    df['A_mean'] = A_mean
    df['B_mean'] = B_mean
    df['C_mean'] = C_mean
    df.to_csv('pair_full_results.txt',sep='\t')
    


##################################################
global prior_alpha,prior_beta,subsample_size,device,na,nb,nc,n,uids

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

uids = C_eff.index.tolist()
prior_alpha = prior_eff['alpha'].values
prior_beta = prior_eff['beta'].values
A_eff = torch.tensor(A_eff.T.values,device=device)
B_eff = torch.tensor(B_eff.T.values,device=device)
C_eff = torch.tensor(C_eff.T.values,device=device)

sa = train_single(model_A,guide_A,'A',A_eff)
sb = train_single(model_B,guide_B,'B',B_eff)
sc = train_single(model_C,guide_C,'C',C_eff)
lis = np.array([sa,sb,sc])
small = lis.min()
wa = small / sa
wb = small / sb
wc = small / sc
print(sa,sb,sc,lis,small,wa,wb,wc)

train_and_infer(model,guide,A_eff,B_eff,C_eff,A_eff,B_eff,C_eff,wa,wb,wc)





