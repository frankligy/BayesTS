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

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

'''auxilary function'''
def compute_y(adata,uids):
    info = adata[uids,:]
    y = info.X.toarray() / adata.var['total_count'].values.reshape(1,-1)
    return y

def compute_scaled_x(adata,uids):
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= 10]
    x = np.zeros((len(uids),len(valid_tissue)))
    for i,tissue in enumerate(valid_tissue):
        sub = adata[uids,adata.var['tissue']==tissue]
        total_count = sub.shape[1]
        c = np.count_nonzero(np.where(sub.X.toarray()<1,0,sub.X.toarray()),axis=1)
        scaled_c = np.round(c * (25/total_count),0)
        x[:,i] = scaled_c
    return x

def compute_x(adata,uids):
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= 10]
    pd.Series(data=valid_tissue).to_csv('valid_tissue.txt',sep='\t')
    x = np.zeros((len(uids),len(valid_tissue)))
    for i,tissue in enumerate(valid_tissue):
        sub = adata[uids,adata.var['tissue']==tissue]
        total_count = sub.shape[1]
        c = np.count_nonzero(np.where(sub.X.toarray()<1,0,sub.X.toarray()),axis=1) / total_count
        x[:,i] = c
    return x

def get_thresholded_adata(adata,cond_Y):
    adata = adata.copy()
    adata.X = csr_matrix(np.where(cond_Y,adata.X.toarray(),0))
    return adata

def weighting(adata,dic,t):
    weights = np.full(t,0.5)
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= 10]
    for t,w in dic.items():
        try:
            i = valid_tissue.index(t)
        except ValueError:   # the tissue is not valid in RNA side, could be a typo or for protein weight
            continue
        weights[i] = w
    return weights

def diagnose(final_path,ylim=(-1,200),output_name='diagnosis.pdf'):
    df = pd.read_csv(final_path,sep='\t',index_col=0)
    fig,ax = plt.subplots()
    im = ax.scatter(df['X_mean'],df['Y_mean'],c=df['mean_sigma'],s=0.5**2,cmap='viridis')
    plt.colorbar(im)
    ax.set_ylabel('average_normalized_counts')
    ax.set_xlabel('average_n_present_samples_per_tissue')
    ax.set_ylim(ylim)
    plt.savefig(output_name,bbox_inches='tight')
    plt.close()

def thresholding_kneedle(cpm,plot=False,S=1,interp_method='polynomial'):
    x = cpm[cpm > 0]  # all non-zero values
    if len(x) <= 2:
        return 0
    else:
        actual_x = np.arange(len(x))
        actual_y = np.sort(x)
        kneedle = KneeLocator(actual_x,actual_y,S=S,curve='convex',direction='increasing',interp_method=interp_method)
        knee = kneedle.knee
        if knee is not None:
            knee_index = round(knee,0)
        else:
            knee_index = 0
        if plot:
            kneedle.plot_knee()
            plt.savefig('kneedle_data.pdf',bbox_inches='tight')
            plt.close()
            kneedle.plot_knee_normalized()
            plt.savefig('kneedle_norm.pdf',bbox_inches='tight')
            plt.close()
        return actual_y[knee_index]

def threshold(cpm,method,**kwargs):
    if method == 'kneedle':
        th = thresholding_kneedle(cpm,**kwargs)
    elif method == 'otsu':
        th = thresholding_otsu(cpm,**kwargs)
    elif method == 'gmm':
        th = thresholding_gmm(cpm,**kwargs)
    elif method == 'hardcode':
        th = thresholding_hardcode(cpm,**kwargs)
    cpm = np.where(cpm>th,cpm,0)
    cond = cpm>th
    return cpm, cond, th

def thresholding_hardcode(cpm,v):
    return v

def thresholding_otsu(cpm,step=0.05,dampen_factor=20):
    x = cpm[cpm > 0]  # all non-zero values
    criteria = []
    ths = np.arange(0,x.max(),step)
    for th in ths:
        thresholded_x = np.where(x>=th,1,0)
        w1 = np.count_nonzero(thresholded_x)/len(x)
        w0 = 1 - w1
        if w1 == 0 or w0 == 0:
            value = np.inf
        else:
            x1 = x[thresholded_x==1]
            x0 = x[thresholded_x==0]
            var1 = x1.var()
            var0 = x0.var()
            value = w0 * var0 + w1 * var1 / dampen_factor
        criteria.append(value)
    best_th = ths[np.argmin(criteria)]
    return best_th


def thresholding_gmm(cpm):
    x = cpm[cpm > 0]  # all non-zero values
    gm = GaussianMixture(n_components=2).fit(cpm.reshape(-1,1))
    means = gm.means_
    bg_index = np.argmin(means.mean(axis=1))
    best_th = means[bg_index,0]
    return best_th

def compute_concordance(uids, X,lookup,external,valid_tissue,method):
    ''' 
    X is n_gene * n_valid_tissue
    '''
    concordance = {}
    external_dic = {gene: sub_df for gene, sub_df in external.groupby(by='Gene')}   # {ensg:sub_df}
    external_ordered_tissue = {t:i for i, t in external_dic['ENSG00000121410']['Tissue'].reset_index(drop=True).to_dict().items()}  # {thyroid gland:0}
    external_dic = {gene: sub_df['nTPM'].values for gene, sub_df in external_dic.items()}
    lookup_dic = lookup.to_dict()  # {internal:external}
    valid_tissue = {t:i for i, t in valid_tissue.to_dict().items()}   # {thyroid: 4}
    avail_indices = []
    avail_external_indices = []
    for inter, exter in lookup_dic.items():
        avail_indices.append(valid_tissue[inter])
        avail_external_indices.append(external_ordered_tissue[exter])
    for i in tqdm(np.arange(X.shape[0]),total=X.shape[0]):
        gene = uids[i]
        v1 = X[i,avail_indices]
        try:
            v2 = external_dic[gene][avail_external_indices]
        except:
            continue
        if np.count_nonzero(v2) < len(v2) * 0.5 and np.count_nonzero(v1) > 0 and np.count_nonzero(v2) > 0:
            if method == 'pearson':
                value = pearsonr(v1,v2)[0]
            elif method == 'spearman':
                value = spearmanr(v1,v2)[0]
            elif method == 'AUPR':
                v2 = np.where(v2>0,1,0)
                precision,recall,_ = precision_recall_curve(v2,v1,pos_label=1)
                value = auc(recall,precision)
            concordance[gene] = value
    return concordance

def compute_z(adata,protein,dic):
    protein = protein.loc[protein['Level'].isin(['High','Medium','Low','Not detected']),:]
    n = []
    genes = []  # 13458
    for gene,sub_df in protein.groupby('Gene'):
        n.append(sub_df.shape[0])
        genes.append(gene)
    # sns.histplot(n)
    # plt.savefig('n_evidence.pdf',bbox_inches='tight')
    # plt.close()
    uids = adata.obs_names.tolist()
    common = list(set(uids).intersection(set(genes)))  # 13306
    adata = adata[common,:]
    uids = adata.obs_names.tolist()

    count = np.empty((len(uids),4),dtype=np.float32)
    dic = {gene: sub_df for gene,sub_df in protein.groupby('Gene')}
    for i,uid in tqdm(enumerate(uids),total=len(uids)):
        sub_df = dic[uid]
        level_values = []
        tissue2level = {p_tissue:sub_df2['Level'].values for p_tissue,sub_df2 in sub_df.groupby('Tissue')}
        # set up weight vector
        weights = {item:0.5 for item in tissue2level.keys()} 
        for t,w in dic.items():
            try:
                weights[t]
            except KeyError:
                continue
            else:
                weights[t] = w
        # let weight scale the distribution of annotations
        for t,w in weights.items():
            l = tissue2level[t]
            if w < 0.5:
                sf = w/0.5
                for level in ['High','Medium','Low']:
                    indices = np.where(l==level)[0]
                    replaced_v = np.random.choice([level,'Not detected'],len(indices),p=[sf,1-sf])
                    l[indices] = replaced_v
            elif w > 0.5:
                sf = w/0.5 - 1
                for level in ['Medium','Low','Not detected']:
                    indices = np.where(l==level)[0]
                    replaced_v = np.random.choice([level,'High'],len(indices),p=[1-sf,sf])
                    l[indices] = replaced_v
            elif w == 0.5:
                l = l
            level_values.extend(l)
        values, counts = np.unique(np.array(level_values),return_counts=True)  # ['High' 'Low' 'Medium' 'Not detected'] [ 4 20 26 34]
        mapping = {v:c for v,c in zip(values,counts)}
        count[i,0] = mapping.get('High',0)
        count[i,1] = mapping.get('Medium',0)
        count[i,2] = mapping.get('Low',0)
        count[i,3] = mapping.get('Not detected',0)
    df = pd.DataFrame(index=uids,data=count,columns=['High','Medium','Low','Not detected'])
    mean_n = np.array(n).mean()
    df = df.apply(lambda x:x/x.values.sum()*mean_n,axis=1)

    total = round(mean_n,0)
    def sample_category(x):
        # x is a series
        t = 0
        d = []
        for i,v in enumerate(x.values[:-1]): # {0:high,1:medium,2:low,3:not detected}, 3 is hold off
            n = round(v,0)
            d.append(np.repeat([i],n))
            t += n
        d.append(np.repeat([3],total-t))
        d = np.concatenate(d)
        np.random.shuffle(d)
        return d
    df = df.apply(sample_category,axis=1,result_type='expand')
    Z = df.values  
    return Z,adata,uids

'''model definition'''
def model_X_Y(X,Y,weights,train):
    subsample_size = 10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    a = torch.tensor(50.,device=device)
    total = pyro.sample('total',dist.Binomial(a,weights).expand([t]).to_event(1))
    scaled_X = torch.round(X * total.unsqueeze(-1))
    if train:
        with pyro.poutine.scale(scale=1/875), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
        with pyro.poutine.scale(scale=1/5000), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1/875), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
        with pyro.poutine.scale(scale=1/5000), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
    return {'c':c,'nc':nc}

def guide_X_Y(X,Y,weights,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma}


def model_Y_Z(Y,Z,train):
    subsample_size=10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 2./3. * (1-sigma)
    not_prob = 1./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=1/5000), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1/5000), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'nc':nc,'pc':pc}

def guide_Y_Z(Y,Z,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    return {'sigma':sigma}

def model_X_Z(X,Z,weights,train):
    subsample_size = 10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    a = torch.tensor(50.,device=device)
    total = pyro.sample('total',dist.Binomial(a,weights).expand([t]).to_event(1))
    scaled_X = torch.round(X * total.unsqueeze(-1))
    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 2./3. * (1-sigma)
    not_prob = 1./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=1/875), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1/875), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'c':c,'pc':pc}

def guide_X_Z(X,Z,weights,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma}

def model_Y(Y,train):
    subsample_size = 10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    if train:
        with pyro.poutine.scale(scale=1), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
    return {'nc':nc}   

def guide_Y(Y,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    return {'sigma':sigma}  

def model_X(X,weights,train):
    subsample_size = 10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    a = torch.tensor(50.,device=device)
    total = pyro.sample('total',dist.Binomial(a,weights).expand([t]).to_event(1))
    scaled_X = torch.round(X * total.unsqueeze(-1))
    if train:
        with pyro.poutine.scale(scale=1), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
    return {'c':c}

def guide_X(X,weights,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma} 

def model_Z(Z,train):
    subsample_size = 10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 2./3. * (1-sigma)
    not_prob = 1./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'pc':pc}  

def guide_Z(Z,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    return {'sigma':sigma}


def model(X,Y,Z,weights,train):
    subsample_size = 10
    a = torch.tensor(2.,device=device)
    b = torch.tensor(2.,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    a = torch.tensor(50.,device=device)
    total = pyro.sample('total',dist.Binomial(a,weights).expand([t]).to_event(1))
    scaled_X = torch.round(X * total.unsqueeze(-1))
    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 2./3. * (1-sigma)
    not_prob = 1./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=1/875), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
        with pyro.poutine.scale(scale=1/5000), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1/875), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
        with pyro.poutine.scale(scale=1/5000), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'c':c,'nc':nc,'pc':pc}


def guide(X,Y,Z,weights,train):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(10.,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma}


def prior_posterior_check(uid,full_result):
    index = uids.index(uid)
    prior_sigma = full_result.loc[uid,'prior_sigma']
    posterior_sigma = full_result.loc[uid,'mean_sigma']
    # nc
    beta_y = 10
    data = Y[:,index].data.cpu().numpy()
    prior = dist.LogNormal(beta_y*prior_sigma,0.5).expand([s]).sample().data.cpu().numpy()
    posterior = dist.LogNormal(beta_y*posterior_sigma,0.5).expand([s]).sample().data.cpu().numpy()
    fig,ax = plt.subplots()
    for item in [data,prior,posterior]:
        sns.histplot(item,ax=ax,stat='density',bins=40,alpha=0.5)
    ax.set_xlim([0,20000])
    ax.set_ylim([0,0.002])
    # axin = ax.inset_axes([0.5,0.5, 0.45, 0.45])
    # for item in [data,prior,posterior]:
    #     sns.histplot(item,ax=axin,stat='density',bins=40,alpha=0.5)
    # axin.set_xlim(-1, 10)
    # axin.set_ylim(0, 0.5)
    plt.savefig('{}_nc.pdf'.format(uid),bbox_inches='tight')
    plt.close()
    # c
    beta_x = 25
    data = X[:,index].data.cpu().numpy()
    prior = dist.Poisson(beta_x*prior_sigma).expand([t]).sample().data.cpu().numpy()
    posterior = dist.Poisson(beta_x*posterior_sigma).expand([t]).sample().data.cpu().numpy()
    fig,ax = plt.subplots()
    for item in [data,prior,posterior]:
        sns.histplot(item,ax=ax,stat='probability',alpha=0.5)
    # ax.set_xlim([-5,150])
    # ax.set_ylim([0,1])
    # axin = ax.inset_axes([0.5,0.5, 0.45, 0.45])
    # for item in [data,prior,posterior]:
    #     sns.histplot(item,ax=axin,stat='density',bins=40,alpha=0.5)
    # axin.set_xlim(-1, 10)
    # axin.set_ylim(0, 0.5)
    plt.savefig('{}_c.pdf'.format(uid),bbox_inches='tight')
    plt.close()
    # pc
    data = Z[:,index].data.cpu().numpy()
    probs = [torch.tensor([2/3*sigma,1/3*sigma,2/3*(1-sigma),1/3*(1-sigma)]) for sigma in [prior_sigma,posterior_sigma]]
    prior = dist.Categorical(probs[0]).expand([p]).sample().data.cpu().numpy()
    posterior = dist.Categorical(probs[1]).expand([p]).sample().data.cpu().numpy()
    fig,axes = plt.subplots(ncols=3,gridspec_kw={'wspace':0.5})
    axes = axes.flatten()
    for i,(item,color,title) in enumerate(zip([data,prior,posterior],['#7995C4','#80BE8E','#E5A37D'],['data','prior','posterior'])):
        sns.histplot(item,ax=axes[i],stat='count',bins=4,alpha=0.5,facecolor=color)
        axes[i].set_ylim([0,90])
        axes[i].set_title(title)
    plt.savefig('{}_pc.pdf'.format(uid),bbox_inches='tight')
    plt.close()



'''main program starts'''

adata = ad.read_h5ad('../gene/coding.h5ad')
protein = pd.read_csv('normal_tissue.tsv',sep='\t')
dic = {}
# Z,adata,uids = compute_z(adata,protein,dic)  # 13306 * 89
# Y = compute_y(adata,uids)  # 13306 * 3644
# Y = np.where(Y==0,1e-5,Y)
# X = compute_scaled_x(adata,uids)  # 13306 * 63
# thresholded_Y = np.empty_like(Y,dtype=np.float32)
# cond_Y = np.empty_like(Y,dtype=bool)
# ths = []
# for i in tqdm(range(Y.shape[0]),total=Y.shape[0]):
#     thresholded_Y[i,:], cond_Y[i,:], th = threshold(Y[i,:],'hardcode',v=0.8)
#     ths.append(th)
# new_adata = get_thresholded_adata(adata,cond_Y)
# raw_X = compute_x(new_adata,uids)
# n = X.shape[0]
# s = Y.shape[1]
# t = X.shape[1]
t = 63
weights = weighting(adata,dic,t)
# with open('uids.p','wb') as f:
#     pickle.dump(uids,f)
# with open('Z.p','wb') as f:
#     pickle.dump(Z,f)
# with open('Y.p','wb') as f:
#     pickle.dump(Y,f)
# with open('X.p','wb') as f:
#     pickle.dump(X,f)
# with open('raw_X.p','wb') as f:
#     pickle.dump(raw_X, f)

with open('uids.p','rb') as f:
    uids = pickle.load(f)
with open('Z.p','rb') as f:
    Z = pickle.load(f)
with open('Y.p','rb') as f:
    Y = pickle.load(f)
with open('X.p','rb') as f:
    X = pickle.load(f)
with open('raw_X.p','rb') as f:
    raw_X = pickle.load(f)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
X = torch.tensor(X.T,device=device)
Y = torch.tensor(Y.T,device=device)
Z = torch.tensor(Z.T,device=device)
n = X.shape[1]
s = Y.shape[0]
t = X.shape[0]
p = Z.shape[0]
weights = torch.tensor(weights,device=device)


adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
elbo = Trace_ELBO()

# trace = pyro.poutine.trace(model).get_trace(X,Y,Z,weights,True)
# trace.compute_log_prob()  
# print(trace.format_shapes())
# # pyro.render_model(model, model_args=(X,Y,Z,weights), render_distributions=True, render_params=True, filename='model.pdf')


# # train
# with pyro.plate('samples',1000,dim=-1):
#     samples = guide(X,Y,Z,weights,True)
# svi_sigma = samples['sigma']  # torch.Size([1000, n])
# prior_sigma = svi_sigma.data.cpu().numpy().mean(axis=0)

# n_steps = 5000
# pyro.clear_param_store()
# svi = SVI(model, guide, adam, loss=Trace_ELBO())
# losses = []
# for step in tqdm(range(n_steps),total=n_steps):  
#     loss = svi.step(X,Y,Z,weights,True)
#     losses.append(loss)
# plt.figure(figsize=(5, 2))
# plt.plot(losses)
# plt.xlabel("SVI step")
# plt.ylabel("ELBO loss")
# plt.savefig('elbo_loss.pdf',bbox_inches='tight')
# plt.close()


# with pyro.plate('samples',1000,dim=-1):
#     samples = guide(X,Y,Z,weights,True)
# svi_sigma = samples['sigma']  # torch.Size([1000, n])
# sigma = svi_sigma.data.cpu().numpy().mean(axis=0)
# alpha = pyro.param('alpha').data.cpu().numpy()
# beta = pyro.param('beta').data.cpu().numpy()
# df = pd.DataFrame(index=uids,data={'mean_sigma':sigma,'alpha':alpha,'beta':beta,'prior_sigma':prior_sigma})
# with open('X.p','rb') as f:
#     X = pickle.load(f)
# with open('Y.p','rb') as f:
#     Y = pickle.load(f)
# with open('Z.p','rb') as f:
#     Z = pickle.load(f)
# Y_mean = Y.mean(axis=1)
# X_mean = X.mean(axis=1)
# Z_mean = Z.mean(axis=1)
# df['Y_mean'] = Y_mean
# df['X_mean'] = X_mean
# df['Z_mean'] = Z_mean
# df.to_csv('full_results.txt',sep='\t')
# diagnose('full_results.txt',output_name='pyro_full_diagnosis.pdf')


'''evaluate'''
# prior and posterior check
full_result = pd.read_csv('full_results.txt',sep='\t',index_col=0)
# uid = 'ENSG00000198681'  # xlim([-5,150]) ylim([0,0.1])
uid = 'ENSG00000150991'  # xlim([0,20000]) ylim([0,0.002])
prior_posterior_check(uid,full_result)
sys.exit('stop')


# 3d plot
from sklearn.preprocessing import MinMaxScaler
result = pd.read_csv('full_results.txt',sep='\t',index_col=0)
sigma = result['mean_sigma'].values
Y_mean = MinMaxScaler().fit_transform(result['Y_mean'].values.reshape(-1,1)).squeeze()
X_mean = MinMaxScaler().fit_transform(result['X_mean'].values.reshape(-1,1)).squeeze()
Z_mean = MinMaxScaler().fit_transform(result['Z_mean'].values.reshape(-1,1)).squeeze()
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xs = X_mean
ys = Y_mean
zs = Z_mean
ax.scatter(xs, ys, zs, marker='o',s=1,c=sigma)
ax.set_xlabel('Tissue Distribution')
ax.set_ylabel('Count Evidence')
ax.set_zlabel('Protein Level')
ax.set_ylim([0,0.3])
ax.tick_params(length=0,labelsize=0,color='w',labelcolor='w')
fig.colorbar(cm.ScalarMappable(norm=None, cmap='viridis'), ax=ax)
plt.savefig('3d.pdf',bbox_inches='tight')
plt.close()


target = pd.read_csv('../gene/CARTargets.txt',sep='\t',index_col=0)
mapping = {e:g for g,e in target['Ensembl ID'].to_dict().items()}
target = target.loc[target['Category']=='in clinical trials',:]['Ensembl ID'].tolist()
result = pd.read_csv('full_results.txt',sep='\t',index_col=0)
target = list(set(result.index).intersection(set(target)))
result = result.loc[target,:]
result['gene'] = result.index.map(mapping).values
result = result.sort_values(by='mean_sigma')
result.to_csv('targets.txt',sep='\t')
fig,ax = plt.subplots()
ax.bar(x=np.arange(result.shape[0]),height=result['mean_sigma'].values)
ax.set_xticks(np.arange(result.shape[0]))
ax.set_xticklabels(result['gene'].values,fontsize=1,rotation=90)
ax.set_ylabel('inferred sigma')
ax.set_title('55 CAR-T Target in clinical trial')
plt.savefig('targets.pdf',bbox_inches='tight')
plt.close()

fig,ax = plt.subplots()
result['Z_mean'] *= -1
ax.imshow(MinMaxScaler().fit_transform(result.loc[:,['Y_mean','X_mean','Z_mean']].values).T)
ax.set_yticks([0,1,2])
ax.set_yticklabels(['Count Evidence','Tissue Distribution','Protein Level'])
ax.set_title('Evidence')
plt.savefig('evidence_targets.pdf',bbox_inches='tight')
plt.close()


