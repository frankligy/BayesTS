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
import math

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'


'''auxilary function'''
def compute_y(adata,uids,tpm=True):
    info = adata[uids,:]
    if tpm:
        y = info.X.toarray() 
    else:
        y = info.X.toarray() / adata.var['total_count'].values.reshape(1,-1)
    return y

def compute_scaled_x(adata,uids,cutoff,min_sample):
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= min_sample]
    x = np.zeros((len(uids),len(valid_tissue)))
    for i,tissue in enumerate(valid_tissue):
        sub = adata[uids,adata.var['tissue']==tissue]
        total_count = sub.shape[1]
        c = np.count_nonzero(np.where(sub.X.toarray()<=cutoff,0,sub.X.toarray()),axis=1)
        scaled_c = np.round(c * (25/total_count),0)
        x[:,i] = scaled_c
    annotated_x = pd.DataFrame(data=x,index=uids,columns=valid_tissue)
    annotated_x.to_csv(os.path.join(outdir,'annotated_x.txt'),sep='\t')
    return x,annotated_x


def get_thresholded_adata(adata,cond_Y):
    adata = adata.copy()
    adata.X = csr_matrix(np.where(cond_Y,adata.X.toarray(),0))
    return adata

def weighting(adata,dic,t,min_sample):
    weights = np.full(t,0.5)  # here the t will be number of valid_tissue
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= min_sample]
    for t,w in dic.items():
        try:
            i = valid_tissue.index(t)
        except ValueError:   # the tissue is not valid in RNA side, could be a typo or for protein weight
            continue
        weights[i] = w
    return weights



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



def compute_concordance(annotated_x,external):

    lookup = {
        'Adipose - Subcutaneous':'adipose tissue',
        'Muscle - Skeletal':None, 
        'Artery - Tibial':None, 
        'Artery - Coronary':None, 
        'Heart - Atrial Appendage':'heart muscle', 
        'Adipose - Visceral (Omentum)':'adipose tissue', 
        'Uterus':None, 
        'Vagina':'vagina', 
        'Breast - Mammary Tissue':'breast', 
        'Skin - Not Sun Exposed (Suprapubic)':'skin', 
        'Minor Salivary Gland':'salivary gland', 
        'Brain - Cortex':'cerebral cortex', 
        'Adrenal Gland':'adrenal gland', 
        'Thyroid':'thyroid gland', 
        'Lung':'lung', 
        'Spleen':'spleen', 
        'Pancreas':'pancreas', 
        'Esophagus - Muscularis':'esophagus', 
        'Esophagus - Mucosa':'esophagus', 
        'Esophagus - Gastroesophageal Junction':'esophagus', 
        'Stomach':'stomach', 
        'Colon - Sigmoid':'colon', 
        'Small Intestine - Terminal Ileum':'small intestine', 
        'Colon - Transverse':'colon', 
        'Prostate':'prostate', 
        'Testis':'testis', 
        'Nerve - Tibial':None, 
        'Skin - Sun Exposed (Lower leg)':'skin', 
        'Heart - Left Ventricle':'heart muscle', 
        'Brain - Cerebellum':'cerebellum', 
        'Whole Blood':'bone marrow', 
        'Artery - Aorta':None, 
        'Pituitary':'pituitary gland', 
        'Brain - Frontal Cortex (BA9)':None, 
        'Brain - Caudate (basal ganglia)':None, 
        'Brain - Nucleus accumbens (basal ganglia)':None, 
        'Brain - Putamen (basal ganglia)':None, 
        'Brain - Hypothalamus':'hypothalamus', 
        'Brain - Spinal cord (cervical c-1)':'spinal cord', 
        'Brain - Hippocampus':'hippocampal formation', 
        'Brain - Anterior cingulate cortex (BA24)':None, 
        'Ovary':'ovary', 
        'Brain - Cerebellar Hemisphere':None, 
        'Liver':'liver',
        'Brain - Substantia nigra':None, 
        'Kidney - Cortex':'kidney', 
        'Brain - Amygdala':'amygdala', 
        'Cervix - Endocervix':'cervix', 
        'Bladder':'urinary bladder',
    }

    valid_tissue_external = [v for k,v in lookup.items() if v is not None]
    valid_tissue_internal = [k for k,v in lookup.items() if v is not None]

    concordance_e = {}
    for gene, sub_df in external.groupby(by='Gene'):
        sub_df.set_index(keys='Tissue',inplace=True)
        try:
            e = sub_df.loc[valid_tissue_external,'nTPM'].values
        except KeyError:
            continue
        else:
            concordance_e[gene] = e
        
    concordance_i = {}
    for gene in annotated_x.index:
        i = annotated_x.loc[gene,valid_tissue_internal].values
        concordance_i[gene] = i


    common = list(set(concordance_e.keys()).intersection(set(concordance_i.keys())))

    spearman = []
    aupr = []
    for gene in common:
        ve = concordance_e[gene]
        vi = concordance_i[gene]
        # spearman
        value = spearmanr(ve,vi)[0]
        spearman.append(value)
        # aupr
        ve = np.where(ve>1,1,0)
        precision,recall,_ = precision_recall_curve(ve,vi,pos_label=1)
        value = auc(recall,precision)
        aupr.append(value)
    result = pd.DataFrame(data={'spearman':spearman,'aupr':aupr},index=common)
    return result

def compute_z(adata,protein,dic_weights):
    protein = protein.loc[protein['Level'].isin(['High','Medium','Low','Not detected']),:]
    n = []
    genes = []  # 13458
    for gene,sub_df in protein.groupby('Gene'):
        n.append(sub_df.shape[0])
        genes.append(gene)

    uids = adata.obs_names.tolist()
    common = list(set(uids).intersection(set(genes)))  # 13350
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
        for t,w in dic_weights.items():
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
    df.to_csv(os.path.join(outdir,'annotated_z.txt'),sep='\t')

    total = round(mean_n,0)
    def sample_category(x):
        # x is a series
        t = 0
        d = []
        for i,v in enumerate(x.values[:-1]): # {0:high,1:medium,2:low,3:not detected}, 3 is hold off
            n = math.floor(v)
            d.append(np.repeat([i],n))
            t += n
        if total-t <= 0:
            n_not_detected = 0
        else:
            n_not_detected = total-t
        d.append(np.repeat([3],n_not_detected))
        d = np.concatenate(d)
        np.random.shuffle(d)
        return d
    df = df.apply(sample_category,axis=1,result_type='expand')
    Z = df.values  
    return Z,adata,uids

'''model definition'''
def model_X_Y(X,Y,weights,ebayes_beta_y,train,w_x,w_y,prior_alpha,prior_beta):
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

    if train:
        with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
        with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))

    else:
        with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
        with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)

    return {'c':c,'nc':nc}

def guide_X_Y(X,Y,weights,ebayes_beta_y,train,w_x,w_y,prior_alpha,prior_beta):
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

def model_Y_Z(Y,Z,ebayes_beta_y,train,w_y,w_z,prior_alpha,prior_beta):

    # now continue
    subsample_size = 10
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(ebayes_beta_y,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))

    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 1./3. * (1-sigma)
    not_prob = 2./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:

        with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))
        with pyro.poutine.scale(scale=w_z), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:

        with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
        with pyro.poutine.scale(scale=w_z), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'nc':nc,'pc':pc}

def guide_Y_Z(Y,Z,ebayes_beta_y,train,w_y,w_z,prior_alpha,prior_beta):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,prior_alpha),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,prior_beta),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(ebayes_beta_y,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    return {'sigma':sigma}

def model_X_Z(X,Z,weights,train,w_x,w_z,prior_alpha,prior_beta):
    # now X is counts referenced to 25, but here we need proportion
    constant = torch.tensor(25.,device=device)
    X = X / constant
    # now continue
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
    low_prob = 1./3. * (1-sigma)
    not_prob = 2./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
        with pyro.poutine.scale(scale=w_z), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
        with pyro.poutine.scale(scale=w_z), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'c':c,'pc':pc}

def guide_X_Z(X,Z,weights,train,w_x,w_z,prior_alpha,prior_beta):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma}

def model_Y(Y,ebayes_beta_y,train,prior_alpha,prior_beta):
    subsample_size = 10
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    a = torch.tensor(ebayes_beta_y,device=device)
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

def guide_Y(Y,ebayes_beta_y,train,prior_alpha,prior_beta):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,prior_alpha),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,prior_beta),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(ebayes_beta_y,device=device)
    b = torch.tensor(1.,device=device)
    beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
    return {'sigma':sigma}  

def model_X(X,weights,train,prior_alpha,prior_beta):
    # now X is counts referenced to 25, but here we need proportion
    constant = torch.tensor(25.,device=device)
    X = X / constant
    # now continue
    subsample_size = 10
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
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

def guide_X(X,weights,train,prior_alpha,prior_beta):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,prior_alpha),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,prior_beta),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    a = torch.tensor(25.,device=device)
    b = torch.tensor(1.,device=device)
    beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
    total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
    return {'sigma':sigma} 

def model_Z(Z,train,prior_alpha,prior_beta):
    subsample_size = 10
    a = torch.tensor(prior_alpha,device=device)
    b = torch.tensor(prior_beta,device=device)
    sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 1./3. * (1-sigma)
    not_prob = 2./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'pc':pc}  

def guide_Z(Z,train,prior_alpha,prior_beta):
    alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,prior_alpha),device=device),constraint=constraints.positive)
    beta = pyro.param('beta',lambda:torch.tensor(np.full(n,prior_beta),device=device),constraint=constraints.positive)
    sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
    return {'sigma':sigma}


def model(X,Y,Z,weights,ebayes_beta_y,train,w_x,w_y,w_z,prior_alpha,prior_beta):
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
    high_prob = 2./3. * sigma
    medium_prob = 1./3. * sigma
    low_prob = 1./3. * (1-sigma)
    not_prob = 2./3. * (1-sigma)
    prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
    if train:
        with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([subsample_size,n]).to_event(1),obs=scaled_X.index_select(0,ind))
        with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([subsample_size,n]).to_event(1),obs=Y.index_select(0,ind))
        with pyro.poutine.scale(scale=w_z), pyro.plate('data_Z',p,subsample_size=subsample_size) as ind:
            ind = ind.to(device=device)
            pc = pyro.sample('pc',dist.Categorical(prob).expand([subsample_size,n]).to_event(1),obs=Z.index_select(0,ind))
    else:
        with pyro.poutine.scale(scale=w_x), pyro.plate('data_X',t):
            c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
        with pyro.poutine.scale(scale=w_y), pyro.plate('data_Y',s):
            nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
        with pyro.poutine.scale(scale=w_z), pyro.plate('data_Z',p):
            pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)
    return {'c':c,'nc':nc,'pc':pc}


def guide(X,Y,Z,weights,ebayes_beta_y,train,w_x,w_y,w_z,prior_alpha,prior_beta):
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





def test_and_graph_model(model,*args):
    trace = pyro.poutine.trace(model).get_trace(*args)
    trace.compute_log_prob()  
    print(trace.format_shapes())
    # pyro.render_model(model, model_args=(*args), render_distributions=True, render_params=True, filename='model.pdf')

def basic_configure(X,Y,Z,weights):
    # basic configure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X = torch.tensor(X.T,device=device)
    n = X.shape[1]
    t = X.shape[0]
    Y = torch.tensor(Y.T,device=device)
    s = Y.shape[0]
    
    if Z is not None:
        Z = torch.tensor(Z.T,device=device)
        n = Z.shape[1]
        p = Z.shape[0]
    else:
        Z = None
        p = None

    weights = torch.tensor(weights,device=device)

    return device,X,Y,Z,n,s,t,p,weights

def train_and_infer(model,guide,*args):

    adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
    clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
    elbo = Trace_ELBO()
    # train
    with pyro.plate('samples',1000,dim=-1):
        samples = guide(*args)
    svi_sigma = samples['sigma']  # torch.Size([1000, n])
    prior_sigma = np.nanmean(svi_sigma.data.cpu().numpy(),axis=0)   # it could be nan

    try:
        n_steps = epoch
        pyro.clear_param_store()
        svi = SVI(model, guide, adam, loss=Trace_ELBO())
        losses = []
        for step in tqdm(range(n_steps),total=n_steps):  
            loss = svi.step(*args)
            losses.append(loss)
        plt.figure(figsize=(5, 2))
        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("ELBO loss")
        plt.savefig(os.path.join(outdir,'elbo_loss.pdf'),bbox_inches='tight')
        plt.close()

        # also save the datafram
        loss_df = pd.DataFrame(data={'step':np.arange(n_steps)+1,'loss':losses})
        loss_df.to_csv(os.path.join(outdir,'loss_df.txt'),sep='\t',index=None)

        with pyro.plate('samples',1000,dim=-1):
            samples = guide(*args)
        svi_sigma = samples['sigma']  # torch.Size([1000, n])
        sigma = np.nanmean(svi_sigma.data.cpu().numpy(),axis=0)

        alpha = pyro.param('alpha').data.cpu().numpy()
        beta = pyro.param('beta').data.cpu().numpy()
        df = pd.DataFrame(index=uids,data={'mean_sigma':sigma,'alpha':alpha,'beta':beta,'prior_sigma':prior_sigma})
        with open(os.path.join(outdir,'X.p'),'rb') as f:
            X = pickle.load(f)
        with open(os.path.join(outdir,'Y.p'),'rb') as f:
            Y = pickle.load(f)
        try:
            with open(os.path.join(outdir,'Z.p'),'rb') as f:
                Z = pickle.load(f)
        except:
            Z = None
        Y_mean = Y.mean(axis=0)
        X_mean = X.mean(axis=0)
        if Z is not None:
            Z_mean = Z.mean(axis=0)
        else:
            Z_mean = np.arange(len(Y_mean))
        df['Y_mean'] = Y_mean
        df['X_mean'] = X_mean
        df['Z_mean'] = Z_mean

        # add a quantile
        df = df.sort_values(by='mean_sigma')
        df['percentile'] = [(i+1)/df.shape[0] for i in np.arange(df.shape[0])]

        df.to_csv(os.path.join(outdir,'full_results.txt'),sep='\t')
    except Exception as e:
        print(e)



def generate_inputs(adata,protein,dic):
    if protein is not None:
        Z,adata,uids = compute_z(adata,protein,dic)  # 13350 * 89
    else:
        adata = adata
        uids = adata.obs_names.tolist()
        Z = None

    Y = compute_y(adata,uids)  # 13350 * 1228

    # derive lambda using ebayes
    mean_each_gene = np.mean(Y,axis=1)
    quantiles = np.linspace(0,1,100)
    quantiles_of_mean = np.quantile(mean_each_gene,quantiles)
    y_var = quantiles_of_mean
    y_var = np.array([1e-5 if item == 0 else item for item in y_var])
    sigma = 0.5
    y_var_adjust = np.log(y_var) - (sigma ** 2 / 2)
    x_var = np.linspace(0,1,100)

    y_var_adjust = y_var_adjust[5:95].reshape(-1,1)
    x_var = x_var[5:95].reshape(-1,1)
    model = LinearRegression()
    model.fit(x_var,y_var_adjust)
    coefficient = model.coef_[0][0]
    intercept = model.intercept_[0]

    fig,ax = plt.subplots()
    ax.scatter(x=x_var.squeeze(),y=y_var_adjust.squeeze())
    ax.plot(x_var.squeeze(),[intercept + coefficient * item for item in x_var.squeeze()],c='black',lw=2)
    ax.text(x=0.5,y=0,s='y={}+{}x'.format(round(intercept,2),round(coefficient,2)))
    plt.savefig(os.path.join(outdir,'eBayes_determine_beta_y.pdf'),bbox_inches='tight')
    plt.close()

    ebayes_beta_y = coefficient

    # continue
    Y = np.where(Y==0,1e-5,Y)

    '''below is to find best cutoff for tissue distribution'''
    # external = pd.read_csv('rna_tissue_consensus.tsv',sep='\t')
    # for cutoff in np.arange(0,5,0.5):
    #     X, annotated_x = compute_scaled_x(adata,uids,cutoff)  # 13350 * 49
    #     result = compute_concordance(annotated_x,external)
    #     annotated_x.to_csv('annotated_x_{}.txt'.format(str(cutoff)),sep='\t')
    #     result.to_csv('result_{}.txt'.format(str(cutoff)),sep='\t')

    best_cutoff = cutoff
    X, annotated_x = compute_scaled_x(adata,uids,best_cutoff,min_sample)  # 13350 * 49

    n = X.shape[0]
    s = Y.shape[1]
    t = X.shape[1]
    weights = weighting(adata,dic,t,min_sample)
    

    return X,Y,Z,weights,uids,ebayes_beta_y




def train_single(model,guide,name,*args):
    adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
    clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
    elbo = Trace_ELBO()

    try:
        n_steps = epoch
        pyro.clear_param_store()
        svi = SVI(model, guide, adam, loss=Trace_ELBO())
        losses = []
        for step in tqdm(range(n_steps),total=n_steps):  
            loss = svi.step(*args)
            losses.append(loss)
        plt.figure(figsize=(5, 2))
        plt.plot(losses)
        plt.xlabel("SVI step")
        plt.ylabel("ELBO loss")
        plt.savefig(os.path.join(outdir,'elbo_loss_train_single_{}.pdf'.format(name)),bbox_inches='tight')
        plt.close()

        largest = np.sort(losses)[-10:]

        # write something to indicate its success
        with open(os.path.join(outdir,'train_single_{}_done'.format(name)),'w') as f:
            f.write('success')
        return np.median(largest)

    except Exception as e:
        print(e)

# diagnose
def diagnose_2d(ylim=(-1,200)):
    df = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
    fig,ax = plt.subplots()
    im = ax.scatter(df['X_mean'],df['Y_mean'],c=df['mean_sigma'],s=0.5**2,cmap='viridis')
    plt.colorbar(im)
    ax.set_ylabel('average_normalized_counts')
    ax.set_xlabel('average_n_present_samples_per_tissue')
    ax.set_ylim(ylim)
    plt.savefig(os.path.join(outdir,'diagnosis_2d.pdf'),bbox_inches='tight')
    plt.close()

def diagnose_3d():
    result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
    result = result.loc[result['Y_mean']<=1500,:]
    sigma = result['mean_sigma'].values
    Y_mean = MinMaxScaler().fit_transform(result['Y_mean'].values.reshape(-1,1)).squeeze()
    X_mean = MinMaxScaler().fit_transform(result['X_mean'].values.reshape(-1,1)).squeeze()
    Z_mean = MinMaxScaler().fit_transform(result['Z_mean'].values.reshape(-1,1)).squeeze()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    xs = X_mean
    ys = Y_mean
    zs = Z_mean
    ax.scatter(xs, ys, zs, marker='o',s=1,c=1-sigma)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig(os.path.join(outdir,'diagnosis_3d.pdf'),bbox_inches='tight')
    plt.close()

def cart_set54_evaluation(target_path):
    target = pd.read_csv(target_path,sep='\t',index_col=0)
    mapping = {e:g for g,e in target['Ensembl ID'].to_dict().items()}
    target = target.loc[target['Category']=='in clinical trials',:]['Ensembl ID'].tolist()
    result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
    target = list(set(result.index).intersection(set(target)))
    result = result.loc[target,:]
    result['gene'] = result.index.map(mapping).values
    result = result.sort_values(by='mean_sigma')
    result.to_csv(os.path.join(outdir,'cart_set54_targets.txt'),sep='\t')
    fig,ax = plt.subplots()
    ax.bar(x=np.arange(result.shape[0]),height=result['mean_sigma'].values)
    ax.set_xticks(np.arange(result.shape[0]))
    ax.set_xticklabels(result['gene'].values,fontsize=1,rotation=90)
    ax.set_ylabel('inferred sigma')
    plt.savefig(os.path.join(outdir,'cart_set54_targets_barplot.pdf'),bbox_inches='tight')
    plt.close()

    fig,ax = plt.subplots()
    result['Z_mean'] *= -1
    ax.imshow(MinMaxScaler().fit_transform(result.loc[:,['Y_mean','X_mean','Z_mean']].values).T)
    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['Y','X','Z'])
    plt.savefig(os.path.join(outdir,'cart_set54_targets_evidence.pdf'),bbox_inches='tight')
    plt.close()

def draw_PR(y_true,y_preds,outdir,outname):
    plt.figure()
    baseline = np.sum(np.array(y_true) == 1) / len(y_true)
    for label,y_pred in y_preds.items():
        precision,recall,_ = precision_recall_curve(y_true,y_pred,pos_label=1)
        area_PR = auc(recall,precision)
        lw = 1
        plt.plot(recall,precision,lw=lw, label='{} (area = {})'.format(label,round(area_PR,2)))
        plt.plot([0, 1], [baseline, baseline], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve example')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(outdir,outname),bbox_inches='tight')
    plt.close()


def benchmark_gs():
    result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
    gs = pd.read_csv('gold_standard.txt',sep='\t')['ensg'].values.tolist()
    result['label'] = [True if item in gs else False for item in result.index]

    hpa = pd.read_csv('proteinatlas.tsv',sep='\t').loc[:,['Ensembl','RNA tissue specificity score']]
    hpa = hpa.loc[hpa['RNA tissue specificity score'].notna(),:]
    mapping = {ensg:score for ensg,score in zip(hpa['Ensembl'].values,hpa['RNA tissue specificity score'].values)}
    result['specificity_score'] = result.index.map(mapping).values

    # against XYZ
    y_preds = {
        'BayesTS':np.negative(result['mean_sigma'].values),
        'RNA TPM':np.negative(result['Y_mean'].values),
        'tissue dist':np.negative(result['X_mean'].values),
        'protein stain':result['Z_mean'].values
    }
    draw_PR(result['label'].values,y_preds,outdir,'PR_curve_versus_xyz.pdf')

    # against hpa reported score
    result = result.loc[result['specificity_score'].notna(),:]
    y_preds = {
        'BayesTS':np.negative(result['mean_sigma'].values),
        'HPA reported specificity':result['specificity_score'].values,
    }
    draw_PR(result['label'].values,y_preds,outdir,'PR_curve_versus_hpa_specificity.pdf')

'''main program starts'''

def main(args):

    global adata,dic,n,s,t,p,device,uids,outdir,prior_alpha,prior_beta,protein,mode,cutoff,min_sample,epoch

    adata = ad.read_h5ad(args.input)
    dic = pd.read_csv(args.weight,sep='\t',index_col=0)['weight'].to_dict()
    outdir = args.outdir
    prior_alpha = args.prior_alpha
    prior_beta = args.prior_beta
    protein = args.protein
    mode = args.mode
    cutoff = args.noise  # this is the only tunable cutoffs that we carefully derive from empirical data
    min_sample = args.min_sample
    epoch = args.epoch

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if protein is not None:
        protein = pd.read_csv(protein,sep='\t')
    else:
        protein = None
    X,Y,Z,weights,uids,ebayes_beta_y = generate_inputs(adata,protein,dic)
    device,X,Y,Z,n,s,t,p,weights = basic_configure(X,Y,Z,weights)


    # check
    print('device:{}'.format(device))
    print('X:{} note: this is not percentage X shown in paper we will later scale it'.format(X.shape))
    print('Y:{}'.format(Y.shape))
    if Z is not None:
        print('Z:{}'.format(Z.shape))
    print('n:{}'.format(n))
    print('s:{}'.format(s))
    print('t:{}'.format(t))
    if Z is not None:
        print('p:{}'.format(p))
    print('weights:{}'.format(weights.shape))
    print('ebayes_beta_y:{}'.format(ebayes_beta_y))

    if mode == 'XYc':
        from custom import generate_and_configure, model_custom, guide_custom, model_X_Y_custom, guide_X_Y_custom
        # generate and configure input to CUSTOM
        CUSTOM, common, order, device = generate_and_configure(uids)
        uids = common
        subset_X = X[:,order]
        subset_Y = Y[:,order]
        n = len(common)
        s_x = train_single(model_X,guide_X,'X',subset_X,weights,True,prior_alpha,prior_beta) 
        while not os.path.exists(os.path.join(outdir,'train_single_X_done')):
            s_x = train_single(model_X,guide_X,'X',subset_X,weights,True,prior_alpha,prior_beta) 
        s_y = train_single(model_Y,guide_Y,'Y',subset_Y,ebayes_beta_y,True,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'train_single_Y_done')):
            s_y = train_single(model_Y,guide_Y,'Y',subset_Y,ebayes_beta_y,True,prior_alpha,prior_beta)
        s_custom = train_single(model_custom,guide_custom,'CUSTOM',CUSTOM,device)
        while not os.path.exists(os.path.join(outdir,'train_single_CUSTOM_done')):
            s_custom = train_single(model_custom,guide_custom,'CUSTOM',CUSTOM,device)
        
        # write out
        with open(os.path.join(outdir,'uids.p'),'wb') as f:
            pickle.dump(uids,f)
        with open(os.path.join(outdir,'X.p'),'wb') as f:
            pickle.dump(subset_X.data.cpu().numpy(),f)
        with open(os.path.join(outdir,'Y.p'),'wb') as f:
            pickle.dump(subset_Y.data.cpu().numpy(),f)
        with open(os.path.join(outdir,'Z.p'),'wb') as f:  # pretend CUSTOM to be Z
            pickle.dump(CUSTOM.data.cpu().numpy(),f)

        # actual infer
        lis = np.array([s_x,s_y,s_custom])
        small = lis.min()
        w_x = small / s_x
        w_y = small / s_y
        w_custom = small / s_custom
        print(lis,small,w_x,w_y,w_custom)
        # run
        pyro.clear_param_store()
        train_and_infer(model_X_Y_custom,guide_X_Y_custom,subset_X,subset_Y,CUSTOM,weights,ebayes_beta_y,t,s,device,w_x,w_y,w_custom,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model_X_Y_custom,guide_X_Y_custom,subset_X,subset_Y,CUSTOM,weights,ebayes_beta_y,t,s,device,w_x,w_y,w_custom,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()
        sys.exit('finished mode XYc')


    # derive w_x, w_y, w_z
    s_x = train_single(model_X,guide_X,'X',X,weights,True,prior_alpha,prior_beta)
    while not os.path.exists(os.path.join(outdir,'train_single_X_done')):
        s_x = train_single(model_X,guide_X,'X',X,weights,True,prior_alpha,prior_beta) 
    s_y = train_single(model_Y,guide_Y,'Y',Y,ebayes_beta_y,True,prior_alpha,prior_beta)
    while not os.path.exists(os.path.join(outdir,'train_single_Y_done')):
        s_y = train_single(model_Y,guide_Y,'Y',Y,ebayes_beta_y,True,prior_alpha,prior_beta)
    if Z is not None:
        s_z = train_single(model_Z,guide_Z,'Z',Z,True,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'train_single_Z_done')):
            s_z = train_single(model_Z,guide_Z,'Z',Z,True,prior_alpha,prior_beta)

    # pickle
    with open(os.path.join(outdir,'uids.p'),'wb') as f:
        pickle.dump(uids,f)
    with open(os.path.join(outdir,'X.p'),'wb') as f:
        pickle.dump(X.data.cpu().numpy(),f)
    with open(os.path.join(outdir,'Y.p'),'wb') as f:
        pickle.dump(Y.data.cpu().numpy(),f)
    if Z is not None:
        with open(os.path.join(outdir,'Z.p'),'wb') as f:
            pickle.dump(Z.data.cpu().numpy(),f)

    if mode == 'XYZ':
        lis = np.array([s_x,s_y,s_z])
        small = lis.min()
        w_x = small / s_x
        w_y = small / s_y
        w_z = small / s_z
        print(lis,small,w_x,w_y,w_z)
        # run
        pyro.clear_param_store()
        train_and_infer(model,guide,X,Y,Z,weights,ebayes_beta_y,True,w_x,w_y,w_z,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model,guide,X,Y,Z,weights,ebayes_beta_y,True,w_x,w_y,w_z,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()

    elif mode == 'XY':

        lis = np.array([s_x,s_y])
        small = lis.min()
        w_x = small / s_x
        w_y = small / s_y
        print(lis,small,w_x,w_y)
        # run
        pyro.clear_param_store()
        train_and_infer(model_X_Y,guide_X_Y,X,Y,weights,ebayes_beta_y,True,w_x,w_y,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model_X_Y,guide_X_Y,X,Y,weights,ebayes_beta_y,True,w_x,w_y,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()

    elif mode == 'YZ':

        lis = np.array([s_y,s_z])
        small = lis.min()
        w_y = small / s_y
        w_z = small / s_z
        print(lis,small,w_y,w_z)
        # run
        pyro.clear_param_store()
        train_and_infer(model_Y_Z,guide_Y_Z,Y,Z,ebayes_beta_y,True,w_y,w_z,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model_Y_Z,guide_Y_Z,Y,Z,ebayes_beta_y,True,w_y,w_z,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()

    elif mode == 'XZ':

        lis = np.array([s_x,s_z])
        small = lis.min()
        w_x = small / s_x
        w_z = small / s_z
        print(lis,small,w_x,w_z)
        # run
        pyro.clear_param_store()
        train_and_infer(model_X_Z,guide_X_Z,X,Z,weights,True,w_x,w_z,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model_X_Z,guide_X_Z,X,Z,weights,True,w_x,w_z,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()

    elif mode == 'X':
        pyro.clear_param_store()
        train_and_infer(model_X,guide_X,X,weights,True,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model_X,guide_X,X,weights,True,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()
    
    elif mode == 'Y':
        pyro.clear_param_store()
        train_and_infer(model_Y,guide_Y,Y,ebayes_beta_y,True,prior_alpha,prior_beta)
        while not os.path.exists(os.path.join(outdir,'elbo_loss.pdf')):
            pyro.clear_param_store()
            train_and_infer(model_Y,guide_Y,Y,ebayes_beta_y,True,prior_alpha,prior_beta)
        diagnose_2d()
        diagnose_3d()
        cart_set54_evaluation('cart_targets.txt')
        benchmark_gs()







if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BayesTS to retrain')
    parser.add_argument('--input',type=str,default='',help='path to the h5ad file')
    parser.add_argument('--weight',type=str,default='',help='path to a txt file with tissue and weights that you want to change')
    parser.add_argument('--mode',type=str,default='XYZ',help='XYZ use full model, XY use only RNA model')
    parser.add_argument('--protein',type=str,default=None,help='path to the protein info downloaded from Synapse')
    parser.add_argument('--outdir',type=str,default='.',help='path to the output directory')
    parser.add_argument('--prior_alpha',type=float,default=2.0,help='alpha for the beta prior')
    parser.add_argument('--prior_beta',type=float,default=2.0,help='beta for the beta prior')
    parser.add_argument('--noise',type=float,default=3.0,help='derived noise signal boundary from the data')
    parser.add_argument('--min_sample',type=int,default=10,help='only consider tissues with more than min_sample')
    parser.add_argument('--epoch',type=int,default=5000,help='total epochs to train')
    args = parser.parse_args()
    main(args)

    '''
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir output_sensitivity_82 --protein normal_tissue.tsv --prior_alpha 8.0 --prior_beta 2.0

    ./BayesTS_rev.py --input combined_normal_cpm.h5ad --weight weights.txt --mode Y --outdir output_splicing_y

    ./BayesTS_rev.py --input adata_final.h5ad --weight weights.txt --mode XYZ --protein final_protein.txt --outdir output_logic_gate_xyz
    '''













