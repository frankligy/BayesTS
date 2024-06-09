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
from sklearn.mixture import GaussianMixture
from scipy.stats import pearsonr, spearmanr
import numpy.ma as ma
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import argparse

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

def compute_scaled_x(adata,uids,cutoff):
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= 10]
    x = np.zeros((len(uids),len(valid_tissue)))
    for i,tissue in enumerate(valid_tissue):
        sub = adata[uids,adata.var['tissue']==tissue]
        total_count = sub.shape[1]
        c = np.count_nonzero(np.where(sub.X.toarray()<=cutoff,0,sub.X.toarray()),axis=1)
        scaled_c = np.round(c * (25/total_count),0)
        x[:,i] = scaled_c
    annotated_x = pd.DataFrame(data=x,index=uids,columns=valid_tissue)
    return x,annotated_x

def compute_x(adata,uids,cutoff):
    total_tissue = adata.var['tissue'].unique()
    valid_tissue = [tissue for tissue in total_tissue if adata[:,adata.var['tissue']==tissue].shape[1] >= 10]
    x = np.zeros((len(uids),len(valid_tissue)))
    for i,tissue in enumerate(valid_tissue):
        sub = adata[uids,adata.var['tissue']==tissue]
        total_count = sub.shape[1]
        c = np.count_nonzero(np.where(sub.X.toarray()<=cutoff,0,sub.X.toarray()),axis=1) / total_count
        x[:,i] = c
    return x

def get_thresholded_adata(adata,cond_Y):
    adata = adata.copy()
    adata.X = csr_matrix(np.where(cond_Y,adata.X.toarray(),0))
    return adata

def weighting(adata,dic,t):
    weights = np.full(t,0.5)  # here the t will be number of valid_tissue
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
    # sns.histplot(n)
    # plt.savefig('n_evidence.pdf',bbox_inches='tight')
    # plt.close()
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

    total = round(mean_n,0)
    def sample_category(x):
        # x is a series
        t = 0
        d = []
        for i,v in enumerate(x.values[:-1]): # {0:high,1:medium,2:low,3:not detected}, 3 is hold off
            n = round(v,0)
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
def model_X_Y(X,Y,weights,train,w_x,w_y):
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

def guide_X_Y(X,Y,weights,train,w_x,w_y):
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


def model(X,Y,Z,weights,train,w_x,w_y,w_z):
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


def guide(X,Y,Z,weights,train,w_x,w_y,w_z):
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


def test_and_graph_model(model,*args):
    trace = pyro.poutine.trace(model).get_trace(*args)
    trace.compute_log_prob()  
    print(trace.format_shapes())
    # pyro.render_model(model, model_args=(*args), render_distributions=True, render_params=True, filename='model.pdf')

def basic_configure(raw_X,Y,Z,weights):
    # basic configure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    X = torch.tensor(raw_X.T,device=device)
    Y = torch.tensor(Y.T,device=device)
    Z = torch.tensor(Z.T,device=device)
    n = X.shape[1]
    s = Y.shape[0]
    t = X.shape[0]
    p = Z.shape[0]
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
    prior_sigma = svi_sigma.data.cpu().numpy().mean(axis=0)

    n_steps = 5000
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
    plt.savefig('elbo_loss.pdf',bbox_inches='tight')
    plt.close()

    with pyro.plate('samples',1000,dim=-1):
        samples = guide(*args)
    svi_sigma = samples['sigma']  # torch.Size([1000, n])
    sigma = svi_sigma.data.cpu().numpy().mean(axis=0)
    alpha = pyro.param('alpha').data.cpu().numpy()
    beta = pyro.param('beta').data.cpu().numpy()
    df = pd.DataFrame(index=uids,data={'mean_sigma':sigma,'alpha':alpha,'beta':beta,'prior_sigma':prior_sigma})
    with open('X.p','rb') as f:
        X = pickle.load(f)
    with open('Y.p','rb') as f:
        Y = pickle.load(f)
    with open('Z.p','rb') as f:
        Z = pickle.load(f)
    Y_mean = Y.mean(axis=1)
    X_mean = X.mean(axis=1)
    Z_mean = Z.mean(axis=1)
    df['Y_mean'] = Y_mean
    df['X_mean'] = X_mean
    df['Z_mean'] = Z_mean
    df.to_csv('full_results.txt',sep='\t')
    diagnose('full_results.txt',output_name='pyro_full_diagnosis.pdf')
    return svi_sigma

def train_and_infer_XY(model,guide,*args):

    adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
    clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
    elbo = Trace_ELBO()
    # train
    with pyro.plate('samples',1000,dim=-1):
        samples = guide(*args)
    svi_sigma = samples['sigma']  # torch.Size([1000, n])
    prior_sigma = svi_sigma.data.cpu().numpy().mean(axis=0)

    n_steps = 5000
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
    plt.savefig('elbo_loss.pdf',bbox_inches='tight')
    plt.close()

    with pyro.plate('samples',1000,dim=-1):
        samples = guide(*args)
    svi_sigma = samples['sigma']  # torch.Size([1000, n])
    sigma = svi_sigma.data.cpu().numpy().mean(axis=0)
    alpha = pyro.param('alpha').data.cpu().numpy()
    beta = pyro.param('beta').data.cpu().numpy()
    df = pd.DataFrame(index=uids,data={'mean_sigma':sigma,'alpha':alpha,'beta':beta,'prior_sigma':prior_sigma})
    with open('X.p','rb') as f:
        X = pickle.load(f)
    with open('Y.p','rb') as f:
        Y = pickle.load(f)
    Y_mean = Y.mean(axis=1)
    X_mean = X.mean(axis=1)
    df['Y_mean'] = Y_mean
    df['X_mean'] = X_mean
    df.to_csv('full_results.txt',sep='\t')
    diagnose('full_results.txt',output_name='pyro_full_diagnosis.pdf')
    return svi_sigma

def generate_inputs(adata,protein,dic):
    Z,adata,uids = compute_z(adata,protein,dic)  # 13350 * 89
    Y = compute_y(adata,uids)  # 13350 * 1228
    Y = np.where(Y==0,1e-5,Y)

    '''below is to find best cutoff for tissue distribution'''
    # external = pd.read_csv('rna_tissue_consensus.tsv',sep='\t')
    # for cutoff in np.arange(0,5,0.5):
    #     X, annotated_x = compute_scaled_x(adata,uids,cutoff)  # 13350 * 49
    #     result = compute_concordance(annotated_x,external)
    #     annotated_x.to_csv('annotated_x_{}.txt'.format(str(cutoff)),sep='\t')
    #     result.to_csv('result_{}.txt'.format(str(cutoff)),sep='\t')

    best_cutoff = 3
    X, annotated_x = compute_scaled_x(adata,uids,best_cutoff)  # 13350 * 49

    thresholded_Y = np.empty_like(Y,dtype=np.float32)
    cond_Y = np.empty_like(Y,dtype=bool)
    ths = []
    for i in tqdm(range(Y.shape[0]),total=Y.shape[0]):
        thresholded_Y[i,:], cond_Y[i,:], th = threshold(Y[i,:],'hardcode',v=best_cutoff)
        ths.append(th)
    new_adata = get_thresholded_adata(adata,cond_Y)

    n = X.shape[0]
    s = Y.shape[1]
    t = X.shape[1]
    weights = weighting(adata,dic,t)

    return X,Y,Z,weights,uids

def generate_input_XY(adata,dic):
    uids = adata.obs_names.tolist()
    Y = compute_y(adata,uids) 
    Y = np.where(Y==0,1e-5,Y)
    X = compute_scaled_x(adata,uids)  
    thresholded_Y = np.empty_like(Y,dtype=np.float32)
    cond_Y = np.empty_like(Y,dtype=bool)
    ths = []
    for i in tqdm(range(Y.shape[0]),total=Y.shape[0]):
        thresholded_Y[i,:], cond_Y[i,:], th = threshold(Y[i,:],'hardcode',v=0.8)
        ths.append(th)
    new_adata = get_thresholded_adata(adata,cond_Y)
    raw_X = compute_x(new_adata,uids)
    n = X.shape[0]
    s = Y.shape[1]
    t = X.shape[1]
    weights = weighting(adata,dic,t)
    with open('uids.p','wb') as f:
        pickle.dump(uids,f)
    with open('weights.p','wb') as f:
        pickle.dump(weights,f)
    with open('Y.p','wb') as f:
        pickle.dump(Y,f)
    with open('X.p','wb') as f:
        pickle.dump(X,f)
    with open('raw_X.p','wb') as f:
        pickle.dump(raw_X, f)   
    return X,Y,raw_X,weights,uids

def basic_configure_XY(raw_X,Y,weights,uids):
    # basic configure
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    X = torch.tensor(raw_X.T,device=device)
    Y = torch.tensor(Y.T,device=device)
    n = X.shape[1]
    s = Y.shape[0]
    t = X.shape[0]
    weights = torch.tensor(weights,device=device)
    return device,X,Y,n,s,t,weights

def load_pre_generated_inputs():
    with open('uids.p','rb') as f:
        uids = pickle.load(f)
    with open('weights.p','rb') as f:
        weights = pickle.load(f)
    with open('Z.p','rb') as f:
        Z = pickle.load(f)
    with open('Y.p','rb') as f:
        Y = pickle.load(f)
    with open('X.p','rb') as f:
        X = pickle.load(f)
    with open('raw_X.p','rb') as f:
        raw_X = pickle.load(f)
    return X,Y,Z,raw_X,weights,uids

def train_single(model,guide,name,*args):
    adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
    clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
    elbo = Trace_ELBO()

    n_steps = 500
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
    plt.savefig('elbo_loss_train_single_{}.pdf'.format(name),bbox_inches='tight')
    plt.close()

    largest10 = np.sort(losses)[-10:]
    # return the scale of this modality
    return np.median(largest10)

'''main program starts'''

def main(args):

    adata = ad.read_h5ad(args.input)
    dic = pd.read_csv(args.weight,sep='\t',index_col=0).to_dict()
    global n,s,t,device,uids
    if args.mode == 'XY':
        X,Y,raw_X,weights,uids = generate_input_XY(adata,dic)
        device,X,Y,n,s,t,weights = basic_configure_XY(raw_X,Y,weights,uids)
        # derive w_x and w_y
        s_x = train_single(model_X,guide_X,'X',X,weights,True)
        s_y = train_single(model_Y,guide_Y,'Y',Y,True)
        lis = np.array([s_x,s_y])
        small = lis.min()
        w_x = small / s_x
        w_y = small / s_y
        # run
        svi_sigma = train_and_infer_XY(model_X_Y,guide_X_Y,X,Y,weights,True,w_x,w_y)
    elif args.mode == 'XYZ':
        global p
        protein = pd.read_csv(args.protein,sep='\t')
        X,Y,Z,weights,uids = generate_inputs(adata,protein,dic)
        device,X,Y,Z,n,s,t,p,weights = basic_configure(X,Y,Z,weights)
        # derive the scale of model_Y using emprical bayes
        # please do
        # derive w_x, w_y, w_z
        s_x = train_single(model_X,guide_X,'X',X,weights,True)
        s_y = train_single(model_Y,guide_Y,'Y',Y,True)
        s_z = train_single(model_Z,guide_Z,'Z',Z,True)
        lis = np.array([s_x,s_y,s_z])
        small = lis.min()
        w_x = small / s_x
        w_y = small / s_y
        w_z = small / s_z
        print(s_x,s_y,s_z,lis,small,w_x,w_y,w_z)
        # run
        svi_sigma = train_and_infer(model,guide,X,Y,Z,weights,True,w_x,w_y,w_z)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run BayesTS to retrain')
    parser.add_argument('--input',type=str,default='',help='path to the h5ad file')
    parser.add_argument('--weight',type=str,default='',help='path to a txt file with tissue and weights that you want to change')
    parser.add_argument('--mode',type=str,default='XYZ',help='XYZ use full model, XY use only RNA model')
    parser.add_argument('--protein',type=str,default='',help='path to the protein info downloaded from Synapse')
    args = parser.parse_args()
    main(args)













