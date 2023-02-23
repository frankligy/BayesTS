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


'''main program starts'''

adata = ad.read_h5ad('../gene/coding.h5ad')
uids = adata.obs_names.tolist()

'''
fig1. visualize the Y, the cpm of count
using these three examples:
'''
from torch.distributions import LogNormal
fig,ax = plt.subplots()
scale = 0.5
ys = {}
for mu in [0.1,2.,4.]:
    m = LogNormal(torch.tensor(mu),torch.tensor(scale))
    y = [m.sample().numpy() for i in range(3644)]
    ys[mu] = y
df = pd.DataFrame(data=ys)
print(df)
sns.histplot(df,ax=ax)
plt.savefig('check.pdf',bbox_inches='tight')
plt.close()
sys.exit('stop')



# Z = df.values  # 13306 * 89
# with open('Z.p','wb') as f:
#     pickle.dump(Z,f)
# Y = compute_y(adata,uids)  # 13306 * 3644
# with open('Y.p','wb') as f:
#     pickle.dump(Y,f)
# X = compute_scaled_x(adata,uids)  # 13306 * 63
# with open('X.p','wb') as f:
#     pickle.dump(X,f)

# thresholded_Y = np.empty_like(Y,dtype=np.float32)
# cond_Y = np.empty_like(Y,dtype=bool)
# ths = []
# for i in tqdm(range(Y.shape[0]),total=Y.shape[0]):
#     thresholded_Y[i,:], cond_Y[i,:], th = threshold(Y[i,:],'hardcode',v=0.8)
#     ths.append(th)
# new_adata = get_thresholded_adata(adata,cond_Y)
# X = compute_x(new_adata,uids)
# with open('raw_X.p','wb') as f:
#     pickle.dump(X, f)

# with open('raw_X.p','rb') as f:
#     X = pickle.load(f)
# with open('Y.p','rb') as f:
#     Y = pickle.load(f)
# with open('Z.p','rb') as f:
#     Z = pickle.load(f)

# # uid = 'ENSG00000156738'
# # index = uids.index(uid)
# # X = X[[index],:]
# # Y = Y[[index],:]
# # print(X.shape,Y.shape)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# Y = np.where(Y==0,1e-5,Y)
# X = torch.tensor(X.T,device=device)
# Y = torch.tensor(Y.T,device=device)
# Z = torch.tensor(Z.T,device=device)
# n = X.shape[1]
# s = Y.shape[0]
# t = X.shape[0]
# p = Z.shape[0]
# weights = np.full(t,0.5)
# # dic = {
# #     'Spleen': 0.9,
# #     'Whole Blood': 0.9
# # }
# # weights = weighting(adata,dic,weights)
# weights = torch.tensor(weights,device=device)


# adam = Adam({'lr': 0.002,'betas':(0.95,0.999)}) 
# clipped_adam = ClippedAdam({'betas':(0.95,0.999)})
# elbo = Trace_ELBO()


# def model(X,Y,Z,weights):
#     a = torch.tensor(2.,device=device)
#     b = torch.tensor(2.,device=device)
#     sigma = pyro.sample('sigma',dist.Beta(a,b).expand([n]).to_event(1))
#     a = torch.tensor(10.,device=device)
#     b = torch.tensor(1.,device=device)
#     beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
#     a = torch.tensor(25.,device=device)
#     b = torch.tensor(1.,device=device)
#     beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
#     a = torch.tensor(50.,device=device)
#     total = pyro.sample('total',dist.Binomial(a,weights).expand([t]).to_event(1))
#     scaled_X = torch.round(X * total.unsqueeze(-1))
#     with pyro.poutine.scale(scale=1/10), pyro.plate('data_X',t):
#         c = pyro.sample('c',dist.Poisson(beta_x*sigma).expand([t,n]).to_event(1),obs=scaled_X)
#     with pyro.poutine.scale(scale=1/10000), pyro.plate('data_Y',s):
#         nc = pyro.sample('nc',dist.LogNormal(beta_y*sigma,0.5).expand([s,n]).to_event(1),obs=Y)
#     high_prob = 2./3. * sigma
#     medium_prob = 1./3. * sigma
#     low_prob = 2./3. * (1-sigma)
#     not_prob = 1./3. * (1-sigma)
#     prob = torch.stack([high_prob,medium_prob,low_prob,not_prob],axis=0).T   # n * p
#     with pyro.poutine.scale(scale=1), pyro.plate('data_Z',p):
#         pc = pyro.sample('pc',dist.Categorical(prob).expand([p,n]).to_event(1),obs=Z)


# def guide(X,Y,Z,weights):
#     alpha = pyro.param('alpha',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
#     beta = pyro.param('beta',lambda:torch.tensor(np.full(n,2.0),device=device),constraint=constraints.positive)
#     sigma = pyro.sample('sigma',dist.Beta(alpha,beta).expand([n]).to_event(1))
#     a = torch.tensor(10.,device=device)
#     b = torch.tensor(1.,device=device)
#     beta_y = pyro.sample('beta_y',dist.Gamma(a,b))
#     a = torch.tensor(25.,device=device)
#     b = torch.tensor(1.,device=device)
#     beta_x = pyro.sample('beta_x',dist.Gamma(a,b))
#     total = pyro.sample('total',dist.Binomial(50,weights).expand([t]).to_event(1))
#     return {'sigma':sigma}

# # trace = pyro.poutine.trace(model).get_trace(X,Y,Z,weights)
# # trace.compute_log_prob()  
# # print(trace.format_shapes())
# # pyro.render_model(model, model_args=(X,Y,Z,weights), render_distributions=True, render_params=True, filename='model.pdf')


# # train
# n_steps = 5000
# pyro.clear_param_store()
# svi = SVI(model, guide, adam, loss=Trace_ELBO())
# losses = []
# for step in tqdm(range(n_steps),total=n_steps):  
#     loss = svi.step(X,Y,Z,weights)
#     losses.append(loss)
# plt.figure(figsize=(5, 2))
# plt.plot(losses)
# plt.xlabel("SVI step")
# plt.ylabel("ELBO loss")
# plt.savefig('elbo_loss.pdf',bbox_inches='tight')
# plt.close()

# with pyro.plate('samples',1000,dim=-1):
#     samples = guide(X,Y,Z,weights)
# svi_sigma = samples['sigma']  # torch.Size([1000, 24290])
# sigma = svi_sigma.data.cpu().numpy().mean(axis=0)
# alpha = pyro.param('alpha').data.cpu().numpy()
# beta = pyro.param('beta').data.cpu().numpy()
# df = pd.DataFrame(index=uids,data={'mean_sigma':sigma,'alpha':alpha,'beta':beta})
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
# ternary plot
# import ternary
from sklearn.preprocessing import MinMaxScaler
result = pd.read_csv('full_results.txt',sep='\t',index_col=0)
sigma = result['mean_sigma'].values
Y_mean = MinMaxScaler().fit_transform(result['Y_mean'].values.reshape(-1,1)).squeeze()
X_mean = MinMaxScaler().fit_transform(result['X_mean'].values.reshape(-1,1)).squeeze()
Z_mean = MinMaxScaler().fit_transform(result['Z_mean'].values.reshape(-1,1)).squeeze()
# scale = 3
# figure, tax = ternary.figure(scale=scale)
# points = [(y,x,z) for y,x,z in zip(Y_mean,X_mean,Z_mean)]
# tax.scatter(points, marker='o', c=sigma)
# tax.boundary(linewidth=2.0)
# tax.set_title("Scatter Plot", fontsize=20)
# tax.gridlines(multiple=5, color="blue")
# tax.ticks(axis='lbr', linewidth=1, multiple=5)
# tax.clear_matplotlib_ticks()
# tax.get_axes().axis('off')
# plt.savefig('check.pdf',bbox_inches='tight')
# plt.close()

# 3d plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
xs = X_mean
ys = Y_mean
zs = Z_mean
ax.scatter(xs, ys, zs, marker='o',s=1,c=1-sigma)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_ylim([0,1])
ax.set_zlim([1,0])
ax.view_init(elev=0,azim=-100)
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
plt.savefig('targets.pdf',bbox_inches='tight')
plt.close()

fig,ax = plt.subplots()
result['Z_mean'] *= -1
ax.imshow(MinMaxScaler().fit_transform(result.loc[:,['Y_mean','X_mean','Z_mean']].values).T)
ax.set_yticks([0,1,2])
ax.set_yticks(['Y','X','Z'])
plt.savefig('evidence_targets.pdf',bbox_inches='tight')
plt.close()


