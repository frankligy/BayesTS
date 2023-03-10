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

# for publication ready figure
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

'''main program starts'''

adata = ad.read_h5ad('../gene/coding.h5ad')
uids = adata.obs_names.tolist()


# # derive some statistics
# var = adata.var
# tcga_var = var.loc[[True if 'TCGA' in item else False for item in var.index],:]
# gtex_var = var.loc[[True if 'TCGA' not in item else False for item in var.index],:]
# print(tcga_var,gtex_var)
# gtex_var['tissue'].value_counts().to_csv('gtex_unique_tissue_counts.txt',sep='\t')
# sys.exit('stop')


# # whether expendable tissues reduction is significant
# df = pd.read_csv('../gene_protein/MS4A1_tissue.txt',sep='\t',index_col=0)
# from scipy.stats import mannwhitneyu,ttest_ind
# print(mannwhitneyu(df['default'].values,df['immune tissue'].values))

# df = pd.read_csv('../gene_protein/CTAG1B_Testis.txt',sep='\t',index_col=0)
# from scipy.stats import mannwhitneyu,ttest_ind
# print(mannwhitneyu(df['0.5'].values,df['0.2'].values))



'''
fig1. visualize the Y, the cpm of count
using these three examples:
'''
from scipy.stats import lognorm,poisson,beta
import matplotlib.lines as mlines
# fig,ax = plt.subplots(figsize=(12,4.8))
# scale = 1
# for s,loc,c in [(1.5,0.1,'#5496BF'),(0.5,1,'#F28749'),(0.3,2.5,'#68BF65')]:
#     y = lognorm.rvs(s=s,loc=loc,scale=scale,size=3644)
#     y = y[y<6]
#     sns.histplot(y,ax=ax,stat='density',binwidth=0.1,alpha=0.5)
#     x = np.linspace(lognorm.ppf(0.01,s=s,loc=loc,scale=scale),lognorm.ppf(0.99,s=s,loc=loc,scale=scale),1000)
#     y = lognorm.pdf(x,s=s,loc=loc,scale=scale)
#     ax.plot(x,y,c=c,linewidth=2)
# ax.legend(handles=[mlines.Line2D([],[],linestyle='-',color=i) for i in ['#5496BF','#F28749','#68BF65']],labels=['LogNormal(0.1,1.5)','LogNormal(1.0,0.5)','LogNormal(2.5,0.3)'],frameon=False,loc='upper left',bbox_to_anchor=(1,1))
# ax.set_xlim([0,6])
# ax.set_xlabel('Normalized Count from RNA-Seq')
# plt.savefig('Y.pdf',bbox_inches='tight')
# plt.close()

'''
fig1. visualize X, the tissue distribution
'''
# fig,axes = plt.subplots(figsize=(2,4.8),nrows=3,gridspec_kw={'hspace':0.2},sharex='all')
# axes = axes.flatten()
# for i,(mu,c) in enumerate(zip([2,12,20],['#5496BF','#F28749','#68BF65'])):
#     y = poisson.rvs(mu=mu,size=10000)
#     sns.histplot(y,ax=axes[i],stat='probability',binwidth=0.6,facecolor='k')
#     x = np.round(np.linspace(poisson.ppf(0.01,mu=mu),poisson.ppf(0.99,mu=mu),1000))
#     y = poisson.pmf(x,mu=mu)
#     x_ = x[y>0]
#     y_ = y[y>0]
#     axes[i].plot(x_,y_,marker='o',markersize=2,linestyle='-',c=c,linewidth=1)
#     axes[i].legend(handles=[mlines.Line2D([],[],linestyle='-',color=c)],labels=['Poisson({})'.format(mu)],frameon=False,loc='upper left',bbox_to_anchor=(1,1))
# axes[2].set_xlabel('Number of Expressed samples per tissue')
# plt.savefig('X.pdf',bbox_inches='tight')
# plt.close()


'''
fig1, bayesTS, beta
'''
# fig,ax = plt.subplots(figsize=(6.4,4.8))
# for a,b,c in [(0.5,5,'#5496BF'),(5,5,'#F28749'),(9,0.1,'#68BF65')]:
#     x = np.linspace(beta.ppf(0.001,a=a,b=b),beta.ppf(0.999,a=a,b=b),1000)
#     y = beta.pdf(x,a=a,b=b)
#     y[y>3] = 3
#     ax.plot(x,y,c=c,linewidth=2)
# ax.legend(handles=[mlines.Line2D([],[],linestyle='-',color=i) for i in ['#5496BF','#F28749','#68BF65']],labels=['Beta(0.5,5)','Beta(5,5)','Beta(9,0.1)'],frameon=False,loc='upper left',bbox_to_anchor=(1,1))
# ax.set_xlabel('Tumor specificity')
# ax.set_ylim([0,3])
# ax.set_ylabel('Probability')
# plt.savefig('sigma.pdf',bbox_inches='tight')
# plt.close()



'''
fig1B, 3d plot, in gene_protien_model
'''


'''
fig2A, tumor importance, in hbm_adv_pyro.model
'''


'''
fig2B, drug discovery
'''
# result = pd.read_csv('../dev/full_results_XY.txt',sep='\t',index_col=0)
# result = result.sort_values(by='mean_sigma')
# fig,ax = plt.subplots(figsize=(12,4.8))
# ax.bar(x=np.arange(result.shape[0]),height=result['mean_sigma'].values)
# index_MEGA1A = result.index.tolist().index('ENSG00000198681')
# index_MS4A1 = result.index.tolist().index('ENSG00000156738')
# ax.axvline(x=index_MEGA1A,c='r',linestyle='--')
# ax.axvline(x=index_MS4A1,c='r',linestyle='--')
# ax.set_xlabel('Targets ranked by tumor specificity')
# ax.set_ylabel('Inferred tumor specificity score')
# plt.savefig('new_targets.pdf',bbox_inches='tight')
# plt.close()

# result = pd.read_csv('../dev/full_results_XY.txt',sep='\t',index_col=0)
# result = result.sort_values(by='mean_sigma')
# fig,ax = plt.subplots(figsize=(12,4.8))
# ax.bar(x=np.arange(result.shape[0]),height=result['mean_sigma'].values)
# index_MEGA1A = result.index.tolist().index('ENSG00000001461:E6.1-E7.1_24445189')
# index_MS4A1 = result.index.tolist().index('ENSG00000149582:E6.2-E7.1')
# ax.axvline(x=index_MEGA1A,c='r',linestyle='--')
# ax.axvline(x=index_MS4A1,c='r',linestyle='--')
# ax.set_xlabel('Targets ranked by tumor specificity')
# ax.set_ylabel('Inferred tumor specificity score')
# plt.savefig('new_targets_splicing.pdf',bbox_inches='tight')
# plt.close()



'''visualize tumor versus normal for gene'''
adata = ad.read_h5ad('../gene/coding.h5ad')
from gtex_viewer import *
gtex_viewer_configuration(adata)
df = pd.read_csv('counts.TCGA-SKCM-steady-state.txt',sep='\t',index_col=0)
# result = pd.read_csv('../gene_protein/full_results_XYZ.txt',sep='\t',index_col=0).sort_values(by='mean_sigma')
# index_MEGA1A = result.index.tolist().index('ENSG00000198681')
# index_MS4A1 = result.index.tolist().index('ENSG00000156738')
# for i in range(index_MEGA1A+1,index_MS4A1):
#     uid = result.index.tolist()[i]
#     try:
#         gtex_visual_combine(uid,norm=True,outdir='new_drug_targets',figsize=(6.4,4.8),tumor=df,ylim=None)
#     except KeyError:
#         continue

uids = [
    'ENSG00000126856',
    'ENSG00000183668',
    'ENSG00000243130',
    'ENSG00000170848',
    'ENSG00000156269',
    'ENSG00000166049',
    'ENSG00000187772',
    'ENSG00000268009'
]

for uid in uids:
    gtex_visual_combine(uid,norm=True,outdir='new_drug_targets_visual_final',figsize=(6.4,4.8),tumor=df,ylim=None)
sys.exit('stop')

'''
fig2C, splicing junction, using snaf
'''



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


