#!/gpfs/data/yarmarkovichlab/Frank/BayesTS/logit_gate_env/bin/python3.7

import os,sys
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import pickle
import pyro.distributions as dist
import torch

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# derive CPM from junction count
# count = pd.read_csv('counts.TCGA-SKCM-steady-state.txt',sep='\t',index_col=0)
# count.rename(columns=lambda x:'-'.join(x.split('-')[:4]),inplace=True)
# cpm = count.values / count.values.sum(axis=0).reshape(1,-1) * 1e6
# cpm = pd.DataFrame(data=cpm,columns=count.columns,index=count.index)

# tpm = pd.read_csv('TCGA_SKCM_gene_tpm.txt',sep='\t',index_col=0)
# common = list(set(cpm.index).intersection(set(tpm.index)))
# samples = count.columns.tolist()

# cpm = cpm.loc[common,samples]
# tpm = tpm.loc[common,samples]

# genes = {
#     'PMEL':'ENSG00000185664',
#     'MLANA':'ENSG00000120215',
#     'TP53':'ENSG00000141510',
#     'GAPDH':'ENSG00000111640',
#     'ACTB':'ENSG00000075624',
#     'UBC':'ENSG00000150991',
#     'HPRT':'ENSG00000165704',
#     'SDHA':'ENSG00000073578',
#     'YWHAZ':'ENSG00000164924',
# }

# for symbol,ensg in genes.items():
#     x = cpm.loc[ensg].values
#     y = tpm.loc[ensg].values
#     s,p = pearsonr(x,y)
#     s,p = round(s,2),round(p,2)
#     fig,ax = plt.subplots()
#     sns.regplot(x=x,y=y,ax=ax)
#     ax.text(0.5,0.1,s='pearsonr: {}\npvalue:{}'.format(s,p),transform=ax.transAxes)
#     ax.set_xlabel('AltAnalyze Counts Per Million (CPM)')
#     ax.set_ylabel('Transcripts Per Million (TPM)')
#     ax.set_title('{},{}'.format(ensg,symbol))
#     plt.savefig('{}_{}.pdf'.format(ensg,symbol),bbox_inches='tight')

# re-sample 25 per tissues for GTEx TPM
# adata_gtex = ad.read_h5ad('gtex_gene_all.h5ad')
# var = adata_gtex.var
# tissues = var['tissue'].unique().tolist()
# tissues.remove('Cells - Cultured fibroblasts')
# tissues.remove('Cells - EBV-transformed lymphocytes')
# all_selected = []
# size = 25
# for t in tissues:
#     var_t = var.loc[var['tissue']==t,:]
#     if var_t.shape[0] < size:
#         actual_size = var_t.shape[0]
#     else:
#         actual_size = size
#     selected_t = np.random.choice(var_t.index.tolist(),size=actual_size,replace=False).tolist()
#     all_selected.extend(selected_t)
# adata_gtex_selected = adata_gtex[:,all_selected]  # 56200 × 1228
# adata_original = ad.read_h5ad('coding.h5ad')  # 24290 × 3644
# adata_gtex_selected.obs_names = [item.split('.')[0] for item in adata_gtex_selected.obs_names]
# cond = np.logical_not(adata_gtex_selected.obs_names.duplicated())
# adata_gtex_selected = adata_gtex_selected[cond,:]  # 56156 × 1228
# adata_gtex_selected.write('gtex_gene_subsample.h5ad')

# check if TPM is lognormal
# adata = ad.read_h5ad('gtex_gene_subsample.h5ad')
# gene = 'ENSG00000177455'
# values = np.log2(adata[gene,:].X.toarray()[0])
# sns.histplot(values)
# plt.savefig('check.pdf',bbox_inches='tight')
# plt.close()



# estimate TPM, the beta parameter using eBayes
# adata = ad.read_h5ad('gtex_gene_subsample.h5ad')

# nuorf = pd.read_csv('nuORF.txt',sep='\t')
# nuorf = nuorf.loc[nuorf['geneType']=='protein_coding',:]
# nuorf['ensg'] = [item.split('.')[0] for item in nuorf['geneId']]
# ensg2pc = {item1:item2 for item1,item2 in zip(nuorf['ensg'],nuorf['geneType'])}

# common = list(set(adata.obs_names).intersection(set(ensg2pc.keys())))
# adata = adata[common,:]

# lower_quantiles = np.quantile(adata.X.toarray(),0.25,axis=1)
# median_quantiles = np.quantile(adata.X.toarray(),0.5,axis=1)
# upper_quantiles = np.quantile(adata.X.toarray(),0.75,axis=1)

# a = median_quantiles / 0.5
# a = a[a>0]
# a = np.sort(a)
# covered = np.arange(len(a)) / len(a)
# df = pd.DataFrame(data={'a':a,'covered':covered})
# cutoff = df.loc[df['covered']<=0.95,:].iloc[-1,0]
# a = a[a<=cutoff]  

# fig,ax = plt.subplots()
# sns.histplot(a,ax=ax,stat='density')

# # if using exponential, which is gamma(1,1/lambda)
# lambda_ = 1/np.mean(a)
# from scipy.stats import expon,gamma
# x = np.linspace(0, cutoff, 1000)
# pdf = expon.pdf(x, scale=1/lambda_)
# pdf = gamma.pdf(x,a=1,scale=1/lambda_)
# ax.plot(x,pdf)
# plt.savefig('expo_a.pdf',bbox_inches='tight')
# plt.close()

# threshold the gene matrix when deriving the tissue distribution
# dic = {}
# for cutoff in np.arange(0,10,0.25):
#     result_path = 'result_{}.txt'.format(str(cutoff))
#     result = pd.read_csv(result_path,sep='\t',index_col=0)
#     repr_spearman = np.nanmean(result['spearman'].values)
#     repr_aupr = np.nanmean(result['aupr'].values)
#     dic[cutoff] = ((repr_spearman,repr_aupr))


# test speed
# outdir = 'output_xyz'
# with open(os.path.join(outdir,'X.p'),'rb') as f:
#     X = pickle.load(f)
# np.savetxt('X.out',X,delimiter='\t')
# with open(os.path.join(outdir,'Y.p'),'rb') as f:
#     Y = pickle.load(f)
# np.savetxt('Y.out',Y,delimiter='\t')
# with open(os.path.join(outdir,'uids.p'),'rb') as f:
#     uids = pickle.load(f)
# with open('uids.out','w') as f:
#     for uid in uids:
#         f.write('{}\n'.format(uid))

n = [10,50,100,200,500]
mcmc = [16.145457983016968, 39.225778341293335, 74.17322897911072, 161.4286584854126, 428.83027958869934]
vi = [5.572661399841309, 7.877718210220337, 12.209128618240356, 22.4102463722229, 58.071091175079346]
fig, ax = plt.subplots()
ax.plot(np.arange(len(n)),mcmc,marker='o',label='mcmc')
ax.plot(np.arange(len(n)),vi,marker='o',label='vi')
ax.legend(frameon=False)
ax.set_xticks(np.arange(len(n)))
ax.set_xticklabels(n)
ax.set_xlabel('Number of genes')
ax.set_ylabel('Time(s)')
plt.savefig('mcmc_vi.pdf',bbox_inches='tight')
plt.close()
sys.exit('stop')



# # combine PR
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
    plt.ylim([0.0, 1.1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve example')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(outdir,outname),bbox_inches='tight')
    plt.close()

# y_preds = {}
# gs = pd.read_csv('gold_standard.txt',sep='\t')['ensg'].values.tolist()
# base_result = pd.read_csv(os.path.join('output_1','full_results.txt'),sep='\t',index_col=0)
# base_order = base_result.index
# for outdir in ['output_sensitivity_82','output_sensitivity_42','output_1','output_sensitivity_24']:
#     result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0).loc[base_order,:]
#     result['label'] = [True if item in gs else False for item in result.index]
#     y_preds[outdir] = np.negative(result['mean_sigma'].values)
# draw_PR(result['label'].values,y_preds,'.','PR_curve_sensitivity_test.pdf')


# deg check
ensg2biotype = pd.read_csv('gene_lfc.txt',sep='\t',index_col=0)['biotype'].to_dict()
hpa = pd.read_csv('proteinatlas.tsv',sep='\t')
hpa = hpa.loc[hpa['RNA tissue specificity'].isin(['Tissue enriched','Group enriched','Not detected','Tissue enhanced']),:]
cond1 = [True if isinstance(item,str) and 'Tumor antigen' in item else False for item in hpa['Molecular function']]
cond2 = [True if isinstance(item,str) and 'melanoma' in item else False for item in hpa['RNA cancer specific FPKM']]
cond = np.any([cond1,cond2],axis=0).tolist()
hpa = hpa.loc[cond,:]

gs = hpa['Ensembl'].values.tolist()
result = pd.read_csv(os.path.join('output_xyc','full_results.txt'),sep='\t',index_col=0)
result['biotype'] = result.index.map(ensg2biotype).values
result = result.loc[result['biotype']=='protein_coding',:]
result['label'] = [True if item in gs else False for item in result.index]
deg = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/BayesTS/revision/deg.txt',sep='\t',index_col='Gene ID')
ensg2adjp = deg['adjp'].to_dict()
ensg2lfc = deg['Log2(Fold Change)'].to_dict()
result['limma_adjp'] = result.index.map(ensg2adjp).values
result['limma_lfc'] = result.index.map(ensg2lfc).values
result = result.loc[result['limma_adjp'].notna(),:]
result = result.loc[result['limma_lfc'].notna(),:]

from scipy.stats import rankdata

result['lfc_rank'] = rankdata(np.negative(result['limma_lfc'].values),method='min')
result['adjp_rank'] = rankdata(result['limma_adjp'].values,method='min')
result['limma_rank'] = [(r1+r2)/2 for r1,r2 in zip(result['lfc_rank'],result['adjp_rank'])]

y_preds = {
    'BayesTS_ext':np.negative(result['mean_sigma'].values),
    'limma_rank':np.negative(result['limma_rank'].values)
}

limma_label = [True if p < 0.05 and s > 0.58 else False for p,s in zip(result['limma_adjp'],result['limma_lfc'])]
result['limma_label'] = limma_label

result.to_csv('deg_bayesTS.txt',sep='\t')
draw_PR(result['label'].values,y_preds,'.','PR_curve_deg.pdf');sys.exit('stop')

result_sorted_bayes = result.sort_values(by='mean_sigma')
result_sorted_lfc = result.sort_values(by='limma_lfc')



# prior and posterior check
def prior_posterior_check(uid,outdir,ebayes_beta_y):
    full_result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
    index = uids.index(uid)
    prior_sigma = full_result.loc[uid,'prior_sigma']
    posterior_sigma = full_result.loc[uid,'mean_sigma']

    # c
    beta_x = 25
    data = X[:,index]
    prior = dist.Poisson(beta_x*prior_sigma).expand([t]).sample().data.cpu().numpy()
    posterior = dist.Poisson(beta_x*posterior_sigma).expand([t]).sample().data.cpu().numpy()
    fig,ax = plt.subplots()
    for item in [data,prior,posterior]:
        sns.histplot(item,ax=ax,stat='probability',alpha=0.5,bins=10)
    plt.savefig(os.path.join(outdir,'{}_c.pdf'.format(uid)),bbox_inches='tight')
    plt.close()

    # nc
    beta_y = ebayes_beta_y
    data = Y[:,index]
    prior = dist.LogNormal(beta_y*prior_sigma,0.5).expand([s]).sample().data.cpu().numpy()
    posterior = dist.LogNormal(beta_y*posterior_sigma,0.5).expand([s]).sample().data.cpu().numpy()
    fig,ax = plt.subplots()
    for item in [data,prior,posterior]:
        sns.histplot(np.log(item),ax=ax,stat='probability',bins=40,alpha=0.5)
    plt.savefig(os.path.join(outdir,'{}_nc.pdf'.format(uid)),bbox_inches='tight')
    plt.close()

    # pc
    data = Z[:,index]
    probs = [torch.tensor([2/3*sigma,1/3*sigma,1/3*(1-sigma),2/3*(1-sigma)]) for sigma in [prior_sigma,posterior_sigma]]
    prior = dist.Categorical(probs[0]).expand([p]).sample().data.cpu().numpy()
    posterior = dist.Categorical(probs[1]).expand([p]).sample().data.cpu().numpy()
    fig,axes = plt.subplots(ncols=3,gridspec_kw={'wspace':0.5})
    axes = axes.flatten()
    for i,(item,color,title) in enumerate(zip([data,prior,posterior],['#7995C4','#E5A37D','#80BE8E'],['data','prior','posterior'])): 
        sns.histplot(item,ax=axes[i],stat='count',bins=4,alpha=0.5,facecolor=color)
        axes[i].set_ylim([0,90])
        axes[i].set_title(title)
    plt.savefig(os.path.join(outdir,'{}_pc.pdf'.format(uid)),bbox_inches='tight')
    plt.close()

outdir = 'output_test'
ebayes_beta_y = 4.301273922770753
s = 1228
t = 49
p = 89
uid = 'ENSG00000111640'  #ENSG00000111640 ENSG00000198681
with open(os.path.join(outdir,'uids.p'),'rb') as f:
    uids = pickle.load(f)
with open(os.path.join(outdir,'X.p'),'rb') as f:
    X = pickle.load(f)
with open(os.path.join(outdir,'Y.p'),'rb') as f:
    Y = pickle.load(f)
with open(os.path.join(outdir,'Z.p'),'rb') as f:
    Z = pickle.load(f)

prior_posterior_check(uid,outdir,ebayes_beta_y)


