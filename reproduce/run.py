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


# diagnose
def diagnose_2d(outdir,ylim=(-1,200)):
    df = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
    fig,ax = plt.subplots()
    im = ax.scatter(df['X_mean'],df['Y_mean'],c=df['mean_sigma'],s=0.5**2,cmap='viridis')
    plt.colorbar(im)
    ax.set_ylabel('average_normalized_counts')
    ax.set_xlabel('average_n_present_samples_per_tissue')
    ax.set_ylim(ylim)
    plt.savefig(os.path.join(outdir,'diagnosis_2d.pdf'),bbox_inches='tight')
    plt.close()

def diagnose_3d(outdir):
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

def cart_set54_benchmark(target_path,outdir):
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

outdir = 'output'

diagnose_2d(outdir)
diagnose_3d(outdir)
cart_set54_benchmark('cart_targets.txt',outdir)
sys.exit('stop')

loss_df = pd.read_csv(os.path.join(outdir,'loss_df.txt'),sep='\t')
ylims = (0,1.0e7)
loss_df = loss_df.loc[loss_df['loss']<ylims[1],:]
plt.figure(figsize=(5, 2))
plt.plot(loss_df['loss'].values)
plt.xlabel("SVI step")
plt.ylabel("ELBO loss")
plt.savefig('loss_check.pdf',bbox_inches='tight')
plt.close()
sys.exit('stop')

result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
sns.histplot(result['prior_sigma'])
plt.savefig('check.pdf',bbox_inches='tight')
plt.close()
