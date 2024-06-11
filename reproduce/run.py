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

# curate gold standard
set65 = {
        'NANOG':'ENSG00000111704',
        'CEACAM8':'ENSG00000124469',
        'TSPAN16':'ENSG00000130167',
        'SLC13A5':'ENSG00000141485',
        'GLRB':'ENSG00000109738',
        'DYRK4':'ENSG00000010219',
        'KCNN4':'ENSG00000104783',
        'SV2C':'ENSG00000122012',
        'SIGLEC8':'ENSG00000105366',
        'RBMXL3':'ENSG00000175718',
        'HIST1H1T':'ENSG00000187475',
        'CCR8':'ENSG00000179934',
        'CCNB3':'ENSG00000147082',
        'ALPPL2':'ENSG00000163286',
        'ZP2':'ENSG00000103310',
        'OTUB2':'ENSG00000089723',
        'LILRA4':'ENSG00000239961',
        'GRM2':'ENSG00000164082',
        'PSG1':'ENSG00000231924',
        'NBPF3':'ENSG00000142794',
        'GYPA':'ENSG00000170180',
        'ALPP':'ENSG00000163283',
        'SPATA19':'ENSG00000166118',
        'SLC6A11':'ENSG00000132164',
        'SLC34A1':'ENSG00000131183',
        'SLC2A14':'ENSG00000173262',
        'SLC22A12':'ENSG00000197891',
        'RBMXL2':'ENSG00000170748',
        'KLK3':'ENSG00000142515',
        'KLK2':'ENSG00000167751',
        'FCRL1':'ENSG00000163534',
        'CACNG3':'ENSG00000006116',
        'UPK3B':'ENSG00000243566',
        'FCRLA':'ENSG00000132185',
        'DCLK2':'ENSG00000170390',
        'IZUMO4':'ENSG00000099840',
        'MUC12':'ENSG00000205277',
        'HEPACAM':'ENSG00000165478',
        'BPI':'ENSG00000101425',
        'ATP6V0A4':'ENSG00000105929',
        'HMMR':'ENSG00000072571',
        'SLC45A3':'ENSG00000158715',
        'SLC4A1':'ENSG00000004939',
        'UPK1A':'ENSG00000105668',
        'CD79B':'ENSG00000007312',
        'CD27':'ENSG00000139193',
        'ADGRV1':'ENSG00000164199',
        'HERC5':'ENSG00000138646',
        'CD37':'ENSG00000104894',
        'CD2':'ENSG00000116824',
        'C3AR1':'ENSG00000171860',
        'SLC7A3':'ENSG00000165349',
        'FASLG':'ENSG00000117560',
        'NGB':'ENSG00000165553',
        'CELSR3':'ENSG00000008300',
        'CD3G':'ENSG00000160654',
        'CEACAM3':'ENSG00000170956',
        'TNFRSF13C':'ENSG00000159958',
        'CD72':'ENSG00000137101',
        'SLC46A2':'ENSG00000119457',
        'MS4A8':'ENSG00000166959',
        'CD79A':'ENSG00000105369',
        'CD3D':'ENSG00000167286',
        'CCR2':'ENSG00000121807',
        'CD83':'ENSG00000112149',
    }
df = pd.DataFrame(data=set65,index=['ensg']).T
df.index.name = 'symbol'
df.to_csv('gold_standard.txt',sep='\t')
sys.exit('stop')

# plot pr
def draw_PR(y_true,y_preds,outdir):
    plt.figure()
    baseline = np.sum(np.array(y_true) == 1) / len(y_true)
    for i,y_pred in enumerate(y_preds):
        precision,recall,_ = precision_recall_curve(y_true,y_pred,pos_label=1)
        area_PR = auc(recall,precision)
        lw = 1
        plt.plot(recall,precision,lw=lw, label='PR curve {} (area = {})'.format(i,round(area_PR,2)))
        plt.plot([0, 1], [baseline, baseline], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 0.15])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve example')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(outdir,'cart_set65_aupr_customized.pdf'),bbox_inches='tight')
    plt.close()

outdir = 'output'
result = pd.read_csv(os.path.join(outdir,'result_aupr.txt'),sep='\t',index_col=0)
y_preds = [
    np.negative(result['mean_sigma'].values),
    np.negative(result['Y_mean'].values),
    np.negative(result['X_mean'].values),
    result['Z_mean'].values
]
draw_PR(result['label'].values,y_preds,outdir)
sys.exit('stop')



# loss_df = pd.read_csv(os.path.join(outdir,'loss_df.txt'),sep='\t')
# ylims = (0,1.0e7)
# loss_df = loss_df.loc[loss_df['loss']<ylims[1],:]
# plt.figure(figsize=(5, 2))
# plt.plot(loss_df['loss'].values)
# plt.xlabel("SVI step")
# plt.ylabel("ELBO loss")
# plt.savefig('loss_check.pdf',bbox_inches='tight')
# plt.close()
# sys.exit('stop')

# result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
# sns.histplot(result['prior_sigma'])
# plt.savefig('check.pdf',bbox_inches='tight')
# plt.close()
