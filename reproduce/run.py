#!/gpfs/data/yarmarkovichlab/Frank/BayesTS/logit_gate_env/bin/python3.7

import os,sys
import pandas as pd
import numpy as np
import anndata as ad
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr,ttest_ind,ttest_rel
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve,auc,roc_curve,accuracy_score
import pickle
import pyro.distributions as dist
import torch
from scipy.sparse import csr_matrix

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'


'''curate a ERV'''
# normal_erv = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/immunopeptidome_project/NeoVerse/GTEx/selected/hg38_telocal_intron/normal_erv.txt',sep='\t',index_col=0)
# normal_erv_aux = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/immunopeptidome_project/NeoVerse/GTEx/selected/hg38_telocal_intron/normal_erv_aux_df.txt',sep='\t',index_col=0)
# erv_data = normal_erv.values / normal_erv_aux['total_count'].values.reshape(1,-1) * 1e6
# adata = ad.AnnData(X=csr_matrix(erv_data),obs=pd.DataFrame(index=normal_erv.index),var=pd.DataFrame(index=normal_erv.columns))
# adata.var['tissue'] = [item.split(',')[1] for item in adata.var_names]
# adata.obs['mean'] = erv_data.mean(axis=1)
# adata.write('adata_erv.h5ad')


'''reproducibility result, which also have tau and each modality'''
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
# base_result = pd.read_csv(os.path.join('reproducibility','output_1','full_results.txt'),sep='\t',index_col=0)
# base_order = base_result.index
# for outdir in ['output_1','output_2','output_3','output_4','output_5']:
#     result = pd.read_csv(os.path.join('reproducibility',outdir,'full_results.txt'),sep='\t',index_col=0).loc[base_order,:]
#     result['label'] = [True if item in gs else False for item in result.index]
#     y_preds[outdir] = np.negative(result['mean_sigma'].values)
# draw_PR(result['label'].values,y_preds,'reproducibility','PR_curve_reproducibility_test.pdf')

'''deg'''
# suggestion to use all normal tissues to run limma
# to run that in altanalyze, I need exp.original-steady-state, groups and comps
# adata = ad.read_h5ad('gtex_gene_subsample.h5ad')
# tpm_normal = adata.to_df()
# tpm_tcga = pd.read_csv('TCGA_SKCM_gene_tpm.txt',sep='\t',index_col=0)
# tpm_all = pd.concat([tpm_tcga,tpm_normal],join='inner',axis=1)
# tpm_all.to_csv('exp.original-steady-state.txt',sep='\t')
# with open('groups.txt','w') as f:
#     for item in tpm_all.columns:
#         if item.startswith('TCGA'):
#             f.write('{}\t{}\t{}\n'.format(item,1,'tumor'))
#         elif item.startswith('GTEX'):
#             f.write('{}\t{}\t{}\n'.format(item,2,'normal'))
# with open('comps.txt','w') as f:
#     f.write('1\t2\n')


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
ensg2adjp_skin_only = deg['adjp'].to_dict()
ensg2lfc_skin_only = deg['Log2(Fold Change)'].to_dict()
deg = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/test_altanalyze/altanalyze_output/ExpressionInput/DEGs-LogFold_0.0_adjp/GE.tumor_vs_normal.txt',sep='\t',index_col=0)
ensg2adjp_all_normal = deg['adjp'].to_dict()
ensg2lfc_all_normal = deg['LogFold'].to_dict()
result['limma_adjp_skin_only'] = result.index.map(ensg2adjp_skin_only).values
result['limma_lfc_skin_only'] = result.index.map(ensg2lfc_skin_only).values
result = result.loc[result['limma_adjp_skin_only'].notna(),:]
result = result.loc[result['limma_lfc_skin_only'].notna(),:]
result['limma_adjp_all_normal'] = result.index.map(ensg2adjp_all_normal).values
result['limma_lfc_all_normal'] = result.index.map(ensg2lfc_all_normal).values
result = result.loc[result['limma_adjp_all_normal'].notna(),:]
result = result.loc[result['limma_lfc_all_normal'].notna(),:]

from scipy.stats import rankdata

result['lfc_rank_skin_only'] = rankdata(np.negative(result['limma_lfc_skin_only'].values),method='min')
result['adjp_rank_skin_only'] = rankdata(result['limma_adjp_skin_only'].values,method='min')
result['limma_rank_skin_only'] = [(r1+r2)/2 for r1,r2 in zip(result['lfc_rank_skin_only'],result['adjp_rank_skin_only'])]

result['lfc_rank_all_normal'] = rankdata(np.negative(result['limma_lfc_all_normal'].values),method='min')
result['adjp_rank_all_normal'] = rankdata(result['limma_adjp_all_normal'].values,method='min')
result['limma_rank_all_normal'] = [(r1+r2)/2 for r1,r2 in zip(result['lfc_rank_all_normal'],result['adjp_rank_all_normal'])]

y_preds = {
    'BayesTS_ext':np.negative(result['mean_sigma'].values),
    'limma_rank_skin_only':np.negative(result['limma_rank_skin_only'].values),
    'limma_rank_all_normal':np.negative(result['limma_rank_all_normal'].values)
}


result.to_csv('deg_bayesTS_new.txt',sep='\t')
draw_PR(result['label'].values,y_preds,'.','PR_curve_deg_new.pdf')
sys.exit('stop')

'''sensitivity'''
# create such 5 beta distribution
# from scipy.stats import beta
# fig,ax = plt.subplots(figsize=(6.4,4.8))
# tups = [(2,2),(2,4),(2,8),(4,2),(8,2)]
# for tup in tups:
#     a,b = tup
#     x = np.linspace(beta.ppf(0.001,a=a,b=b),beta.ppf(0.999,a=a,b=b),1000)
#     y = beta.pdf(x,a=a,b=b)
#     ax.plot(x,y,linewidth=2,label='{}_{}'.format(a,b))
# ax.legend()
# ax.set_xlabel('Tumor specificity')
# ax.set_ylabel('Probability')
# plt.savefig(os.path.join('sensitivity','schema.pdf'),bbox_inches='tight')
# plt.close()

# y_preds = {}
# gs = pd.read_csv('gold_standard.txt',sep='\t')['ensg'].values.tolist()
# base_result = pd.read_csv(os.path.join('sensitivity','output_2_2','full_results.txt'),sep='\t',index_col=0)
# base_order = base_result.index
# for outdir in ['output_2_2','output_2_4','output_2_8','output_4_2','output_8_2']:
#     result = pd.read_csv(os.path.join('sensitivity',outdir,'full_results.txt'),sep='\t',index_col=0).loc[base_order,:]
#     result['label'] = [True if item in gs else False for item in result.index]
#     y_preds[outdir] = np.negative(result['mean_sigma'].values)
# draw_PR(result['label'].values,y_preds,'sensitivity','PR_curve_sensitivity_test.pdf')

# fig,ax = plt.subplots()
# for outdir in ['output_2_2','output_2_4','output_2_8','output_4_2','output_8_2']:
#     result = pd.read_csv(os.path.join('sensitivity',outdir,'full_results.txt'),sep='\t',index_col=0)
#     data = result['mean_sigma'].values
#     sns.ecdfplot(data)
# plt.savefig(os.path.join('sensitivity','ecdf_plots.pdf'),bbox_inches='tight')
# plt.close()

'''speed test'''
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


# n = [10,50,100,200,500,1000]
# mcmc = [16.145457983016968, 39.225778341293335, 74.17322897911072, 161.4286584854126, 428.83027958869934,1462.81707406044]
# vi = [5.572661399841309, 7.877718210220337, 12.209128618240356, 22.4102463722229, 58.071091175079346,166.65992164611816]
# fig, ax = plt.subplots()
# ax.plot(np.arange(len(n)),mcmc,marker='o',label='mcmc')
# ax.plot(np.arange(len(n)),vi,marker='o',label='vi')
# ax.legend(frameon=False)
# ax.set_xticks(np.arange(len(n)))
# ax.set_xticklabels(n)
# ax.set_xlabel('Number of genes')
# ax.set_ylabel('Time(s)')
# plt.savefig('mcmc_vi.pdf',bbox_inches='tight')
# plt.close()

'''weight adjust'''

# outdir = 'output_check'
# annotated_x = pd.read_csv(os.path.join(outdir,'annotated_x.txt'),sep='\t',index_col=0)
# markers = {
#     'ENSG00000079112':'CDH17',
#     'ENSG00000185686':'PRAME',
#     'ENSG00000177455':'CD19',
# }

# marker_x = annotated_x.loc[list(markers.keys()),:].T
# marker_x.to_csv('marker_x.txt',sep='\t')

# organs = ['testis', 'immune', 'gi']
# ensgs = ['ENSG00000185686', 'ENSG00000177455', 'ENSG00000079112']
# ylims = [(0,0.15),(0,0.20),(0,0.18)]
# for organ,ensg,ylim in zip(organs,ensgs,ylims):
#     dic = {}
#     for weight in [0.9,0.5,0.1]:
#         each_weight = []
#         for i in [1,2,3,4,5]:
#             outdir = os.path.join('weight_adjust','output_weights_{}_{}_{}'.format(organ,weight,i))
#             result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
#             each_weight.append(result.at[ensg,'mean_sigma'])
#         dic[weight] = each_weight
#     t1,s1 = ttest_rel(dic[0.9],dic[0.5])
#     t2,s2 = ttest_rel(dic[0.5],dic[0.1])
#     t3,s3 = ttest_rel(dic[0.9],dic[0.1])

#     fig,ax = plt.subplots()
#     sns.swarmplot(list(dic.values()),ax=ax)
#     ax.set_ylim(ylim)
#     ax.set_xticks((0,1,2))
#     ax.set_xticklabels(('upweight','default','downweight'))
#     ax.set_ylabel('{} BayesTS'.format(ensg))
#     ax.set_title('{}_{}_{}'.format(s1,s2,s3))
#     plt.savefig(os.path.join('weight_adjust','{}_{}.pdf'.format(organ,ensg)),bbox_inches='tight')
#     plt.close()

'''find new targets'''
# result = pd.read_csv(os.path.join('ablation','output_xyz','full_results.txt'),sep='\t',index_col=0)
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

def gtex_visual_combine(adata,uid,outdir='.',figsize=(6.4,4.8),tumor=None,ylim=None):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    query = uid
    try:
        info = adata[[query],:]
    except:
        print('{} not detected in gtex, impute as zero'.format(query))
        info = ad.AnnData(X=csr_matrix(np.full((1,adata.shape[1]),0)),obs=pd.DataFrame(data={'mean':[0]},index=[uid]),var=adata.var) 
    title = query
    identifier = query.replace(':','_')
    df = pd.DataFrame(data={'value':info.X.toarray().squeeze(),'tissue':info.var['tissue'].values},index=info.var_names)
    tmp = []
    for tissue,sub_df in df.groupby(by='tissue'):
        tmp.append((sub_df,sub_df['value'].mean()))
    sorted_sub_df_list = list(list(zip(*sorted(tmp,key=lambda x:x[1])))[0])
    tumor_query_value = tumor.loc[query,:].values.squeeze()
    tumor_sub_df = pd.DataFrame(data={'value':tumor_query_value,'tissue':['tumor']*tumor.shape[1]},index=tumor.columns)
    sorted_sub_df_list.append(tumor_sub_df)

    fig,ax = plt.subplots(figsize=figsize)
    x = 0
    x_list = []
    y_list = []
    v_delimiter = [0]
    xticklabel = []
    total_number_tissues = len(sorted_sub_df_list)
    c_list_1 = np.concatenate([np.array(['g']*sub_df.shape[0]) for sub_df in sorted_sub_df_list[:-1]]).tolist()
    c_list_2 = ['r'] * sorted_sub_df_list[-1].shape[0]
    c_list = c_list_1 + c_list_2

    sorted_sub_df_list = [df,tumor_sub_df]

    for i,sub_df in enumerate(sorted_sub_df_list):
        sub_df.sort_values(by='value',inplace=True)
        n = sub_df.shape[0]
        xticklabel.append(sub_df['tissue'].iloc[0])
        for j,v in enumerate(sub_df['value']):
            x_list.append(x)
            y_list.append(v)
            x += 1
            if j == n-1:
                v_delimiter.append(x)
                x += 1
    ax.scatter(x_list,y_list,s=2,c=c_list,marker='o')

    for v in v_delimiter[1:-1]:
        ax.axvline(v,linestyle='--',linewidth=0.5)
    xtick = [(v + v_delimiter[i+1])/2 for i,v in enumerate(v_delimiter[:-1])]
    ax.set_xticks(xtick)
    ax.set_xticklabels(xticklabel,rotation=90,fontsize=1)
    
    ax.set_title(title)
    ylabel = 'Raw read counts'
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Normal Tissues --> Tumor')
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.savefig(os.path.join(outdir,'gtex_visual_combine_{}.pdf'.format(identifier)),bbox_inches='tight')
    plt.close()

# adata = ad.read_h5ad('gtex_gene_subsample.h5ad')
# outdir = '.'
# tumor = pd.read_csv('TCGA_SKCM_gene_tpm.txt',sep='\t',index_col=0)
# gtex_visual_combine(adata,uid='ENSG00000126856',outdir=outdir,tumor=tumor)  # PRDM7
# gtex_visual_combine(adata,uid='ENSG00000183668',outdir=outdir,tumor=tumor)  # PSG9
# gtex_visual_combine(adata,uid='ENSG00000243130',outdir=outdir,tumor=tumor)  # PSG11
# gtex_visual_combine(adata,uid='ENSG00000170848',outdir=outdir,tumor=tumor)  # PSG6   killed
# gtex_visual_combine(adata,uid='ENSG00000156269',outdir=outdir,tumor=tumor)  # NAA11
# gtex_visual_combine(adata,uid='ENSG00000166049',outdir=outdir,tumor=tumor)  # PASD1
# gtex_visual_combine(adata,uid='ENSG00000187772',outdir=outdir,tumor=tumor)  # LIN28B
# gtex_visual_combine(adata,uid='ENSG00000268009',outdir=outdir,tumor=tumor)  # SSX4  killed
# gtex_visual_combine(adata,uid='ENSG00000231924',outdir=outdir,tumor=tumor)  # PSG1
# gtex_visual_combine(adata,uid='ENSG00000242221',outdir=outdir,tumor=tumor)  # PSG2
# gtex_visual_combine(adata,uid='ENSG00000221826',outdir=outdir,tumor=tumor)  # PSG3
# gtex_visual_combine(adata,uid='ENSG00000243137',outdir=outdir,tumor=tumor)  # PSG4
# gtex_visual_combine(adata,uid='ENSG00000204941',outdir=outdir,tumor=tumor)  # PSG5
# gtex_visual_combine(adata,uid='ENSG00000221878',outdir=outdir,tumor=tumor)  # PSG7

'''junctions'''
# for TUFM, we use SRR1072223 chr16 to confirm absence in normal skin as well
# result = pd.read_csv(os.path.join('output_splicing_xy','full_results.txt'),sep='\t',index_col=0)
# result = result.sort_values(by='mean_sigma')
# fig,ax = plt.subplots(figsize=(6,4.8))
# ax.bar(x=np.arange(result.shape[0]),height=result['mean_sigma'].values)
# # index_MEGA1A = result.index.tolist().index('ENSG00000001461:E6.1-E7.1_24445189')
# # index_MS4A1 = result.index.tolist().index('ENSG00000149582:E6.2-E7.1')
# # ax.axvline(x=index_MEGA1A,c='r',linestyle='--')
# # ax.axvline(x=index_MS4A1,c='r',linestyle='--')
# ax.set_xlabel('Targets ranked by tumor specificity')
# ax.set_ylabel('Inferred tumor specificity score')
# plt.savefig('new_targets_splicing_just_plot.pdf',bbox_inches='tight')
# plt.close()

'''compare CPM and TPM'''
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


''' re-sample 25 per tissues for GTEx TPM'''
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




''' junction cpm'''
# adata = ad.read_h5ad('combined_normal_count.h5ad')
# cpm = adata.X.toarray() / adata.var['total_count'].values.reshape(1,-1)
# adata = ad.AnnData(X=csr_matrix(cpm),var=adata.var,obs=adata.obs)
# adata.write('combined_normal_cpm.h5ad')




'''prior and posterior check'''
# def prior_posterior_check(uid,outdir,ebayes_beta_y):
#     full_result = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
#     index = uids.index(uid)
#     prior_sigma = full_result.loc[uid,'prior_sigma']
#     posterior_sigma = full_result.loc[uid,'mean_sigma']

#     # c
#     beta_x = 25
#     data = X[:,index]
#     prior = dist.Poisson(beta_x*prior_sigma).expand([t]).sample().data.cpu().numpy()
#     posterior = dist.Poisson(beta_x*posterior_sigma).expand([t]).sample().data.cpu().numpy()
#     fig,ax = plt.subplots()
#     for item in [data,prior,posterior]:
#         sns.histplot(item,ax=ax,stat='probability',alpha=0.5,bins=10)
#     plt.savefig(os.path.join(outdir,'{}_c.pdf'.format(uid)),bbox_inches='tight')
#     plt.close()

#     # nc
#     beta_y = ebayes_beta_y
#     data = Y[:,index]
#     prior = dist.LogNormal(beta_y*prior_sigma,0.5).expand([s]).sample().data.cpu().numpy()
#     posterior = dist.LogNormal(beta_y*posterior_sigma,0.5).expand([s]).sample().data.cpu().numpy()
#     fig,ax = plt.subplots()
#     for item in [data,prior,posterior]:
#         sns.histplot(np.log(item),ax=ax,stat='probability',bins=40,alpha=0.5)
#     plt.savefig(os.path.join(outdir,'{}_nc.pdf'.format(uid)),bbox_inches='tight')
#     plt.close()

#     # pc
#     data = Z[:,index]
#     probs = [torch.tensor([2/3*sigma,1/3*sigma,1/3*(1-sigma),2/3*(1-sigma)]) for sigma in [prior_sigma,posterior_sigma]]
#     prior = dist.Categorical(probs[0]).expand([p]).sample().data.cpu().numpy()
#     posterior = dist.Categorical(probs[1]).expand([p]).sample().data.cpu().numpy()
#     fig,axes = plt.subplots(ncols=3,gridspec_kw={'wspace':0.5})
#     axes = axes.flatten()
#     for i,(item,color,title) in enumerate(zip([data,prior,posterior],['#7995C4','#E5A37D','#80BE8E'],['data','prior','posterior'])): 
#         sns.histplot(item,ax=axes[i],stat='count',bins=4,alpha=0.5,facecolor=color)
#         axes[i].set_ylim([0,90])
#         axes[i].set_title(title)
#     plt.savefig(os.path.join(outdir,'{}_pc.pdf'.format(uid)),bbox_inches='tight')
#     plt.close()

# outdir = 'ablation/output_xyz'
# ebayes_beta_y = 4.301273922770753
# s = 1228
# t = 49
# p = 89
# uid = 'ENSG00000111640'  #ENSG00000111640 ENSG00000198681
# with open(os.path.join(outdir,'uids.p'),'rb') as f:
#     uids = pickle.load(f)
# with open(os.path.join(outdir,'X.p'),'rb') as f:
#     X = pickle.load(f)
# with open(os.path.join(outdir,'Y.p'),'rb') as f:
#     Y = pickle.load(f)
# with open(os.path.join(outdir,'Z.p'),'rb') as f:
#     Z = pickle.load(f)

# prior_posterior_check(uid,outdir,ebayes_beta_y)


'''threshold the gene matrix when deriving the tissue distribution'''
# dic = {}
# for cutoff in np.arange(0,10,0.25):
#     result_path = 'result_{}.txt'.format(str(cutoff))
#     result = pd.read_csv(result_path,sep='\t',index_col=0)
#     repr_spearman = np.nanmean(result['spearman'].values)
#     repr_aupr = np.nanmean(result['aupr'].values)
#     dic[cutoff] = ((repr_spearman,repr_aupr))

'''add gene symbol'''
def ensemblgene_to_symbol(query,species):
    # assume query is a list, will also return a list
    import mygene
    mg = mygene.MyGeneInfo()
    out = mg.querymany(query,scopes='ensemblgene',fileds='symbol',species=species,returnall=True,as_dataframe=True,df_index=True)
    result = out['out']['symbol'].fillna('unknown_gene').tolist()
    try:
        assert len(query) == len(result)
    except AssertionError:    # have duplicate results
        df = out['out']
        df_unique = df.loc[~df.index.duplicated(),:]
        result = df_unique['symbol'].fillna('unknown_gene').tolist()
    return result

# outdir = 'ablation/output_xyz'
# df = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
# ensgs = list(set(df.index.tolist()))
# symbols = ensemblgene_to_symbol(ensgs,'human')
# mapping = {item1:item2 for item1,item2 in zip(ensgs,symbols)}
# df['gene_symbol'] = df.index.map(mapping).values
# df.rename(columns={'mean_sigma':'BayesTS'},inplace=True)
# df.to_csv('full_results_XYZ.txt',sep='\t')

# outdir = 'output_no_xy'
# df = pd.read_csv(os.path.join(outdir,'full_results.txt'),sep='\t',index_col=0)
# ensgs = list(set(df.index.tolist()))
# symbols = ensemblgene_to_symbol(ensgs,'human')
# mapping = {item1:item2 for item1,item2 in zip(ensgs,symbols)}
# df['gene_symbol'] = df.index.map(mapping).values
# df.rename(columns={'mean_sigma':'BayesTS'},inplace=True)
# df.to_csv('full_results_XY.txt',sep='\t')


