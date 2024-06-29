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
from tqdm import tqdm
from scipy.sparse import csr_matrix

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

target = pd.read_csv(os.path.join('output_xyz','cart_set54_targets.txt'),sep='\t',index_col=0)
ensgs = target.index.tolist()

cart = pd.read_csv('cart_targets.txt',sep='\t',index_col=1)
ensg2symbol = cart['CART target'].to_dict()

adata = ad.read_h5ad('gtex_gene_subsample.h5ad')
protein = pd.read_csv('normal_tissue.tsv',sep='\t')
valid_level = set(['High','Medium','Low','Not detected'])
protein = protein.loc[protein['Level'].isin(valid_level),:]
from itertools import combinations
pairs = list(combinations(ensgs,2))

# ## build pair version of adata input
# adata = adata[ensgs,:]
# adata_list = []
# for pair in tqdm(pairs,total=len(pairs)):
#     no1, no2 = pair
#     adata1 = adata[no1,:]
#     adata2 = adata[no2,:]
#     x1 = adata1.X.toarray()
#     x2 = adata2.X.toarray()
#     x12 = np.concatenate([x1,x2],axis=0).min(axis=0)[np.newaxis,:]
#     mean_ = x12.mean()
#     symbol_ = ','.join([ensg2symbol[no1],ensg2symbol[no2]])
#     adata12 = ad.AnnData(X=x12,var=adata.var,obs=pd.DataFrame(index=[','.join([no1,no2])],data={'mean':[mean_],'symbol':[symbol_]}))
#     adata_list.append(adata12)
# adata_final = ad.concat(adata_list,axis=0,join='inner',merge='first')
# adata_final.X = csr_matrix(adata_final.X)
# adata_final.write('adata_final.h5ad')

# ## build pair version of protein input
# dic = {}
# ensgs_set = set(ensgs)
# for gene,sub_df in protein.groupby(by='Gene'):
#     if gene in ensgs_set:
#         dic[gene] = sub_df


# correspondance = {'High':0,
#                   'Medium':1,
#                   'Low':2,
#                   'Not detected':3}
# reverse_correspondance = {v:k for k,v in correspondance.items()}
# mapping_vec = np.vectorize(correspondance.get)

# pair_df_list = []
# for pair in tqdm(pairs,total=len(pairs)):
#     no1, no2 = pair
#     no1_df = dic[no1]
#     no2_df = dic[no2]
#     com_df = pd.concat([no1_df,no2_df],axis=0)
#     data = []
#     for t,sub_df in com_df.groupby(by='Tissue'):
#         tmp = list(sub_df.groupby(by='Gene'))   # [(ensg,sub_df),(ensg,sub_df)]
#         if len(tmp) == 1:
#             continue
#         dfs = [item.reset_index() for item in list(zip(*tmp))[1]]
#         tmps = dfs[0].join(other=dfs[1],how='inner',lsuffix='_1',rsuffix='_2')
#         try:
#             v = mapping_vec(tmps.loc[:,['Level_1','Level_2']].values).max(axis=1)
#         except:
#             print(tmps.loc[:,['Level_1','Level_2']].values)
#             raise Exception
#         for item in v:
#             add = (','.join([no1,no2]),','.join([ensg2symbol[no1],ensg2symbol[no2]]),t,reverse_correspondance[item])
#             data.append(add)
#     pair_df = pd.DataFrame.from_records(data,columns=['Gene','Gene name','Tissue','Level'])
#     pair_df_list.append(pair_df)
# final_protein = pd.concat(pair_df_list,axis=0).reset_index()
# final_protein.to_csv('final_protein.txt',sep='\t',index=None)


# post-analysis
result = pd.read_csv(os.path.join('output_logic_gate_xyz','full_results.txt'),sep='\t',index_col=0)
col = []
for item in result.index:
    no1,no2 = item.split(',')
    gene1 = ensg2symbol[no1]
    gene2 = ensg2symbol[no2]
    col.append(','.join([gene1,gene2]))
result['gene_symbols'] = col

# compare to their respective single arm
db = pd.read_csv(os.path.join('output_xyz','full_results.txt'),sep='\t',index_col=0)
gene2sigma = db['mean_sigma'].to_dict()
col1 = []
col2 = []
col3 = []
col4 = []
for item,sigma in zip(result.index.values,result['mean_sigma']):
    t1,t2 = item.split(',')
    sigma1 = gene2sigma[t1]
    sigma2 = gene2sigma[t2]
    delta1 = sigma - sigma1
    delta2 = sigma - sigma2
    col1.append(sigma1)
    col2.append(sigma2)
    col3.append(delta1)
    col4.append(delta2)
result['sigma1'] = col1
result['sigma2'] = col2
result['delta1'] = col3
result['delta2'] = col4
result.to_csv(os.path.join('output_logic_gate_xyz','full_results_post.txt'),sep='\t')


# heatmap
reversed_result = result.copy()
reversed_result['gene_symbols'] = [','.join([item.split(',')[1],item.split(',')[0]]) for item in result['gene_symbols']]
result = pd.concat([result,reversed_result],axis=0)

col1 = [item.split(',')[0] for item in result['gene_symbols']]
col2 = [item.split(',')[1] for item in result['gene_symbols']]
result['gene1'] = col1
result['gene2'] = col2
result_new = result.loc[:,['mean_sigma','gene1','gene2']]

df = result_new.groupby(by=['gene1','gene2'])['mean_sigma'].mean().unstack(fill_value=1)

u_indices = np.triu_indices(df.shape[0],k=1)
u_values = df.values[u_indices]
reshaped_u_indices = [(i,j) for i,j in zip(*u_indices)] 
new_reshaped_u_indices = []
for u_i,u_v in zip(reshaped_u_indices,u_values):
    if u_v > 0:
        new_reshaped_u_indices.append(u_i)
new_u_indices = list(zip(*new_reshaped_u_indices))
new_u_indices = tuple([np.array(item) for item in new_u_indices])
tmp = df.values.T
tmp[new_u_indices] = df.values[new_u_indices]
new = tmp.T
new_df = pd.DataFrame(data=new,index=df.index,columns=df.columns)

sorted_gene = [ensg2symbol[item] for item in target.sort_values(by='mean_sigma',ascending=True).index]
new_df = new_df.loc[sorted_gene,sorted_gene]

fig,ax = plt.subplots()
sns.heatmap(new_df,mask=np.triu(new_df.values,k=0),cmap='viridis',xticklabels=1,yticklabels=1,ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=1)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=1)
plt.savefig(os.path.join('output_logic_gate_xyz','heatmap_lower_new.pdf'),bbox_inches='tight')
plt.close()
sys.exit('stop')

# visualize pair

def gtex_visual_combine_barplot(adata,uid,outdir='.',ylim=None,facecolor='#D99066',figsize=(6.4,4.8)):
    '''
    facecolor : #D99066, #50BF80 #795D8C
    '''
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    query = uid
    try:
        info = adata[[query],:]
    except:
        print('{} not detected in gtex, impute as zero'.format(query))
        info = ad.AnnData(X=csr_matrix(np.full((1,adata.shape[1]),0)),obs=pd.DataFrame(data={'mean':[0]},index=[uid]),var=adata.var)  # weired , anndata 0.7.6 can not modify the X in place? anndata 0.7.2 can do that in scTriangulate
    title = query
    identifier = query.replace(':','_')
    df = pd.DataFrame(data={'value':info.X.toarray().squeeze(),'tissue':info.var['tissue'].values},index=info.var_names)
    tmp = []
    for tissue,sub_df in df.groupby(by='tissue'):
        tmp.append((sub_df,sub_df['value'].mean(),tissue))
    fig,ax = plt.subplots(figsize=figsize)
    ax.bar(x=np.arange(len(tmp)),height=[item[1] for item in tmp],color=facecolor)
    ax.set_xticks(ticks=np.arange(len(tmp)),labels=[item[2] for item in tmp],fontsize=1,rotation=90)
    ax.set_xlabel('Tissues')
    ax.set_ylabel('TPM')
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.savefig(os.path.join(outdir,'gtex_visual_combine_barplot_norm_{}.pdf'.format(identifier)),bbox_inches='tight')
    plt.close()

# ENSG00000240505,ENSG00000196083
# TNFRSF13B,IL1RAP
adata_single = ad.read_h5ad('gtex_gene_subsample.h5ad')
adata_pair = ad.read_h5ad('adata_final.h5ad')
gtex_visual_combine_barplot(adata_single,'ENSG00000240505','output_logic_gate_xyz',ylim=(0,30),facecolor='#D99066')
gtex_visual_combine_barplot(adata_single,'ENSG00000196083','output_logic_gate_xyz',ylim=(0,30),facecolor='#50BF80')
gtex_visual_combine_barplot(adata_pair,'ENSG00000240505,ENSG00000196083','output_logic_gate_xyz',ylim=(0,30),facecolor='#795D8C')

