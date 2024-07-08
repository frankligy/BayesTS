#!/data/salomonis2/LabFiles/Frank-Li/scTriangulate/spatial_slide/sctri_spatial_env/bin/python3.7

import scanpy as sc
import anndata as ad
import pandas as pd
import numpy as np
import subprocess
from tqdm import tqdm
import os,sys
sys.path.insert(0,'/data/salomonis2/software')
from sctriangulate import *
from sctriangulate.preprocessing import *
from sctriangulate.colors import bg_greyed_cmap

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

# # load the data
# adata = small_txt_to_adata('./GSM3828672_Smartseq2_GBM_IDHwt_processed_TPM.tsv',gene_is_index=True,sep='\t')  # 7930 × 23686

# # process
# adata.var_names_make_unique()
# adata.var['mt'] = adata.var_names.str.startswith('MT-')
# sc.pp.calculate_qc_metrics(adata,qc_vars=['mt'],percent_top=None,inplace=True,log1p=False)
# sc.pp.highly_variable_genes(adata,flavor='seurat',n_top_genes=3000)
# adata.raw = adata
# adata = adata[:,adata.var['highly_variable']]
# sc.pp.regress_out(adata,['total_counts','pct_counts_mt'])
# sc.pp.scale(adata,max_value=10)
# sc.tl.pca(adata,n_comps=50)
# sc.pp.neighbors(adata)
# sc.tl.leiden(adata,resolution=1,key_added='leiden_1')
# sc.tl.umap(adata)
# adata = adata.raw.to_adata()
# if not issparse(adata.X):
#     adata.X = csr_matrix(adata.X)
# adata.write('GBM_data_processed.h5ad')

# visualize and conduct diffential
adata = sc.read('GBM_data_processed.h5ad')
# adata.obs['sample'] = [item.split('-')[0] for item in adata.obs_names]
# umap_dual_view_save(adata,cols=['leiden_1','sample'])
# sc.tl.rank_genes_groups(adata,groupby='leiden_1')  # ['params', 'names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
# for cluster in adata.obs['leiden_1'].unique():
#     n = adata.uns['rank_genes_groups']['names'][cluster]
#     lfc = adata.uns['rank_genes_groups']['logfoldchanges'][cluster]
#     p = adata.uns['rank_genes_groups']['pvals'][cluster]
#     padj = adata.uns['rank_genes_groups']['pvals_adj'][cluster]
#     df = pd.DataFrame({'name':n,'lfc':lfc,'pval':p,'padj':padj})
#     df.to_csv('DEG/cluster_{}_results.txt'.format(cluster),sep='\t')
# adata.obs.to_csv('cell_annotations.txt',sep='\t')

# inspect and alternative approach
# adata.var.to_csv('var.txt',sep='\t')

# sc.pl.umap(adata,color=['SLC7A6OS','RNF139','STX5','E2F4','WDTC1','RCE1','DLG5-AS1','OTUD4','RHBDD1','INO80'],color_map=bg_greyed_cmap('viridis'),vmin=1e-5)
# sc.pl.umap(adata,color=['CXorf51A','CXorf51B','GSC2','KRTAP19-5','MIR20B','MIR519B','MIR550A2','OR2V2','AA06','BSPH1'],color_map=bg_greyed_cmap('viridis'),vmin=1e-5)
# sc.pl.umap(adata,color=['HBB','CRYAA','INS-IGF2','SERPINB2','C8orf22','SAA1','CCL7','CPA3','TH','CXCL1'],color_map=bg_greyed_cmap('viridis'),vmin=1e-5)
sc.pl.umap(adata,color=['ITGAV','SEC61G'],color_map=bg_greyed_cmap('viridis'),vmin=1e-5)
plt.savefig('bayesTS_targets.pdf',bbox_inches='tight')
plt.close()
sys.exit('stop')





# lis = []
# for cluster in adata.obs['leiden_1'].unique():
#     adata_c = adata[adata.obs['leiden_1']==cluster,:]
#     ave = adata_c.to_df().mean(axis=0)
#     ave.name = cluster
#     lis.append(ave)
# result = pd.concat(lis,axis=1)
# cv = result.apply(func=lambda x:x.std()/x.mean(),axis=1,result_type='reduce')
# cv.name = 'cv'
# cv.to_csv('cv_frank.txt',sep='\t')












