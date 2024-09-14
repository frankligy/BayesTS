#!/gpfs/data/yarmarkovichlab/Frank/test_bayesTS/test_bayes_env/bin/python3.7

import pandas as pd
import numpy as np
import sys,os
import anndata as ad
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'



# # inspection
# adata = ad.read_h5ad('gtex_gene_subsample_essential.h5ad')
# values = list(adata['ENSG00000139352',:].X.toarray()[0])
# fig,ax = plt.subplots()
# sns.histplot(values,ax=ax)
# values = list(adata['ENSG00000167751',:].X.toarray()[0])
# sns.histplot(values,ax=ax)
# plt.savefig('check.pdf',bbox_inches='tight')
# plt.close()


# first completely remove that from your adata

# adata = ad.read_h5ad('gtex_gene_subsample.h5ad')
# fixed_remove = [
#     'Ovary',
#     'Prostate',
#     'Testis',
#     'Vagina',
#     'Adrenal Gland',
#     'Cervix - Endocervix',
#     'Cervix - Ectocervix',
#     'Fallopian Tube',
#     'Pituitary'
# ]

# cond = [False if item in fixed_remove else True for item in adata.var['tissue']]
# adata = adata[:,cond]
# adata.write('gtex_gene_subsample_essential.h5ad')



# then please also make sure non-essential tissue, in protein tissue, is set to zero

# now should be all good to run

# add gene symbol
ori = pd.read_csv('/gpfs/data/yarmarkovichlab/Frank/pan_cancer/gene/full_results_XY_v2.txt',sep='\t',index_col=0)
mapping = ori['gene_symbol']
df = pd.read_csv(os.path.join('essential_output','full_results.txt'),sep='\t',index_col=0)
df['gene_symbol'] = df.index.map(mapping).values
df.rename(columns={'mean_sigma':'BayesTS'},inplace=True)
df.to_csv('full_results_XY_essential_tissues.txt',sep='\t')