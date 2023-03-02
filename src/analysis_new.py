#!/data/salomonis2/LabFiles/Frank-Li/refactor/neo_env/bin/python3.7

import os
import sys
import pandas as pd
sys.path.insert(0,'/data/salomonis2/software')
import snaf
import numpy as np
import anndata as ad
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import subprocess

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'

'''mainly for generating sashimi plot'''


# # get reduced junction
# df = snaf.get_reduced_junction_matrix(pc='counts.TCGA-SKCM.txt',pea='Hs_RNASeq_top_alt_junctions-PSI_EventAnnotation.txt')

# # run SNAF
# netMHCpan_path = '/data/salomonis2/LabFiles/Frank-Li/refactor/external/netMHCpan-4.1/netMHCpan'
# db_dir = '/data/salomonis2/LabFiles/Frank-Li/neoantigen/revision/data'
# tcga_ctrl_db = ad.read_h5ad(os.path.join(db_dir,'controls','tcga_matched_control_junction_count.h5ad'))
# gtex_skin_ctrl_db = ad.read_h5ad(os.path.join(db_dir,'controls','gtex_skin_count.h5ad'))
# add_control = {'tcga_control':tcga_ctrl_db,'gtex_skin':gtex_skin_ctrl_db}

# snaf.initialize(df=df,db_dir=db_dir,binding_method='netMHCpan',software_path=netMHCpan_path,add_control=add_control)
# surface.initialize(db_dir=db_dir)


'''
# task 1: ENSG00000178952:I8.1_28843525-E9.1 TCGA-EE-A3AG-06A-31R-A18S-07 chr16:28843148-28843525 (1500,1500)
# task 2: ENSG00000000419:E6.1-E8.2 TCGA-EB-A5VU-01A-21R-A32P-07 chr20:50940933-50942031 (1500,1500)
'''

'''
inspect those candidates for junction validity and tumor specificity
'''

# ts = pd.read_csv('../dev/full_results_XY.txt',sep='\t',index_col=0)
# result = pd.read_csv('/data/salomonis2/LabFiles/Frank-Li/neoantigen/revision/TCGA_melanoma/result_new/frequency_stage0_verbosity1_uid_gene_symbol_coord_mean_mle.txt',sep='\t',index_col=0)
# count = snaf.remove_trailing_coord('../bayesian/counts.TCGA-SKCM.txt')
# count_max = pd.Series(index=count.index,data=count.values.max(axis=1)).to_dict()
# count_n = pd.Series(index=count.index,data=np.count_nonzero(count.values,axis=1)).to_dict()
# s2g = pd.Series(index=[item.split(',')[0] for item in result.index],data=result['symbol'].values)
# s2c = pd.Series(index=[item.split(',')[0] for item in result.index],data=result['coord'].values)
# subset = ts.loc[(ts['mean_sigma']<0.87) & (ts['mean_sigma']>0.86),:]
# subset['max'] = subset.index.map(count_max).values
# subset['n'] = subset.index.map(count_n).values
# subset['gene'] = subset.index.map(s2g).values
# subset['coord'] = subset.index.map(s2c).values
# subset.to_csv('area0.86_0.87.txt',sep='\t')


uid = 'ENSG00000000419:E6.1-E8.2'
# count = snaf.remove_trailing_coord('../bayesian/counts.TCGA-SKCM.txt')
# count.loc[uid,:].to_csv('each_check.txt',sep='\t')
# sys.exit('stop')


control_bam_path_list = ['/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-BREAST-CANCER/TCGA-files-Ens91/bams/TCGA-BH-A208-11A-51R-A157-07.bam',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-Kidney/KIRC/TCGA-CJ-5680-11A-01R-1541-07.bam']
control_bai_path_list = ['/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-BREAST-CANCER/TCGA-files-Ens91/bams/TCGA-BH-A208-11A-51R-A157-07.bam.bai',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-Kidney/KIRC/TCGA-CJ-5680-11A-01R-1541-07.bam.bai']

def unit_run(uid,base_sample,region,task_name):
    uid = uid
    sample = base_sample + '.bed'
    region = region
    bam_path_list = ['/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-SKCM/TCGA_SKCM-BAMs/bams/{}.bam'.format(base_sample)] + control_bam_path_list
    bai_path_list = ['/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-SKCM/TCGA_SKCM-BAMs/bams/{}.bam.bai'.format(base_sample)] + control_bai_path_list
    sif_anno_path = '/data/salomonis2/software/ggsashimi'
    outdir = 'Frank_inspection/sashimi'
    bam_contig_rename = [False,False,False,False,False,False]
    snaf.prepare_sashimi_plot(bam_path_list,bai_path_list,outdir,sif_anno_path,bam_contig_rename,query_region=region,skip_copy=False, min_junction=1,task_name=task_name)

def flank_chrom(chrom,offset):
    chrom = chrom.split('(')[0]
    sec1 = chrom.split(':')[0]
    sec2 = chrom.split(':')[1].split('-')[0]
    sec3 = chrom.split(':')[1].split('-')[1]
    new_sec2 = int(sec2) - offset[0]
    new_sec3 = int(sec3) + offset[1]
    assemble = '{}:{}-{}'.format(sec1,new_sec2,new_sec3)
    return assemble


base_sample = 'TCGA-EB-A5VU-01A-21R-A32P-07'
coord = 'chr20:50940933-50942031'
region = flank_chrom(coord,(1500,1500))
unit_run(uid,base_sample,region,'task2')


