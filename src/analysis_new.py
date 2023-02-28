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
inspect those candidates for junction validity and tumor specificity
'''
control_bam_path_list = ['/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-BREAST-CANCER/TCGA-files-Ens91/bams/TCGA-BH-A208-11A-51R-A157-07.bam',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-Kidney/KIRC/TCGA-CJ-5680-11A-01R-1541-07.bam',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-STAD/TCGA-BR-6453-11A-01R-1802-13.bam',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-ESCA/TCGA-L5-A4OG-11A-12R-A260-31.bam',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-THCA/TCGA-EL-A3ZP-11A.bam',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-BREAST-CANCER/TCGA-files-Ens91/bams/TCGA-BH-A0H7-11A-13R-A089-07.bam']
control_bai_path_list = ['/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-BREAST-CANCER/TCGA-files-Ens91/bams/TCGA-BH-A208-11A-51R-A157-07.bam.bai',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-Kidney/KIRC/TCGA-CJ-5680-11A-01R-1541-07.bam.bai',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-STAD/TCGA-BR-6453-11A-01R-1802-13.bam.bai',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-ESCA/TCGA-L5-A4OG-11A-12R-A260-31.bam.bai',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-THCA/TCGA-EL-A3ZP-11A.bam.bai',
                 '/data/salomonis-archive/BAMs/NCI-R01/TCGA/TCGA-BREAST-CANCER/TCGA-files-Ens91/bams/TCGA-BH-A0H7-11A-13R-A089-07.bam.bai']

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

# unit_run('ENSG00000001461:E6.1-E7.1_24445189','TCGA-EE-A2GD-06A-11R-A18T-07',flank_chrom('chr1:24442226-24445189',(1000,1000)),'task1')
unit_run('ENSG00000149582:E6.2-E7.1','TCGA-EE-A2GD-06A-11R-A18T-07',flank_chrom('chr11:118534129-118534266',(100,100)),'task2')





