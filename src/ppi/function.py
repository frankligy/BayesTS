#!/data/salomonis2/LabFiles/Frank-Li/neoantigen/revision/ts/bayesian/pytorch_pyro_mamba_env/bin/python3.7

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'Arial'


df = pd.read_csv('full_results_XYZ.txt',sep='\t',index_col=0)

# extract 6 tier genes 
df1 = df.loc[(df['mean_sigma']>=0) & (df['mean_sigma']<0.05),:].index.to_series().to_csv('df1_0_0.05.txt',sep='\t',index=None,header=None)
df2 = df.loc[(df['mean_sigma']>=0.05) & (df['mean_sigma']<0.1),:].index.to_series().to_csv('df2_0.05_0.1.txt',sep='\t',index=None,header=None)
df3 = df.loc[(df['mean_sigma']>=0.1) & (df['mean_sigma']<0.15),:].index.to_series().to_csv('df3_0.1_0.15.txt',sep='\t',index=None,header=None)
df4 = df.loc[(df['mean_sigma']>=0.15) & (df['mean_sigma']<0.2),:].index.to_series().to_csv('df4_0.15_0.2.txt',sep='\t',index=None,header=None)
df5 = df.loc[(df['mean_sigma']>=0.2) & (df['mean_sigma']<0.25),:].index.to_series().to_csv('df5_0.2_0.25.txt',sep='\t',index=None,header=None)
df6 = df.loc[(df['mean_sigma']>=0.25) & (df['mean_sigma']<0.3),:].index.to_series().to_csv('df5_0.25_0.3.txt',sep='\t',index=None,header=None)