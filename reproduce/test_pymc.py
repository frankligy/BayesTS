#!/gpfs/home/lig08/.conda/envs/pymc_env/bin/python3.12

import numpy as np
import pandas as pd
import os,sys

import pymc as pm
import time

def infer_parameters_vectorize(uids,solver):

    n = len(uids)
    s = Y.shape[1]
    t = X.shape[1]
    m = pm.Model()
    with m:
        sigma = pm.Beta('sigma',alpha=2,beta=2,shape=n)
        nc = pm.LogNormal('nc',mu=sigma * 4, sigma=0.5, observed=Y.T)  # since sigma is of shape n, so each draw will be a row of length n (support dimention), and we draw s times, then the resultant matrix, we transpose
        c = pm.Poisson('c',mu=sigma * 25,observed=X.T)
    with m:
        if solver == 'mcmc':
            start_time = time.time()
            trace = pm.sample(draws=1000,step=pm.NUTS(),tune=1000,cores=1,progressbar=True)
            end_time = time.time()
            duration = end_time - start_time
            print('###',n,solver,duration)
        elif solver == 'vi':
            start_time = time.time()
            mean_field = pm.fit(method='advi',progressbar=True)
            trace = mean_field.sample(1000)
            end_time = time.time()
            duration = end_time - start_time
            print('###',n,solver,duration)



ori_X = np.loadtxt('X.out',delimiter='\t').T
ori_Y = np.loadtxt('Y.out',delimiter='\t').T
with open('uids.out','r') as f:
    ori_uids = np.array([item.rstrip('\n') for item in f.readlines()])


for size in [10,50,100,200,500,1000]:
    index_draw = np.random.choice(np.arange(len(ori_uids)),replace=False,size=size)

    uids = ori_uids[index_draw]
    X = ori_X[index_draw,:]
    Y = ori_Y[index_draw,:]

    infer_parameters_vectorize(uids=uids,solver='mcmc')
    infer_parameters_vectorize(uids=uids,solver='vi')


'''
all will be draw 1000 samples, 1 cpu core. 
                                                
svi_advi                                                
mcmc_nuts (tune 1000 samples, 1 chain)                            


### 10 mcmc 16.145457983016968
### 10 vi 5.572661399841309
### 50 mcmc 39.225778341293335
### 50 vi 7.877718210220337
### 100 mcmc 74.17322897911072
### 100 vi 12.209128618240356
### 200 mcmc 161.4286584854126
### 200 vi 22.4102463722229
### 500 mcmc 428.83027958869934
### 500 vi 58.071091175079346
### 1000 mcmc 1462.81707406044
### 1000 vi 166.65992164611816
'''