A bit of these source code files:

**analysis.py**: I used for extracting the junction data from my TCGA melanoma analysis, so only junctions included in the TCGA SNAF analysis, here can find code for generating combined_normal_count.h5ad. Regarding how those source data were curated, that is a SNAF question

**build_db.py**: I used for curating gene count matrix from multiple sources

**hbm.py**: the first iteration of the model using pyro VI, transitioning from pymc old code, here can find the code for generating coding.h5ad

**hbm_adv_pyro.py**: the second iteration of the model, here tissue importance has been included and rigorous tests are preformed here

**gene_protein_model.py**: the third iteration of the model, preceding the release code BayesTS.py, contain the most advaced and complete model

**gtex_viewer.py**: the main viewer utils, inhereting SNAF.gtex_viewer, but have a lot of updates which are fed back to a future release of SNAF as well

**junction_model.py**: I initialy used this script to test the possibility of applying it on splicing data

**paper.py**: codes for gnerating some figures in the manuscript

**analysis_new.py**: codes for auxiliary functions for generating some splicing figures
