# BayesTS
Quantifying Tumor Specificity using Bayesian probabilistic modeling for drug target discovery and prioritization

 - Access BayesTS database for the tumor specifcity of 13,350 protein coding genes with both RNA and protein information [here](./database/full_results_XYZ.txt).

 - Retrain the BayesTS model to adjust tissue importance see [here](https://github.com/frankligy/BayesTS?tab=readme-ov-file#retrain-or-adjust-tissue-importance).

 - Extend BayesTS by incorporting additional modalities (i.e. logFC) see [here](https://github.com/frankligy/BayesTS/tree/main/extension).

 - Apply BayesTS to [SNAF](https://github.com/frankligy/SNAF) splicing junctions see [here](https://github.com/frankligy/BayesTS?tab=readme-ov-file#interface-with-snaf)

 Please feel free to contact me if I can help clarify anything, contact is at the bottom.


 # Overview

 ![overview](./images/fig1.png)


 ## Installation

A `linux` system, `fresh conda` environment, `python 3.7`

 ```bash
# env
conda create -n BayesTS_env python=3.7

# dependencies
pip install sctriangulate
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pyro-ppl==1.8.4

# code
git clone https://github.com/frankligy/BayesTS.git
 ```

 ## Retrain or Adjust tissue importance

 You first prepare a plain txt (demiliter is tab) file like this, the valid tissues can be found in [valid_tisse](./database/valid_tissues.txt). I didn't implement any magic function to convert the strings, so please go over these two list (both protein and rna), and choose all tissues, tissue name can be different in RNA and protein, for example, Testis (RNA) and testis (protein), so just include all like below:

 ```
tissue     weight
tonsil      0.1
appendix    0.1
testis      0.1
Testis      0.1
 ```

If you don't want to change any weight (all 0.5), I provided a `weights.txt` file you can use as input.

You need to download RNA or Protein data from [this synapse folder](https://www.synapse.org/Synapse:syn61670083).

```bash
# help information
python BayesTS.py --help

# trian using full model including protein
python BayesTS.py --input "./gtex_gene_subsample.h5ad"  # download gene count from synapse
                   --weight "./weights.txt"   # see above
                   --mode "XYZ"      # XYZ is full model, XY is RNA model
                   --protein "./normal_tissue.tsv"  # download from synapses

# train using RNA model if protein is not available
python BayesTS.py --input "./coding.h5ad"  # download gene count from synapse
                   --weight "./weights.txt"   # see above
                   --mode "XY"                   
```

Full prompt:

```
usage: BayesTS_rev.py [-h] [--input INPUT] [--weight WEIGHT] [--mode MODE]
                      [--protein PROTEIN] [--outdir OUTDIR]
                      [--prior_alpha PRIOR_ALPHA] [--prior_beta PRIOR_BETA]

Run BayesTS to retrain

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         path to the h5ad file
  --weight WEIGHT       path to a txt file with tissue and weights that you
                        want to change
  --mode MODE           XYZ use full model, XY use only RNA model
  --protein PROTEIN     path to the protein info downloaded from Synapse
  --outdir OUTDIR       path to the output directory
  --prior_alpha PRIOR_ALPHA
                        alpha for the beta prior
  --prior_beta PRIOR_BETA
                        beta for the beta prior
```



## Interface with SNAF

In [SNAF](https://github.com/frankligy/SNAF), you are able to [get a h5ad file](https://snaf.readthedocs.io/en/latest/api.html#get-all-normal-h5ad) with all the splicing junctions you'd like to query, let's say the file name is `junction.h5ad`

```bash
python BayesTS.py --input "junction.h5ad"
                  --weight "weights.txt"
                  --mode "XY"
```


## Citation

- If you are using Version 1:

Li, Guangyuan, Anukana Bhattacharjee, and Nathan Salomonis. 2023. “Quantifying Tumor Specificity Using Bayesian Probabilistic Modeling for Drug Target Discovery and Prioritization.” bioRxiv. https://www.biorxiv.org/content/10.1101/2023.03.03.530994v1

Guangyuan Li et al. ,Splicing neoantigen discovery with SNAF reveals shared targets for cancer immunotherapy.Sci. Transl. Med.16,eade2886(2024).DOI:10.1126/scitranslmed.ade2886 (https://www.science.org/doi/10.1126/scitranslmed.ade2886)

- If you are using Version 2 (Current Version):

Quantifying tumor specificity using Bayesian probabilistic modeling for drug and immunotherapeutic target discovery, In Revision

## Contact

Guangyuan(Frank) Li

Email: li2g2@mail.uc.edu

PhD student, Biomedical Informatics

Cincinnati Children’s Hospital Medical Center(CCHMC)

University of Cincinnati, College of Medicine

