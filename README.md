[![DOI](https://zenodo.org/badge/605715442.svg)](https://doi.org/10.5281/zenodo.13922316)


# BayesTS
Quantifying Tumor Specificity using Bayesian probabilistic modeling for drug target discovery and prioritization

 - Access BayesTS database for the tumor specifcity of human genes [has protien stain](./database/full_results_XYZ.txt) or [all genes with RNA](./database/full_results_XY.txt).

 - Retrain the BayesTS model to adjust tissue importance see [here](https://github.com/frankligy/BayesTS?tab=readme-ov-file#retrain-or-adjust-tissue-importance).

 - Extend BayesTS by incorporting additional modalities (i.e. logFC, splicing junction) see [here](https://github.com/frankligy/BayesTS/tree/main/extension).

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

 You first prepare a plain txt (demiliter is tab) file like this, the valid tissues can be found in [valid_tisse](./database/valid_tissues.csv). I didn't implement any magic function to convert the strings, so please go over these two list (both protein and rna), and choose all tissues, tissue name can be different in RNA and protein, for example, Testis (RNA) and testis (protein), so just include all like below:

 ```
# weights.txt
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

# trian using full model including protein, below assume you are within BayesTS code folder, but modify accordingly
python BayesTS.py --input "./gtex_gene_subsample.h5ad"  # download gene count from synapse
                  --weight "./weights.txt"   # see above
                  --mode "XYZ"      # XYZ is full model, XY is RNA model
                  --protein "./normal_tissue.tsv"  # download from synapses
                  --outdir "./test"

# train using RNA model if protein is not available
python BayesTS.py --input "./coding.h5ad"  # download gene count from synapse
                   --weight "./weights.txt"   # see above
                   --mode "XY"                   
```

#### Full prompt:

```
usage: BayesTS_rev.py [-h] [--input INPUT] [--weight WEIGHT] [--mode MODE]
                      [--protein PROTEIN] [--outdir OUTDIR]
                      [--prior_alpha PRIOR_ALPHA] [--prior_beta PRIOR_BETA]
                      [--noise NOISE] [--min_sample MIN_SAMPLE]
                      [--epoch EPOCH]

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
  --noise NOISE         derived noise signal boundary from the data
  --min_sample MIN_SAMPLE
                        only consider tissues with more than min_sample
  --epoch EPOCH         total epochs to train
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

- If you are using Version 2 (Current Version):

Guangyuan Li, Daniel Schnell, Anukana Bhattacharjee, Mark Yarmarkovich, Nathan Salomonis. 2024 “Quantifying tumor specificity using Bayesian probabilistic modeling for drug and immunotherapeutic target discovery.” Cell Reports Methods. https://doi.org/10.1016/j.crmeth.2024.100900 

## Contact

Guangyuan(Frank) Li

Email: guangyuan.li@nyulangone.org

Postdoc, Perlmutter Cancer Center

NYU Grossman School of Medicine



