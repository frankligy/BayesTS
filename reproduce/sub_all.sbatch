#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=100Gb
#SBATCH --job-name="bayesTS"
#SBATCH --output=/gpfs/data/yarmarkovichlab/Frank/job_dump/%j_%x.out
#SBATCH --error=/gpfs/data/yarmarkovichlab/Frank/job_dump/%j_%x.err
#SBATCH --gres=gpu:a100:1


# module load anaconda3
# conda activate pymc_env
# unset PYTHONPATH

# ./test_pymc.py


# module load anaconda3
# conda activate /gpfs/data/yarmarkovichlab/Frank/BayesTS/logit_gate_env
# unset PYTHONPATH

# ./enrich.py


module load anaconda3
conda activate /gpfs/data/yarmarkovichlab/Frank/BayesTS/logit_gate_env
unset PYTHONPATH


# # reproducibility test
# for i in 1 2 3 4 5
# do
#     ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir reproducibility/output_${i} --protein normal_tissue.tsv
# done

# # ablation test
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir ablation/output_xyz --protein normal_tissue.tsv
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XY --outdir ablation/output_xy --protein normal_tissue.tsv
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode YZ --outdir ablation/output_yz --protein normal_tissue.tsv
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XZ --outdir ablation/output_xz --protein normal_tissue.tsv

# # sensitivity test
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir sensitivity/output_2_2 --protein normal_tissue.tsv --prior_alpha 2.0 --prior_beta 2.0
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir sensitivity/output_4_2 --protein normal_tissue.tsv --prior_alpha 4.0 --prior_beta 2.0
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir sensitivity/output_8_2 --protein normal_tissue.tsv --prior_alpha 8.0 --prior_beta 2.0
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir sensitivity/output_2_4 --protein normal_tissue.tsv --prior_alpha 2.0 --prior_beta 4.0
# ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir sensitivity/output_2_8 --protein normal_tissue.tsv --prior_alpha 2.0 --prior_beta 8.0

# weight adjust test
# 1. testis
# for i in 1 2 3 4 5
# do 
#     ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weight_adjust/weights_testis_0.9.txt --mode XYZ --outdir weight_adjust/output_weights_testis_0.9_${i} --protein normal_tissue.tsv
# done

# for i in 1 2 3 4 5
# do 
#     ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir weight_adjust/output_weights_testis_0.5_${i} --protein normal_tissue.tsv
# done

# for i in 1 2 3 4 5
# do 
#     ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weight_adjust/weights_testis_0.1.txt --mode XYZ --outdir weight_adjust/output_weights_testis_0.1_${i} --protein normal_tissue.tsv
# done

# 2. immune
for i in 1 2 3 4 5
do 
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weight_adjust/weights_immune_0.9.txt --mode XYZ --outdir weight_adjust/output_weights_immune_0.9_${i} --protein normal_tissue.tsv
done

for i in 1 2 3 4 5
do 
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir weight_adjust/output_weights_immune_0.5_${i} --protein normal_tissue.tsv
done

for i in 1 2 3 4 5
do 
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weight_adjust/weights_immune_0.1.txt --mode XYZ --outdir weight_adjust/output_weights_immune_0.1_${i} --protein normal_tissue.tsv
done

# 2. gi
for i in 1 2 3 4 5
do 
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weight_adjust/weights_gi_0.9.txt --mode XYZ --outdir weight_adjust/output_weights_gi_0.9_${i} --protein normal_tissue.tsv
done

for i in 1 2 3 4 5
do 
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir weight_adjust/output_weights_gi_0.5_${i} --protein normal_tissue.tsv
done

for i in 1 2 3 4 5
do 
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weight_adjust/weights_gi_0.1.txt --mode XYZ --outdir weight_adjust/output_weights_gi_0.1_${i} --protein normal_tissue.tsv
done


