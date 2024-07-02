#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=150Gb
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


# reproducibility test
for i in 1 2 3 4 5
do
    ./BayesTS_rev.py --input gtex_gene_subsample.h5ad --weight weights.txt --mode XYZ --outdir reproducibility/output_${i} --protein normal_tissue.tsv
done




