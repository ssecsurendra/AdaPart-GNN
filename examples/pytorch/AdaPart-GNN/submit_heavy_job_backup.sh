#!/bin/bash
#SBATCH --job-name=heavy_metis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=heavy_metis_%j.out
#SBATCH --error=heavy_metis_%j.err

# Load requested environment modules in order
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3
module load cuda-12.1
module load gcc-12.3.0
module load cmake-3.29.2

# Activate dgl-cling conda environment
source /data/surendra/workspace/anaconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate dgl-cling

echo "=================================================="
echo "Running on node: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# Run heavy edge graph partitioning backup script for ogbn-arxiv dataset with 20 clusters
python3 heavy_edge_metis_partitioning_backup.py --dataset ogbn-arxiv --num_clusters 20 --lambda_val 1.0
