#!/bin/bash
#SBATCH --job-name=eigen_metis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=eigen_metis_%j.out
#SBATCH --error=eigen_metis_%j.err

# Load environment modules in order
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3
module load cuda-11.7
module load gcc-12.3.0
module load cmake-3.29.2

# Activate dgl_cluster conda environment
source /apps/anaconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate /data/conda_envs/surendra/dgl_cluster

echo "=================================================="
echo "Running on node: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# 1. Run heavy edge graph partitioning with eigenvector centrality for ogbn-arxiv (20 clusters)
python3 heavy_edge_metis_partitioning_eigenvector.py --dataset ogbn-arxiv --num_clusters 20 --lambda_val 1.0

# 2. Run heavy edge graph partitioning with eigenvector centrality for reddit (20 clusters)
# python3 heavy_edge_metis_partitioning_eigenvector.py --dataset reddit --num_clusters 20 --lambda_val 1.0

# 3. Run heavy edge graph partitioning with eigenvector centrality for ogbn-products (20 clusters)
# python3 heavy_edge_metis_partitioning_eigenvector.py --dataset ogbn-products --num_clusters 20 --lambda_val 1.0
