#!/bin/bash
#SBATCH --job-name=pure_metis
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=pure_metis_%j.out
#SBATCH --error=pure_metis_%j.err

# Load environment modules in order
source /etc/profile.d/modules.sh 2>/dev/null || true
module load anaconda3
module load cuda-11.7
module load gcc-12.3.0
module load cmake-3.29.2

# Activate dgl_cluster conda environment
source /apps/anaconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate /data/conda_envs/surendra/dgl_cluster

export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Running on node: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# 1. Run unweighted METIS partitioning for ogbn-arxiv (20 clusters)
echo ">>> [1/3] Running Pure Unweighted METIS for ogbn-arxiv..."
python3 metis_partitioning_unweighted.py --dataset ogbn-arxiv --num_clusters 20 --output_dir ./cluster_unweighted_metis

# 2. Run unweighted METIS partitioning for reddit (20 clusters)
echo ">>> [2/3] Running Pure Unweighted METIS for reddit..."
python3 metis_partitioning_unweighted.py --dataset reddit --num_clusters 20 --output_dir ./cluster_unweighted_metis

# 3. Run unweighted METIS partitioning for ogbn-products (20 clusters)
echo ">>> [3/3] Running Pure Unweighted METIS for ogbn-products..."
python3 metis_partitioning_unweighted.py --dataset ogbn-products --num_clusters 20 --output_dir ./cluster_unweighted_metis

# Copy newly created unweighted partition directory to sgsgnn training directory
echo ">>> Copying partition directory to sgsgnn workspace..."
mkdir -p /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN/cluster_unweighted_metis
cp -v ./cluster_unweighted_metis/*_cluster_id.txt /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN/cluster_unweighted_metis/

echo "=================================================="
echo "All unweighted METIS partitions generated and copied successfully!"
echo "=================================================="

