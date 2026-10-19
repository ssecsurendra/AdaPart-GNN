#!/bin/bash
#SBATCH --job-name=gcn_unweighted
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=gcn_unweighted_%j.out
#SBATCH --error=gcn_unweighted_%j.err

# Load environment modules in order
source /etc/profile.d/modules.sh 2>/dev/null || true
module unload cuda-11.7 2>/dev/null || true
module load cuda-12.1
module load anaconda3
module load gcc-12.3.0
module load cmake-3.29.2

# Activate dgl-subgraph-sampling conda environment
source /apps/anaconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate /data/conda_envs/surendra/dgl-subgraph-sampling

# Export DGL C++ library path for DGL 2.4.0
export DGL_LIBRARY_PATH=/data/conda_envs/surendra/dgl-subgraph-sampling/dgl
export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Running on node: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# 1. Run Subgraph GCN on ogbn-arxiv with unweighted METIS partitions (20 clusters, 100 epochs, lr=0.001, weight_decay=5e-4)
echo ">>> [1/3] Training on ogbn-arxiv with pure unweighted METIS partitions..."
python3 subgraph_gcn.py --dataset ogbn-arxiv --cluster_file ./cluster_unweighted_metis/ogbn-arxiv_cluster_id.txt --num_clusters 20 --batch_clusters 1 --epochs 100 --lr 0.001 --weight_decay 5e-4

# 2. Run Subgraph GCN on reddit with unweighted METIS partitions (20 clusters, 100 epochs, lr=0.001, weight_decay=5e-4)
echo ">>> [2/3] Training on reddit with pure unweighted METIS partitions..."
python3 subgraph_gcn.py --dataset reddit --cluster_file ./cluster_unweighted_metis/reddit_cluster_id.txt --num_clusters 20 --batch_clusters 1 --epochs 100 --lr 0.001 --weight_decay 5e-4

# 3. Run Subgraph GCN on ogbn-products with unweighted METIS partitions (20 clusters, 100 epochs, lr=0.001, weight_decay=5e-4)
echo ">>> [3/3] Training on ogbn-products with pure unweighted METIS partitions..."
python3 subgraph_gcn.py --dataset ogbn-products --cluster_file ./cluster_unweighted_metis/ogbn-products_cluster_id.txt --num_clusters 20 --batch_clusters 1 --epochs 100 --lr 0.001 --weight_decay 5e-4

