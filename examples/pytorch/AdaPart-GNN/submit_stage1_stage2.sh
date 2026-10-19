#!/bin/bash
#SBATCH --job-name=stage1_2_autok
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=stage1_2_autok_%j.out
#SBATCH --error=stage1_2_autok_%j.err

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
echo "Running Stage 1 & Stage 2 on node: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# Create dedicated directory for Auto-K partitions
mkdir -p ./cluster_autok

# 1. ogbn-arxiv: Stage 1 (Auto-K) + Stage 2 (Partitioning)
echo ">>> [1/3] Running Stage 1 & Stage 2 for ogbn-arxiv..."
python3 partition_engine_autok.py --dataset ogbn-arxiv --output_dir ./cluster_autok --lambda_val 1.0

# 2. reddit: Stage 1 (Auto-K) + Stage 2 (Partitioning)
echo ">>> [2/3] Running Stage 1 & Stage 2 for reddit..."
python3 partition_engine_autok.py --dataset reddit --output_dir ./cluster_autok --lambda_val 1.0

# 3. ogbn-products: Stage 1 (Auto-K) + Stage 2 (Partitioning)
echo ">>> [3/3] Running Stage 1 & Stage 2 for ogbn-products..."
python3 partition_engine_autok.py --dataset ogbn-products --output_dir ./cluster_autok --lambda_val 1.0

# Copy partition directory to sgsgnn training directory
echo ">>> Copying cluster_autok to sgsgnn workspace..."
mkdir -p /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN/cluster_autok
cp -v ./cluster_autok/* /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN/cluster_autok/

echo "=================================================="
echo "Stage 1 & Stage 2 completed successfully!"
echo "=================================================="

