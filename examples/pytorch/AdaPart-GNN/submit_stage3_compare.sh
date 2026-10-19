#!/bin/bash
#SBATCH --job-name=stage3_compare
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=stage3_compare_%j.out
#SBATCH --error=stage3_compare_%j.err

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
echo "Running Stage 3 (Setting A vs Setting B) on: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# ==============================================================================
# EXPERIMENT A: Old Partitions (K=20 from ./cluster_eigenvector/) + Stage 3 Model
# ==============================================================================
echo "################################################################"
echo "### [SETTING A] Old Partitions (K=20) + Stage 3 EigenPrompt ###"
echo "################################################################"

echo ">>> [A1] Training ogbn-arxiv with Old Partitions (K=20)..."
python3 train_eigenprompt.py \
    --dataset ogbn-arxiv \
    --cluster_file ./cluster_eigenvector/ogbn-arxiv_cluster_id.txt \
    --model sage \
    --use_prompts \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 5e-4

echo ">>> [A2] Training reddit with Old Partitions (K=20)..."
python3 train_eigenprompt.py \
    --dataset reddit \
    --cluster_file ./cluster_eigenvector/reddit_cluster_id.txt \
    --model sage \
    --use_prompts \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 5e-4


# ==============================================================================
# EXPERIMENT B: New Auto-K* Partitions (from ./cluster_autok/) + Stage 3 Model
# ==============================================================================
echo "################################################################"
echo "### [SETTING B] New Auto-K* Partitions + Stage 3 EigenPrompt ###"
echo "################################################################"

echo ">>> [B1] Training ogbn-arxiv with Auto-K* Partitions..."
python3 train_eigenprompt.py \
    --dataset ogbn-arxiv \
    --cluster_file ./cluster_autok/ogbn-arxiv_cluster_id.txt \
    --centrality_file ./cluster_autok/ogbn-arxiv_eigen_centrality.npy \
    --model sage \
    --use_prompts \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 5e-4

echo ">>> [B2] Training reddit with Auto-K* Partitions..."
python3 train_eigenprompt.py \
    --dataset reddit \
    --cluster_file ./cluster_autok/reddit_cluster_id.txt \
    --centrality_file ./cluster_autok/reddit_eigen_centrality.npy \
    --model sage \
    --use_prompts \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 5e-4

echo "=================================================="
echo "Stage 3 Comparative Experiments Completed!"
echo "=================================================="

