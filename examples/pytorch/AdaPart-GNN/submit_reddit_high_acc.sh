#!/bin/bash
#SBATCH --job-name=reddit_93
#SBATCH --output=reddit_93_%j.log
#SBATCH --error=reddit_93_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00

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
export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=/data/conda_envs/surendra/dgl-subgraph-sampling/dgl
export PYTHONUNBUFFERED=1

cd /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN
DATASET_DIR="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset"

echo "=========================================================================="
echo "RUN 1: REDDIT 3-LAYER GCN + EIGENPROMPT + COSINE LR (TARGET >= 93%)"
echo "=========================================================================="

python reddit_high_acc_engine.py \
    --dataset_dir $DATASET_DIR \
    --cluster_file ./cluster_eigenvector/reddit_cluster_id.txt \
    --centrality_file ./cluster_autok/reddit_eigen_centrality.npy \
    --model gcn \
    --n_hidden 256 \
    --n_layers 3 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --epochs 100 \
    --use_prompts \
    --use_cosine_lr \
    --checkpoint best_reddit_gcn.pt \
    --device 0

echo "=========================================================================="
echo "RUN 2: REDDIT 3-LAYER SAGE + LAYERNORM + RESIDUAL + PROMPTS"
echo "=========================================================================="

python reddit_high_acc_engine.py \
    --dataset_dir $DATASET_DIR \
    --cluster_file ./cluster_eigenvector/reddit_cluster_id.txt \
    --centrality_file ./cluster_autok/reddit_eigen_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --epochs 100 \
    --use_layer_norm \
    --use_residual \
    --use_prompts \
    --use_cosine_lr \
    --checkpoint best_reddit_sage.pt \
    --device 0

echo "=========================================================================="
echo "REDDIT HIGH-ACCURACY BENCHMARKS COMPLETED"
echo "=========================================================================="

