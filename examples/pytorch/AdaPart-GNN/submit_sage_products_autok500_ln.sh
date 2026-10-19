#!/bin/bash
#SBATCH --job-name=prod_k500_ln
#SBATCH --output=prod_k500_ln_%j.log
#SBATCH --error=prod_k500_ln_%j.err
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
echo "EXPERIMENT: OGBN-PRODUCTS with Strict 3-Layer GraphSAGE + LayerNorm"
echo "Partitions: cluster_autok (K=500) | lr=0.001, wd=5e-4, hidden=256, dropout=0.5"
echo "=========================================================================="

python train_sota_engine.py \
    --dataset ogbn-products \
    --dataset_dir $DATASET_DIR \
    --cluster_file ./cluster_autok/ogbn-products_cluster_id.txt \
    --centrality_file ./cluster_autok/ogbn-products_eigen_centrality.npy \
    --model sage \
    --norm_type layernorm \
    --n_hidden 256 \
    --n_layers 3 \
    --dropout 0.5 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --use_cosine_lr \
    --epochs 100 \
    --q_clusters 1 \
    --use_prompts \
    --device 0

echo "=========================================================================="
echo "COMPLETED: OGBN-PRODUCTS 3-LAYER GRAPHSAGE (K=500, LAYERNORM) BENCHMARK"
echo "=========================================================================="

