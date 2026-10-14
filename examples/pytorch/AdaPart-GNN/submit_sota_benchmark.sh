#!/bin/bash
#SBATCH --job-name=sota_3layer
#SBATCH --output=sota_3layer_%j.log
#SBATCH --error=sota_3layer_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

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

cd /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/sgsgnn
DATASET_DIR="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset"

echo "=========================================================================="
echo "EXPERIMENT 1: REDDIT (Strict 3-Layer SAGE + Feature Norm + EigenPrompt)"
echo "TARGET: Test Acc ~96% | Time < 75s (vs ProDo-GNN 312.03s)"
echo "=========================================================================="

python train_sota_engine.py \
    --dataset reddit \
    --dataset_dir $DATASET_DIR \
    --cluster_file ./cluster_eigenvector/reddit_cluster_id.txt \
    --centrality_file ./cluster_autok/reddit_eigen_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --dropout 0.5 \
    --lr 0.003 \
    --weight_decay 5e-4 \
    --epochs 100 \
    --q_clusters 1 \
    --eval_interval 10 \
    --use_prompts \
    --device 0

echo "=========================================================================="
echo "EXPERIMENT 2: OGBN-ARXIV (Strict 3-Layer SAGE + Bidirected Augmentation)"
echo "TARGET: Test Acc ~56% | Time < 25s (vs ProDo-GNN 95.37s)"
echo "=========================================================================="

python train_sota_engine.py \
    --dataset ogbn-arxiv \
    --dataset_dir $DATASET_DIR \
    --cluster_file ./cluster_autok/ogbn-arxiv_cluster_id.txt \
    --centrality_file ./cluster_autok/ogbn-arxiv_eigen_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --dropout 0.5 \
    --lr 0.003 \
    --weight_decay 5e-4 \
    --epochs 100 \
    --q_clusters 1 \
    --eval_interval 10 \
    --use_prompts \
    --device 0

echo "=========================================================================="
echo "EXPERIMENT 3: OGBN-PRODUCTS (Strict 3-Layer SAGE + EigenPrompt)"
echo "TARGET: Test Acc ~78% | Time < 160s (vs ProDo-GNN 236.50s)"
echo "=========================================================================="

python train_sota_engine.py \
    --dataset ogbn-products \
    --dataset_dir $DATASET_DIR \
    --cluster_file ./cluster_autok/ogbn-products_cluster_id.txt \
    --centrality_file ./cluster_autok/ogbn-products_eigen_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --dropout 0.3 \
    --lr 0.003 \
    --weight_decay 5e-4 \
    --epochs 100 \
    --q_clusters 1 \
    --eval_interval 10 \
    --use_prompts \
    --device 0

echo "=========================================================================="
echo "ALL SOTA 3-LAYER BENCHMARKS COMPLETED"
echo "=========================================================================="
