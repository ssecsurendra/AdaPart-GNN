#!/bin/bash
#SBATCH --job-name=gcn_bench
#SBATCH --output=gcn_bench_%j.log
#SBATCH --error=gcn_bench_%j.err
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

cd /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/gcn_benchmark
DATASET_DIR="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset"

echo "=========================================================================="
echo "BENCHMARK 1: OGBN-ARXIV (3-Layer GCN, lr=0.001, wd=0.0005, hidden=256)"
echo "=========================================================================="

python train.py \
    --dataset ogbn-arxiv \
    --dataset_dir $DATASET_DIR \
    --n_layers 3 \
    --n_hidden 256 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --epochs 100 \
    --device 0

echo "=========================================================================="
echo "BENCHMARK 2: REDDIT (3-Layer GCN, lr=0.001, wd=0.0005, hidden=256)"
echo "=========================================================================="

python train.py \
    --dataset reddit \
    --dataset_dir $DATASET_DIR \
    --n_layers 3 \
    --n_hidden 256 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --epochs 100 \
    --device 0

echo "=========================================================================="
echo "BENCHMARK 3: OGBN-PRODUCTS (3-Layer GCN, lr=0.001, wd=0.0005, hidden=256)"
echo "=========================================================================="

python train.py \
    --dataset ogbn-products \
    --dataset_dir $DATASET_DIR \
    --n_layers 3 \
    --n_hidden 256 \
    --lr 0.001 \
    --weight_decay 0.0005 \
    --dropout 0.5 \
    --epochs 100 \
    --device 0

echo "=========================================================================="
echo "ALL GCN BENCHMARKS COMPLETED"
echo "=========================================================================="
