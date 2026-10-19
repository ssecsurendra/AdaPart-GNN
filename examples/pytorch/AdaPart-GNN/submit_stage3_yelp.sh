#!/bin/bash
#SBATCH --job-name=stg3_yelp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN/stg3_yelp_%j.out
#SBATCH --error=/data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN/stg3_yelp_%j.err

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

# Export DGL C++ library path for DGL
export DGL_LIBRARY_PATH=/data/conda_envs/surendra/dgl-subgraph-sampling/dgl
export PYTHONUNBUFFERED=1

cd /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN

echo "=================================================="
echo "Running Stage 3 Setting A (K=20) for Yelp on: $(hostname)"
echo "Loaded Modules:"
module list 2>&1
echo "Python binary: $(which python3)"
nvidia-smi
echo "=================================================="

# Yelp: Setting A (K=20 Eigenvector Partitions + Stage 3 SAGE with BCEWithLogitsLoss)
python3 train_stage3_yelp_igb.py \
    --dataset yelp \
    --dataset_dir ./dataset \
    --cluster_file ./cluster_eigenvector/yelp_cluster_id.txt \
    --epochs 100 \
    --lr 0.001 \
    --weight_decay 5e-4 \
    --n_hidden 256 \
    --eval_interval 10

echo "=================================================="
echo "YELP EXPERIMENT COMPLETED!"
echo "=================================================="

