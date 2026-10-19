#!/bin/bash
#SBATCH --job-name=hw_part_all5
#SBATCH --output=hw_part_all5_%j.log
#SBATCH --error=hw_part_all5_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Load environment modules in order (dgl_cluster uses cuda-11.7)
source /etc/profile.d/modules.sh 2>/dev/null || true
module unload cuda-12.1 2>/dev/null || true
module load gcc-12.3.0
module load cuda-11.7
module load anaconda3
module load cmake-3.29.2

# Activate dgl_cluster conda environment (contains CuPy, PyMETIS, DGL, PyTorch)
source /apps/anaconda3/etc/profile.d/conda.sh 2>/dev/null || eval "$(conda shell.bash hook)"
conda activate /data/conda_envs/surendra/dgl_cluster

# Set LD_PRELOAD for modern libstdc++.so.6 from gcc-12.3.0
export LD_PRELOAD=/apps/gcc/12.3.0/lib64/libstdc++.so.6
export PYTHONUNBUFFERED=1

cd /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/AdaPart-GNN
DATASET_DIR="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset"
OUT_DIR="./cluster_hardware_adaptive"

mkdir -p $OUT_DIR
mkdir -p /data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/cluster_hardware_adaptive

echo "=========================================================================="
echo "HARDWARE-ADAPTIVE CENTRALITY DOMINATING SET PARTITIONING (ALL 5 DATASETS)"
echo "Host: $(hostname) | Python: $(which python3)"
echo "Output Directory: $OUT_DIR"
echo "=========================================================================="

echo -e "\n\n>>> DATASET 1/5: OGBN-ARXIV <<<"
python3 hardware_adaptive_dominating_partitioner.py \
    --dataset ogbn-arxiv \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUT_DIR \
    --lambda_val 1.0 \
    --device 0

echo -e "\n\n>>> DATASET 2/5: REDDIT <<<"
python3 hardware_adaptive_dominating_partitioner.py \
    --dataset reddit \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUT_DIR \
    --lambda_val 1.0 \
    --device 0

echo -e "\n\n>>> DATASET 3/5: YELP <<<"
python3 hardware_adaptive_dominating_partitioner.py \
    --dataset yelp \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUT_DIR \
    --lambda_val 1.0 \
    --device 0

echo -e "\n\n>>> DATASET 4/5: IGB-SMALL <<<"
python3 hardware_adaptive_dominating_partitioner.py \
    --dataset igb-small \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUT_DIR \
    --lambda_val 1.0 \
    --device 0

echo -e "\n\n>>> DATASET 5/5: OGBN-PRODUCTS <<<"
python3 hardware_adaptive_dominating_partitioner.py \
    --dataset ogbn-products \
    --dataset_dir $DATASET_DIR \
    --output_dir $OUT_DIR \
    --lambda_val 1.0 \
    --device 0

# Copy generated partitions to dgl_cluster directory as well
cp -r $OUT_DIR/* /data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/cluster_hardware_adaptive/ 2>/dev/null || true

echo "=========================================================================="
echo "ALL 5 DATASETS PARTITIONED WITH HARDWARE-ADAPTIVE DOMINATING SET"
echo "=========================================================================="

