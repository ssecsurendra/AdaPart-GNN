#!/bin/bash
#SBATCH --job-name=part_def_metis
#SBATCH --output=part_default_metis_%j.log
#SBATCH --error=part_default_metis_%j.err
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
OUTPUT_DIR="./cluster_default_metis"

mkdir -p $OUTPUT_DIR

echo "=========================================================================="
echo "DEFAULT DGL METIS PARTITIONING SUITE (K=20, All 5 Datasets)"
echo "Host: $(hostname) | Date: $(date)"
echo "=========================================================================="

echo -e "\n\n>>> DATASET 1/5: OGBN-ARXIV (Default METIS, K=20) <<<"
python default_metis_partitioner.py \
    --dataset ogbn-arxiv \
    --dataset_dir $DATASET_DIR \
    --num_clusters 20 \
    --output_dir $OUTPUT_DIR

echo -e "\n\n>>> DATASET 2/5: REDDIT (Default METIS, K=20) <<<"
python default_metis_partitioner.py \
    --dataset reddit \
    --dataset_dir $DATASET_DIR \
    --num_clusters 20 \
    --output_dir $OUTPUT_DIR

echo -e "\n\n>>> DATASET 3/5: YELP (Default METIS, K=20) <<<"
python default_metis_partitioner.py \
    --dataset yelp \
    --dataset_dir $DATASET_DIR \
    --num_clusters 20 \
    --output_dir $OUTPUT_DIR

echo -e "\n\n>>> DATASET 4/5: IGB-SMALL (Default METIS, K=20) <<<"
python default_metis_partitioner.py \
    --dataset igb-small \
    --dataset_dir $DATASET_DIR \
    --num_clusters 20 \
    --output_dir $OUTPUT_DIR

echo -e "\n\n>>> DATASET 5/5: OGBN-PRODUCTS (Default METIS, K=20) <<<"
python default_metis_partitioner.py \
    --dataset ogbn-products \
    --dataset_dir $DATASET_DIR \
    --num_clusters 20 \
    --output_dir $OUTPUT_DIR

echo "=========================================================================="
echo "DEFAULT METIS PARTITIONING COMPLETED FOR ALL 5 DATASETS"
echo "=========================================================================="
ls -lh $OUTPUT_DIR/*.txt
