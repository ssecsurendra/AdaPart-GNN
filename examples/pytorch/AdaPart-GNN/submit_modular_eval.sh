#!/bin/bash
#SBATCH --job-name=modular_eval
#SBATCH --output=modular_eval_%j.log
#SBATCH --error=modular_eval_%j.err
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

export DGLBACKEND=pytorch
export DGL_LIBRARY_PATH=/data/conda_envs/surendra/dgl-subgraph-sampling/dgl
source /data/miniconda3/etc/profile.d/conda.sh
conda activate /data/conda_envs/surendra/dgl-subgraph-sampling

cd /data/surendra/workspace/dgl-subgraph-sampling/examples/pytorch/sgsgnn

echo "======================================================================"
echo "RUNNING MODULAR TRAINER WITH DUAL-MODE STREAMING EVALUATION"
echo "======================================================================"

python train_modular_streaming.py \
    --dataset ogbn-arxiv \
    --dataset_dir /data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset \
    --cluster_file ./cluster_autok/ogbn-arxiv_autok_15.txt \
    --centrality_file ./cluster_autok/ogbn-arxiv_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --epochs 15 \
    --eval_interval 5 \
    --use_prompts \
    --device 0

echo "======================================================================"
echo "TESTING STANDALONE INFERENCE RUNNER (MODE A: SUBGRAPH)"
echo "======================================================================"

python run_inference_standalone.py \
    --mode subgraph \
    --dataset ogbn-arxiv \
    --dataset_dir /data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset \
    --cluster_file ./cluster_autok/ogbn-arxiv_autok_15.txt \
    --centrality_file ./cluster_autok/ogbn-arxiv_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --use_prompts \
    --device 0

echo "======================================================================"
echo "TESTING STANDALONE INFERENCE RUNNER (MODE B: LAYER-WISE STREAMING)"
echo "======================================================================"

python run_inference_standalone.py \
    --mode layerwise \
    --dataset ogbn-arxiv \
    --dataset_dir /data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset \
    --cluster_file ./cluster_autok/ogbn-arxiv_autok_15.txt \
    --centrality_file ./cluster_autok/ogbn-arxiv_centrality.npy \
    --model sage \
    --n_hidden 256 \
    --n_layers 3 \
    --use_prompts \
    --device 0

echo "======================================================================"
echo "MODULAR EVALUATION BENCHMARK COMPLETED SUCCESSFULLY"
echo "======================================================================"
