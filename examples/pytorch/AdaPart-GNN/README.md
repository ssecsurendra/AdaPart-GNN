# AdaPart-GNN: Adaptive Graph Partitioning & Boundary-Preserving Training Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![DGL](https://img.shields.io/badge/DGL-1.1+-green.svg)](https://www.dgl.ai/)

**AdaPart-GNN** is an adaptive, communication-free framework for training large-scale Graph Neural Networks (GNNs) on memory-constrained GPUs. By dynamically computing optimal partition counts ($K^*$), preserving dense topological structures through **Jaccard-Eigenvector weighting**, and restoring severed cross-partition signals using a lightweight **Boundary Eigen-Prompt module**, AdaPart-GNN achieves:

- **Up to 31.2$\times$ Speedup** (16.5$\times$ average) compared to standard mini-batch DGL GraphSAGE.
- **Up to 64.4% Peak VRAM Reduction**, enabling training of million-scale graphs on consumer/single GPUs.
- **+11.7% Accuracy Gain** on `ogbn-arxiv` and **+13.2% Micro-F1 Gain** on `Yelp` over classical partition baselines.

---

## 🏛️ Architecture Overview

```
                      Large-Scale Graph Input (V, E)
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │  Stage 1: Scale-Adaptive Partition Count Selection (K*)       │
    │  - Hardware Budget: M_budget = VRAM_avail - M_model - M_buf   │
    │  - Bound Formulation: K* = ceil( (V_graph * M_node) / M_budget)│
    └───────────────────────────────┬───────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │  Stage 2: Topology-Aware Graph Partitioning                   │
    │  - Jaccard Structural Overlap: S_J(u, v)                      │
    │  - Eigenvector Centrality Bias: w_topo(u, v)                  │
    │  - Heavy-Edge METIS: Preserves dense communities & boundaries │
    └───────────────────────────────┬───────────────────────────────┘
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │  Stage 3: Boundary EigenPrompt Bridging                       │
    │  - Boundary Node Identification: B_k                          │
    │  - Eigen-Scale Weighting: w_u = (EC(u))^γ (γ = 1.5)           │
    │  - Prompt Projection: P_k = α_p * MLP_prompt(w_u ⊙ h_u)       │
    │  - Seamless Re-injection: h_u^(l+1) = GNN_layer(h_u^(l) + P_k)│
    └───────────────────────────────────────────────────────────────┘
```

---

## 📊 Benchmark Results

Evaluated on NVIDIA A100 / RTX GPUs across 5 standard graph benchmarks:

| Dataset | Metric | DGL GraphSAGE | Classical METIS | AdaPart-GNN (Ours) | Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ogbn-arxiv** | Accuracy | 59.50% | 60.10% | **71.80%** | **+11.7%** |
| **Yelp** | Micro-F1 | 51.58% | 51.60% | **64.80%** | **+13.2%** |
| **ogbn-products** | Accuracy | 75.32% | 76.10% | **79.40%** | **+3.3%** |
| **Reddit** | Micro-F1 | 93.10% | 93.40% | **95.60%** | **+2.2%** |
| **IGB-Small** | Accuracy | 68.20% | 68.50% | **71.30%** | **+2.8%** |

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
conda create -n adapart python=3.10 -y
conda activate adapart

# Install PyTorch & DGL matching your CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html
pip install ogb scikit-learn scipy numpy networkx pymetis
```

### 2. Run Scale-Adaptive Partitioning (Stage 1 & 2)

```bash
# Compute adaptive K* and perform Jaccard-Eigenvector partitioning
python hardware_scale_adaptive_partitioner.py \
    --dataset ogbn-arxiv \
    --vram_gb 16.0 \
    --output_dir cluster_scale_adaptive/
```

Pre-computed cluster assignments are provided in:
- `cluster_scale_adaptive/`
- `cluster_hardware_adaptive/`
- `cluster_default_metis/`

### 3. Train with Boundary EigenPrompt (Stage 3)

```bash
# Train GCN / GraphSAGE with EigenPrompt bridging
python train_unified_graphsage.py \
    --dataset ogbn-arxiv \
    --cluster_dir cluster_scale_adaptive \
    --use_eigenprompt \
    --alpha_prompt 0.1 \
    --gamma 1.5 \
    --epochs 100 \
    --lr 0.001
```

### 4. Slurm Job Scripts

Ready-to-submit batch scripts for cluster environments:
```bash
sbatch submit_train_partition_scheme_b.sh
sbatch submit_scale_adaptive_partitioning_all5.sh
```

---

## 📁 Repository Structure

```
├── cluster_batch_sampler.py               # Batch loader for partition-based training
├── cluster_scale_adaptive/                # Pre-computed adaptive partition assignments
├── cluster_hardware_adaptive/             # Hardware-constrained partition assignments
├── dataset_sota_loader.py                 # Graph dataset loader (OGB, Yelp, Reddit, IGB)
├── default_metis_partitioner.py           # Standard METIS baseline partitioner
├── dissimilarity_eigenvector_centrality.py# Eigenvector centrality & topological scoring
├── eigen_prompt.py                        # Boundary EigenPrompt injection module
├── hardware_scale_adaptive_partitioner.py # Dynamic K* selection & METIS pipeline
├── hipc_srs_paper/                        # Conference research paper LaTeX sources & PDF
├── models_gnn.py                          # GCN and GraphSAGE model architectures
├── train_unified_gcn.py                   # Unified GCN training engine
├── train_unified_graphsage.py             # Unified GraphSAGE training engine
├── submit_*.sh                            # Slurm workload manager submission scripts
└── README.md
```

---

## 📄 Paper & Documentation

- **Conference Paper**: [`hipc_srs_paper/Paper.pdf`](hipc_srs_paper/Paper.pdf)
- **Mathematical Derivation**: [`AdaPart_Equation_1_Explanation.pdf`](AdaPart_Equation_1_Explanation.pdf)
- **Hardware-Constrained Formulation**: [`Equation_1_Hardware_Derivation_and_Explanation.pdf`](Equation_1_Hardware_Derivation_and_Explanation.pdf)

---

## 👤 Author & Contact

- **Surendra Bana**
- **Email**: [bana.surendra@gmail.com](mailto:bana.surendra@gmail.com)
- **GitHub**: [@ssecsurendra](https://github.com/ssecsurendra)

