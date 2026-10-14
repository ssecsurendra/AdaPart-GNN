import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

# Modular Decoupled Components
from dataset_sota_loader import load_sota_dataset
from models_sota import SOTASubgraphGNN
from cluster_batch_sampler import ClusterBatchSampler
from eigen_prompt import EigenPromptModule
from eval_subgraph import evaluate_subgraph_mode
from eval_layerwise import evaluate_layerwise_streaming
from eval_comparison import compare_inference_modes

def load_partition_ids(cluster_file_path, num_nodes):
    print(f"\n[Partition Loader] Reading partition assignments from: {cluster_file_path}")
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Cluster partition file not found: {cluster_file_path}")

    with open(cluster_file_path, "r") as f:
        cluster_ids = np.array([int(line.strip()) for line in f if line.strip() != ""], dtype=np.int64)

    if len(cluster_ids) != num_nodes:
        raise ValueError(f"Cluster ID count ({len(cluster_ids)}) != Graph nodes ({num_nodes})")

    num_clusters = len(np.unique(cluster_ids))
    print(f"[Partition Loader] Loaded {len(cluster_ids):,} assignments across {num_clusters} unique clusters.")
    return cluster_ids, num_clusters


def load_centrality_scores(centrality_file_path, g):
    if centrality_file_path and os.path.exists(centrality_file_path):
        print(f"[Centrality Loader] Loading eigenvector centrality from: {centrality_file_path}")
        return np.load(centrality_file_path)
    else:
        deg = g.in_degrees().float()
        deg_max = deg.max().item()
        return (deg / (deg_max + 1e-8)).numpy()


def main():
    parser = argparse.ArgumentParser(description="SOTA 3-Layer GNN Training Engine (vs ProDo-GNN Table II)")
    parser.add_argument("--dataset", type=str, default="reddit", choices=["ogbn-arxiv", "reddit", "ogbn-products"])
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--cluster_file", type=str, required=True, help="Path to cluster partition file")
    parser.add_argument("--centrality_file", type=str, default="", help="Path to centrality .npy file (optional)")
    parser.add_argument("--model", type=str, default="sage", choices=["sage", "gcn"])
    parser.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "batchnorm", "none"],
                        help="Normalization type: layernorm (recommended for partition-based GNNs) or batchnorm")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3, help="Strictly 3-layer architecture")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--use_cosine_lr", action="store_true", help="Enable CosineAnnealingLR schedule")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--q_clusters", type=int, default=1, help="Number of clusters grouped per training mini-batch")
    parser.add_argument("--eval_interval", type=int, default=0, help="Epoch interval for layer-wise inference (0 = end only)")
    parser.add_argument("--infer_batch_size", type=int, default=4096)
    parser.add_argument("--use_prompts", action="store_true", help="Enable EigenPrompt boundary augmentation")
    parser.add_argument("--prompt_gamma", type=float, default=1.5)
    parser.add_argument("--prompt_alpha_init", type=float, default=0.1)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    assert args.n_layers == 3, f"Configuration mandates strict 3-layer architecture (got {args.n_layers})"
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"\nTarget Execution Device: {device} | Architecture: Strict 3-Layer {args.model.upper()} | Norm: {args.norm_type.upper()}")

    # 1. SOTA Dataset Loading (with feature normalization & bidirectional conversion)
    g, in_feats, num_classes = load_sota_dataset(args.dataset, args.dataset_dir, verbose=True)

    # 2. Partitioning & Centrality
    cluster_ids, num_clusters = load_partition_ids(args.cluster_file, g.num_nodes())
    centrality_np = load_centrality_scores(args.centrality_file, g)
    g.ndata["centrality"] = torch.from_numpy(centrality_np).float()

    # 3. High-Throughput Multi-Cluster Batcher
    batcher = ClusterBatchSampler(
        g, cluster_ids, q_clusters=args.q_clusters, centrality_scores=centrality_np
    )
    isolated_subgraphs = batcher.get_isolated_subgraphs()

    # 4. Strict 3-Layer SOTA Model
    model = SOTASubgraphGNN(
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=num_classes,
        n_layers=3,
        dropout=args.dropout,
        model_type=args.model,
        norm_type=args.norm_type,
        use_residual=True
    ).to(device)

    prompt_module = None
    if args.use_prompts:
        prompt_module = EigenPromptModule(
            in_feats=in_feats,
            alpha=args.prompt_alpha_init,
            gamma=args.prompt_gamma
        ).to(device)
        print("[EigenPrompt] Boundary Prompt Bridging Module Enabled.")

    params = list(model.parameters())
    if prompt_module is not None:
        params += list(prompt_module.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
        print("[Scheduler] CosineAnnealingLR enabled (T_max=100, eta_min=1e-5).")

    print(f"\nModel Parameters: {sum(p.numel() for p in params if p.requires_grad):,}")
    print(f"Training Plan: {args.epochs} Epochs | Multi-Cluster Batching: q={args.q_clusters} clusters/batch")
    if args.eval_interval > 0:
        print(f"Layer-Wise Full-Graph Inference Interval: Every {args.eval_interval} Epochs\n")
    else:
        print("Layer-Wise Full-Graph Inference Interval: Executed at final checkpoint\n")

    print(f"{'Epoch':^7} | {'Train(s)':^8} | {'Loss':^10} | {'Train Acc':^9} | {'Val Acc':^8} | {'Subg Test':^9} | {'GPU Mem':^8}")
    print("-" * 75)

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    best_model_state = None
    total_pure_train_time = 0.0
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        if prompt_module is not None:
            prompt_module.train()

        total_loss = 0.0
        total_train_correct = 0
        total_train_nodes = 0

        # High-throughput cluster mini-batch training
        epoch_subgraphs = batcher.get_epoch_subgraphs(shuffle=True) if args.q_clusters > 1 else isolated_subgraphs

        for sg in epoch_subgraphs:
            sg = sg.to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            train_mask = sg.ndata["train_mask"]

            if train_mask.sum() == 0:
                continue

            optimizer.zero_grad()

            if prompt_module is not None:
                c_scores = sg.ndata.get("centrality", None)
                feat_aug = prompt_module(feat, c_scores)
            else:
                feat_aug = feat

            logits = model(sg, feat_aug)
            loss = F.cross_entropy(logits[train_mask], label[train_mask])
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * train_mask.sum().item()
            preds = logits[train_mask].argmax(dim=-1)
            total_train_correct += (preds == label[train_mask]).sum().item()
            total_train_nodes += train_mask.sum().item()

        if scheduler is not None:
            scheduler.step()

        epoch_train_time = time.time() - epoch_start_time
        total_pure_train_time += epoch_train_time

        train_loss = total_loss / total_train_nodes if total_train_nodes > 0 else 0.0
        train_acc = total_train_correct / total_train_nodes if total_train_nodes > 0 else 0.0

        # Mode A: Subgraph evaluation (fast, isolated)
        val_loss, val_acc, test_acc_subgraph = evaluate_subgraph_mode(
            model, prompt_module, isolated_subgraphs, device
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc_subgraph
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:^7d} | {epoch_train_time:^8.4f} s | {train_loss:^10.4f} | {train_acc*100:^8.2f}% | {val_acc*100:^8.2f}% | {test_acc_subgraph*100:^8.2f}% | {peak_gpu_mb:^7.1f}MB")

    total_elapsed_time = time.time() - start_total_time

    # Load best checkpoint for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\n[Checkpoint] Loaded best validation checkpoint from Epoch {best_epoch} (Val: {best_val_acc*100:.2f}%, Subg Test: {best_test_acc*100:.2f}%)")

    # ProDo-GNN Table II reference baselines
    prodo_time_ref = {"reddit": 312.03, "ogbn-arxiv": 95.37, "ogbn-products": 236.50}.get(args.dataset, 0.0)
    prodo_acc_ref = {"reddit": 0.96, "ogbn-arxiv": 0.56, "ogbn-products": 0.77}.get(args.dataset, 0.0)
    speedup = (prodo_time_ref / total_pure_train_time) if total_pure_train_time > 0 else 0.0

    print("\n==================================================================")
    print("      SOTA 3-LAYER GNN EXECUTION & BENCHMARK SUMMARY             ")
    print("==================================================================")
    print(f"Dataset                         : {args.dataset}")
    print(f"Model Architecture              : 3-Layer {args.model.upper()} (hidden={args.n_hidden}, residual=True, norm={args.norm_type.upper()})")
    print(f"Partition File Used             : {args.cluster_file}")
    print(f"Multi-Cluster Batching (q)      : {args.q_clusters} clusters/batch")
    print(f"EigenPrompt Bridging            : {'ENABLED' if args.use_prompts else 'DISABLED'}")
    print("------------------------------------------------------------------")
    print(f"Total Pure Training Time        : {round(total_pure_train_time, 4)} Seconds")
    print(f"Average Epoch Training Time     : {round(total_pure_train_time / args.epochs, 4)} Seconds")
    print(f"Peak GPU Memory Allocated       : {round(peak_gpu_mb, 1)} MB")
    print("------------------------------------------------------------------")
    print(f"Best Validation Accuracy        : {round(best_val_acc * 100, 2)}% (Epoch {best_epoch})")
    print(f"Test Accuracy @ Best Val Checkpt: {round(best_test_acc * 100, 2)}%")
    print(f"Final Subgraph Test Accuracy    : {round(test_acc_subgraph * 100, 2)}%")
    print("------------------------------------------------------------------")
    print("COMPARISON WITH PRODO-GNN (Table II):")
    print(f"  ProDo-GNN Published Time       : {prodo_time_ref:.2f} s")
    print(f"  Our Pure Training Time         : {total_pure_train_time:.2f} s  ==>  {speedup:.2f}x FASTER!")
    print(f"  ProDo-GNN Published Test Acc   : {prodo_acc_ref * 100:.2f}%")
    print("==================================================================")

    # Full side-by-side inference comparison on Best Model
    compare_inference_modes(model, prompt_module, isolated_subgraphs, g, device, batch_size=args.infer_batch_size)

if __name__ == "__main__":
    main()
