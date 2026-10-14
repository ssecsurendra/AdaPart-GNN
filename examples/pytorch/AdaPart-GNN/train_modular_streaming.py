import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset

# Modular decoupled components
from models_gnn import SubgraphGNN
from eigen_prompt import EigenPromptModule
from eval_subgraph import evaluate_subgraph_mode
from eval_layerwise import evaluate_layerwise_streaming
from eval_comparison import compare_inference_modes

def load_dataset(dataset_name, dataset_dir="./dataset"):
    print(f"Loading dataset: {dataset_name} ...")
    t0 = time.time()
    if dataset_name == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = data[0]
    elif dataset_name == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = data[0]
    elif dataset_name == "reddit":
        data = RedditDataset(raw_dir=dataset_dir)
        g = data[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if "label" in g.ndata:
        g.ndata["label"] = g.ndata["label"].squeeze().long()

    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if mask_name in g.ndata:
            g.ndata[mask_name] = g.ndata[mask_name].bool()

    in_feats = g.ndata["feat"].shape[1]
    num_classes = data.num_classes
    print(f"Dataset loaded in {time.time() - t0:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,} | In-Feats: {in_feats} | Classes: {num_classes}")
    return g, in_feats, num_classes


def create_partitioned_subgraphs(g, cluster_file_path, centrality_file_path=""):
    print(f"Reading cluster partition IDs from: {cluster_file_path}")
    t0 = time.time()
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Cluster partition file not found: {cluster_file_path}")

    with open(cluster_file_path, "r") as f:
        cluster_ids = np.array([int(line.strip()) for line in f if line.strip() != ""], dtype=np.int64)

    if len(cluster_ids) != g.num_nodes():
        raise ValueError(f"Cluster ID count ({len(cluster_ids)}) != Graph nodes ({g.num_nodes()})")

    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    print(f"Loaded {len(cluster_ids):,} node assignments ({num_clusters} unique clusters)")

    if centrality_file_path and os.path.exists(centrality_file_path):
        print(f"Loading eigenvector centrality scores from: {centrality_file_path}")
        centrality_np = np.load(centrality_file_path)
    else:
        deg = g.in_degrees().float()
        deg_max = deg.max().item()
        centrality_np = (deg / (deg_max + 1e-8)).numpy()

    centrality_tensor = torch.from_numpy(centrality_np).float()
    g.ndata["centrality"] = centrality_tensor

    subgraphs = []
    print("Extracting partitioned subgraphs with local topologies & self-loops...")
    for c_id in range(num_clusters):
        node_indices = torch.where(torch.from_numpy(cluster_ids == c_id))[0]
        if len(node_indices) == 0:
            continue
        sg = g.subgraph(node_indices)
        sg = dgl.remove_self_loop(sg)
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)

    print(f"Constructed {len(subgraphs)} subgraphs in {time.time() - t0:.2f}s")
    return subgraphs, num_clusters


def main():
    parser = argparse.ArgumentParser(description="Modular Training & Dual-Mode Inference Engine")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", choices=["ogbn-arxiv", "reddit", "ogbn-products"])
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--cluster_file", type=str, required=True, help="Path to cluster partition file")
    parser.add_argument("--centrality_file", type=str, default="", help="Path to centrality .npy file (optional)")
    parser.add_argument("--model", type=str, default="sage", choices=["sage", "gcn"])
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=10, help="Epoch interval for layer-wise inference")
    parser.add_argument("--infer_batch_size", type=int, default=4096, help="Batch size for layer-wise dataloader")
    parser.add_argument("--use_prompts", action="store_true", help="Enable EigenPrompt boundary augmentation")
    parser.add_argument("--prompt_gamma", type=float, default=1.5, help="Centrality decay exponent")
    parser.add_argument("--prompt_alpha_init", type=float, default=0.1, help="Prompt scaling init")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"Execution Target Device: {device}")

    g, in_feats, num_classes = load_dataset(args.dataset, args.dataset_dir)
    subgraphs, num_clusters = create_partitioned_subgraphs(g, args.cluster_file, args.centrality_file)

    model = SubgraphGNN(
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=num_classes,
        n_layers=args.n_layers,
        dropout=args.dropout,
        model_type=args.model,
        use_batch_norm=True,
        use_residual=True
    ).to(device)

    prompt_module = None
    if args.use_prompts:
        prompt_module = EigenPromptModule(
            in_feats=in_feats,
            gamma=args.prompt_gamma,
            alpha_init=args.prompt_alpha_init
        ).to(device)
        print("EigenPrompt Bridging Module Initialized.")

    params = list(model.parameters())
    if prompt_module is not None:
        params += list(prompt_module.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    print(f"\nModel Configuration: {args.model.upper()} | Layers: {args.n_layers} | Hidden: {args.n_hidden}")
    print(f"Total Model Parameters: {sum(p.numel() for p in params if p.requires_grad):,}")
    print(f"Training Epochs: {args.epochs} | Streaming Inference Eval Interval: {args.eval_interval}\n")

    print(f"{'Epoch':^7} | {'Train(s)':^8} | {'Loss':^10} | {'Train Acc':^9} | {'Val Acc':^8} | {'Subg Test':^9} | {'Stream Test':^11} | {'GPU Mem':^8}")
    print("-" * 86)

    best_val_acc = 0.0
    best_stream_acc = 0.0
    best_epoch = 0
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

        # Cluster mini-batch training
        for sg in subgraphs:
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

        epoch_train_time = time.time() - epoch_start_time
        total_pure_train_time += epoch_train_time

        train_loss = total_loss / total_train_nodes if total_train_nodes > 0 else 0.0
        train_acc = total_train_correct / total_train_nodes if total_train_nodes > 0 else 0.0

        # Mode A: Subgraph evaluation (fast, isolated)
        val_loss, val_acc, test_acc_subgraph = evaluate_subgraph_mode(model, prompt_module, subgraphs, device)

        # Mode B: True Layer-by-Layer Streaming (periodic or final)
        stream_test_acc = test_acc_subgraph
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            _, _, stream_test_acc = evaluate_layerwise_streaming(
                model, prompt_module, g, device, batch_size=args.infer_batch_size
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_stream_acc = stream_test_acc
            best_epoch = epoch

        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:^7d} | {epoch_train_time:^8.4f} s | {train_loss:^10.4f} | {train_acc*100:^8.2f}% | {val_acc*100:^8.2f}% | {test_acc_subgraph*100:^8.2f}% | {stream_test_acc*100:^10.2f}% | {peak_gpu_mb:^7.1f}MB")

    total_elapsed_time = time.time() - start_total_time

    print("\n==================================================")
    print("      TRAINING EXECUTION COMPLETE                 ")
    print("==================================================")
    print(f"Total Pure Training Time        : {round(total_pure_train_time, 4)}s")
    print(f"Average Epoch Training Time     : {round(total_pure_train_time / args.epochs, 4)}s")
    print(f"Peak GPU Memory Allocated       : {round(peak_gpu_mb, 1)} MB")
    print(f"Best Validation Accuracy        : {round(best_val_acc * 100, 2)}% (Epoch {best_epoch})")
    print(f"Streaming Test Acc @ Best Val   : {round(best_stream_acc * 100, 2)}%")

    # Run comprehensive side-by-side comparison of both inference modes
    compare_inference_modes(model, prompt_module, subgraphs, g, device, batch_size=args.infer_batch_size)

if __name__ == "__main__":
    main()
