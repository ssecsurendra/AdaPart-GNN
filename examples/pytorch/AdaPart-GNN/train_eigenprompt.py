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

# Modular components
from models_gnn import SubgraphGNN
from eigen_prompt import EigenPromptModule
from inference_engine import evaluate_subgraphs, evaluate_full_graph_layerwise

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

    # Ensure labels are 1D LongTensor
    if "label" in g.ndata:
        g.ndata["label"] = g.ndata["label"].squeeze().long()

    # Ensure masks are boolean
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

    # Load centrality if provided, otherwise compute normalized degrees
    if centrality_file_path and os.path.exists(centrality_file_path):
        print(f"Loading eigenvector centrality scores from: {centrality_file_path}")
        centrality_np = np.load(centrality_file_path)
    else:
        deg = g.in_degrees().float()
        deg_max = deg.max().item()
        centrality_np = (deg / (deg_max + 1e-8)).numpy()

    g.ndata["centrality"] = torch.from_numpy(centrality_np).float()

    subgraphs = []
    print("Extracting induced subgraphs for each cluster...")
    for cid in unique_clusters:
        node_indices = np.where(cluster_ids == cid)[0]
        if len(node_indices) == 0:
            continue
        sg = dgl.node_subgraph(g, node_indices, relabel_nodes=True)
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)

    print(f"Created {len(subgraphs)} subgraphs in {time.time() - t0:.2f}s")
    return subgraphs, num_clusters


def main():
    parser = argparse.ArgumentParser(description="Stage 3: EigenPrompt-GNN Modular Training Pipeline")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--dataset_dir", type=str, default="./dataset", help="Directory where datasets are stored")
    parser.add_argument("--cluster_file", type=str, required=True, help="Path to cluster ID file")
    parser.add_argument("--centrality_file", type=str, default="", help="Path to normalized centrality npy file")
    parser.add_argument("--model", type=str, default="sage", choices=["sage", "gcn"], help="Backbone model: sage or gcn")
    parser.add_argument("--use_prompts", action="store_true", default=True, help="Enable EigenPrompt global bridging")
    parser.add_argument("--num_prompts", type=int, default=0, help="Number of prompts (0 = auto-set to num_classes)")
    parser.add_argument("--prompt_alpha", type=float, default=0.5, help="Prompt strength alpha")
    parser.add_argument("--prompt_gamma", type=float, default=1.0, help="Centrality modulation exponent gamma")
    parser.add_argument("--n_hidden", type=int, default=256, help="Hidden embedding dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--eval_interval", type=int, default=5, help="Epoch interval for full-graph layerwise eval")
    args = parser.parse_args()

    # Device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 1. Load full graph
    g, in_feats, num_classes = load_dataset(args.dataset, args.dataset_dir)

    # 2. Extract subgraphs
    subgraphs, num_clusters = create_partitioned_subgraphs(g, args.cluster_file, args.centrality_file)

    # 3. Initialize Backbone Model
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

    # 4. Initialize EigenPrompt Module if enabled
    prompt_module = None
    if args.use_prompts:
        m_prompts = args.num_prompts if args.num_prompts > 0 else num_classes
        prompt_module = EigenPromptModule(
            in_feats=in_feats,
            num_prompts=m_prompts,
            alpha=args.prompt_alpha,
            gamma=args.prompt_gamma
        ).to(device)
        print(f"EigenPrompt Module: ENABLED ({m_prompts} prompts, alpha={args.prompt_alpha}, gamma={args.prompt_gamma})")
    else:
        print("EigenPrompt Module: DISABLED (Standard Subgraph GNN Baseline)")

    # 5. Optimizer
    params = list(model.parameters())
    if prompt_module is not None:
        params += list(prompt_module.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    print("\n==================================================")
    print(f"STAGE 3: EIGENPROMPT-GNN TRAINING ({args.epochs} EPOCHS)")
    print(f"Dataset: {args.dataset} | Backbone: {args.model.upper()} | Clusters: {num_clusters}")
    print(f"In: {in_feats} -> Hidden: {args.n_hidden} -> Classes: {num_classes}")
    print("==================================================\n")

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    total_pure_train_time = 0.0

    print(f"{'Epoch':^7} | {'Train Time':^10} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Acc':^9} | {'Test Acc':^9} | {'Full-Test':^9} | {'GPU Mem':^9}")
    print("-" * 95)

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        model.train()
        if prompt_module is not None:
            prompt_module.train()

        total_loss = 0.0
        total_train_correct = 0
        total_train_nodes = 0

        # Shuffle cluster training order
        shuffled_order = torch.randperm(num_clusters).tolist()

        for cid in shuffled_order:
            sg = subgraphs[cid].to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            train_mask = sg.ndata["train_mask"]

            if train_mask.sum() == 0:
                continue

            optimizer.zero_grad()

            # Apply EigenPrompt augmentation
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

        # Evaluation
        val_loss, val_acc, test_acc_subgraph = evaluate_subgraphs(model, prompt_module, subgraphs, device)

        # Full-graph evaluation at intervals or epoch 100
        full_test_acc = test_acc_subgraph
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            _, _, full_test_acc = evaluate_full_graph_layerwise(model, prompt_module, g, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = full_test_acc
            best_epoch = epoch

        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:^7d} | {epoch_train_time:^8.4f} s | {train_loss:^10.4f} | {train_acc*100:^8.2f}% | {val_acc*100:^8.2f}% | {test_acc_subgraph*100:^8.2f}% | {full_test_acc*100:^8.2f}% | {peak_gpu_mb:^7.1f}MB")

    total_elapsed_time = time.time() - start_total_time

    print("\n==================================================")
    print("      STAGE 3 EXECUTION & ACCURACY SUMMARY        ")
    print("==================================================")
    print(f"Dataset                         : {args.dataset}")
    print(f"Partition File Used             : {args.cluster_file}")
    print(f"Total Clusters                  : {num_clusters}")
    print(f"Model Architecture              : {args.model.upper()} ({args.n_layers} layers, hidden={args.n_hidden})")
    print(f"EigenPrompt Bridging            : {'ENABLED' if args.use_prompts else 'DISABLED'}")
    print(f"Total Pure Training Time        : {round(total_pure_train_time, 4)} Seconds")
    print(f"Average Epoch Time              : {round(total_pure_train_time / args.epochs, 4)} Seconds")
    print(f"Total Elapsed Time (with Eval)  : {round(total_elapsed_time, 4)} Seconds")
    print(f"Peak GPU Memory Allocated       : {round(peak_gpu_mb, 1)} MB")
    print("--------------------------------------------------")
    print(f"Final Train Accuracy            : {round(train_acc * 100, 2)}%")
    print(f"Final Validation Accuracy       : {round(val_acc * 100, 2)}%")
    print(f"Final Subgraph Test Accuracy    : {round(test_acc_subgraph * 100, 2)}%")
    print(f"Final Full-Graph Test Accuracy  : {round(full_test_acc * 100, 2)}%")
    print(f"Best Validation Accuracy        : {round(best_val_acc * 100, 2)}% (Epoch {best_epoch})")
    print(f"Test Accuracy @ Best Validation : {round(best_test_acc * 100, 2)}%")
    print("==================================================")


if __name__ == "__main__":
    main()
