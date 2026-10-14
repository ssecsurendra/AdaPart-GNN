import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset


class SubgraphGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        if n_layers == 1:
            self.layers.append(dglnn.GraphConv(in_feats, n_classes, allow_zero_in_degree=True))
        else:
            self.layers.append(dglnn.GraphConv(in_feats, n_hidden, allow_zero_in_degree=True))
            for _ in range(n_layers - 2):
                self.layers.append(dglnn.GraphConv(n_hidden, n_hidden, allow_zero_in_degree=True))
            self.layers.append(dglnn.GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, g, x):
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def load_dataset(dataset_name):
    print(f"Loading dataset: {dataset_name} ...")
    t0 = time.time()
    if dataset_name == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root="./dataset"))
        g = data[0]
    elif dataset_name == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root="./dataset"))
        g = data[0]
    elif dataset_name == "reddit":
        data = RedditDataset(raw_dir="./dataset")
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


def create_cluster_subgraphs(g, cluster_file_path, num_clusters):
    print(f"Reading cluster partition IDs from: {cluster_file_path}")
    t0 = time.time()
    
    if not os.path.exists(cluster_file_path):
        raise FileNotFoundError(f"Cluster partition file not found: {cluster_file_path}")

    with open(cluster_file_path, "r") as f:
        cluster_ids = np.array([int(line.strip()) for line in f if line.strip() != ""], dtype=np.int64)

    if len(cluster_ids) != g.num_nodes():
        raise ValueError(f"Cluster ID count ({len(cluster_ids)}) does not match graph nodes ({g.num_nodes()})")

    actual_clusters = len(np.unique(cluster_ids))
    print(f"Loaded {len(cluster_ids):,} node assignments ({actual_clusters} unique clusters)")

    subgraphs = []
    print("Extracting induced subgraphs for each cluster...")
    for cid in range(num_clusters):
        node_indices = np.where(cluster_ids == cid)[0]
        if len(node_indices) == 0:
            continue
        
        # Extract node-induced subgraph with all features/labels/masks retained
        sg = dgl.node_subgraph(g, node_indices, relabel_nodes=True)
        # Add self-loops to avoid zero in-degree issues in GCN
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)

    print(f"Created {len(subgraphs)} subgraphs in {time.time() - t0:.2f}s")
    return subgraphs


def evaluate(model, subgraphs, device):
    model.eval()
    total_val_loss = 0.0
    total_val_correct = 0
    total_val_nodes = 0

    total_test_correct = 0
    total_test_nodes = 0

    with torch.no_grad():
        for sg in subgraphs:
            sg = sg.to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            val_mask = sg.ndata["val_mask"]
            test_mask = sg.ndata["test_mask"]

            logits = model(sg, feat)

            if val_mask.sum() > 0:
                val_loss = F.cross_entropy(logits[val_mask], label[val_mask], reduction="sum")
                total_val_loss += val_loss.item()
                preds = logits[val_mask].argmax(dim=-1)
                total_val_correct += (preds == label[val_mask]).sum().item()
                total_val_nodes += val_mask.sum().item()

            if test_mask.sum() > 0:
                preds = logits[test_mask].argmax(dim=-1)
                total_test_correct += (preds == label[test_mask]).sum().item()
                total_test_nodes += test_mask.sum().item()

    val_acc = (total_val_correct / total_val_nodes) if total_val_nodes > 0 else 0.0
    val_loss = (total_val_loss / total_val_nodes) if total_val_nodes > 0 else 0.0
    test_acc = (total_test_correct / total_test_nodes) if total_test_nodes > 0 else 0.0
    return val_loss, val_acc, test_acc, total_val_correct, total_val_nodes, total_test_correct, total_test_nodes


def main():
    parser = argparse.ArgumentParser(description="Partition-Based Subgraph GCN Training")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Dataset name")
    parser.add_argument("--cluster_file", type=str, default="", help="Path to cluster ID file")
    parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters")
    parser.add_argument("--batch_clusters", type=int, default=1, help="Number of clusters per training batch")
    parser.add_argument("--n_hidden", type=int, default=256, help="Hidden embedding dimension")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID (-1 for CPU)")
    args = parser.parse_args()

    # Determine device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Set default cluster file if not provided
    if not args.cluster_file:
        args.cluster_file = f"./cluster/{args.dataset}_cluster_id.txt"

    # 1. Load dataset
    g, in_feats, num_classes = load_dataset(args.dataset)

    # 2. Extract subgraphs based on pre-computed partitions
    subgraphs = create_cluster_subgraphs(g, args.cluster_file, args.num_clusters)
    num_subgraphs = len(subgraphs)

    # 3. Initialize Model & Optimizer
    model = SubgraphGCN(
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=num_classes,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"\n==================================================")
    print(f"Model Architecture: {args.n_layers}-Layer Subgraph GCN")
    print(f"In: {in_feats} -> Hidden: {args.n_hidden} -> Classes: {num_classes}")
    print(f"Batch Clusters per step: {args.batch_clusters} | Total Clusters: {num_subgraphs}")
    print(f"==================================================\n")

    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0
    best_test_correct = 0
    best_test_nodes = 0
    total_pure_train_time = 0.0

    print(f"Starting Subgraph GCN Training for {args.epochs} Epochs...")
    print(f"{'Epoch':^7} | {'Train Time':^10} | {'Train Loss':^10} | {'Train Acc':^9} | {'Val Acc':^9} | {'Test Acc':^9} | {'GPU Mem':^9}")
    print("-" * 80)

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_train_start = time.time()
        model.train()

        # Shuffle cluster order each epoch
        perm = torch.randperm(num_subgraphs).tolist()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        # Iterate through subgraphs in batches of size args.batch_clusters
        for start_idx in range(0, num_subgraphs, args.batch_clusters):
            batch_cids = perm[start_idx : start_idx + args.batch_clusters]
            if len(batch_cids) == 1:
                batch_g = subgraphs[batch_cids[0]].to(device)
            else:
                batch_g = dgl.batch([subgraphs[cid] for cid in batch_cids]).to(device)

            feat = batch_g.ndata["feat"]
            label = batch_g.ndata["label"]
            train_mask = batch_g.ndata["train_mask"]

            if train_mask.sum() == 0:
                continue

            logits = model(batch_g, feat)
            loss = F.cross_entropy(logits[train_mask], label[train_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * train_mask.sum().item()
            preds = logits[train_mask].argmax(dim=-1)
            train_correct += (preds == label[train_mask]).sum().item()
            train_total += train_mask.sum().item()

        epoch_train_time = time.time() - t_train_start
        total_pure_train_time += epoch_train_time

        train_loss = epoch_loss / train_total if train_total > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Evaluate across subgraphs
        val_loss, val_acc, test_acc, val_corr, val_n, test_corr, test_n = evaluate(model, subgraphs, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_correct = test_corr
            best_test_nodes = test_n

        gpu_mem = f"{torch.cuda.max_memory_allocated() / 1024**2:.1f}MB" if torch.cuda.is_available() else "N/A"

        if args.epochs <= 10 or epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:^7d} | {epoch_train_time:^9.4f}s | {train_loss:^10.4f} | {train_acc:^9.4f} | {val_acc:^9.4f} | {test_acc:^9.4f} | {gpu_mem:^9}")

    total_elapsed_time = time.time() - start_total_time

    print("-" * 80)
    print("\n==================================================")
    print("           EXECUTION TIME & ACCURACY SUMMARY      ")
    print("==================================================")
    print(f"Total Pure Training Time (only training) : {total_pure_train_time:.4f} Seconds")
    print(f"Average Training Time per Epoch          : {total_pure_train_time / args.epochs:.4f} Seconds")
    print(f"Total Elapsed Time (train + eval)        : {total_elapsed_time:.4f} Seconds")
    print("--------------------------------------------------")
    print(f"Final Train Accuracy                     : {train_acc * 100:.2f}% ({train_correct}/{train_total} nodes)")
    print(f"Final Validation Accuracy                : {val_acc * 100:.2f}% ({val_corr}/{val_n} nodes)")
    print(f"Final Test Accuracy (on test nodes)      : {test_acc * 100:.2f}% ({test_corr}/{test_n} nodes)")
    print("--------------------------------------------------")
    print(f"Best Validation Accuracy                 : {best_val_acc * 100:.2f}% (Achieved at Epoch {best_epoch})")
    print(f"Test Accuracy @ Best Validation          : {best_test_acc * 100:.2f}% ({best_test_correct}/{best_test_nodes} nodes)")
    print(f"Peak GPU Memory Allocated                : {gpu_mem}")
    print("==================================================\n")


if __name__ == "__main__":
    main()

