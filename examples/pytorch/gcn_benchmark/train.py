import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from dgl import AddSelfLoop
from dgl.data import AsNodePredDataset, RedditDataset, CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset
from ogb.nodeproppred import DglNodePropPredDataset


class GCN(nn.Module):
    """
    3-Layer Graph Convolutional Network (GCN) Backbone
    """
    def __init__(self, in_size, hid_size, out_size, n_layers=3, dropout=0.5):
        super().__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers

        for l in range(n_layers):
            dim_in = in_size if l == 0 else hid_size
            dim_out = out_size if l == n_layers - 1 else hid_size
            # GraphConv with ReLU on hidden layers and allow_zero_in_degree
            act = F.relu if l < n_layers - 1 else None
            self.layers.append(dglnn.GraphConv(dim_in, dim_out, activation=act, allow_zero_in_degree=True))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h


def evaluate(g, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def load_data(dataset_name, dataset_dir="./dataset"):
    print(f"\n[Data Loader] Loading dataset: {dataset_name} from '{dataset_dir}' ...")
    t0 = time.time()

    # Handle dataset aliases
    ds = dataset_name.lower().strip()
    if ds in ["ogbn-arxiv", "arxiv", "ogbn-reddit"]:
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = data[0]
        num_classes = data.num_classes
        actual_name = "ogbn-arxiv"
    elif ds in ["ogbn-products", "products"]:
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = data[0]
        num_classes = data.num_classes
        actual_name = "ogbn-products"
    elif ds in ["reddit"]:
        data = RedditDataset(raw_dir=dataset_dir)
        g = data[0]
        num_classes = data.num_classes
        actual_name = "reddit"
    elif ds == "cora":
        data = CoraGraphDataset(transform=AddSelfLoop())
        g = data[0]
        num_classes = data.num_classes
        actual_name = "cora"
    elif ds == "citeseer":
        data = CiteseerGraphDataset(transform=AddSelfLoop())
        g = data[0]
        num_classes = data.num_classes
        actual_name = "citeseer"
    elif ds == "pubmed":
        data = PubmedGraphDataset(transform=AddSelfLoop())
        g = data[0]
        num_classes = data.num_classes
        actual_name = "pubmed"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standardize label tensor
    if "label" in g.ndata:
        g.ndata["label"] = g.ndata["label"].squeeze().long()

    # Standardize masks
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if mask_name in g.ndata:
            g.ndata[mask_name] = g.ndata[mask_name].bool()

    # Ensure self-loops
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    elapsed = time.time() - t0
    print(f"[Data Loader] Loaded {actual_name} in {elapsed:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,} | Classes: {num_classes}")
    return g, num_classes, actual_name


def main():
    parser = argparse.ArgumentParser(description="3-Layer Full-Graph GCN Benchmark")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", help="Dataset name ('ogbn-arxiv', 'reddit', 'ogbn-products').")
    parser.add_argument("--dataset_dir", type=str, default="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset")
    parser.add_argument("--n_layers", type=int, default=3, help="Number of GCN layers")
    parser.add_argument("--n_hidden", type=int, default=256, help="Hidden feature dimension")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout probability")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"Target Execution Device: {device} | GCN Layers: {args.n_layers} | Hidden: {args.n_hidden}")

    # Load dataset
    g, num_classes, dataset_name = load_data(args.dataset, args.dataset_dir)
    g = g.to(device)
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    in_size = features.shape[1]
    out_size = num_classes

    # Instantiate 3-layer GCN
    model = GCN(
        in_size=in_size,
        hid_size=args.n_hidden,
        out_size=out_size,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)

    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Model Configuration: {args.n_layers}-Layer GCN | Hidden: {args.n_hidden} | LR: {args.lr} | Weight Decay: {args.weight_decay}")
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Training for {args.epochs} epochs...\n")

    print(f"{'Epoch':^7} | {'Train Time(s)':^14} | {'Train Loss':^12} | {'Val Acc':^10} | {'Test Acc':^10} | {'GPU Mem':^9}")
    print("-" * 75)

    total_pure_train_time = 0.0
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_epoch = 0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Synchronize before training step timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_step_start = time.time()

        logits = model(g, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_step = time.time() - t_step_start
        total_pure_train_time += t_step

        # Evaluation
        val_acc = evaluate(g, features, labels, val_mask, model)
        test_acc = evaluate(g, features, labels, test_mask, model)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            best_epoch = epoch

        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:^7d} | {t_step:^14.4f} | {loss.item():^12.4f} | {val_acc*100:^9.2f}% | {test_acc*100:^9.2f}% | {peak_gpu_mb:^8.1f}MB")

    final_test_acc = evaluate(g, features, labels, test_mask, model)

    print("\n==================================================================")
    print(f"      3-LAYER GCN BENCHMARK RESULTS: {dataset_name.upper()}      ")
    print("==================================================================")
    print(f"Dataset                         : {dataset_name}")
    print(f"Architecture                    : {args.n_layers}-Layer GCN (Hidden={args.n_hidden})")
    print(f"Hyperparameters                 : lr={args.lr}, weight_decay={args.weight_decay}, dropout={args.dropout}")
    print(f"Total Epochs                    : {args.epochs}")
    print("------------------------------------------------------------------")
    print(f"Total Pure Training Time        : {total_pure_train_time:.4f} Seconds")
    print(f"Average Epoch Training Time     : {total_pure_train_time / args.epochs:.4f} Seconds")
    print(f"Peak GPU Memory Allocated       : {peak_gpu_mb:.1f} MB")
    print("------------------------------------------------------------------")
    print(f"Best Validation Accuracy        : {best_val_acc * 100:.2f}% (Epoch {best_epoch})")
    print(f"Test Accuracy @ Best Validation : {best_test_acc * 100:.2f}%")
    print(f"Final Test Accuracy (Epoch 100) : {final_test_acc * 100:.2f}%")
    print("==================================================================\n")


if __name__ == "__main__":
    main()
