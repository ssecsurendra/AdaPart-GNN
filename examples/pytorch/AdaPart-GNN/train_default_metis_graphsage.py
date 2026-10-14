#!/usr/bin/env python3
"""
Baseline 3-Layer GraphSAGE on Default DGL METIS Partitions (K=20)
Features:
- Strict 3-Layer GraphSAGE (hidden=256, dropout=0.5, mean aggregator)
- LayerNorm on intermediate representations
- Residual skip connections on intermediate representations
- CosineAnnealingLR (100 Epochs, lr=0.001, wd=5e-4)
- Partitioning: Default DGL METIS (K=20 unweighted) from ./cluster_default_metis/
- ABLATION / BASELINE: NO Boundary EigenPrompt (direct raw features)
- Yelp: Multi-Label Accuracy via MF.accuracy + Micro-F1 / Macro-F1
"""

import os
import gc
import sys
import copy
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.metrics import f1_score
import torchmetrics.functional as MF
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler


# ---------------------------------------------------------------------------
# Strict 3-Layer GraphSAGE with LayerNorm & Residual Connections (NO Prompt)
# ---------------------------------------------------------------------------
class GraphSAGEBaseline(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers=3, dropout=0.5, is_multilabel=False):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout = dropout
        self.is_multilabel = is_multilabel

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.residuals = nn.ModuleList()

        for l in range(n_layers):
            dim_in = in_feats if l == 0 else n_hidden
            dim_out = n_classes if l == n_layers - 1 else n_hidden

            # SAGEConv with mean aggregator
            self.convs.append(dglnn.SAGEConv(dim_in, dim_out, aggregator_type="mean"))

            # LayerNorm & Residual connection on intermediate layers
            if l < n_layers - 1:
                self.norms.append(nn.LayerNorm(dim_out))
                if dim_in != dim_out:
                    self.residuals.append(nn.Linear(dim_in, dim_out, bias=False))
                else:
                    self.residuals.append(nn.Identity())

        self.drop = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, g, x):
        """Full subgraph forward pass (direct features, NO EigenPrompt)."""
        h = x
        for l in range(self.n_layers):
            h_in = h
            h_out = self.convs[l](g, h_in)
            if l < self.n_layers - 1:
                h_res = self.residuals[l](h_in)
                h_out = self.norms[l](h_out + h_res)
                h = self.drop(self.activation(h_out))
            else:
                h = h_out
        return h

    def forward_layer(self, block, h, l):
        """Processes 1 layer on bipartite block (for True Layer-by-Layer Streaming)."""
        num_dst = block.num_dst_nodes()
        h_dst = h[:num_dst]
        h_out = self.convs[l](block, (h, h_dst))
        if l < self.n_layers - 1:
            h_res = self.residuals[l](h_dst)
            h_out = self.norms[l](h_out + h_res)
            h_out = self.activation(h_out)
            if self.training:
                h_out = self.drop(h_out)
        return h_out


# ---------------------------------------------------------------------------
# Dataset Loader
# ---------------------------------------------------------------------------
def load_unified_dataset(dataset_name, dataset_dir):
    print(f"\n[Dataset Loader] Loading dataset: {dataset_name} from '{dataset_dir}' ...")
    t0 = time.time()
    is_multilabel = False

    if dataset_name == "ogbn-arxiv":
        ds = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = ds[0]
        n_classes = ds.num_classes
    elif dataset_name == "reddit":
        ds = RedditDataset(raw_dir=dataset_dir)
        g = ds[0]
        n_classes = ds.num_classes
    elif dataset_name == "yelp":
        ds = YelpDataset(raw_dir=dataset_dir)
        g = ds[0]
        n_classes = ds.num_classes
        is_multilabel = True
    elif dataset_name == "igb-small":
        load_path = os.path.join(dataset_dir, "igb_small.dgl")
        if not os.path.exists(load_path):
            alt_path = "./dataset/igb_small.dgl"
            if os.path.exists(alt_path):
                load_path = alt_path
            else:
                raise FileNotFoundError(f"Cannot find igb_small.dgl in {dataset_dir} or ./dataset")
        graphs, _ = dgl.load_graphs(load_path)
        g = graphs[0]
        labels = g.ndata["label"].squeeze()
        n_classes = int(labels.max().item()) + 1
        is_multilabel = False
        g.ndata["label"] = labels.long()
    elif dataset_name == "ogbn-products":
        ds = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = ds[0]
        n_classes = ds.num_classes
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Ensure self-loops
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # Standardize masks
    for mask_key in ["train_mask", "val_mask", "test_mask"]:
        if mask_key in g.ndata:
            g.ndata[mask_key] = g.ndata[mask_key].bool()

    if not is_multilabel:
        g.ndata["label"] = g.ndata["label"].squeeze().long()
    else:
        g.ndata["label"] = g.ndata["label"].float()

    in_feats = g.ndata["feat"].shape[1]
    elapsed = time.time() - t0
    print(f"[Dataset Loader] Loaded in {elapsed:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,} | In-Feats: {in_feats} | Classes: {n_classes} | Multi-Label: {is_multilabel}")
    return g, in_feats, n_classes, is_multilabel


# ---------------------------------------------------------------------------
# Subgraph Extraction
# ---------------------------------------------------------------------------
def load_and_extract_subgraphs(g, cluster_file):
    print(f"\n[Partition Loader] Reading partition assignments from: {cluster_file}")
    t0 = time.time()
    cluster_ids = np.loadtxt(cluster_file, dtype=np.int64)
    assert len(cluster_ids) == g.num_nodes(), f"Mismatch: {len(cluster_ids)} cluster IDs vs {g.num_nodes()} nodes"

    unique_clusters = np.unique(cluster_ids)
    k_clusters = len(unique_clusters)
    print(f"[Partition Loader] Extracting {k_clusters} node-induced subgraphs...")

    subgraphs = []
    for c_id in unique_clusters:
        c_nodes = np.where(cluster_ids == c_id)[0]
        c_nodes_t = torch.from_numpy(c_nodes).long()
        sg = dgl.node_subgraph(g, c_nodes_t, relabel_nodes=True, store_ids=True)
        sg = dgl.remove_self_loop(sg)
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)

    print(f"[Partition Loader] Extracted {k_clusters} subgraphs in {time.time()-t0:.2f}s.")
    return subgraphs, k_clusters


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------
def evaluate_subgraph_mode(model, subgraphs, device, is_multilabel=False, num_classes=100, is_logging_epoch=False):
    model.eval()
    val_preds, val_targets = [], []
    test_preds, test_targets = [], []

    with torch.no_grad():
        for sg in subgraphs:
            sg = sg.to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            val_mask = sg.ndata["val_mask"].bool()

            logits = model(sg, feat)

            if val_mask.sum() > 0:
                val_preds.append(logits[val_mask].cpu())
                val_targets.append(label[val_mask].cpu())

            if is_logging_epoch:
                test_mask = sg.ndata["test_mask"].bool()
                if test_mask.sum() > 0:
                    test_preds.append(logits[test_mask].cpu())
                    test_targets.append(label[test_mask].cpu())

    if is_multilabel:
        v_logits = torch.cat(val_preds, dim=0)
        v_targets = torch.cat(val_targets, dim=0)
        val_acc = MF.accuracy(v_logits, v_targets.long(), task="multilabel", num_labels=num_classes, threshold=0.5).item()

        if is_logging_epoch and test_preds:
            t_logits = torch.cat(test_preds, dim=0)
            t_targets = torch.cat(test_targets, dim=0)
            test_acc = MF.accuracy(t_logits, t_targets.long(), task="multilabel", num_labels=num_classes, threshold=0.5).item()

            y_v_pred = (v_logits.sigmoid().numpy() > 0.5).astype(np.int32)
            y_t_pred = (t_logits.sigmoid().numpy() > 0.5).astype(np.int32)
            val_mic = f1_score(v_targets.numpy(), y_v_pred, average="micro", zero_division=0)
            test_mic = f1_score(t_targets.numpy(), y_t_pred, average="micro", zero_division=0)
            test_mac = f1_score(t_targets.numpy(), y_t_pred, average="macro", zero_division=0)
        else:
            test_acc, val_mic, test_mic, test_mac = 0.0, 0.0, 0.0, 0.0

        return val_acc, test_acc, val_mic, test_mic, test_mac
    else:
        v_p = torch.cat(val_preds, dim=0).argmax(dim=-1)
        v_t = torch.cat(val_targets, dim=0)
        val_acc = (v_p == v_t).float().mean().item() if len(v_t) > 0 else 0.0

        if is_logging_epoch and test_preds:
            t_p = torch.cat(test_preds, dim=0).argmax(dim=-1)
            t_t = torch.cat(test_targets, dim=0)
            test_acc = (t_p == t_t).float().mean().item() if len(t_t) > 0 else 0.0
        else:
            test_acc = 0.0

        return val_acc, test_acc, val_acc, test_acc, 0.0


def evaluate_layerwise_streaming(model, g, device, batch_size=4096, is_multilabel=False, num_classes=100):
    try:
        model.eval()
        with torch.no_grad():
            num_nodes = g.num_nodes()
            cur_feat = g.ndata["feat"]
            val_mask = g.ndata["val_mask"].bool()
            test_mask = g.ndata["test_mask"].bool()

            sampler = MultiLayerFullNeighborSampler(1)
            dataloader = DataLoader(
                g,
                torch.arange(num_nodes),
                sampler,
                device=device,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0
            )

            for l in range(model.n_layers):
                dim_out = model.n_classes if l == model.n_layers - 1 else model.n_hidden
                y = torch.empty((num_nodes, dim_out), dtype=torch.float32)

                for input_nodes, output_nodes, blocks in dataloader:
                    block = blocks[0].to(device)
                    h = cur_feat[input_nodes.cpu()].to(device)
                    h = model.forward_layer(block, h, l)
                    y[output_nodes.cpu()] = h.cpu()

                cur_feat = y

            logits = cur_feat
            labels = g.ndata["label"]

            if is_multilabel:
                s_t_acc = MF.accuracy(logits[test_mask], labels[test_mask].long(), task="multilabel", num_labels=num_classes, threshold=0.5).item()
                y_pred = (logits[test_mask].sigmoid().numpy() > 0.5).astype(np.int32)
                s_t_mic = f1_score(labels[test_mask].numpy(), y_pred, average="micro", zero_division=0)
                s_t_mac = f1_score(labels[test_mask].numpy(), y_pred, average="macro", zero_division=0)
                return s_t_acc, s_t_mic, s_t_mac
            else:
                preds = logits[test_mask].argmax(dim=-1)
                s_t_acc = (preds == labels[test_mask]).float().mean().item()
                return s_t_acc, s_t_acc, 0.0
    except Exception as e:
        print(f"[Streaming Warning] Streaming inference skipped: {e}")
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def run_default_metis_graphsage(args):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print(f" BASELINE 3-LAYER GRAPHSAGE BENCHMARK (DEFAULT DGL METIS, K=20, NO PROMPT)")
    print(f" Dataset: {args.dataset.upper()} | Device: {device} | Partition: {args.cluster_file}")
    print(f" Arch: 3-Layer GraphSAGE | Hidden: {args.n_hidden} | LayerNorm + Residual + CosineAnnealingLR")
    print("=" * 80)

    g, in_feats, n_classes, is_multilabel = load_unified_dataset(args.dataset, args.dataset_dir)
    subgraphs, k_clusters = load_and_extract_subgraphs(g, args.cluster_file)

    model = GraphSAGEBaseline(
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=n_classes,
        n_layers=3,
        dropout=args.dropout,
        is_multilabel=is_multilabel
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    metric_name = "Micro-F1" if is_multilabel else "Accuracy"
    loss_fn_name = "BCEWithLogitsLoss" if is_multilabel else "CrossEntropyLoss"

    print(f"\nModel Configuration: 3-Layer GraphSAGE (Baseline) | Parameters: {total_params:,}")
    print(f"Loss Function: {loss_fn_name} | Primary Metric: {metric_name}")
    print(f"Clusters (K): {k_clusters} | Training Plan: {args.epochs} Epochs\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_score = 0.0
    best_test_score = 0.0
    best_epoch = 0
    best_model_state = None
    total_pure_train_time = 0.0

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    if is_multilabel:
        print(f" Epoch  | Train(s) |   Loss    | Val Acc  | Val Mic  | Subg Test Acc | Subg Test Mic | GPU Mem ")
        print("-" * 88)
    else:
        print(f" Epoch  | Train(s) |   Loss    |  Val Acc  | Subg Test  | GPU Mem ")
        print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()

        epoch_start_time = time.time()
        perm = np.random.permutation(k_clusters)
        total_loss = 0.0
        total_train_nodes = 0

        for c_idx in perm:
            sg = subgraphs[c_idx].to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            train_mask = sg.ndata["train_mask"].bool()

            if train_mask.sum() == 0:
                continue

            optimizer.zero_grad()
            logits = model(sg, feat)

            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits[train_mask], label[train_mask].float())
            else:
                loss = F.cross_entropy(logits[train_mask], label[train_mask])

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * train_mask.sum().item()
            total_train_nodes += train_mask.sum().item()

        scheduler.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_train_time = time.time() - epoch_start_time
        total_pure_train_time += epoch_train_time

        train_loss = total_loss / total_train_nodes if total_train_nodes > 0 else 0.0

        is_logging = (epoch % 10 == 0 or epoch == 1 or epoch == args.epochs)

        # Subgraph Evaluation
        v_acc, t_acc, v_mic, t_mic, t_mac = evaluate_subgraph_mode(
            model, subgraphs, device,
            is_multilabel=is_multilabel, num_classes=n_classes, is_logging_epoch=is_logging
        )

        tracking_score = v_mic if is_multilabel else v_acc

        if tracking_score > best_val_score:
            best_val_score = tracking_score
            best_test_score = t_mic if is_multilabel else t_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if is_logging:
            if is_multilabel:
                print(f"{epoch:^7d} | {epoch_train_time:^8.4f} s | {train_loss:^9.4f} | {v_acc*100:^8.2f}% | {v_mic*100:^8.2f}% | {t_acc*100:^12.2f}% | {t_mic*100:^12.2f}% | {peak_gpu_mb:^7.1f} MB")
            else:
                print(f"{epoch:^7d} | {epoch_train_time:^8.4f} s | {train_loss:^9.4f} | {v_acc*100:^8.2f} % | {t_acc*100:^9.2f} % | {peak_gpu_mb:^7.1f} MB")

    # Load best checkpoint
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n[Checkpoint] Loaded best validation checkpoint from Epoch {best_epoch} (Val {metric_name}: {best_val_score*100:.2f}%)")

    # Final Subgraph Evaluation
    final_v_acc, final_t_acc, final_v_mic, final_t_mic, final_t_mac = evaluate_subgraph_mode(
        model, subgraphs, device, is_multilabel=is_multilabel, num_classes=n_classes, is_logging_epoch=True
    )

    # True Layer-Wise Full-Graph Streaming Inference
    print("[Streaming Inference] Running True Layer-by-Layer Full-Graph Streaming Inference...")
    s_t_acc, s_t_mic, s_t_mac = evaluate_layerwise_streaming(
        model, g, device, batch_size=4096, is_multilabel=is_multilabel, num_classes=n_classes
    )

    avg_epoch_time = total_pure_train_time / args.epochs

    print("\n" + "=" * 80)
    print(f" BENCHMARK SUMMARY: DEFAULT DGL METIS GRAPHSAGE ON {args.dataset.upper()} (K={k_clusters})")
    print("=" * 80)
    print(f" Partition File Used              : {args.cluster_file}")
    print(f" Total Clusters (K)               : {k_clusters}")
    print(f" Model Architecture               : 3-Layer GraphSAGE (Baseline) + LayerNorm + Residual Highway")
    print(f" Boundary Prompting               : NONE (Direct raw features)")
    print(f" Optimization                     : CosineAnnealingLR (100 Epochs, lr={args.lr}, wd={args.weight_decay})")
    print("--------------------------------------------------------------------------------")
    print(f" Total Pure Training Time (100 Ep): {round(total_pure_train_time, 4)} Seconds")
    print(f" Average Epoch Training Time      : {round(avg_epoch_time, 4)} Seconds")
    print(f" Peak GPU Training Memory         : {round(peak_gpu_mb, 1)} MB")
    print("--------------------------------------------------------------------------------")
    if is_multilabel:
        print(f" Best Validation Micro-F1         : {round(best_val_score * 100, 2)}% (Epoch {best_epoch})")
        print(f" Subgraph Multi-Label Accuracy    : {round(final_t_acc * 100, 2)}%")
        print(f" Subgraph Micro-F1                : {round(final_t_mic * 100, 2)}%")
        print(f" Subgraph Macro-F1                : {round(final_t_mac * 100, 2)}%")
        if s_t_acc > 0:
            print(f" Stream Multi-Label Accuracy      : {round(s_t_acc * 100, 2)}%")
            print(f" Stream Micro-F1                  : {round(s_t_mic * 100, 2)}%")
            print(f" Stream Macro-F1                  : {round(s_t_mac * 100, 2)}%")
    else:
        print(f" Best Validation Accuracy         : {round(best_val_score * 100, 2)}% (Epoch {best_epoch})")
        print(f" Subgraph Test Accuracy           : {round(final_t_acc * 100, 2)}%")
        if s_t_acc > 0:
            print(f" Stream Full-Graph Test Accuracy  : {round(s_t_acc * 100, 2)}%")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["ogbn-arxiv", "reddit", "yelp", "igb-small", "ogbn-products"])
    parser.add_argument("--dataset_dir", type=str, default="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset")
    parser.add_argument("--cluster_file", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    run_default_metis_graphsage(args)


if __name__ == "__main__":
    main()
