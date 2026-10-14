#!/usr/bin/env python3
"""
Evaluate Yelp with exact Accuracy metric from node_classification_yelp_surendra.py
Compares Scheme A (K=10) vs Scheme B (K=26) for 100 Epochs.
Computes:
  1. Multi-Label Accuracy via torchmetrics.functional.accuracy(task="multilabel", num_labels=100, threshold=0.5)
  2. Micro-F1 via sklearn.metrics.f1_score(average='micro')
  3. Macro-F1 via sklearn.metrics.f1_score(average='macro')
Both in Subgraph evaluation and Full-Graph Layer-Wise Streaming Inference.
"""

import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.metrics import f1_score
import torchmetrics.functional as MF
from dgl.data import YelpDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler


class GraphSAGEModel(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers=3, dropout=0.5):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_linears = nn.ModuleList()

        for l in range(n_layers):
            in_dim = in_feats if l == 0 else n_hidden
            out_dim = n_classes if l == n_layers - 1 else n_hidden

            self.layers.append(dglnn.SAGEConv(in_dim, out_dim, aggregator_type="mean"))

            if l < n_layers - 1:
                self.norms.append(nn.LayerNorm(out_dim))
                if in_dim != out_dim:
                    self.res_linears.append(nn.Linear(in_dim, out_dim, bias=False))
                else:
                    self.res_linears.append(nn.Identity())

        self.drop = nn.Dropout(dropout)

    def forward(self, g, x):
        h = x
        for l in range(self.n_layers):
            h_res = self.res_linears[l](h) if l < self.n_layers - 1 else None
            h = self.layers[l](g, h)
            if l < self.n_layers - 1:
                h = self.norms[l](h)
                h = F.relu(h)
                if h_res is not None:
                    h = h + h_res
                h = self.drop(h)
        return h

    def forward_layer(self, block, h, layer_idx):
        h_res = self.res_linears[layer_idx](h[:block.num_dst_nodes()]) if layer_idx < self.n_layers - 1 else None
        h_out = self.layers[layer_idx](block, h)
        if layer_idx < self.n_layers - 1:
            h_out = self.norms[layer_idx](h_out)
            h_out = F.relu(h_out)
            if h_res is not None:
                h_out = h_out + h_res
            h_out = self.drop(h_out)
        return h_out


class EigenPrompt(nn.Module):
    def __init__(self, in_feats, alpha=0.1, gamma=1.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma
        self.prompt_proj = nn.Linear(in_feats, in_feats, bias=False)
        nn.init.eye_(self.prompt_proj.weight)

    def forward(self, feat, centrality_scores=None):
        prompt = self.prompt_proj(feat)
        if centrality_scores is not None:
            c = centrality_scores.unsqueeze(-1).to(feat.device)
            modulation = (1.0 - c) ** self.gamma
            return feat + self.alpha * (modulation * prompt)
        return feat + self.alpha * prompt


def load_yelp(dataset_dir):
    print(f"\n[Loader] Loading Yelp from: {dataset_dir} ...")
    t0 = time.time()
    ds = YelpDataset(raw_dir=dataset_dir)
    g = ds[0]
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    print(f"[Loader] Yelp loaded in {time.time()-t0:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,} | Classes: {ds.num_classes}")
    return g, ds.num_classes


def compute_yelp_metrics(logits, labels, num_classes, compute_f1=False):
    """
    Exact metric implementation matching node_classification_yelp_surendra.py:
    1. MF.accuracy(..., task="multilabel", num_labels=num_classes, threshold=0.5)
    2. f1_score(y_true, y_pred, average='micro')
    3. f1_score(y_true, y_pred, average='macro')
    """
    acc = MF.accuracy(
        logits,
        labels.long(),
        task="multilabel",
        num_labels=num_classes,
        threshold=0.5
    ).item()

    if compute_f1:
        y_true = labels.cpu().numpy()
        y_pred = (logits.sigmoid().cpu().numpy() > 0.5).astype(np.int32)
        f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        return acc, f1_micro, f1_macro

    return acc, 0.0, 0.0


def eval_subgraphs(model, prompt_module, subgraphs, device, num_classes, is_logging_epoch=False):
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    val_logits, val_labels = [], []
    test_logits, test_labels = [], []

    with torch.no_grad():
        for sg in subgraphs:
            sg = sg.to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            val_mask = sg.ndata["val_mask"].bool()

            if prompt_module is not None:
                c_scores = sg.ndata.get("centrality", None)
                feat = prompt_module(feat, c_scores)

            logits = model(sg, feat)

            if val_mask.sum() > 0:
                val_logits.append(logits[val_mask].cpu())
                val_labels.append(label[val_mask].cpu())

            if is_logging_epoch:
                test_mask = sg.ndata["test_mask"].bool()
                if test_mask.sum() > 0:
                    test_logits.append(logits[test_mask].cpu())
                    test_labels.append(label[test_mask].cpu())

    v_acc, v_mic, v_mac = compute_yelp_metrics(
        torch.cat(val_logits), torch.cat(val_labels), num_classes, compute_f1=is_logging_epoch
    )

    if is_logging_epoch and test_logits:
        t_acc, t_mic, t_mac = compute_yelp_metrics(
            torch.cat(test_logits), torch.cat(test_labels), num_classes, compute_f1=True
        )
    else:
        t_acc, t_mic, t_mac = 0.0, 0.0, 0.0

    return v_acc, v_mic, v_mac, t_acc, t_mic, t_mac


def eval_streaming(model, prompt_module, g, device, num_classes, batch_size=4096):
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    with torch.no_grad():
        num_nodes = g.num_nodes()
        feat = g.ndata["feat"]
        label = g.ndata["label"]
        val_mask = g.ndata["val_mask"].bool()
        test_mask = g.ndata["test_mask"].bool()

        if prompt_module is not None:
            c_scores = g.ndata.get("centrality", None)
            cur_feat = torch.empty_like(feat)
            chunk_size = 32768
            for i in range(0, num_nodes, chunk_size):
                end_i = min(i + chunk_size, num_nodes)
                c_chunk = c_scores[i:end_i] if c_scores is not None else None
                cur_feat[i:end_i] = prompt_module(feat[i:end_i].to(device), c_chunk).cpu()
        else:
            cur_feat = feat

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
        v_acc, v_mic, v_mac = compute_yelp_metrics(logits[val_mask], label[val_mask], num_classes, compute_f1=True)
        t_acc, t_mic, t_mac = compute_yelp_metrics(logits[test_mask], label[test_mask], num_classes, compute_f1=True)
        return v_acc, v_mic, v_mac, t_acc, t_mic, t_mac


def run_experiment(scheme_name, g, num_classes, cluster_file, device_id=0, epochs=100):
    print("\n" + "=" * 80)
    print(f" EXPERIMENT: YELP | {scheme_name}")
    print(f" Partition file: {cluster_file}")
    print("=" * 80)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Load partition
    cluster_ids = np.loadtxt(cluster_file, dtype=np.int64)
    unique_clusters = np.unique(cluster_ids)
    k = len(unique_clusters)
    print(f"[Partition] Loaded {k} clusters for Yelp ({len(cluster_ids):,} nodes).")

    # Extract subgraphs
    t0 = time.time()
    subgraphs = []
    for c_id in unique_clusters:
        c_nodes = np.where(cluster_ids == c_id)[0]
        c_nodes_t = torch.from_numpy(c_nodes).long()
        sg = dgl.node_subgraph(g, c_nodes_t, relabel_nodes=True, store_ids=True)
        sg = dgl.remove_self_loop(sg)
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)
    print(f"[Subgraphs] Extracted {k} subgraphs in {time.time()-t0:.2f}s.")

    # Init model & optimizer
    in_feats = g.ndata["feat"].shape[1]
    model = GraphSAGEModel(in_feats, 256, num_classes, n_layers=3, dropout=0.5).to(device)
    prompt_module = EigenPrompt(in_feats).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(prompt_module.parameters()),
        lr=0.001,
        weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nEpoch   | Train(s) |  Loss    | Val Acc   | Val Mic  | Subg Test Acc | Subg Test Mic | GPU Mem")
    print("-" * 88)

    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    total_train_time = 0.0

    torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, epochs + 1):
        model.train()
        prompt_module.train()
        t_ep_start = time.time()

        perm = np.random.permutation(len(subgraphs))
        total_loss = 0.0
        total_train_nodes = 0

        for idx in perm:
            sg = subgraphs[idx].to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            train_mask = sg.ndata["train_mask"].bool()

            if train_mask.sum() == 0:
                continue

            optimizer.zero_grad()
            c_scores = sg.ndata.get("centrality", None)
            feat_aug = prompt_module(feat, c_scores)
            logits = model(sg, feat_aug)

            loss = F.binary_cross_entropy_with_logits(logits[train_mask], label[train_mask].float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * train_mask.sum().item()
            total_train_nodes += train_mask.sum().item()

        scheduler.step()
        torch.cuda.synchronize()
        ep_time = time.time() - t_ep_start
        total_train_time += ep_time

        avg_loss = total_loss / total_train_nodes if total_train_nodes > 0 else 0.0

        is_logging = (epoch % 10 == 0 or epoch == 1 or epoch == epochs)
        # Subgraph Eval
        v_acc, v_mic, v_mac, t_acc, t_mic, t_mac = eval_subgraphs(
            model, prompt_module, subgraphs, device, num_classes, is_logging_epoch=is_logging
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        if is_logging:
            print(f"{epoch:^7d} | {ep_time:^8.4f} | {avg_loss:^8.4f} | {v_acc*100:^8.2f}% | {v_mic*100:^8.2f}% | {t_acc*100:^12.2f}% | {t_mic*100:^12.2f}% | {mem_mb:^7.1f}MB")

    # Load best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    # Subgraph Final
    print(f"\n[Checkpoint] Loaded best checkpoint from Epoch {best_epoch} (Val Acc: {best_val_acc*100:.2f}%)")
    v_acc, v_mic, v_mac, t_acc_subg, t_mic_subg, t_mac_subg = eval_subgraphs(
        model, prompt_module, subgraphs, device, num_classes, is_logging_epoch=True
    )

    # Streaming Final
    print("[Inference] Running Full-Graph Layer-Wise Streaming Evaluation...")
    s_v_acc, s_v_mic, s_v_mac, s_t_acc, s_t_mic, s_t_mac = eval_streaming(model, prompt_module, g, device, num_classes)

    return {
        "scheme": scheme_name,
        "k": k,
        "total_train_time": total_train_time,
        "avg_epoch_time": total_train_time / epochs,
        "peak_mem_mb": torch.cuda.max_memory_allocated(device) / (1024 * 1024),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc * 100,
        "subg_test_acc": t_acc_subg * 100,
        "subg_test_micro": t_mic_subg * 100,
        "subg_test_macro": t_mac_subg * 100,
        "stream_test_acc": s_t_acc * 100,
        "stream_test_micro": s_t_mic * 100,
        "stream_test_macro": s_t_mac * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    g, num_classes = load_yelp(args.dataset_dir)

    file_a = "./cluster_hardware_adaptive/yelp_cluster_id.txt"
    file_b = "./cluster_scale_adaptive/yelp_cluster_id.txt"

    res_a = run_experiment("Scheme A (Hardware-Adaptive, K=10)", g, num_classes, file_a, device_id=args.device, epochs=args.epochs)
    res_b = run_experiment("Scheme B (Scale-Adaptive, K=26)", g, num_classes, file_b, device_id=args.device, epochs=args.epochs)

    print("\n" + "=" * 90)
    print(" FINAL COMPARATIVE SUMMARY: YELP DATASET (EXACT ACCURACY & F1 METRICS)")
    print("=" * 90)
    print(f"{'Metric':<38} | {'Scheme A (K=10)':<20} | {'Scheme B (K=26)':<20} | {'Delta / Gain':<10}")
    print("-" * 90)
    print(f"{'Total Clusters (K)':<38} | {res_a['k']:<20} | {res_b['k']:<20} | {res_b['k'] - res_a['k']:+d}")
    print(f"{'Multi-Label Test Accuracy (Subgraph)':<38} | {res_a['subg_test_acc']:.2f}%{'':<15} | {res_b['subg_test_acc']:.2f}%{'':<15} | {res_b['subg_test_acc'] - res_a['subg_test_acc']:+.2f}%")
    print(f"{'Multi-Label Test Accuracy (Stream)':<38} | {res_a['stream_test_acc']:.2f}%{'':<15} | {res_b['stream_test_acc']:.2f}%{'':<15} | {res_b['stream_test_acc'] - res_a['stream_test_acc']:+.2f}%")
    print(f"{'Test Micro-F1 (Subgraph)':<38} | {res_a['subg_test_micro']:.2f}%{'':<15} | {res_b['subg_test_micro']:.2f}%{'':<15} | {res_b['subg_test_micro'] - res_a['subg_test_micro']:+.2f}%")
    print(f"{'Test Micro-F1 (Stream)':<38} | {res_a['stream_test_micro']:.2f}%{'':<15} | {res_b['stream_test_micro']:.2f}%{'':<15} | {res_b['stream_test_micro'] - res_a['stream_test_micro']:+.2f}%")
    print(f"{'Test Macro-F1 (Subgraph)':<38} | {res_a['subg_test_macro']:.2f}%{'':<15} | {res_b['subg_test_macro']:.2f}%{'':<15} | {res_b['subg_test_macro'] - res_a['subg_test_macro']:+.2f}%")
    print(f"{'Test Macro-F1 (Stream)':<38} | {res_a['stream_test_macro']:.2f}%{'':<15} | {res_b['stream_test_macro']:.2f}%{'':<15} | {res_b['stream_test_macro'] - res_a['stream_test_macro']:+.2f}%")
    print(f"{'Peak GPU Memory':<38} | {res_a['peak_mem_mb']:.1f} MB{'':<12} | {res_b['peak_mem_mb']:.1f} MB{'':<12} | {res_b['peak_mem_mb'] - res_a['peak_mem_mb']:+.1f} MB")
    print(f"{'Avg Epoch Training Time':<38} | {res_a['avg_epoch_time']:.4f} s{'':<13} | {res_b['avg_epoch_time']:.4f} s{'':<13} | {res_b['avg_epoch_time'] - res_a['avg_epoch_time']:+.4f} s")
    print(f"{'Total Pure Training Time (100 Ep)':<38} | {res_a['total_train_time']:.2f} s{'':<14} | {res_b['total_train_time']:.2f} s{'':<14} | {res_b['total_train_time'] - res_a['total_train_time']:+.2f} s")
    print("=" * 90)


if __name__ == "__main__":
    main()
