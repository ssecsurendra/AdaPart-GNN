import argparse
import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn
from sklearn.metrics import f1_score
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

# ---------------------------------------------------------------------------
# Strict 3-Layer GraphSAGE with LayerNorm & Residual Connections
# ---------------------------------------------------------------------------
class UnifiedGraphSAGE(nn.Module):
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
        """Full subgraph forward pass."""
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
# Boundary EigenPrompt Bridging Module
# ---------------------------------------------------------------------------
class EigenPrompt(nn.Module):
    def __init__(self, in_feats, alpha=0.1, gamma=1.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.gamma = gamma
        self.prompt_weight = nn.Parameter(torch.randn(in_feats) * 0.02)
        self.prompt_bias = nn.Parameter(torch.zeros(in_feats))

    def forward(self, feat, centrality_scores=None):
        if centrality_scores is None:
            return feat
        c = centrality_scores.unsqueeze(-1) if centrality_scores.dim() == 1 else centrality_scores
        boundary_gate = torch.pow(torch.clamp(1.0 - c, 0.0, 1.0), self.gamma)
        prompt = boundary_gate * (feat * self.prompt_weight + self.prompt_bias)
        return feat + self.alpha * prompt


# ---------------------------------------------------------------------------
# Universal SOTA Dataset Loading (5 Datasets)
# ---------------------------------------------------------------------------
def load_unified_dataset(dataset_name, dataset_dir="./dataset"):
    print(f"\n[Dataset Loader] Loading dataset: {dataset_name} from '{dataset_dir}' ...")
    t0 = time.time()

    if dataset_name == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = data[0]
        num_classes = data.num_classes
        is_multilabel = False
        print("[Dataset Loader] Augmenting ogbn-arxiv to bidirected citation graph...")
        g = dgl.to_bidirected(g, copy_ndata=True)

    elif dataset_name == "reddit":
        data = RedditDataset(raw_dir=dataset_dir)
        g = data[0]
        num_classes = data.num_classes
        is_multilabel = False
        print("[Dataset Loader] Applying in-place feature standardization (mean=0, std=1) on Reddit...")
        feat = g.ndata["feat"]
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True)
        g.ndata["feat"] = (feat - mean) / (std + 1e-6)

    elif dataset_name == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = data[0]
        num_classes = data.num_classes
        is_multilabel = False

    elif dataset_name == "yelp":
        try:
            ds = YelpDataset(raw_dir=dataset_dir)
        except Exception:
            ds = YelpDataset()
        g = ds[0]
        num_classes = ds.num_classes
        is_multilabel = True
        if g.ndata["label"].dtype != torch.float32:
            g.ndata["label"] = g.ndata["label"].float()

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
        num_classes = int(labels.max().item()) + 1
        is_multilabel = False
        g.ndata["label"] = labels.long()

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Standardize label tensor
    if not is_multilabel and "label" in g.ndata:
        g.ndata["label"] = g.ndata["label"].squeeze().long()

    # Standardize masks
    for m in ["train_mask", "val_mask", "test_mask"]:
        if m in g.ndata:
            g.ndata[m] = g.ndata[m].bool()

    in_feats = g.ndata["feat"].shape[1]
    elapsed = time.time() - t0
    print(f"[Dataset Loader] Loaded in {elapsed:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,} | In-Feats: {in_feats} | Classes: {num_classes} | Multi-Label: {is_multilabel}")
    return g, in_feats, num_classes, is_multilabel


# ---------------------------------------------------------------------------
# Partition Loading & Subgraph Extraction
# ---------------------------------------------------------------------------
def partition_graph(g, cluster_file):
    print(f"\n[Partition Loader] Reading partition assignments from: {cluster_file}")
    if not os.path.exists(cluster_file):
        raise FileNotFoundError(f"Cluster file not found: {cluster_file}")

    with open(cluster_file, "r") as f:
        cluster_ids = np.array([int(line.strip()) for line in f if line.strip() != ""], dtype=np.int64)

    if len(cluster_ids) != g.num_nodes():
        raise ValueError(f"Cluster count ({len(cluster_ids)}) != Graph nodes ({g.num_nodes()})")

    k_clusters = len(np.unique(cluster_ids))
    print(f"[Partition Loader] Extracting {k_clusters} node-induced subgraphs...")
    t0 = time.time()

    subgraphs = []
    for c_id in range(k_clusters):
        c_nodes = np.where(cluster_ids == c_id)[0]
        sg = g.subgraph(torch.from_numpy(c_nodes), relabel_nodes=True)
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)

    print(f"[Partition Loader] Extracted {len(subgraphs)} subgraphs in {time.time() - t0:.2f}s.")
    return subgraphs, cluster_ids


# ---------------------------------------------------------------------------
# Mode A: Subgraph Evaluation (Isolated Subgraphs)
# ---------------------------------------------------------------------------
def evaluate_subgraph_mode(model, prompt_module, subgraphs, device, is_multilabel=False):
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    val_preds, val_targets = [], []
    test_preds, test_targets = [], []

    with torch.no_grad():
        for sg in subgraphs:
            sg = sg.to(device)
            feat = sg.ndata["feat"]
            val_mask = sg.ndata["val_mask"]
            test_mask = sg.ndata["test_mask"]

            if val_mask.sum() == 0 and test_mask.sum() == 0:
                continue

            c_scores = sg.ndata.get("centrality", None)
            feat_aug = prompt_module(feat, c_scores) if prompt_module is not None else feat
            logits = model(sg, feat_aug)

            if val_mask.sum() > 0:
                val_preds.append(logits[val_mask].cpu())
                val_targets.append(sg.ndata["label"][val_mask].cpu())

            if test_mask.sum() > 0:
                test_preds.append(logits[test_mask].cpu())
                test_targets.append(sg.ndata["label"][test_mask].cpu())

    if is_multilabel:
        v_preds = torch.cat(val_preds, dim=0).numpy()
        v_targets = torch.cat(val_targets, dim=0).numpy()
        t_preds = torch.cat(test_preds, dim=0).numpy()
        t_targets = torch.cat(test_targets, dim=0).numpy()

        val_bin = (v_preds > 0.0).astype(int)
        test_bin = (t_preds > 0.0).astype(int)

        val_score = f1_score(v_targets, val_bin, average="micro")
        test_score = f1_score(t_targets, test_bin, average="micro")
        val_acc = (val_bin == v_targets).mean()
        test_acc = (test_bin == t_targets).mean()
        return val_score, test_score, val_acc, test_acc
    else:
        v_p = torch.cat(val_preds, dim=0).argmax(dim=-1)
        v_t = torch.cat(val_targets, dim=0)
        t_p = torch.cat(test_preds, dim=0).argmax(dim=-1)
        t_t = torch.cat(test_targets, dim=0)

        val_acc = (v_p == v_t).float().mean().item() if len(v_t) > 0 else 0.0
        test_acc = (t_p == t_t).float().mean().item() if len(t_t) > 0 else 0.0
        return val_acc, test_acc, val_acc, test_acc


# ---------------------------------------------------------------------------
# Mode B: True Layer-Wise Full-Graph Streaming Inference
# ---------------------------------------------------------------------------
def evaluate_layerwise_streaming(model, prompt_module, g, device, batch_size=4096, is_multilabel=False):
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    with torch.no_grad():
        num_nodes = g.num_nodes()
        feat = g.ndata["feat"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]

        # Apply prompts in chunks
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

        # 1-hop layer-wise streaming sampler
        g_eval = dgl.remove_self_loop(g)
        g_eval = dgl.add_self_loop(g_eval)
        sampler = MultiLayerFullNeighborSampler(1)
        dataloader = DataLoader(
            g_eval,
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
                h = cur_feat[input_nodes].to(device)
                h = model.forward_layer(block, h, l)
                y[output_nodes] = h.cpu()

            cur_feat = y

        logits = cur_feat
        labels = g.ndata["label"]

        if is_multilabel:
            v_logits = logits[val_mask].numpy()
            v_labels = labels[val_mask].numpy()
            t_logits = logits[test_mask].numpy()
            t_labels = labels[test_mask].numpy()

            v_bin = (v_logits > 0.0).astype(int)
            t_bin = (t_logits > 0.0).astype(int)

            val_score = f1_score(v_labels, v_bin, average="micro")
            test_score = f1_score(t_labels, t_bin, average="micro")
            val_acc = (v_bin == v_labels).mean()
            test_acc = (t_bin == t_labels).mean()
            return val_score, test_score, val_acc, test_acc
        else:
            v_pred = logits[val_mask].argmax(dim=-1)
            v_lab = labels[val_mask]
            t_pred = logits[test_mask].argmax(dim=-1)
            t_lab = labels[test_mask]

            val_acc = (v_pred == v_lab).float().mean().item()
            test_acc = (t_pred == t_lab).float().mean().item()
            return val_acc, test_acc, val_acc, test_acc


# ---------------------------------------------------------------------------
# Master Training & Benchmark Pipeline
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified 3-Layer GraphSAGE Benchmark Engine")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["ogbn-arxiv", "reddit", "ogbn-products", "yelp", "igb-small"])
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--cluster_file", type=str, required=True, help="Path to cluster partition file")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print("=" * 80)
    print(f" UNIFIED 3-LAYER GRAPHSAGE BENCHMARK ENGINE")
    print(f" Dataset: {args.dataset.upper()} | Device: {device} | Partition: {args.cluster_file}")
    print(f" Arch: 3-Layer GraphSAGE | Hidden: {args.n_hidden} | LayerNorm + Residual + CosineAnnealingLR")
    print("=" * 80)

    # 1. Load Dataset
    g, in_feats, num_classes, is_multilabel = load_unified_dataset(args.dataset, args.dataset_dir)

    # 2. Extract Subgraphs
    subgraphs, cluster_ids = partition_graph(g, args.cluster_file)
    k_clusters = len(subgraphs)

    # 3. Model & EigenPrompt Setup
    model = UnifiedGraphSAGE(
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=num_classes,
        n_layers=3,
        dropout=args.dropout,
        is_multilabel=is_multilabel
    ).to(device)

    prompt_module = EigenPrompt(in_feats=in_feats, alpha=0.1, gamma=1.5).to(device)

    params = list(model.parameters()) + list(prompt_module.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    metric_name = "Micro-F1" if is_multilabel else "Accuracy"
    loss_fn_name = "BCEWithLogitsLoss" if is_multilabel else "CrossEntropyLoss"

    print(f"\nModel Configuration: 3-Layer GraphSAGE | Parameters: {sum(p.numel() for p in params if p.requires_grad):,}")
    print(f"Loss Function: {loss_fn_name} | Primary Metric: {metric_name}")
    print(f"Clusters (K): {k_clusters} | Training Plan: {args.epochs} Epochs\n")

    print(f"{'Epoch':^7} | {'Train(s)':^8} | {'Loss':^9} | {'Val ' + metric_name[:3]:^9} | {'Subg Test':^10} | {'GPU Mem':^8}")
    print("-" * 65)

    best_val_score = 0.0
    best_test_score = 0.0
    best_epoch = 0
    best_model_state = None
    total_pure_train_time = 0.0
    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start_time = time.time()

        model.train()
        prompt_module.train()
        total_loss = 0.0
        total_train_nodes = 0

        # Shuffled subgraph training loop
        perm = torch.randperm(k_clusters)
        for idx in perm:
            sg = subgraphs[idx].to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            train_mask = sg.ndata["train_mask"]

            if train_mask.sum() == 0:
                continue

            optimizer.zero_grad()
            c_scores = sg.ndata.get("centrality", None)
            feat_aug = prompt_module(feat, c_scores)
            logits = model(sg, feat_aug)

            if is_multilabel:
                loss = F.binary_cross_entropy_with_logits(logits[train_mask], label[train_mask])
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

        # Isolated Subgraph Evaluation
        val_score, test_score, val_acc, test_acc = evaluate_subgraph_mode(
            model, prompt_module, subgraphs, device, is_multilabel=is_multilabel
        )

        if val_score > best_val_score:
            best_val_score = val_score
            best_test_score = test_score
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())

        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"{epoch:^7d} | {epoch_train_time:^8.4f} s | {train_loss:^9.4f} | {val_score*100:^8.2f}% | {test_score*100:^9.2f}% | {peak_gpu_mb:^7.1f}MB")

    # Load best checkpoint for final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"\n[Checkpoint] Loaded best validation checkpoint from Epoch {best_epoch} (Val {metric_name}: {best_val_score*100:.2f}%)")

    # Final Subgraph Evaluation
    final_val_subg, final_test_subg, final_val_acc_s, final_test_acc_s = evaluate_subgraph_mode(
        model, prompt_module, subgraphs, device, is_multilabel=is_multilabel
    )

    # True Layer-Wise Full-Graph Streaming Evaluation
    print("[Streaming Inference] Running True Layer-by-Layer Full-Graph Streaming Inference...")
    t_stream0 = time.time()
    stream_val_score, stream_test_score, stream_val_acc, stream_test_acc = evaluate_layerwise_streaming(
        model, prompt_module, g, device, batch_size=4096, is_multilabel=is_multilabel
    )
    t_stream = time.time() - t_stream0

    # ProDo-GNN Reference baselines
    prodo_time_ref = {"reddit": 312.03, "ogbn-arxiv": 95.37, "ogbn-products": 236.50, "yelp": 195.40, "igb-small": 280.00}.get(args.dataset, 0.0)
    speedup = (prodo_time_ref / total_pure_train_time) if total_pure_train_time > 0 and prodo_time_ref > 0 else 0.0

    print("\n" + "=" * 80)
    print(f" BENCHMARK SUMMARY FOR {args.dataset.upper()} (K={k_clusters})")
    print("=" * 80)
    print(f" Partition File Used              : {args.cluster_file}")
    print(f" Total Clusters (K)               : {k_clusters}")
    print(f" Model Architecture               : 3-Layer GraphSAGE + LayerNorm + Residual Highway")
    print(f" Optimization                     : CosineAnnealingLR (100 Epochs, lr={args.lr}, wd={args.weight_decay})")
    print("--------------------------------------------------------------------------------")
    print(f" Total Pure Training Time (100 Ep): {round(total_pure_train_time, 4)} Seconds")
    print(f" Average Epoch Training Time      : {round(total_pure_train_time / args.epochs, 4)} Seconds")
    print(f" Peak GPU Training Memory         : {round(peak_gpu_mb, 1)} MB")
    print("--------------------------------------------------------------------------------")
    print(f" Best Validation {metric_name:10s}     : {round(best_val_score * 100, 2)}% (Epoch {best_epoch})")
    print(f" Subgraph Test {metric_name:12s}   : {round(final_test_subg * 100, 2)}%")
    print(f" Stream Full-Graph Test {metric_name:5s}   : {round(stream_test_score * 100, 2)}%")
    if is_multilabel:
        print(f" Subgraph Multi-Label Accuracy    : {round(final_test_acc_s * 100, 2)}%")
        print(f" Stream Multi-Label Accuracy      : {round(stream_test_acc * 100, 2)}%")
    print("--------------------------------------------------------------------------------")
    if prodo_time_ref > 0:
        print(f" ProDo-GNN Reference Time         : {prodo_time_ref:.2f} s")
        print(f" Speedup vs. ProDo-GNN            : {speedup:.2f}x FASTER!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
