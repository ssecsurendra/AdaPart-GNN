import os
import time
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset

def load_sota_dataset(dataset_name, dataset_dir="./dataset", verbose=True):
    """
    SOTA Graph Dataset Loader with Dataset-Specific Topological & Feature Optimizations.
    
    Optimizations applied:
      - Reddit: Feature Standardization (X <- (X - mean) / std). Essential for ~96% test accuracy.
      - Ogbn-Arxiv: Bidirectional Citation Augmentation (dgl.to_bidirected). Allows 2-way message passing.
      - Ogbn-Products: Clean label flattening and boolean mask validation.
      
    Parameters:
        dataset_name (str): 'ogbn-arxiv', 'reddit', or 'ogbn-products'
        dataset_dir (str): Directory containing dataset files
        verbose (bool): Whether to print dataset statistics
        
    Returns:
        g (DGLGraph): Processed graph
        in_feats (int): Input feature dimension
        num_classes (int): Number of target classes
    """
    t0 = time.time()
    if verbose:
        print(f"\n[Dataset Loader] Loading dataset: {dataset_name} from '{dataset_dir}' ...")

    if dataset_name == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = data[0]
        num_classes = data.num_classes

        # Bidirectional graph augmentation: allows citation flow in both directions
        if verbose:
            print(f"[Dataset Loader] Augmenting ogbn-arxiv to bidirected citation graph...")
        g = dgl.to_bidirected(g, copy_ndata=True)

    elif dataset_name == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = data[0]
        num_classes = data.num_classes

    elif dataset_name == "reddit":
        data = RedditDataset(raw_dir=dataset_dir)
        g = data[0]
        num_classes = data.num_classes

        # Feature Standardization: critical for Reddit ~96% SOTA accuracy
        if verbose:
            print(f"[Dataset Loader] Applying in-place feature standardization (mean=0, std=1) on Reddit...")
        feat = g.ndata["feat"]
        mean = feat.mean(dim=0, keepdim=True)
        std = feat.std(dim=0, keepdim=True)
        g.ndata["feat"] = (feat - mean) / (std + 1e-6)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Standardize label tensor to 1D LongTensor
    if "label" in g.ndata:
        g.ndata["label"] = g.ndata["label"].squeeze().long()

    # Standardize masks to boolean tensors
    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if mask_name in g.ndata:
            g.ndata[mask_name] = g.ndata[mask_name].bool()

    in_feats = g.ndata["feat"].shape[1]
    elapsed = time.time() - t0

    if verbose:
        print(f"[Dataset Loader] Loaded in {elapsed:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,} | In-Feats: {in_feats} | Classes: {num_classes}")

    return g, in_feats, num_classes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="reddit", choices=["ogbn-arxiv", "reddit", "ogbn-products"])
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    args = parser.parse_args()

    g, in_feats, num_classes = load_sota_dataset(args.dataset, args.dataset_dir)
    print(f"Validation summary: Features mean={g.ndata['feat'].mean():.4f}, std={g.ndata['feat'].std():.4f}")

