#!/usr/bin/env python3
"""
Default DGL METIS Graph Partitioner (K=20, Unweighted Baseline)
Standard unweighted METIS partitioning using DGL's native C++ METIS binding.
Generates cluster partitions without edge weights or centrality scores.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Default DGL METIS Graph Partitioning (K=20 Unweighted)")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["ogbn-arxiv", "reddit", "yelp", "igb-small", "ogbn-products"],
                        help="Dataset name")
    parser.add_argument("--dataset_dir", type=str,
                        default="/data/surendra/workspace/dgl_cluster/examples/pytorch/graphsage/dataset",
                        help="Path to dataset directory")
    parser.add_argument("--num_clusters", type=int, default=20,
                        help="Number of partitions (Default: 20)")
    parser.add_argument("--output_dir", type=str, default="./cluster_default_metis",
                        help="Dedicated directory to save partition files")
    return parser.parse_args()


def load_graph(dataset_name, dataset_dir):
    print(f"\n[Loader] Loading dataset: {dataset_name} from '{dataset_dir}' ...")
    t0 = time.time()

    if dataset_name == "ogbn-arxiv":
        ds = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = ds[0]
    elif dataset_name == "reddit":
        ds = RedditDataset(raw_dir=dataset_dir)
        g = ds[0]
    elif dataset_name == "yelp":
        ds = YelpDataset(raw_dir=dataset_dir)
        g = ds[0]
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
    elif dataset_name == "ogbn-products":
        ds = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = ds[0]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Remove self-loops for partitioning (METIS requires simple graph)
    g = dgl.remove_self_loop(g)

    elapsed = time.time() - t0
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    print(f"[Loader] Loaded in {elapsed:.2f}s | Nodes: {num_nodes:,} | Undirected Edges: {num_edges:,}")
    return g


def partition_default_metis(g, num_clusters):
    print(f"\n[METIS Engine] Running default DGL METIS partitioning into K={num_clusters} partitions...")
    t0 = time.time()

    # DGL native METIS partition assignment (unweighted)
    cluster_ids = dgl.metis_partition_assignment(g, k=num_clusters)
    cluster_ids_np = cluster_ids.cpu().numpy()

    metis_time = time.time() - t0
    print(f"[METIS Engine] METIS partitioning finished in {metis_time:.2f} seconds.")

    # Compute edge cuts
    print("[METIS Engine] Evaluating inter-partition edge cuts...")
    src, dst = g.edges()
    src_c = cluster_ids[src.long()]
    dst_c = cluster_ids[dst.long()]
    cut_mask = (src_c != dst_c)
    total_cut_edges = cut_mask.sum().item() // 2  # Bidirectional -> divide by 2
    total_edges = g.num_edges() // 2
    cut_ratio = (total_cut_edges / total_edges) * 100.0 if total_edges > 0 else 0.0

    print(f"[METIS Engine] Total Cut Edges: {total_cut_edges:,} / {total_edges:,} ({cut_ratio:.2f}%)")
    return cluster_ids_np, metis_time, total_cut_edges, cut_ratio


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    total_start = time.time()
    g = load_graph(args.dataset, args.dataset_dir)
    cluster_ids, metis_time, cut_edges, cut_ratio = partition_default_metis(g, args.num_clusters)
    total_time = time.time() - total_start

    # Save to dedicated output directory
    out_file = os.path.join(args.output_dir, f"{args.dataset}_cluster_id.txt")
    print(f"\n[Saver] Saving cluster assignments to: {out_file} ...")
    np.savetxt(out_file, cluster_ids, fmt="%d")

    print("\n" + "=" * 80)
    print(f" DEFAULT DGL METIS SUMMARY: {args.dataset.upper()} (K={args.num_clusters})")
    print("=" * 80)
    print(f" Total Nodes                      : {len(cluster_ids):,}")
    print(f" Partitions (K)                   : {args.num_clusters}")
    print(f" Total Cut Edges                  : {cut_edges:,} ({cut_ratio:.2f}%)")
    print(f" Pure METIS Partitioning Time     : {metis_time:.4f} Seconds")
    print(f" Total Wall-Clock Preprocessing   : {total_time:.4f} Seconds")
    print(f" Partition Output File            : {out_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
