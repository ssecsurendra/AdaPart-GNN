import numpy as np
import sys
import dgl
import time
import torch as th
import pymetis
import os
import cupy as cp
import psutil
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch
import math
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# ------------------------------------- Custom CuPy Kernel ----------------#
similarity_kernel = cp.RawKernel(r'''
extern "C" __global__
void similarity_kernel(
    const float* d_prime, const int* indptr, const int* indices, 
    float* edge_weights, float lambda_val
) {
    int src = blockIdx.x;
    int edge_start = indptr[src];
    int edge_end = indptr[src + 1];

    for (int e = edge_start + threadIdx.x; e < edge_end; e += blockDim.x) {
        int dst = indices[e];

        // Jaccard similarity between src and dst
        int count_intersection = 0;
        int i = indptr[src], j = indptr[dst];

        while (i < indptr[src + 1] && j < indptr[dst + 1]) {
            int a = indices[i];
            int b = indices[j];
            if (a == b) {
                count_intersection++;
                i++; j++;
            } else if (a < b) {
                i++;
            } else {
                j++;
            }
        }
        int count_union = (indptr[src + 1] - indptr[src]) + (indptr[dst + 1] - indptr[dst]) - count_intersection;
        float J_uv = count_union > 0 ? (float)count_intersection / count_union : 0.0f;

        // Geometric mean of normalized degrees
        float G_uv = sqrtf(d_prime[src] * d_prime[dst]);

        // Composite similarity
        float S_uv = powf(J_uv, 0.7f) * powf(G_uv, 0.3f);

        // Final edge weight
        edge_weights[e] = 1.0f + lambda_val * S_uv;
    }
}
''', 'similarity_kernel')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--num_clusters", type=int, default=10, help="Number of clusters")
    parser.add_argument("--lambda_val", type=float, default=1.0, help="Lambda parameter in weighting scheme")
    parser.add_argument("--scale_factor", type=float, default=1000.0, help="Multiplier for METIS integer edge weights")
    args = parser.parse_args()

    total_start_time = time.time()

    print(f"==================================================")
    print(f"Dataset: {args.dataset}, Clusters: {args.num_clusters}, Lambda: {args.lambda_val}")
    print(f"==================================================")

    # Load dataset
    try:
        if args.dataset == "cora":
            data = CoraGraphDataset()
        elif args.dataset == "citeseer":
            data = CiteseerGraphDataset()
        elif args.dataset == "pubmed":
            data = PubmedGraphDataset()
        elif args.dataset == "wisconsin":
            data = WisconsinDataset()
        elif args.dataset == "flickr":
            data = FlickrDataset()
        elif args.dataset == "reddit":
            data = RedditDataset()
        elif args.dataset == "yelp":
            data = YelpDataset()
        elif args.dataset == "ogbn-products":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
        elif args.dataset == "ogbn-arxiv":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv"))
        elif args.dataset == "ogbn-papers":
            data = AsNodePredDataset(DglNodePropPredDataset("ogbn-papers100M"))
        elif args.dataset == "amazon_products":
            load_path = './dataset/amazon_products.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-tiny":
            load_path = './dataset/igb_tiny.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-small":
            load_path = './dataset/igb_small.dgl'
            data, _ = dgl.load_graphs(load_path)
        elif args.dataset == "igb-medium":
            load_path = './dataset/igb_medium.dgl'
            data, _ = dgl.load_graphs(load_path)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    G = data[0]

    # Convert to bidirected & remove self-loops for symmetric METIS format
    g_pre_time_start = time.time()
    G = dgl.to_bidirected(G)
    G = dgl.remove_self_loop(G)
    g_pre_time_end = time.time()
    g_pre_time = g_pre_time_end - g_pre_time_start
    print(f"preprocessing Time  : {round(g_pre_time, 4)} Seconds")
    num_nodes = G.num_nodes()
    num_edges = G.num_edges()
    print(f"Graph nodes: {num_nodes}, edges: {num_edges}")

    # Extract CSR representation
    row_ptr = np.array(G.adj_tensors('csr')[0], dtype=np.int32)
    col_idx = np.array(G.adj_tensors('csr')[1], dtype=np.int32)

    # Step 1: Weight Calculation Phase
    weight_calc_start = time.time()
    degrees = row_ptr[1:] - row_ptr[:-1]
    d_max = np.max(degrees)
    d_prime = np.log1p(degrees) / np.log1p(d_max)
    d_prime_cp = cp.asarray(d_prime, dtype=cp.float32)

    row_ptr_cp = cp.asarray(row_ptr, dtype=cp.int32)
    col_idx_cp = cp.asarray(col_idx, dtype=cp.int32)
    edge_weights_cp = cp.zeros(num_edges, dtype=cp.float32)

    # Launch GPU Kernel for custom edge weighting
    block_size = 128
    grid_size = num_nodes
    similarity_kernel(
        (grid_size,), (block_size,),
        (d_prime_cp, row_ptr_cp, col_idx_cp, edge_weights_cp, np.float32(args.lambda_val))
    )
    cp.cuda.Device(0).synchronize()

    # Scale floating edge weights to positive integers for METIS
    float_weights = cp.asnumpy(edge_weights_cp)
    int_edge_weights = np.rint(float_weights * args.scale_factor).astype(np.int32)
    int_edge_weights = np.maximum(int_edge_weights, 1)  # Ensure minimum positive weight of 1
    print("First 20 edge weights:")
    for i, w in enumerate(int_edge_weights[:20]):
        print(f"edge weight[{i}] = {w}")
    weight_calc_end = time.time()
    weight_calc_time = weight_calc_end - weight_calc_start

    # Step 2: METIS Partitioning Phase (PyMETIS with explicit edge weights)
    print(f"Starting PyMETIS Partitioning into {args.num_clusters} parts with explicit edge weights...")
    partition_start = time.time()
    cuts, node_parts = pymetis.part_graph(
        nparts=args.num_clusters,
        xadj=row_ptr,
        adjncy=col_idx,
        eweights=int_edge_weights
    )
    partition_end = time.time()
    partition_time = partition_end - partition_start
    print(f"Partitioning Done! Weighted Edge Cut (METIS objective): {cuts}")

    # Step 3: Calculate unweighted cut edges count
    src = np.repeat(np.arange(num_nodes), row_ptr[1:] - row_ptr[:-1])
    dst = col_idx
    cut_mask = np.array(node_parts)[src] != np.array(node_parts)[dst]
    unweighted_cut_edges_directed = int(np.sum(cut_mask))
    unweighted_cut_edges_undirected = unweighted_cut_edges_directed // 2
    total_undirected_edges = num_edges // 2
    cut_percentage = round(100.0 * unweighted_cut_edges_undirected / total_undirected_edges, 2)

    print(f"Unweighted Cut Edges Count (Undirected): {unweighted_cut_edges_undirected} / {total_undirected_edges} ({cut_percentage}%)")

    # Step 4: Save cluster IDs into cluster/ folder inside graphsage
    output_dir = './cluster'
    os.makedirs(output_dir, exist_ok=True)
    cluster_file_path = os.path.join(output_dir, f"{args.dataset}_cluster_id.txt")
    
    with open(cluster_file_path, "w") as f:
        for pid in node_parts:
            f.write(f"{pid}\n")
    print(f"Saved cluster partition IDs to: {cluster_file_path}")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Output Execution Time Summary
    print("\n==================================================")
    print(f"Weight Computation Time : {round(weight_calc_time, 4)} Seconds")
    print(f"Partitioning Time       : {round(partition_time, 4)} Seconds")
    print(f"Overall Time            : {round(total_time, 4)} Seconds")
    print("==================================================")
