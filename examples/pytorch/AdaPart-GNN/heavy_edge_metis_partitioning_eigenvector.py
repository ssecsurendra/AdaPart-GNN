import numpy as np
import sys
import dgl
import time
import torch as th
import pymetis
import os
import cupy as cp
from cupy.sparse import csr_matrix
import psutil
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch
import math
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset, WisconsinDataset, FlickrDataset, RedditDataset, YelpDataset

# ------------------ Cosine Dissimilarity Kernel for Weighted Eigenvector Centrality ------------------ #
cosine_dissimilarity_kernel = cp.RawKernel(r'''
extern "C" __global__
void calculate_cosine_similarity(
    const float* features, 
    const int* indptr, 
    const int* indices, 
    float* edge_weights, 
    int feature_dim,
    int num_nodes
) {
    int src = blockIdx.x;
    if (src >= num_nodes) return;

    int edge_start = indptr[src];
    int edge_end = indptr[src + 1];

    for (int e = edge_start + threadIdx.x; e < edge_end; e += blockDim.x) {
        int dst = indices[e];

        float dot = 0.0f;
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        const float* feature_a = features + (size_t)src * feature_dim;
        const float* feature_b = features + (size_t)dst * feature_dim;

        for (int i = 0; i < feature_dim; ++i) {
            float a = feature_a[i];
            float b = feature_b[i];
            dot += a * b;
            norm_a += a * a;
            norm_b += b * b;
        }

        float raw_cosine = 0.0f;
        if (norm_a > 0.0f && norm_b > 0.0f) {
            raw_cosine = dot / (sqrtf(norm_a) * sqrtf(norm_b));
        }
        
        // Dissimilarity weight clamped to [0, 1]
        edge_weights[e] = 1.0f - fmaxf(raw_cosine, 0.0f);
    }
}
''', 'calculate_cosine_similarity')

# ------------------ Custom CuPy Kernel: Jaccard Similarity & Eigenvector Centrality Weighting ------------------ #
similarity_kernel = cp.RawKernel(r'''
extern "C" __global__
void similarity_kernel(
    const float* c_prime, const int* indptr, const int* indices, 
    const float* e_prime, const int* indptr, const int* indices, 
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

        // Geometric mean of normalized eigenvector centralities
        float G_uv = sqrtf(c_prime[src] * c_prime[dst]);
        // Geometric mean of normalized eigenvector centralities: G_uv = sqrt(e'_u * e'_v)
        float G_uv = sqrtf(e_prime[src] * e_prime[dst]);

        // Composite similarity: S_uv = (J_uv)^0.7 * (G_uv)^0.3
        float S_uv = powf(J_uv, 0.7f) * powf(G_uv, 0.3f);

        // Final edge weight
        edge_weights[e] = 1.0f + lambda_val * S_uv;
    }
}
''', 'similarity_kernel')


def compute_weighted_eigenvector_centrality(row_ptr_cp, col_idx_cp, features_cp, num_nodes, num_edges, feature_dim, max_iter=100, tol=1e-6):
    """
    Computes weighted eigenvector centrality directly on GPU via power iteration,
    using cosine dissimilarity edge weights as defined in the reference code.
    """
    edge_weights_cp = cp.empty(num_edges, dtype=cp.float32)

    if features_cp is not None and feature_dim > 0:
        threads_per_block = 128
        blocks_per_grid = num_nodes
        cosine_dissimilarity_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (features_cp, row_ptr_cp, col_idx_cp, edge_weights_cp, feature_dim, num_nodes)
        )
        cp.cuda.Device(0).synchronize()
    else:
        edge_weights_cp.fill(1.0)

    # Construct sparse CSR adjacency matrix on GPU
    adj_matrix = csr_matrix((edge_weights_cp, col_idx_cp, row_ptr_cp), shape=(num_nodes, num_nodes))

    # Power Iteration
    x = cp.ones(num_nodes, dtype=cp.float32)
    for i in range(max_iter):
        x_old = cp.copy(x)
        x = adj_matrix.dot(x)
        norm = cp.linalg.norm(x)
        if norm == 0:
            return cp.asnumpy(x)
        x /= norm
        if cp.linalg.norm(x - x_old) < tol:
            print(f"Eigenvector power iteration converged after {i+1} iterations.")
            break
    else:
        print(f"Eigenvector power iteration completed {max_iter} iterations.")
    
    return cp.asnumpy(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heavy Edge METIS Partitioning with On-the-Fly Eigenvector Centrality")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv")
    parser.add_argument("--num_clusters", type=int, default=20, help="Number of clusters")
    parser.add_argument("--lambda_val", type=float, default=1.0, help="Lambda parameter in weighting scheme")
    parser.add_argument("--scale_factor", type=float, default=1000.0, help="Multiplier for METIS integer edge weights")
    parser.add_argument("--output_dir", type=str, default="./cluster_eigenvector", help="Dedicated folder to store cluster partition IDs")
    parser.add_argument("--max_iter", type=int, default=100, help="Max iterations for eigenvector power iteration")
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance for power iteration")
    args = parser.parse_args()

    total_start_time = time.time()

    print(f"==================================================")
    print(f"Heavy Edge METIS with Eigenvector Centrality (On-The-Fly Computation)")
    print(f"Dataset: {args.dataset} | Clusters: {args.num_clusters} | Lambda: {args.lambda_val}")
    print(f"Output Directory: {args.output_dir}")
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

    # ------------------ Ultra-Fast O(1) Symmetry Check ------------------ #
    sym_check_start = time.time()
    in_deg = G.in_degrees()
    out_deg = G.out_degrees()
    if not torch.equal(in_deg, out_deg):
        is_undirected = False
    else:
        sample_size = min(10000, G.num_edges())
        sample_idx = torch.randint(0, G.num_edges(), (sample_size,))
        src_sample, dst_sample = G.find_edges(sample_idx)
        is_undirected = G.has_edges_between(dst_sample, src_sample).all().item()
    sym_check_end = time.time()
    print(f"Symmetry check time: {round(sym_check_end - sym_check_start, 6)} Seconds")

    if not is_undirected:
        print("Graph is directed. Converting to bidirected for METIS...")
        G = dgl.to_bidirected(G)
    else:
        print("Graph is already undirected. Skipping to_bidirected() overhead!")

    # Remove self-loops for symmetric METIS format
    G = dgl.remove_self_loop(G)

    num_nodes = G.num_nodes()
    num_edges = G.num_edges()
    print(f"Graph nodes: {num_nodes}, edges: {num_edges}")

    # Extract CSR representation
    row_ptr = np.array(G.adj_tensors('csr')[0], dtype=np.int32)
    col_idx = np.array(G.adj_tensors('csr')[1], dtype=np.int32)

    row_ptr_cp = cp.asarray(row_ptr, dtype=cp.int32)
    col_idx_cp = cp.asarray(col_idx, dtype=cp.int32)

    # ---------------- Phase 1: Eigenvector Centrality Computation ---------------- #
    print("\n--- Step 1: Computing Eigenvector Centrality (Power Iteration on GPU) ---")
    centrality_start_time = time.time()

    features_cp = cp.asarray(G.ndata['feat'], dtype=cp.float32) if 'feat' in G.ndata else None
    feat_dim = features_cp.shape[1] if features_cp is not None and features_cp.ndim > 1 else 0

    eigen_centrality = compute_weighted_eigenvector_centrality(
        row_ptr_cp=row_ptr_cp,
        col_idx_cp=col_idx_cp,
        features_cp=features_cp,
        num_nodes=num_nodes,
        num_edges=num_edges,
        feature_dim=feat_dim,
        max_iter=args.max_iter,
        tol=args.tol
    )
    cp.cuda.Device(0).synchronize()
    centrality_end_time = time.time()
    centrality_time = centrality_end_time - centrality_start_time
    print(f"Eigenvector Centrality Computation Time: {round(centrality_time, 4)} Seconds")

    # Normalize eigenvector centrality according to: e'_v = EC_v / EC_max
    EC_max = float(np.max(eigen_centrality))
    if EC_max > 0:
        e_prime = eigen_centrality / EC_max
    else:
        e_prime = np.ones_like(eigen_centrality, dtype=np.float32)

    e_prime_cp = cp.asarray(e_prime, dtype=cp.float32)

    # ---------------- Phase 2: Jaccard & Edge Weight Computation ---------------- #
    print("\n--- Step 2: Computing Jaccard Similarity & Edge Weights (GPU Kernel) ---")
    jaccard_start_time = time.time()

    edge_weights_cp = cp.zeros(num_edges, dtype=cp.float32)

    # Launch GPU Kernel for Jaccard and composite edge weighting
    block_size = 128
    grid_size = num_nodes
    similarity_kernel(
        (grid_size,), (block_size,),
        (e_prime_cp, row_ptr_cp, col_idx_cp, edge_weights_cp, np.float32(args.lambda_val))
    )
    cp.cuda.Device(0).synchronize()

    # Scale floating edge weights to positive integers for METIS
    float_weights = cp.asnumpy(edge_weights_cp)
    int_edge_weights = np.rint(float_weights * args.scale_factor).astype(np.int32)
    int_edge_weights = np.maximum(int_edge_weights, 1)  # Ensure minimum positive weight of 1

    jaccard_end_time = time.time()
    jaccard_time = jaccard_end_time - jaccard_start_time
    print(f"Jaccard Computation Time: {round(jaccard_time, 4)} Seconds")

    # ---------------- Phase 3: PyMETIS Partitioning ---------------- #
    print(f"\n--- Step 3: PyMETIS Partitioning ({args.num_clusters} Clusters) ---")
    partition_start_time = time.time()
    cuts, node_parts = pymetis.part_graph(
        nparts=args.num_clusters,
        xadj=row_ptr,
        adjncy=col_idx,
        eweights=int_edge_weights
    )
    partition_end_time = time.time()
    partition_time = partition_end_time - partition_start_time
    print(f"Partitioning Done! Weighted Edge Cut (METIS objective): {cuts}")

    # ---------------- Phase 4: Unweighted Cut Edges Count ---------------- #
    src = np.repeat(np.arange(num_nodes), row_ptr[1:] - row_ptr[:-1])
    dst = col_idx
    cut_mask = np.array(node_parts)[src] != np.array(node_parts)[dst]
    unweighted_cut_edges_directed = int(np.sum(cut_mask))
    unweighted_cut_edges_undirected = unweighted_cut_edges_directed // 2
    total_undirected_edges = num_edges // 2
    cut_percentage = round(100.0 * unweighted_cut_edges_undirected / total_undirected_edges, 2)

    print(f"Unweighted Cut Edges Count (Undirected): {unweighted_cut_edges_undirected} / {total_undirected_edges} ({cut_percentage}%)")

    # ---------------- Phase 5: Save Cluster IDs into Dedicated Folder ---------------- #
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cluster_file_path = os.path.join(output_dir, f"{args.dataset}_cluster_id.txt")
    
    with open(cluster_file_path, "w") as f:
        for pid in node_parts:
            f.write(f"{pid}\n")
    print(f"Saved cluster partition IDs to: {cluster_file_path}")

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # ---------------- Final Execution Time Summary ---------------- #
    print("\n==================================================")
    print("           EXECUTION TIME & BENCHMARK SUMMARY     ")
    print("==================================================")
    print(f"Centrality Computation Time     : {round(centrality_time, 4)} Seconds")
    print(f"Jaccard Computation Time        : {round(jaccard_time, 4)} Seconds")
    print(f"Partitioning Time (PyMETIS)     : {round(partition_time, 4)} Seconds")
    print(f"Overall Execution Time          : {round(total_time, 4)} Seconds")
    print("==================================================")
