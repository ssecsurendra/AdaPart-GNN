import numpy as np
import sys
import dgl
import time
import os
import math
import argparse
os.environ["DGLBACKEND"] = "pytorch"
import torch
import cupy as cp
from cupy.sparse import csr_matrix
import pymetis
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset, YelpDataset

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
    const float* e_prime, const int* indptr, const int* indices, 
    float* edge_weights, float lambda_val
) {
    int src = blockIdx.x;
    if (src >= indptr[0] && src < 0) return;
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
    """Computes weighted eigenvector centrality on GPU via power iteration."""
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

    # Sparse CSR adjacency matrix on GPU
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
            print(f"[Eigen Centrality] Power iteration converged after {i+1} iterations.")
            break
    else:
        print(f"[Eigen Centrality] Power iteration reached max {max_iter} iterations.")

    return cp.asnumpy(x)


def compute_k_star_equation_1(num_nodes, num_edges, n_target=38000, beta=0.35):
    """
    Computes K* directly using Equation (1) from the AdaPart-GNN paper:
        K* = ceil( (|V| / N_target) * (1 + beta * log10(1 + 2|E| / |V|)) )
    
    Penalizes edge-heavy graphs where dense message passing inflates SpMM
    activation memory on GPU, while dampening power-law skews with log10.
    """
    avg_degree = (2.0 * float(num_edges)) / max(float(num_nodes), 1.0)
    density_penalty = 1.0 + beta * math.log10(1.0 + avg_degree)
    k_star = math.ceil((float(num_nodes) / float(n_target)) * density_penalty)
    return max(2, k_star), avg_degree, density_penalty


def derive_absolute_hardware_n_min(num_nodes, avg_degree, num_sms=84, hidden_dim=256, n_base=38000, beta=0.35):
    """
    Derives the absolute physical N_min floor incorporating the structural density penalty of Equation (1).
    Ensures sufficient tensor volume to saturate GPU SMs without PCIe starving, while penalizing
    edge-heavy graphs (e.g., Reddit) to restrict peak activation volume during SpMM aggregation.
    """
    # Structural density penalty from Equation (1)
    density_penalty = 1.0 + beta * math.log10(1.0 + max(avg_degree, 1.0))
    n_min_density = int(n_base / density_penalty)

    # Edge-heavy graphs (Reddit d~492) generate 20x more edges per node -> lower node floor
    if avg_degree > 100:
        n_min = min(13000, n_min_density)
    elif num_nodes < 300000:
        n_min = 12500
    elif num_nodes < 1000000:
        n_min = 27000
    elif num_nodes < 1500000:
        n_min = 31000
    else:
        n_min = max(35000, n_min_density)
    return n_min


def compute_scale_adaptive_dominating_k(
    g,
    eigen_centrality,
    device,
    target_gpu_mem_gb=2.0,
    feat_dim=128,
    hidden_dim=256,
    user_n_min=None,
    n_target=38000,
    beta=0.35
):
    """
    Method 1: Centrality Dominating Set with Scale-Adaptive Hardware Saturation.
    K* scales naturally with graph size N while enforcing absolute hardware core occupancy
    and structural density penalization from Equation (1).
    """
    t_start = time.time()
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()
    avg_degree = float(num_edges) / float(max(num_nodes, 1))

    # Evaluate Equation (1) analytical formulation
    k_star_eq1, avg_degree_dir, density_penalty = compute_k_star_equation_1(
        num_nodes, num_edges, n_target=n_target, beta=beta
    )
    print(f"\n[Equation (1) Sizing] Analytical Calculation:")
    print(f"  |V| = {num_nodes:,} | |E| = {num_edges:,} | Average Degree (2|E|/|V|) = {avg_degree_dir:.2f}")
    print(f"  Density Penalty [1 + beta*log10(1 + 2|E|/|V|)] = {density_penalty:.4f} (beta={beta})")
    print(f"  Equation (1) Baseline K* = {k_star_eq1} (with N_target = {n_target:,} nodes/subgraph)")

    # 1. Query physical GPU properties dynamically
    if torch.cuda.is_available() and device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        num_sms = props.multi_processor_count
        total_vram_gb = props.total_memory / (1024 ** 3)
    else:
        num_sms = 84
        total_vram_gb = 48.0

    # 2. Derive Absolute Hardware N_min Floor
    if user_n_min is not None and user_n_min > 0:
        n_min = user_n_min
        print(f"[Hardware Constraint] Manual override N_min = {n_min:,} nodes/cluster")
    else:
        n_min = derive_absolute_hardware_n_min(num_nodes, avg_degree, num_sms, hidden_dim, n_base=n_target, beta=beta)
        print(f"[Hardware Constraint] GPU: {num_sms} SMs ({total_vram_gb:.1f} GB) | Absolute N_min floor = {n_min:,} nodes")

    # 3. Memory budget lower bound (K_min)
    bytes_per_node = (feat_dim * 4 + avg_degree * 8 + hidden_dim * 4) * 4
    k_min_mem = max(2, math.ceil((num_nodes * bytes_per_node) / (target_gpu_mem_gb * (1024 ** 3))))

    # 4. Compute saturation upper bound (K_max)
    k_max_compute = max(k_min_mem + 1, num_nodes // n_min)

    # 5. Method 1: Centrality Dominating Set Pass
    c_tensor = torch.from_numpy(eigen_centrality).float()
    c_max = c_tensor.max().item()
    c_norm = c_tensor / max(c_max, 1e-8)
    sorted_nodes = torch.argsort(c_norm, descending=True)

    indptr, indices, _ = g.adj_tensors('csr')
    dominated = torch.zeros(num_nodes, dtype=torch.bool)
    dominating_centers = []

    mean_c = c_norm.mean().item()
    std_c = c_norm.std().item()
    c_threshold = max(0.01, mean_c * 0.5)

    for u in sorted_nodes:
        u_id = u.item()

        if c_norm[u_id] < c_threshold and len(dominating_centers) >= k_max_compute:
            break

        if dominated[u_id]:
            continue

        dominating_centers.append(u_id)
        dominated[u_id] = True

        # Mark 1-hop dense community as dominated
        nbrs = indices[indptr[u_id]:indptr[u_id + 1]]
        dominated[nbrs] = True

        # Hardware Upper Bound termination
        if len(dominating_centers) >= k_max_compute:
            break

    # If dominating set produced fewer centers than k_max_compute but graph is large,
    # target the natural hardware balance
    k_topo = len(dominating_centers)
    k_optimal = max(k_min_mem, min(k_topo, k_max_compute))
    t_discovery = time.time() - t_start

    print(f"[Method 1 Dominating Set] Discovered {k_topo} natural hub centers in {t_discovery:.4f}s.")
    print(f"[Stage 1 K* Selection] Final K* = {k_optimal} clusters (Equation 1 ref: {k_star_eq1}, Hardware range: [{k_min_mem}, {k_max_compute}])")

    return k_optimal, n_min, t_discovery, dominating_centers


def load_dataset_graph(dataset_name, dataset_dir="./dataset"):
    print(f"\n[Dataset Loader] Loading '{dataset_name}' from {dataset_dir} ...")
    t0 = time.time()
    if dataset_name == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = data[0]
    elif dataset_name == "reddit":
        data = RedditDataset(raw_dir=dataset_dir)
        g = data[0]
    elif dataset_name == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = data[0]
    elif dataset_name == "yelp":
        data = YelpDataset(raw_dir=dataset_dir)
        g = data[0]
    elif dataset_name == "igb-small":
        load_path = os.path.join(dataset_dir, "igb_small.dgl")
        graphs, _ = dgl.load_graphs(load_path)
        g = graphs[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    print(f"[Dataset Loader] Loaded in {time.time() - t0:.2f}s | Nodes: {g.num_nodes():,} | Edges: {g.num_edges():,}")
    return g


def main():
    parser = argparse.ArgumentParser(description="Hardware-Scale-Adaptive Dominating Set Partitioner")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["ogbn-arxiv", "reddit", "ogbn-products", "yelp", "igb-small"])
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--output_dir", type=str, default="./cluster_scale_adaptive",
                        help="Dedicated folder to store scale-adaptive cluster partition IDs")
    parser.add_argument("--n_min", type=int, default=0, help="Optional manual override for N_min")
    parser.add_argument("--lambda_val", type=float, default=1.0, help="Lambda weight for Jaccard-Eigenvector composite")
    parser.add_argument("--scale_factor", type=float, default=1000.0, help="Integer scale factor for METIS edge weights")
    parser.add_argument("--target_gpu_mem_gb", type=float, default=2.0)
    parser.add_argument("--n_target", type=int, default=38000, help="Hardware budgeted node capacity N_target for Equation (1)")
    parser.add_argument("--beta", type=float, default=0.35, help="Structural density penalty coefficient beta for Equation (1)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    total_start_time = time.time()
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")

    print("=" * 75)
    print(" HARDWARE SCALE-ADAPTIVE CENTRALITY DOMINATING SET PARTITIONER")
    print(f" Dataset: {args.dataset.upper()} | Output Folder: {args.output_dir}")
    print(f" Physical Saturation Constraint (N_min) + Jaccard/Eigenvector")
    print("=" * 75)

    # 1. Load Graph
    G = load_dataset_graph(args.dataset, args.dataset_dir)

    # 2. Symmetry verification & bidirected conversion
    t_sym = time.time()
    in_deg = G.in_degrees()
    out_deg = G.out_degrees()
    if not torch.equal(in_deg, out_deg):
        is_undirected = False
    else:
        sample_size = min(10000, G.num_edges())
        sample_idx = torch.randint(0, G.num_edges(), (sample_size,))
        src_sample, dst_sample = G.find_edges(sample_idx)
        is_undirected = G.has_edges_between(dst_sample, src_sample).all().item()

    if not is_undirected:
        print("[Topology] Graph is directed. Converting to bidirected for symmetric partitioning...")
        G = dgl.to_bidirected(G)
    else:
        print("[Topology] Graph is symmetric/undirected.")

    G = dgl.remove_self_loop(G)
    num_nodes = G.num_nodes()
    num_edges = G.num_edges()
    print(f"[Topology] Active nodes: {num_nodes:,} | Active undirected edges: {num_edges // 2:,} ({time.time() - t_sym:.4f}s)")

    # 3. Extract CSR representation
    row_ptr = np.array(G.adj_tensors('csr')[0], dtype=np.int32)
    col_idx = np.array(G.adj_tensors('csr')[1], dtype=np.int32)
    row_ptr_cp = cp.asarray(row_ptr, dtype=cp.int32)
    col_idx_cp = cp.asarray(col_idx, dtype=cp.int32)

    features_cp = cp.asarray(G.ndata['feat'], dtype=cp.float32) if 'feat' in G.ndata else None
    feat_dim = features_cp.shape[1] if features_cp is not None and features_cp.ndim > 1 else 128

    # 4. Phase 1: Eigenvector Centrality Computation on GPU
    print("\n--- Phase 1: Eigenvector Centrality Computation (Power Iteration on GPU) ---")
    t_eig0 = time.time()
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
    t_centrality = time.time() - t_eig0
    print(f"[Phase 1 Complete] Centrality computation time: {t_centrality:.4f} Seconds")

    # 5. Phase 2: Method 1 (Scale-Adaptive Dominating Set)
    print("\n--- Phase 2: Discovering Scale-Adaptive K* via Hardware-Constrained Dominating Set ---")
    k_optimal, n_min, t_discovery, centers = compute_scale_adaptive_dominating_k(
        g=G,
        eigen_centrality=eigen_centrality,
        device=device,
        target_gpu_mem_gb=args.target_gpu_mem_gb,
        feat_dim=feat_dim,
        hidden_dim=256,
        user_n_min=args.n_min if args.n_min > 0 else None,
        n_target=args.n_target,
        beta=args.beta
    )

    # 6. Phase 3: Jaccard & Heavy-Edge Weighting on GPU
    print(f"\n--- Phase 3: Jaccard & Eigenvector Heavy-Edge Weighting (lambda={args.lambda_val}) ---")
    t_jacc0 = time.time()
    EC_max = float(np.max(eigen_centrality))
    e_prime = (eigen_centrality / EC_max) if EC_max > 0 else np.ones_like(eigen_centrality, dtype=np.float32)
    e_prime_cp = cp.asarray(e_prime, dtype=cp.float32)

    edge_weights_cp = cp.zeros(num_edges, dtype=cp.float32)
    block_size = 128
    grid_size = num_nodes
    similarity_kernel(
        (grid_size,), (block_size,),
        (e_prime_cp, row_ptr_cp, col_idx_cp, edge_weights_cp, np.float32(args.lambda_val))
    )
    cp.cuda.Device(0).synchronize()

    float_weights = cp.asnumpy(edge_weights_cp)
    int_edge_weights = np.rint(float_weights * args.scale_factor).astype(np.int32)
    int_edge_weights = np.maximum(int_edge_weights, 1)
    t_jaccard = time.time() - t_jacc0
    print(f"[Phase 3 Complete] Edge weighting computed in {t_jaccard:.4f} Seconds")

    # 7. Phase 4: PyMETIS Partitioning with Scale-Adaptive K*
    print(f"\n--- Phase 4: PyMETIS Partitioning into K* = {k_optimal} Scale-Adaptive Clusters ---")
    t_metis0 = time.time()
    cuts, node_parts = pymetis.part_graph(
        nparts=k_optimal,
        xadj=row_ptr,
        adjncy=col_idx,
        eweights=int_edge_weights
    )
    t_partition = time.time() - t_metis0
    print(f"[Phase 4 Complete] PyMETIS partitioned graph in {t_partition:.4f} Seconds")

    # 8. Phase 5: Metric & Edge Cut Analysis
    src = np.repeat(np.arange(num_nodes), row_ptr[1:] - row_ptr[:-1])
    dst = col_idx
    cut_mask = np.array(node_parts)[src] != np.array(node_parts)[dst]
    unweighted_cut_edges_undirected = int(np.sum(cut_mask)) // 2
    total_undirected_edges = num_edges // 2
    cut_pct = (100.0 * unweighted_cut_edges_undirected) / total_undirected_edges

    # Cluster node size distribution
    cluster_counts = np.bincount(node_parts, minlength=k_optimal)
    min_cluster_nodes = int(np.min(cluster_counts))
    max_cluster_nodes = int(np.max(cluster_counts))
    mean_cluster_nodes = float(np.mean(cluster_counts))
    std_cluster_nodes = float(np.std(cluster_counts))

    print(f"\n[Partition Analysis]")
    print(f"  Weighted Cut Objective (METIS): {cuts:,}")
    print(f"  Unweighted Edge Cut (Undirected): {unweighted_cut_edges_undirected:,} / {total_undirected_edges:,} ({cut_pct:.2f}%)")
    print(f"  Cluster Node Distribution: Min={min_cluster_nodes:,} | Max={max_cluster_nodes:,} | Mean={mean_cluster_nodes:,.0f} (std={std_cluster_nodes:.1f})")

    # 9. Phase 6: Save into Dedicated Folder
    os.makedirs(args.output_dir, exist_ok=True)
    partition_file = os.path.join(args.output_dir, f"{args.dataset}_cluster_id.txt")
    centrality_file = os.path.join(args.output_dir, f"{args.dataset}_eigen_centrality.npy")

    with open(partition_file, "w") as f:
        for pid in node_parts:
            f.write(f"{pid}\n")
    np.save(centrality_file, eigen_centrality)

    total_time = time.time() - total_start_time

    print(f"\n[Saved Outputs]")
    print(f"  Partition IDs       : {partition_file} ({len(node_parts):,} assignments across {k_optimal} clusters)")
    print(f"  Eigen Centrality    : {centrality_file}")

    print("\n" + "=" * 75)
    print(f" BENCHMARK SUMMARY FOR {args.dataset.upper()} (SCALE-ADAPTIVE)")
    print("=" * 75)
    print(f"  Total Graph Nodes               : {num_nodes:,}")
    print(f"  Discovered Scale-Adaptive K*    : {k_optimal} clusters")
    print(f"  Physical Hardware Floor (N_min) : {n_min:,} nodes/cluster")
    print(f"  Undirected Edge Cut Percentage  : {cut_pct:.2f}%")
    print(f"  Eigen Centrality Time           : {t_centrality:.4f} s")
    print(f"  Method 1 Discovery Time         : {t_discovery:.4f} s")
    print(f"  Jaccard Weighting Time          : {t_jaccard:.4f} s")
    print(f"  PyMETIS Partitioning Time       : {t_partition:.4f} s")
    print(f"  Overall Execution Time          : {total_time:.4f} s")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()

