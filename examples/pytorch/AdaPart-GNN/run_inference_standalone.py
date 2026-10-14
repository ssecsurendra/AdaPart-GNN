import argparse
import os
import time
import numpy as np
import torch
import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import AsNodePredDataset, RedditDataset

from models_gnn import SubgraphGNN
from eigen_prompt import EigenPromptModule
from eval_subgraph import evaluate_subgraph_mode, profile_subgraph_mode
from eval_layerwise import evaluate_layerwise_streaming, profile_layerwise_streaming
from eval_comparison import compare_inference_modes

def load_dataset(dataset_name, dataset_dir="./dataset"):
    print(f"Loading dataset: {dataset_name} ...")
    if dataset_name == "ogbn-arxiv":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-arxiv", root=dataset_dir))
        g = data[0]
    elif dataset_name == "ogbn-products":
        data = AsNodePredDataset(DglNodePropPredDataset("ogbn-products", root=dataset_dir))
        g = data[0]
    elif dataset_name == "reddit":
        data = RedditDataset(raw_dir=dataset_dir)
        g = data[0]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if "label" in g.ndata:
        g.ndata["label"] = g.ndata["label"].squeeze().long()

    for mask_name in ["train_mask", "val_mask", "test_mask"]:
        if mask_name in g.ndata:
            g.ndata[mask_name] = g.ndata[mask_name].bool()

    return g, g.ndata["feat"].shape[1], data.num_classes


def create_subgraphs(g, cluster_file_path, centrality_file_path=""):
    with open(cluster_file_path, "r") as f:
        cluster_ids = np.array([int(line.strip()) for line in f if line.strip() != ""], dtype=np.int64)

    if centrality_file_path and os.path.exists(centrality_file_path):
        centrality_np = np.load(centrality_file_path)
    else:
        deg = g.in_degrees().float()
        centrality_np = (deg / (deg.max().item() + 1e-8)).numpy()

    g.ndata["centrality"] = torch.from_numpy(centrality_np).float()
    unique_clusters = np.unique(cluster_ids)
    
    subgraphs = []
    for c_id in unique_clusters:
        node_indices = torch.where(torch.from_numpy(cluster_ids == c_id))[0]
        if len(node_indices) == 0:
            continue
        sg = g.subgraph(node_indices)
        sg = dgl.remove_self_loop(sg)
        sg = dgl.add_self_loop(sg)
        subgraphs.append(sg)

    return subgraphs


def main():
    parser = argparse.ArgumentParser(description="Standalone Inference Runner (Subgraph vs Layer-Wise Streaming)")
    parser.add_argument("--mode", type=str, default="both", choices=["subgraph", "layerwise", "both"],
                        help="Inference mode: 'subgraph' (isolated), 'layerwise' (full-graph streaming), or 'both'")
    parser.add_argument("--dataset", type=str, default="ogbn-arxiv", choices=["ogbn-arxiv", "reddit", "ogbn-products"])
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--cluster_file", type=str, required=True, help="Path to cluster partition file")
    parser.add_argument("--centrality_file", type=str, default="")
    parser.add_argument("--model", type=str, default="sage", choices=["sage", "gcn"])
    parser.add_argument("--checkpoint", type=str, default="", help="Path to saved model checkpoint (.pt)")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--use_prompts", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device >= 0 else "cpu")
    print(f"Device: {device} | Evaluation Mode Selected: {args.mode.upper()}")

    g, in_feats, num_classes = load_dataset(args.dataset, args.dataset_dir)
    subgraphs = create_subgraphs(g, args.cluster_file, args.centrality_file)

    model = SubgraphGNN(
        in_feats=in_feats,
        n_hidden=args.n_hidden,
        n_classes=num_classes,
        n_layers=args.n_layers,
        model_type=args.model,
        use_batch_norm=True,
        use_residual=True
    ).to(device)

    prompt_module = None
    if args.use_prompts:
        prompt_module = EigenPromptModule(in_feats=in_feats).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model"])
        if prompt_module is not None and "prompt" in ckpt:
            prompt_module.load_state_dict(ckpt["prompt"])

    if args.mode == "subgraph":
        res = profile_subgraph_mode(model, prompt_module, subgraphs, device)
        print(f"\n[Subgraph Evaluation Mode A]")
        print(f"Validation Loss : {res['val_loss']:.4f}")
        print(f"Validation Acc  : {res['val_acc']*100:.2f}%")
        print(f"Test Acc        : {res['test_acc']*100:.2f}%")
        print(f"Latency         : {res['latency_sec']:.4f}s")
        print(f"Peak GPU Memory : {res['peak_gpu_mb']:.1f} MB")
    elif args.mode == "layerwise":
        res = profile_layerwise_streaming(model, prompt_module, g, device, batch_size=args.batch_size)
        print(f"\n[True Layer-by-Layer Streaming Mode B]")
        print(f"Validation Loss : {res['val_loss']:.4f}")
        print(f"Validation Acc  : {res['val_acc']*100:.2f}%")
        print(f"Test Acc        : {res['test_acc']*100:.2f}%")
        print(f"Latency         : {res['latency_sec']:.4f}s")
        print(f"Peak GPU Memory : {res['peak_gpu_mb']:.1f} MB")
    else:  # both
        compare_inference_modes(model, prompt_module, subgraphs, g, device, batch_size=args.batch_size)

if __name__ == "__main__":
    main()
