import argparse
import os
import sys
import torch
from eval_subgraph import evaluate_subgraph_mode, profile_subgraph_mode
from eval_layerwise import evaluate_layerwise_streaming, profile_layerwise_streaming

def compare_inference_modes(model, prompt_module, subgraphs, g, device, batch_size=4096):
    """
    Runs both Subgraph Evaluation (Mode A) and True Layer-by-Layer Streaming (Mode B),
    profiles latency & peak GPU memory, and prints a side-by-side comparison table.
    
    Parameters:
        model: Trained PyTorch GNN model
        prompt_module: EigenPromptModule or None
        subgraphs: List of DGL subgraphs
        g: Full DGLGraph
        device: torch.device
        batch_size: Batch size for layer-wise dataloader
        
    Returns:
        dict with metrics from both modes
    """
    print("\n" + "=" * 80)
    print(f"{'INFERENCE BENCHMARK & COMPARISON':^80}")
    print("=" * 80)
    print("Executing Mode A: Subgraph Evaluation (isolated subgraphs)...")
    res_subgraph = profile_subgraph_mode(model, prompt_module, subgraphs, device)
    
    print("Executing Mode B: True Layer-by-Layer Streaming (full graph with cut-edge recovery)...")
    res_layerwise = profile_layerwise_streaming(model, prompt_module, g, device, batch_size=batch_size)

    # Compute metric deltas
    acc_delta = (res_layerwise["test_acc"] - res_subgraph["test_acc"]) * 100.0
    mem_delta = res_layerwise["peak_gpu_mb"] - res_subgraph["peak_gpu_mb"]
    time_delta = res_layerwise["latency_sec"] - res_subgraph["latency_sec"]

    print("\n" + "-" * 80)
    print(f"{'Metric':<30} | {'Mode A (Subgraph)':<22} | {'Mode B (Layer-Wise Streaming)':<22}")
    print("-" * 80)
    print(f"{'Graph Edge Coverage':<30} | {'Within-Cluster Only':<22} | {'100% Full Graph':<22}")
    print(f"{'Cross-Partition Cut Edges':<30} | {'Discarded (0%)':<22} | {'Fully Restored (100%)':<22}")
    print(f"{'Validation Accuracy':<30} | {res_subgraph['val_acc']*100:>19.2f}% | {res_layerwise['val_acc']*100:>19.2f}%")
    print(f"{'Test Accuracy':<30} | {res_subgraph['test_acc']*100:>19.2f}% | {res_layerwise['test_acc']*100:>19.2f}%")
    print(f"{'Evaluation Latency':<30} | {res_subgraph['latency_sec']:>19.4f}s | {res_layerwise['latency_sec']:>19.4f}s")
    print(f"{'Peak GPU Memory':<30} | {res_subgraph['peak_gpu_mb']:>19.1f}MB| {res_layerwise['peak_gpu_mb']:>19.1f}MB")
    print("-" * 80)
    print(f"Accuracy Gain from Full-Graph Topology : {acc_delta:+.2f}%")
    print(f"Latency Overhead                       : {time_delta:+.4f}s")
    print(f"Memory Overhead                        : {mem_delta:+.1f} MB")
    print("=" * 80 + "\n")

    return {
        "subgraph_mode": res_subgraph,
        "layerwise_mode": res_layerwise,
        "accuracy_gain_pct": acc_delta
    }

