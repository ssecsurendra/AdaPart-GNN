import time
import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

def evaluate_layerwise_streaming(model, prompt_module, g, device, batch_size=4096):
    """
    Conducts True Layer-by-Layer Streaming Inference over the full graph (Mode B).
    
    Restores 100% of graph topology and cross-partition cut edges without GPU OOM.
    Iterates sequentially through GNN layers, streaming 1-hop neighborhood blocks 
    into a CPU buffer via MultiLayerFullNeighborSampler(1).
    
    Parameters:
        model: PyTorch GNN model with forward_layer() support
        prompt_module: EigenPromptModule or None
        g: DGLGraph (full graph)
        device: torch.device (GPU device for computation)
        batch_size: int (chunk size of destination nodes per batch)
        
    Returns:
        val_loss (float): Cross-entropy loss on validation nodes
        val_acc (float): Validation accuracy
        test_acc (float): Test accuracy
    """
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    with torch.no_grad():
        num_nodes = g.num_nodes()
        
        # Ensure self-loops are present so neighborhood aggregation matches training
        g_eval = dgl.remove_self_loop(g)
        g_eval = dgl.add_self_loop(g_eval)

        feat = g.ndata["feat"]
        label = g.ndata["label"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]

        # Apply prompts if present (in chunks to avoid memory spikes)
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

        # 1-hop layer-wise neighbor sampler
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

        buffer_device = torch.device("cpu")
        pin_memory = (buffer_device != device)

        # Iterate sequentially through GNN layers
        for l in range(model.n_layers):
            out_dim = model.n_classes if l == model.n_layers - 1 else model.n_hidden
            next_feat = torch.empty(
                num_nodes,
                out_dim,
                dtype=torch.float32,
                device=buffer_device,
                pin_memory=pin_memory
            )

            for input_nodes, output_nodes, blocks in dataloader:
                block = blocks[0].to(device)
                h = cur_feat[input_nodes.cpu()].to(device)
                h_out = model.forward_layer(block, h, l)
                next_feat[output_nodes.cpu()] = h_out.to(buffer_device)

            cur_feat = next_feat

        logits = cur_feat

        # Calculate metrics
        val_acc = 0.0
        val_loss = 0.0
        if val_mask.sum() > 0:
            val_loss = F.cross_entropy(logits[val_mask], label[val_mask]).item()
            preds_val = logits[val_mask].argmax(dim=-1)
            val_acc = (preds_val == label[val_mask]).float().mean().item()

        test_acc = 0.0
        if test_mask.sum() > 0:
            preds_test = logits[test_mask].argmax(dim=-1)
            test_acc = (preds_test == label[test_mask]).float().mean().item()

    return val_loss, val_acc, test_acc


def profile_layerwise_streaming(model, prompt_module, g, device, batch_size=4096):
    """
    Profiles execution time and peak memory consumption of True Layer-by-Layer Streaming Inference.
    
    Returns:
        metrics: dict containing val_loss, val_acc, test_acc, latency_sec, peak_gpu_mb
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.time()
    val_loss, val_acc, test_acc = evaluate_layerwise_streaming(
        model, prompt_module, g, device, batch_size=batch_size
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_gpu_mb = 0.0

    latency_sec = time.time() - t0

    return {
        "mode": "True Layer-by-Layer Streaming (Mode B)",
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "latency_sec": latency_sec,
        "peak_gpu_mb": peak_gpu_mb
    }

# Aliases for drop-in compatibility
evaluate_full_graph_layerwise = evaluate_layerwise_streaming
layerwise_infer = evaluate_layerwise_streaming

