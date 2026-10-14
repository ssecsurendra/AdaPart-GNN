import time
import torch
import torch.nn.functional as F

def evaluate_subgraph_mode(model, prompt_module, subgraphs, device):
    """
    Evaluates GNN performance on isolated subgraphs (Mode A: Subgraph Evaluation).
    
    Each partition subgraph is evaluated independently without accessing
    cross-partition cut edges, yielding ultra-low memory consumption.
    
    Parameters:
        model: PyTorch GNN model
        prompt_module: EigenPromptModule or None
        subgraphs: List of DGL subgraphs
        device: torch.device
        
    Returns:
        val_loss (float): Average validation loss across validation nodes
        val_acc (float): Overall validation accuracy
        test_acc (float): Overall test accuracy
    """
    model.eval()
    if prompt_module is not None:
        prompt_module.eval()

    total_val_loss = 0.0
    total_val_correct = 0
    total_val_nodes = 0

    total_test_correct = 0
    total_test_nodes = 0

    with torch.no_grad():
        for sg in subgraphs:
            sg = sg.to(device)
            feat = sg.ndata["feat"]
            label = sg.ndata["label"]
            val_mask = sg.ndata["val_mask"]
            test_mask = sg.ndata["test_mask"]

            # Apply EigenPrompt if enabled
            if prompt_module is not None:
                c_scores = sg.ndata.get("centrality", None)
                feat = prompt_module(feat, c_scores)

            logits = model(sg, feat)

            if val_mask.sum() > 0:
                val_loss = F.cross_entropy(logits[val_mask], label[val_mask], reduction="sum")
                total_val_loss += val_loss.item()
                preds = logits[val_mask].argmax(dim=-1)
                total_val_correct += (preds == label[val_mask]).sum().item()
                total_val_nodes += val_mask.sum().item()

            if test_mask.sum() > 0:
                preds = logits[test_mask].argmax(dim=-1)
                total_test_correct += (preds == label[test_mask]).sum().item()
                total_test_nodes += test_mask.sum().item()

    val_acc = (total_val_correct / total_val_nodes) if total_val_nodes > 0 else 0.0
    val_loss = (total_val_loss / total_val_nodes) if total_val_nodes > 0 else 0.0
    test_acc = (total_test_correct / total_test_nodes) if total_test_nodes > 0 else 0.0

    return val_loss, val_acc, test_acc


def profile_subgraph_mode(model, prompt_module, subgraphs, device):
    """
    Profiles execution time and peak memory consumption of isolated Subgraph Evaluation.
    
    Returns:
        metrics: dict containing val_loss, val_acc, test_acc, latency_sec, peak_gpu_mb
    """
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()

    t0 = time.time()
    val_loss, val_acc, test_acc = evaluate_subgraph_mode(model, prompt_module, subgraphs, device)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_gpu_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        peak_gpu_mb = 0.0

    latency_sec = time.time() - t0

    return {
        "mode": "Subgraph Evaluation (Mode A)",
        "val_loss": val_loss,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "latency_sec": latency_sec,
        "peak_gpu_mb": peak_gpu_mb
    }

# Alias for drop-in compatibility
evaluate_subgraphs = evaluate_subgraph_mode

