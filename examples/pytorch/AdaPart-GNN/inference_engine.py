import torch
import torch.nn.functional as F
import dgl
from dgl.dataloading import DataLoader, MultiLayerFullNeighborSampler

def evaluate_subgraphs(model, prompt_module, subgraphs, device):
    """
    Evaluates GNN performance on isolated subgraphs (Mode A: low-memory evaluation).
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


def evaluate_full_graph_layerwise(model, prompt_module, g, device, batch_size=4096):
    """
    Conducts true layer-wise streaming inference over the full graph (Mode B: SOTA protocol).
    Ensures self-loops are present (matching training distribution) and streams 
    1-hop neighborhood blocks layer by layer using MultiLayerFullNeighborSampler(1).
    Restores all cross-partition cut edges without loading full graph into GPU memory.
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

        # Layer-wise neighbor sampler: 1-hop aggregation per layer
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

        # Iterate through GNN layers one by one
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


