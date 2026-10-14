import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class SOTASubgraphGNN(nn.Module):
    """
    State-of-the-Art 3-Layer Graph Neural Network Backbone.
    
    Equipped with:
      - 3-Layer GraphSAGE (SAGEConv-mean) or GCN
      - Layer Normalization (LayerNorm) or Batch Normalization (BatchNorm1d)
      - Residual Skip Connections (Identity/Linear) to prevent oversmoothing
      - Bipartite block execution support (forward_layer) for Layer-Wise Streaming Inference
    """
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers=3,
        dropout=0.5,
        model_type="sage",
        norm_type="layernorm",  # 'layernorm', 'batchnorm', or 'none'
        use_batch_norm=None,    # For backwards compatibility
        use_residual=True
    ):
        super().__init__()
        assert n_layers == 3, f"SOTASubgraphGNN is configured for strict 3-layer architecture (received {n_layers})"
        
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.model_type = model_type.lower()
        self.use_residual = use_residual

        # Backwards compatibility handling
        if use_batch_norm is not None:
            self.norm_type = "batchnorm" if use_batch_norm else "none"
        else:
            self.norm_type = norm_type.lower() if norm_type else "none"

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() if self.norm_type != "none" else None
        self.res_layers = nn.ModuleList() if use_residual else None

        for l in range(n_layers):
            dim_in = in_feats if l == 0 else n_hidden
            dim_out = n_classes if l == n_layers - 1 else n_hidden

            # 1. Graph Convolution / SAGE Convolution
            if self.model_type == "sage":
                self.conv_layers.append(dglnn.SAGEConv(dim_in, dim_out, aggregator_type="mean"))
            else:
                self.conv_layers.append(dglnn.GraphConv(dim_in, dim_out, allow_zero_in_degree=True))

            # 2. Normalization (LayerNorm or BatchNorm on intermediate hidden representations)
            if self.norm_type != "none" and l < n_layers - 1:
                if self.norm_type == "layernorm":
                    self.norm_layers.append(nn.LayerNorm(dim_out))
                elif self.norm_type == "batchnorm":
                    self.norm_layers.append(nn.BatchNorm1d(dim_out))
                else:
                    raise ValueError(f"Unknown norm_type: {self.norm_type}")

            # 3. Residual Skip Connections (on intermediate hidden representations)
            if use_residual and l < n_layers - 1:
                if dim_in != dim_out:
                    self.res_layers.append(nn.Linear(dim_in, dim_out, bias=False))
                else:
                    self.res_layers.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, g, x):
        """Standard full subgraph forward pass."""
        h = x
        for l in range(self.n_layers):
            h_in = h
            h_out = self.conv_layers[l](g, h_in)

            if l < self.n_layers - 1:
                if self.use_residual:
                    h_res = self.res_layers[l](h_in)
                    h_out = h_out + h_res

                if self.norm_type != "none":
                    h_out = self.norm_layers[l](h_out)

                h = self.activation(h_out)
                h = self.dropout(h)
            else:
                h = h_out  # Output logits

        return h

    def forward_layer(self, block, h, l):
        """Processes a single layer on a bipartite block (for True Layer-by-Layer Streaming)."""
        num_dst = block.num_dst_nodes()
        h_dst = h[:num_dst]

        if self.model_type == "sage":
            h_out = self.conv_layers[l](block, (h, h_dst))
        else:
            h_out = self.conv_layers[l](block, h)

        if l < self.n_layers - 1:
            if self.use_residual:
                h_res = self.res_layers[l](h_dst)
                h_out = h_out + h_res

            if self.norm_type != "none":
                h_out = self.norm_layers[l](h_out)

            h_out = self.activation(h_out)
            if self.training:
                h_out = self.dropout(h_out)

        return h_out
