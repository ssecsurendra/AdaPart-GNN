import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn

class SubgraphGNN(nn.Module):
    """
    Modular GNN Backbone supporting GraphSAGE (SAGEConv-mean) and GCN,
    equipped with Batch Normalization and Residual Skip Connections.
    """
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers=3,
        dropout=0.5,
        model_type="sage",  # 'sage' or 'gcn'
        use_batch_norm=True,
        use_residual=True
    ):
        super().__init__()
        self.in_feats = in_feats
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.model_type = model_type.lower()
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual

        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_batch_norm else None
        self.res_layers = nn.ModuleList() if use_residual else None

        for l in range(n_layers):
            dim_in = in_feats if l == 0 else n_hidden
            dim_out = n_classes if l == n_layers - 1 else n_hidden

            # 1. Graph Convolution / SAGE Convolution
            if self.model_type == "sage":
                self.conv_layers.append(dglnn.SAGEConv(dim_in, dim_out, aggregator_type="mean"))
            else:
                self.conv_layers.append(dglnn.GraphConv(dim_in, dim_out, allow_zero_in_degree=True))

            # 2. Batch Normalization (on intermediate hidden representations)
            if use_batch_norm and l < n_layers - 1:
                self.bn_layers.append(nn.BatchNorm1d(dim_out))

            # 3. Residual Projection (if dimensions match or linear projection)
            if use_residual and l < n_layers - 1:
                if dim_in != dim_out:
                    self.res_layers.append(nn.Linear(dim_in, dim_out, bias=False))
                else:
                    self.res_layers.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

    def forward(self, g, x):
        h = x
        for l in range(self.n_layers):
            h_in = h
            # Message Passing
            h_out = self.conv_layers[l](g, h_in)

            # Apply Residual, BatchNorm, ReLU, and Dropout for hidden layers
            if l < self.n_layers - 1:
                if self.use_residual:
                    h_res = self.res_layers[l](h_in)
                    h_out = h_out + h_res

                if self.use_batch_norm:
                    h_out = self.bn_layers[l](h_out)

                h = self.activation(h_out)
                h = self.dropout(h)
            else:
                h = h_out  # Final logits

        return h

    def forward_layer(self, block, h, l):
        """Processes a single layer on a bipartite block (for layer-wise inference)."""
        num_dst = block.num_dst_nodes()
        h_dst = h[:num_dst]
        
        # Bipartite convolution
        if self.model_type == "sage":
            h_out = self.conv_layers[l](block, (h, h_dst))
        else:
            h_out = self.conv_layers[l](block, h)

        if l < self.n_layers - 1:
            if self.use_residual:
                h_res = self.res_layers[l](h_dst)
                h_out = h_out + h_res

            if self.use_batch_norm:
                h_out = self.bn_layers[l](h_out)

            h_out = self.activation(h_out)
            if self.training:
                h_out = self.dropout(h_out)

        return h_out


