import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class EigenPromptModule(nn.Module):
    """
    Centrality-Weighted Global Prompt Module (EigenPrompt)
    Learns a shared global prompt dictionary P in R^{M x d} to bridge subgraph boundaries.
    Attaches prompts to node features weighted by normalized eigenvector centrality (e'_v).
    """
    def __init__(self, in_feats, num_prompts=40, alpha=0.5, gamma=1.0):
        super().__init__()
        self.in_feats = in_feats
        self.num_prompts = num_prompts
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
        self.gamma = gamma

        # Learnable global prompt embeddings: P in R^{M x d}
        self.prompts = nn.Parameter(torch.empty(num_prompts, in_feats))
        nn.init.xavier_uniform_(self.prompts)

    def select_prompts(self, x):
        """
        Dynamically selects the best prompt for each node via cosine / normalized dot product similarity.
        x: (N, d)
        Returns: selected_prompts: (N, d)
        """
        # Normalize node features and prompt embeddings
        x_norm = F.normalize(x, p=2, dim=-1, eps=1e-8)
        p_norm = F.normalize(self.prompts, p=2, dim=-1, eps=1e-8)

        # Similarity scores: (N, M)
        sim = torch.matmul(x_norm, p_norm.t())

        # Select index with highest similarity score
        best_indices = torch.argmax(sim, dim=-1)  # (N,)

        # Extract corresponding prompt embeddings
        selected_prompts = self.prompts[best_indices]  # (N, d)
        return selected_prompts

    def forward(self, x, centrality_scores=None):
        """
        Attaches selected prompts to node features, modulated by normalized centrality.
        x: (N, d) node feature tensor
        centrality_scores: (N,) tensor in [0, 1] representing normalized eigenvector centrality.
        """
        selected_prompts = self.select_prompts(x)

        if centrality_scores is not None:
            # Ensure centrality is 2D: (N, 1) and matches device/dtype
            c = centrality_scores.to(x.device).to(x.dtype)
            if c.dim() == 1:
                c = c.unsqueeze(-1)

            # Centrality-weighted attachment: h'_v = h_v + alpha * (e'_v)^gamma * P_m
            weight = self.alpha * torch.pow(c.clamp(min=1e-6, max=1.0), self.gamma)
            x_augmented = x + weight * selected_prompts
        else:
            # Standard uniform prompt attachment (fallback)
            x_augmented = x + self.alpha * selected_prompts

        return x_augmented
