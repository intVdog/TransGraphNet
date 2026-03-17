import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np
import torch

# >>>>>>>>>> Fix all random seeds <<<<<<<<<<
SEED = 44
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# >>>>>>>>>> End of seed setting <<<<<<<<<<

class AdaptiveFusionModule(nn.Module):
    """
    Attention-based gated fusion module
    Input: A list containing multiple branch features [f_temp, f_graph, ...]
    Output: Fused classification results
    """

    def __init__(self, hidden_dim, out_dim, num_branches=3):
        super(AdaptiveFusionModule, self).__init__()
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim

        # 1. Scoring network: Assign scores to features of each branch
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Output scalar score
        )

        # 2. Feature transformation layers: Align feature spaces of different branches (optional but recommended)
        # Create an independent Linear layer for each branch
        self.transform_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_branches)
        ])

        # 3. Final classifier
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, features_list):
        """
        Parameters:
            features_list: List containing [f_temp, f_graph]
                           Shape could be (Nodes, Dim) or (1, Nodes, Dim) or (Batch, Nodes, Dim)
        Returns:
            output: Classification results
        """
        # print("=== Fusion Module Input Shapes ===")
        # for idx, f in enumerate(features_list):
        #     print(f"Feature {idx}: {f.shape}")
        # print("=================================")
        transformed_features = []

        # --- Step 0: Dimension alignment (Key fix) ---
        # Ensure all features have 3 dimensions: (Batch, Nodes, Dim)
        # If single sample (Nodes, Dim), unsqueeze(0) to (1, Nodes, Dim)
        normalized_features = []
        for f in features_list:
            if f.dim() == 1:  # ← Handle cases like (26,)
                f = f.unsqueeze(-1)  # Convert to (26, 1), automatic alignment later

            if f.dim() == 2:  # (26, D) → (1, 26, D)
                f = f.unsqueeze(0)

            elif f.dim() == 3:
                pass  # Remove squeeze operation to maintain consistent batch dimension (even if batch=1)

            else:
                raise ValueError(f"Unsupported feature dimension: {f.dim()}, shape: {f.shape}")

            normalized_features.append(f)

        batch_size = normalized_features[0].size(0)
        num_nodes = normalized_features[0].size(1)

        # --- Step 1: Feature transformation ---
        # Note: transform_layers are Linear layers, usually applied to the last dimension (Dim)
        # To apply Linear, reshape (Batch, Nodes, Dim) to (Batch*Nodes, Dim)
        # Or use torch.nn.functional.linear with broadcasting, but reshape is more reliable

        for i, f in enumerate(normalized_features):
            # f shape: (Batch, Nodes, Dim)
            # Reshape for Linear layer: (Batch * Nodes, Dim)
            f_reshaped = f.view(-1, self.hidden_dim)

            # Pass through transformation layer
            h_reshaped = self.transform_layers[i](f_reshaped)

            # Restore shape: (Batch, Nodes, Dim)
            h = h_reshaped.view(batch_size, num_nodes, self.hidden_dim)
            transformed_features.append(h)

        # Stack: [Batch, Num_Branches, Nodes, Dim]  <-- Note added Nodes dimension here
        # Previous code stacked on dim=1 to get [Batch, Branches, Dim], which was for globally pooled features
        # For node-level classification tasks, calculate attention on each node
        stacked_features = torch.stack(transformed_features, dim=1)
        # Current shape: (Batch, 2, Nodes, Dim)

        # --- Step 2: Calculate attention scores (node-level attention) ---
        # Need to score each branch for each node
        # Reshape: (Batch * Nodes, Num_Branches, Dim) -> easier calculation?
        # Or simpler: reshape to (Batch * Nodes * Num_Branches, Dim) then score

        b, branches, n, d = stacked_features.shape

        # Reshape to (Batch * Nodes * Branches, Dim)
        reshaped_for_score = stacked_features.view(-1, d)

        # Calculate scores: (Batch * Nodes * Branches, 1)
        scores = self.score_net(reshaped_for_score)

        # Restore shape: (Batch, Nodes, Branches, 1) -> adjust dimension order for Softmax
        scores = scores.view(b, n, branches, 1)
        # Transpose to (Batch, Nodes, Branches, 1) for Softmax on Branches dimension
        # The above view already gives (B, N, Br, 1), apply softmax directly on dim=2 (Branches)
        attention_weights = F.softmax(scores, dim=2)
        # Shape: (Batch, Nodes, Branches, 1)

        # --- Step 3: Weighted sum ---
        # stacked_features: (Batch, Branches, Nodes, Dim)
        # attention_weights: (Batch, Nodes, Branches, 1) -> need to adjust to (Batch, Branches, Nodes, 1) to match

        # Adjust weight dimensions to match stacked_features (B, Br, N, D)
        # Current weights: (B, N, Br, 1) -> permute to (B, Br, N, 1)
        weights_permuted = attention_weights.permute(0, 2, 1, 3)

        # Weighting: (B, Br, N, 1) * (B, Br, N, D) -> (B, Br, N, D)
        weighted_features = weights_permuted * stacked_features

        # Sum over Branches dimension (dim=1) -> (Batch, Nodes, Dim)
        fused_feature = torch.sum(weighted_features, dim=1)

        # --- Step 4: Classification (node-level) ---
        # fused_feature: (Batch, Nodes, Dim)
        # Reshape to (Batch * Nodes, Dim) for classifier
        fused_reshaped = fused_feature.view(-1, self.hidden_dim)
        logits_reshaped = self.classifier(fused_reshaped)  # (Batch * Nodes, Out_Dim)

        # Restore shape: (Batch, Nodes, Out_Dim)
        output = logits_reshaped.view(batch_size, num_nodes, -1)

        if output.size(-1) == 1:  # Handle out_dim=1 case
            output = output.squeeze(-1)  # (B, N, 1) -> (B, N)
        if output.dim() == 2 and output.size(0) == 1:
            output = output.squeeze(0)  # (1, N) -> (N,)
        # If input is single sample without batch dimension (original logic processes sample by sample), squeeze batch dim
        # But for unified training loop, keep (1, Nodes, Out_Dim) and let upper layer handle squeeze
        return output