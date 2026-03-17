import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from TransGCN import AdaptiveFusionModule
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
# NN model
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # torch.manual_seed(1234)
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.out_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.lin1(x).relu()
        h = self.lin2(h).relu()

        output = self.out_layer(h).squeeze()
        return output


# GCN model
class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.6):
        super().__init__()
        # torch.manual_seed(1234)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout1(h)

        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout2(h)

        out = self.classifier(h).squeeze()
        return out


# Positional Encoding - Essential component for Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Dim)
        return x + self.pe[:, :x.size(1), :]


# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, nhead=2, num_layers=2, seq_len=10):
        """
        in_dim: Input feature dimension
        hidden_dim: Hidden layer dimension (d_model)
        out_dim: Output dimension
        nhead: Number of attention heads
        num_layers: Number of Encoder layers
        seq_len: Sequence length (used for positional encoding initialization, dynamically determined by input)
        """
        super().__init__()
        # torch.manual_seed(1234)

        # 1. Input projection layer: map input to hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 2. Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # 3. Transformer Encoder layer
        # dropout=0.1 is a common default value
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.4,
            batch_first=True  # Important: Set to True to match (Batch, Seq, Dim) format
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Classification/Regression head
        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.hidden_dim = hidden_dim

    def forward(self, x, mask=None):
        """
        x: Input tensor with shape (Num_Nodes, In_Dim) or (Batch, Num_Nodes, In_Dim)
        Returns: (Num_Nodes, Out_Dim) or (Batch, Num_Nodes, Out_Dim)
                 Preserve node dimension for subsequent filtering with action_mask
        """
        # 1. Ensure input is 3D: (Batch, Seq_Len, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)

        batch_size, seq_len, _ = x.shape

        # 2. Linear projection
        h = self.input_proj(x) * math.sqrt(self.hidden_dim)

        # 3. Add positional encoding
        h = self.pos_encoder(h)

        # 4. Transformer Encoder
        # Note: If seq_len is large, memory consumption will be O(N^2), which is a characteristic of Transformer
        h = self.transformer_encoder(h, mask=mask)

        # 5. Activation function
        h = F.relu(h)

        # 6. Output layer (Classifier)
        # Input: (Batch, Seq_Len, Hidden) -> Output: (Batch, Seq_Len, Out_Dim)
        out = self.classifier(h)

        # 7. Dimension adjustment
        # If out_dim=1, remove the last dimension: (B, N, 1) -> (B, N)
        if out.size(-1) == 1:
            out = out.squeeze(-1)

        # If input was originally 2D (single graph), return 2D (N, ) to match GCN/GAT convention
        # But for unified batch processing in training loop, it's better to keep 3D or let training loop handle squeeze
        # Here we keep (Batch, N) or (Batch, N, C)
        if x.dim() == 2:
            out = out.squeeze(0)

        return out


class DilatedCNNBranch(nn.Module):
    """
    Dilated CNN Branch designed based on TransGraphNet paper.
    Used to extract multi-scale spatial patterns from CICIDS-2017 traffic features.

    Parameters:
        input_dim (int): Dimension of input features (e.g., usually 78 or 80 for CICIDS-2017)
        hidden_channels (int): Number of hidden layer channels, default to 64 as per paper
        dropout_rate (float): Dropout ratio
        reshape_dims (tuple): Shape (H, W) to reshape 1D features into 2D, must satisfy H*W >= input_dim
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.3, reshape_dims=(12, 12)):
        super(DilatedCNNBranch, self).__init__()

        self.input_dim = in_dim
        self.reshape_dims = reshape_dims  # (Height, Width)
        total_pixels = reshape_dims[0] * reshape_dims[1]

        if total_pixels < in_dim:
            raise ValueError(
                f"Reshape dimensions {reshape_dims} (total {total_pixels}) are too small for input_dim {in_dim}")

        # Define dilation rate sequence [1, 2, 4, 8]
        dilation_rates = [1, 2, 4, 8]

        layers = []
        in_channels = 1  # Input treated as single-channel image (Batch, 1, H, W)

        for i, dilation in enumerate(dilation_rates):
            # Dynamically calculate output channels, gradually increase complexity, stabilize at hidden_channels finally
            out_channels = hidden_dim if i > 0 else 32

            # Conv2d parameters:
            # kernel_size=3
            # padding=dilation (to ensure output size matches input size, i.e., Same Padding)
            # dilation=dilation (core dilated parameter)
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=True
            )

            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(out_channels))  # Add BN to accelerate convergence and stabilize training
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout_rate * 0.5))  # 2D Dropout

            in_channels = out_channels  # Input channels of next layer equal to output channels of current layer

        self.conv_stack = nn.Sequential(*layers)

        # Global average pooling, compress (B, C, H, W) to (B, hidden_dim, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully connected layer for specified output
        self.fc_output = nn.Linear(hidden_dim, out_dim)
        # Final Dropout and fully connected layer (optional, FC not needed for direct fusion, return pure feature vector here)
        self.final_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
    Supports two types of input:
      1. (Batch, in_dim)          → Normal case
      2. (Batch, Num_Action_Nodes, in_dim) → Current training scenario (action nodes)
    """
        original_shape = None
        if x.dim() == 3:  # ← Key fix
            B, N, F = x.shape
            x = x.view(B * N, F)  # Treat each action node as independent sample
            original_shape = (B, N)
        elif x.dim() == 2:
            B = x.size(0)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # ==================== Original padding + reshape logic ====================
        h, w = self.reshape_dims
        total_pixels = h * w

        if self.input_dim < total_pixels:
            padded_x = torch.zeros(x.size(0), total_pixels,
                                   device=x.device, dtype=x.dtype)
            padded_x[:, :self.input_dim] = x
            x = padded_x

        x = x.view(-1, 1, h, w)  # (B*N or B, 1, H, W)

        x = self.conv_stack(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_output(x)
        x = self.final_dropout(x)

        # Restore node dimension if flattened before
        if original_shape is not None:
            x = x.view(original_shape[0], original_shape[1], -1)  # (B, 7, out_dim)

        # Auto squeeze when out_dim=1 for convenient criterion calculation later
        if x.size(-1) == 1:
            x = x.squeeze(-1)  # → (B, 7)

        return x


class MainModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, choice):
        super(MainModel, self).__init__()
        self.rt_meas_dim = 46
        # 1. Instantiate each branch model (only once during initialization)
        self.branch_gcn = GCN(in_dim, hidden_dim, hidden_dim)
        self.branch_transformer = TransformerModel(in_dim, hidden_dim, hidden_dim)
        self.DCNN = DilatedCNNBranch(self.rt_meas_dim, hidden_dim, hidden_dim)
        self.choice = choice
        # 2. Instantiate fusion module (specify 2 branches)
        self.fusion_module = AdaptiveFusionModule(hidden_dim, out_dim, num_branches=3)

    def forward(self, x, edge_index):
        # 3. Extract features separately (dynamically calculated, different for each forward pass)
        # GCN requires edge_index
        if (self.choice != 1):
            f_graph = self.branch_gcn(x, edge_index)

        # Transformer usually only needs x (depending on your TransformerModel implementation)
        if (self.choice != 2):
            f_temp = self.branch_transformer(x)

        # DilatedCNN
        if (self.choice != 3):
            meas_x = x[..., -self.rt_meas_dim:]
            f_DCNN = self.DCNN(meas_x)

        # 4. Pass feature list to fusion module
        # Note: Pass calculated feature tensors here, not raw data

        if (self.choice == 1):
            output = self.fusion_module([f_temp, f_DCNN])

        elif (self.choice == 2):
            output = self.fusion_module([f_graph, f_DCNN])

        elif (self.choice == 3):
            output = self.fusion_module([f_temp, f_graph])
        else:
            output = self.fusion_module([f_temp, f_graph, f_DCNN])

        return output