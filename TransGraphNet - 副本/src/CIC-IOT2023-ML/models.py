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

# >>>>>>>>>> 固定所有随机种子 <<<<<<<<<<
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


# 位置编码 (Positional Encoding) - Transformer 必需组件
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
        in_dim: 输入特征维度
        hidden_dim: 隐藏层维度 (d_model)
        out_dim: 输出维度
        nhead: 注意力头数
        num_layers: Encoder 层数
        seq_len: 序列长度 (用于位置编码初始化，实际由输入动态决定)
        """
        super().__init__()
        # torch.manual_seed(1234)

        # 1. 输入投影层：将输入映射到 hidden_dim
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # 3. Transformer Encoder 层
        # dropout=0.1 是常见默认值
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=0.4,
            batch_first=True  # 重要：设置为 True 以匹配 (Batch, Seq, Dim) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 分类/回归头
        self.classifier = nn.Linear(hidden_dim, out_dim)

        self.hidden_dim = hidden_dim

    def forward(self, x, mask=None):
        """
        x: 输入张量，形状 (Num_Nodes, In_Dim) 或 (Batch, Num_Nodes, In_Dim)
        返回: (Num_Nodes, Out_Dim) 或 (Batch, Num_Nodes, Out_Dim)
              保持节点维度，以便后续使用 action_mask 筛选
        """
        # 1. 确保输入是 3D: (Batch, Seq_Len, Dim)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # (N, D) -> (1, N, D)

        batch_size, seq_len, _ = x.shape

        # 2. 线性投影
        h = self.input_proj(x) * math.sqrt(self.hidden_dim)

        # 3. 添加位置编码
        h = self.pos_encoder(h)

        # 4. Transformer Encoder
        # 注意：如果 seq_len 很大，显存消耗会是 O(N^2)，这是 Transformer 的特性
        h = self.transformer_encoder(h, mask=mask)

        # 5. 激活函数
        h = F.relu(h)

        # 6. 输出层 (Classifier)
        # 输入: (Batch, Seq_Len, Hidden) -> 输出: (Batch, Seq_Len, Out_Dim)
        out = self.classifier(h)

        # 7. 维度调整
        # 如果 out_dim=1，去掉最后一个维度: (B, N, 1) -> (B, N)
        if out.size(-1) == 1:
            out = out.squeeze(-1)

        # 如果输入原本是 2D (单图)，我们也返回 2D (N, ) 以匹配 GCN/GAT 的习惯
        # 但为了训练循环统一处理 batch，保持 3D 或让训练循环处理 squeeze 更好
        # 这里我们保持 (Batch, N) 或 (Batch, N, C)
        if x.dim() == 2:
            out = out.squeeze(0)

        return out


class DilatedCNNBranch(nn.Module):
    """
    基于 TransGraphNet 论文设计的空洞卷积分支 (Dilated CNN Branch)。
    用于提取 CICIDS-2017 流量特征中的多尺度空间模式。

    参数:
        input_dim (int): 输入特征的维度 (例如 CICIDS-2017 通常为 78 或 80)
        hidden_channels (int): 隐藏层通道数，默认根据论文设为 64
        dropout_rate (float): Dropout 比率
        reshape_dims (tuple): 将 1D 特征重塑为 2D 的形状 (H, W)，需满足 H*W >= input_dim
    """

    def __init__(self, in_dim, hidden_dim, out_dim, dropout_rate=0.3, reshape_dims=(12, 12)):
        super(DilatedCNNBranch, self).__init__()

        self.input_dim = in_dim
        self.reshape_dims = reshape_dims  # (Height, Width)
        total_pixels = reshape_dims[0] * reshape_dims[1]

        if total_pixels < in_dim:
            raise ValueError(
                f"Reshape dimensions {reshape_dims} (total {total_pixels}) are too small for input_dim {in_dim}")

        # 定义膨胀率序列 [1, 2, 4, 8]
        dilation_rates = [1, 2, 4, 8]

        layers = []
        in_channels = 1  # 输入视为单通道图像 (Batch, 1, H, W)

        for i, dilation in enumerate(dilation_rates):
            # 动态计算输出通道，逐步增加复杂度，最后稳定在 hidden_channels
            out_channels = hidden_dim if i > 0 else 32

            # Conv2d 参数:
            # kernel_size=3
            # padding=dilation (为了保证输出尺寸与输入尺寸一致，即 Same Padding)
            # dilation=dilation (核心空洞参数)
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=True
            )

            layers.append(conv_layer)
            layers.append(nn.BatchNorm2d(out_channels))  # 添加 BN 加速收敛并稳定训练
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout2d(p=dropout_rate * 0.5))  # 2D Dropout

            in_channels = out_channels  # 下一层的输入通道等于当前层的输出通道

        self.conv_stack = nn.Sequential(*layers)

        # 全局平均池化，将 (B, C, H, W) 压缩为(B, hidden_dim, 1, 1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层指定输出
        self.fc_output = nn.Linear(hidden_dim, out_dim)
        # 最终的 Dropout 和 全连接层 (可选，如果直接融合则不需要 FC，这里输出纯特征向量)
        self.final_dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
    支持两种输入:
      1. (Batch, in_dim)          → 普通情况
      2. (Batch, Num_Action_Nodes, in_dim) → 当前训练场景（action nodes）
    """
        original_shape = None
        if x.dim() == 3:  # ← 关键修复
            B, N, F = x.shape
            x = x.view(B * N, F)  # 把每个 action node 当成独立样本
            original_shape = (B, N)
        elif x.dim() == 2:
            B = x.size(0)
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")

        # ==================== 原来填充 + reshape 逻辑 ====================
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

        # 如果之前展平了节点维度，现在恢复回来
        if original_shape is not None:
            x = x.view(original_shape[0], original_shape[1], -1)  # (B, 7, out_dim)

        # out_dim=1 时自动 squeeze，方便后面 criterion
        if x.size(-1) == 1:
            x = x.squeeze(-1)  # → (B, 7)

        return x


class MainModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, choice):
        super(MainModel, self).__init__()
        self.rt_meas_dim = 46
        # 1. 实例化各个分支模型 (只在初始化时做一次)
        self.branch_gcn = GCN(in_dim, hidden_dim, hidden_dim)
        self.branch_transformer = TransformerModel(in_dim, hidden_dim, hidden_dim)
        self.DCNN = DilatedCNNBranch(self.rt_meas_dim, hidden_dim, hidden_dim)
        self.choice = choice
        # 2. 实例化融合模块 (指定有 2 个分支)
        self.fusion_module = AdaptiveFusionModule(hidden_dim, out_dim, num_branches=3)

    def forward(self, x, edge_index):
        # 3. 分别提取特征 (动态计算，每次 forward 都不同)
        # GCN 需要 edge_index
        if (self.choice != 1):
            f_graph = self.branch_gcn(x, edge_index)

        # Transformer 通常只需要 x (具体看你的 TransformerModel 实现)
        if (self.choice != 2):
            f_temp = self.branch_transformer(x)

        # DilatedCNN
        if (self.choice != 3):
            meas_x = x[..., -self.rt_meas_dim:]
            f_DCNN = self.DCNN(meas_x)

        # 4. 将特征列表传入融合模块
        # 注意：这里传入的是计算好的特征 tensor，而不是原始数据

        if (self.choice == 1):
            output = self.fusion_module([f_temp, f_DCNN])

        elif (self.choice == 2):
            output = self.fusion_module([f_graph, f_DCNN])

        elif (self.choice == 3):
            output = self.fusion_module([f_temp, f_graph])
        else:
            output = self.fusion_module([f_temp, f_graph, f_DCNN])

        return output

