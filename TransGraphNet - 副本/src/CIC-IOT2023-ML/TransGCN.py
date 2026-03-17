import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch.nn as nn
import torch.nn.functional as F
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

class AdaptiveFusionModule(nn.Module):
    """
    基于注意力的门控融合模块
    输入: 一个包含多个分支特征的列表 [f_temp, f_graph, ...]
    输出: 融合后的分类结果
    """

    def __init__(self, hidden_dim, out_dim, num_branches=3):
        super(AdaptiveFusionModule, self).__init__()
        self.num_branches = num_branches
        self.hidden_dim = hidden_dim

        # 1. 评分网络：为每个分支的特征打分
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # 输出标量分数
        )

        # 2. 特征变换层：对齐不同分支的特征空间 (可选，但推荐)
        # 为每个分支创建一个独立的 Linear 层
        self.transform_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_branches)
        ])

        # 3. 最终分类器
        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, features_list):
        """
        参数:
            features_list: 列表，包含 [f_temp, f_graph]
                           形状可能是 (Nodes, Dim) 或 (1, Nodes, Dim) 或 (Batch, Nodes, Dim)
        返回:
            output: 分类结果
        """
        # print("=== Fusion Module Input Shapes ===")
        # for idx, f in enumerate(features_list):
        #     print(f"Feature {idx}: {f.shape}")
        # print("=================================")
        transformed_features = []

        # --- 步骤 0: 维度对齐 (关键修复) ---
        # 确保所有特征都有 3 个维度: (Batch, Nodes, Dim)
        # 如果是单个样本 (Nodes, Dim)，unsqueeze(0) 变成 (1, Nodes, Dim)
        normalized_features = []
        for f in features_list:
            if f.dim() == 1:  # ← 处理 (26,) 这种情况
                f = f.unsqueeze(-1)  # 变成 (26, 1)，后续会自动对齐

            if f.dim() == 2:  # (26, D) → (1, 26, D)
                f = f.unsqueeze(0)

            elif f.dim() == 3:
                pass  # 移除 squeeze 操作，以保持一致的 batch 维度（即使 batch=1）

            else:
                raise ValueError(f"不支持的特征维度: {f.dim()}, 形状: {f.shape}")

            normalized_features.append(f)

        batch_size = normalized_features[0].size(0)
        num_nodes = normalized_features[0].size(1)

        # --- 步骤 1: 特征变换 ---
        # 注意：transform_layers 是 Linear 层，通常作用于最后一维 (Dim)
        # 为了应用 Linear，我们需要将 (Batch, Nodes, Dim) 重塑为 (Batch*Nodes, Dim)
        # 或者使用 torch.nn.functional.linear 配合广播，但重塑最稳妥

        for i, f in enumerate(normalized_features):
            # f shape: (Batch, Nodes, Dim)
            # 重塑以便通过 Linear 层: (Batch * Nodes, Dim)
            f_reshaped = f.view(-1, self.hidden_dim)

            # 通过变换层
            h_reshaped = self.transform_layers[i](f_reshaped)

            # 恢复形状: (Batch, Nodes, Dim)
            h = h_reshaped.view(batch_size, num_nodes, self.hidden_dim)
            transformed_features.append(h)

        # 堆叠: [Batch, Num_Branches, Nodes, Dim]  <-- 注意这里多了 Nodes 维度
        # 之前的代码是 stack dim=1 得到 [Batch, Branches, Dim]，那是针对全局池化后的特征
        # 如果你的任务是节点级分类 (Node Classification)，我们需要在每个节点上计算注意力
        stacked_features = torch.stack(transformed_features, dim=1)
        # 现在形状: (Batch, 2, Nodes, Dim)

        # --- 步骤 2: 计算注意力分数 (节点级注意力) ---
        # 我们需要对每个节点的每个分支打分
        # 重塑: (Batch * Nodes, Num_Branches, Dim) -> 方便计算?
        # 或者更简单：重塑为 (Batch * Nodes * Num_Branches, Dim) 然后打分

        b, branches, n, d = stacked_features.shape

        # 重塑为 (Batch * Nodes * Branches, Dim)
        reshaped_for_score = stacked_features.view(-1, d)

        # 计算分数: (Batch * Nodes * Branches, 1)
        scores = self.score_net(reshaped_for_score)

        # 恢复形状: (Batch, Nodes, Branches, 1) -> 然后调整维度顺序以便 Softmax
        scores = scores.view(b, n, branches, 1)
        # 转置为 (Batch, Nodes, Branches, 1) 以便在 Branches 维度做 Softmax
        # 上面的 view 已经是 (B, N, Br, 1)，直接在 dim=2 (Branches) 做 softmax
        attention_weights = F.softmax(scores, dim=2)
        # 形状: (Batch, Nodes, Branches, 1)

        # --- 步骤 3: 加权求和 ---
        # stacked_features: (Batch, Branches, Nodes, Dim)
        # attention_weights: (Batch, Nodes, Branches, 1) -> 需要调整为 (Batch, Branches, Nodes, 1) 以匹配

        # 调整权重维度以匹配 stacked_features (B, Br, N, D)
        # 当前 weights: (B, N, Br, 1) -> permute to (B, Br, N, 1)
        weights_permuted = attention_weights.permute(0, 2, 1, 3)

        # 加权: (B, Br, N, 1) * (B, Br, N, D) -> (B, Br, N, D)
        weighted_features = weights_permuted * stacked_features

        # 在 Branches 维度 (dim=1) 求和 -> (Batch, Nodes, Dim)
        fused_feature = torch.sum(weighted_features, dim=1)

        # --- 步骤 4: 分类 (节点级) ---
        # fused_feature: (Batch, Nodes, Dim)
        # 重塑为 (Batch * Nodes, Dim) 通过 classifier
        fused_reshaped = fused_feature.view(-1, self.hidden_dim)
        logits_reshaped = self.classifier(fused_reshaped)  # (Batch * Nodes, Out_Dim)

        # 恢复形状: (Batch, Nodes, Out_Dim)
        output = logits_reshaped.view(batch_size, num_nodes, -1)

        if output.size(-1) == 1:  # 处理 out_dim=1 的情况
            output = output.squeeze(-1)  # (B, N, 1) -> (B, N)
        if output.dim() == 2 and output.size(0) == 1:
            output = output.squeeze(0)  # (1, N) -> (N,)
        # 如果输入是单个样本且没有 batch 维 (原始逻辑是逐个样本处理)，我们可以 squeeze 掉 batch 维
        # 但为了训练循环统一，通常保留 (1, Nodes, Out_Dim) 让上层去 squeeze
        return output