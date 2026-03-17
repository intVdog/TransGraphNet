import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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

def prepare_raw_pools(csv_file):
    """第一步：读取原始CSV并预先划分为三个互不相交的流量池"""
    print("Reading and Splitting Raw CIC-IDS2017 Pools...")
    df = pd.read_csv(csv_file, low_memory=False)

    # 基本清理
    df.columns = df.columns.str.strip()
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).str.strip().copy()

    # ========== 【关键修复】只保留实验实际用到的类别 ==========
    target_classes = [
        'BENIGN',
        'DoS slowloris',
        'FTP-Patator',
        'SSH-Patator',
        'DDoS',
        'Bot',
        'PortScan'
        # 如果你后面想加 DoS Hulk / DoS GoldenEye 等，可以在这里加
    ]

    print("Original class distribution (top 10):")
    print(y.value_counts().head(10))

    mask = np.isin(y, target_classes)
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print(f"After filtering to target classes, remaining samples: {len(y)}")
    # ==========================================================

    # 继续清理 NaN 和 inf
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = x.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        x[col] = x[col].fillna(x[col].median())

    # 转换为 numpy
    x_values = x.values.astype(np.float32)
    y_values = y.values

    # 按照 8:1:1 划分（现在可以放心使用 stratify）
    indices = np.arange(len(y_values))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_values)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_values[temp_idx])

    pools = {
        'train': (x_values[train_idx], y_values[train_idx]),
        'val': (x_values[val_idx], y_values[val_idx]),
        'test': (x_values[test_idx], y_values[test_idx])
    }
    return pools


import numpy as np
import torch


def load_graphs_from_pool(pool_data, num_benign, num_malic, action_node_idx):
    """
    第二步：从指定的流量池中抽取流量构造图 (已优化：随机化注入位置与全类型攻击识别)
    """
    x_values, y_values = pool_data
    num_action_nodes = len(action_node_idx)

    # 1. 准备良性流量索引
    benign_indices = np.where(y_values == 'BENIGN')[0]

    # 计算构造所有图（良性图 + 恶意图背景）所需的良性样本总数
    total_benign_needed = (num_benign + num_malic) * num_action_nodes

    if len(benign_indices) < total_benign_needed:
        # 如果池子太小（验证集/测试集），允许有放回抽样
        print("正常样本不够")
        all_benign_idx = np.random.choice(benign_indices, size=total_benign_needed, replace=True)
    else:
        # 训练集通常够大，进行无放回打乱抽取
        np.random.shuffle(benign_indices)
        all_benign_idx = benign_indices[:total_benign_needed]

    # --- 构造良性图 (Label 全为 0) ---
    x_benign = torch.from_numpy(x_values[all_benign_idx[:num_benign * num_action_nodes]]).reshape(
        num_benign, num_action_nodes, -1
    ).float()
    y_benign = torch.zeros(num_benign, num_action_nodes)

    # --- 构造恶意图背景 (初始填充良性流量) ---
    bg_start = num_benign * num_action_nodes
    x_malic = torch.from_numpy(x_values[all_benign_idx[bg_start:]]).reshape(
        num_malic, num_action_nodes, -1
    ).float()
    # 恶意图的标签初始化为 0，后续注入攻击后再改为 1
    y_malic = torch.zeros(num_malic, num_action_nodes)

    # 2. 注入攻击流量
    # 定义可用的攻击类型（排除良性）
    attack_types_pool = ['DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'BENIGN','DDoS', 'Bot', 'PortScan']

    print(f"开始构造恶意图：随机注入位置并标记全攻击类型...")

    for i in range(num_malic):
        # --- 【关键改进 1】随机决定注入几个攻击节点 (1 到 全部节点) ---
        num_to_infect = np.random.randint(1, num_action_nodes + 1)
        # --- 【关键改进 2】随机选择注入的具体位置索引 ---
        target_node_indices = np.random.choice(range(num_action_nodes), size=num_to_infect, replace=False)

        for node_idx in target_node_indices:
            # --- 【关键改进 3】从攻击池中随机选一种攻击 ---
            target_attack = np.random.choice(attack_types_pool)

            attack_samples_idx = np.where(y_values == target_attack)[0]
            if len(attack_samples_idx) == 0:
                continue

            # 随机抽取该类型的一个样本
            selected_sample_idx = np.random.choice(attack_samples_idx)

            # 注入特征矩阵
            x_malic[i, node_idx, :] = torch.from_numpy(x_values[selected_sample_idx]).float()

            if target_attack =="DDoS":
                y_malic[i, node_idx] = 1.0

    return x_benign, y_benign, x_malic, y_malic