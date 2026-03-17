import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

# >>>>>>>>>> 固定所有随机种子 <<<<<<<<<<
SEED = 44
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def prepare_raw_pools(csv_file):
    print("Loading CSE-CIC-IDS2018 dataset...")
    # 使用 float64 读取以防止读取阶段溢出
    df = pd.read_csv(csv_file, low_memory=False, on_bad_lines='skip', nrows=5000000)
    df.columns = df.columns.str.strip()

    leakage_columns = ['Timestamp', 'Flow ID', 'Src IP', 'Src Port', 'Dst IP', 'Dst Port', 'Protocol']
    df = df.drop(columns=[col for col in leakage_columns if col in df.columns], errors='ignore')

    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).str.strip().str.upper().copy()

    target_classes = [
        'BENIGN', 'DDOS ATTACK-HOIC', 'DOS ATTACKS-HULK', 'FTP-BRUTEFORCE',
        'SSH-BRUTEFORCE', 'DOS ATTACKS-SLOWHTTPTEST', 'DOS ATTACKS-GOLDENEYE',
        'DOS ATTACKS-SLOWLORIS', 'DDOS ATTACK-LOIC-UDP'
    ]
    mask = y.isin(target_classes)
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    # --- 彻底修复 ValueError ---
    # 1. 处理无穷大
    x.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. 识别数值列
    numeric_cols = x.select_dtypes(include=[np.number]).columns

    # 3. 填充与极端值截断 (针对 float32 的安全上限)
    # float32 最大约 3.4e38, 我们取 1e30 作为安全边界
    MAX_VAL = 1e30
    for col in numeric_cols:
        median_val = x[col].median()
        x[col] = x[col].fillna(median_val)
        x[col] = x[col].clip(lower=-MAX_VAL, upper=MAX_VAL)

    # 4. 转换为 float32 之前再次确认无 NaN
    x_values = x.values.astype(np.float32)
    # 如果此时还有 NaN (比如某一列全是 NaN)，强制归零
    x_values = np.nan_to_num(x_values)
    y_values = y.values

    indices = np.arange(len(y_values))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_values)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_values[temp_idx])

    return {
        'train': (x_values[train_idx], y_values[train_idx]),
        'val': (x_values[val_idx], y_values[val_idx]),
        'test': (x_values[test_idx], y_values[test_idx])
    }


def load_graphs_from_pool(pool_data, num_benign, num_malic, action_node_idx):
    """极致速度优化版：预映射索引 + 批量采样"""
    x_values, y_values = pool_data
    num_action_nodes = len(action_node_idx)
    feat_dim = x_values.shape[1]

    attack_prefixes = {
        'DDOS': ['DDOS ATTACK-HOIC', 'DDOS ATTACK-LOIC-UDP'],
        'DOS': ['DOS ATTACKS-HULK', 'DOS ATTACKS-SLOWHTTPTEST', 'DOS ATTACKS-GOLDENEYE', 'DOS ATTACKS-SLOWLORIS'],
        'BENIGN': ['BENIGN'],
        'FTP-BRUTEFORCE': ['FTP-BRUTEFORCE'],
        'SSH-BRUTEFORCE': ['SSH-BRUTEFORCE']
    }

    # 预先分类所有索引（避免在循环中使用 np.where）
    idx_map = {k: np.where(np.isin(y_values, v))[0] for k, v in attack_prefixes.items()}
    attack_types = list(idx_map.keys())

    # --- 1. 批量构造良性图 ---
    benign_indices = idx_map['BENIGN']
    total_needed_bg = (num_benign + num_malic) * num_action_nodes

    # 统一抽取所有背景流量索引
    sampled_bg_idx = np.random.choice(benign_indices, size=total_needed_bg, replace=True)

    # 分配给良性图和恶意图背景
    all_bg_feats = x_values[sampled_bg_idx].reshape(-1, num_action_nodes, feat_dim)
    x_benign = torch.from_numpy(all_bg_feats[:num_benign])
    y_benign = torch.zeros(num_benign, num_action_nodes)

    # --- 2. 构造恶意图 ---
    x_malic = torch.from_numpy(all_bg_feats[num_benign:])
    y_malic = torch.zeros(num_malic, num_action_nodes)

    # 仅针对恶意注入点进行循环，这部分逻辑复杂，保留循环但优化内部查找
    for i in range(num_malic):
        num_to_infect = np.random.randint(1, num_action_nodes + 1)
        target_node_indices = np.random.choice(range(num_action_nodes), size=num_to_infect, replace=False)

        for node_idx in target_node_indices:
            target_type = random.choice(attack_types)
            pool = idx_map[target_type]
            if len(pool) > 0:
                # O(1) 随机抽取
                s_idx = pool[np.random.randint(len(pool))]
                x_malic[i, node_idx] = torch.from_numpy(x_values[s_idx])
                if target_type == 'DDOS':
                    y_malic[i, node_idx] = 1.0

    return x_benign, y_benign, x_malic, y_malic


# if __name__ == '__main__':
#     pools = prepare_raw_pools('../../datasets/CSE-CIC-IDS2018.csv')
#     scaler = MinMaxScaler()
#     X_train_raw, _ = pools['train']
#     print("Fitting scaler...")
#     scaler.fit(X_train_raw)  # 这里现在是安全的了
#
#
#     # 3. 如果需要应用到所有池，建议定义一个简单的 scale 函数
#     def apply_scale(pool, scaler):
#         x, y = pool
#         # scaler.transform 期望 2D 输入，x_values 已经是 2D 的
#         return scaler.transform(x), y
#
#
#     # 应用归一化
#     # pools['train'] = apply_scale(pools['train'], scaler)
#     # pools['val'] = apply_scale(pools['val'], scaler)
#     # pools['test'] = apply_scale(pools['test'], scaler)
#
#     print("Data preparation complete and cleaned.")
#
# # >>> 示例用法及测试 <<<
# # pools = prepare_raw_pools('data.csv')
# # scaler = MinMaxScaler()
# # X_train_raw, _ = pools['train']
# # # 展平进行 fit
# # scaler.fit(X_train_raw)