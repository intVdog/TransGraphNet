# import os
# import pandas as pd
# import numpy as np
# import torch
# from sklearn.preprocessing import MinMaxScaler
#
#
# def load_CICIDS(num_benign, num_malic, action_node_idx):
#     csv_file = '../../datasets/CIC-IOT2023.csv'
#     if not os.path.exists(csv_file):
#         raise FileNotFoundError("Dataset file not found")
#     print("Reading dataset...")
#
#     df = pd.read_csv(csv_file, low_memory=False)
#     df.drop_duplicates(inplace=True)
#     x = df.drop('label', axis=1)
#     y = df['label']
#
#     # 转换为大写并清理
#     y = y.astype(str).str.strip().str.upper()
#     print("Unique labels:", y.unique()[:20])
#
#     # ================= 清理异常值 =================
#     x.replace([np.inf, -np.inf], np.nan, inplace=True)
#
#     numeric_cols = x.select_dtypes(include=[np.number]).columns
#     object_cols = x.select_dtypes(include=['object', 'category']).columns
#
#     # 数值列用中位数填充
#     for col in numeric_cols:
#         if x[col].isnull().any():
#             median_val = x[col].median()
#             x[col] = x[col].fillna(median_val)
#
#     # 类别列用众数填充
#     for col in object_cols:
#         if x[col].isnull().any():
#             mode_result = x[col].mode()
#             if not mode_result.empty:
#                 x[col] = x[col].fillna(mode_result[0])
#             else:
#                 x[col].fillna('Unknown', inplace=True)
#
#     # 标签列缺失值用众数填充
#     if y.isnull().any():
#         mode_y = y.mode()[0] if not y.mode().empty else 'Unknown'
#         y.fillna(mode_y, inplace=True)
#
#     # 转为 numpy
#     x_values = x.values
#     y_values = y.values
#     num_action_nodes = len(action_node_idx)
#
#     # ==================== 定义攻击前缀匹配规则 ====================
#     # 注意：CIC-IOT2023 的标签本身就是 'DDOS-XXX', 'MIRAI-XXX' 等
#     attack_prefixes = {
#         'DDOS': 'DDOS',
#         'MIRAI': 'MIRAI',
#         'DOS': 'DOS',
#         'BENIGNTRAFFIC':'BENIGNTRAFFIC',
#         'DNS_SPOOFING': 'DNS_SPOOFING'
#     }
#
#     # 攻击类型顺序（与你要求一致）
#     attack_types = ['DDOS', 'MIRAI','BENIGNTRAFFIC', 'DOS','DNS_SPOOFING']
#     ddos_attack_type = 'DDOS'  # 用于标记正样本
#
#     # 良性池：只包含 BENIGNTRAFFIC
#     benign_pool_idx = np.where(y_values == 'BENIGNTRAFFIC')[0]
#     if len(benign_pool_idx) == 0:
#         raise ValueError("No BENIGNTRAFFIC samples found!")
#
#     # ==================== 构造良性样本集 (X_benign) ====================
#     total_benign_needed = num_benign * num_action_nodes
#     selected_benign_idx = np.random.choice(benign_pool_idx, size=total_benign_needed, replace=True)
#     x_benign = torch.from_numpy(x_values[selected_benign_idx]).float().reshape(num_benign, num_action_nodes,
#                                                                                x_values.shape[1])
#     y_benign = torch.zeros(num_benign, num_action_nodes)
#
#     # ==================== 构造恶意样本集 (X_malic) ====================
#
#     x_malic = torch.zeros(num_malic * num_action_nodes, num_action_nodes, x_values.shape[1])
#     y_malic = torch.zeros(num_malic * num_action_nodes, num_action_nodes)
#
#     # ==================== 注入攻击流量（前缀匹配） ====================
#     for i in range(num_action_nodes):
#         start = i * num_malic
#         end = start + num_malic
#
#         attack_key = attack_types[i % len(attack_types)]
#         prefix = attack_prefixes[attack_key]
#
#         # 使用 startswith 匹配（关键改动！）
#         if attack_key in ['BENIGNTRAFFIC', 'DNS_SPOOFING']:
#             # 精确匹配（因为它们不是前缀，而是完整标签）
#             attack_indices = np.where(y_values == prefix)[0]
#         else:
#             # 前缀匹配：DDOS, MIRAI, DOS
#             attack_indices = np.where([lbl.startswith(prefix) for lbl in y_values])[0]
#
#         if len(attack_indices) == 0:
#             print(
#                 f"Warning: No samples found for attack type '{attack_key}' (prefix/label: '{prefix}'). Background remains.")
#             continue
#
#         print(f"✅ Matched {len(attack_indices)} samples for {attack_key} (using prefix/label: '{prefix}')")
#
#         selected_attack_idx = np.random.choice(attack_indices, size=num_malic, replace=True)
#         x_malic[start:end, i, :] = torch.from_numpy(x_values[selected_attack_idx]).float()
#
#         # 只有 DDOS 类型标记为 1
#         if attack_key == ddos_attack_type:
#             y_malic[start:end, i] = 1
#             print(f"{attack_key}匹配成功")
#
#     print(f"✅ 变异性增强完成: 背景节点现在是随机良性流量。")
#     return x_benign, y_benign, x_malic, y_malic
#
#
# def gene_dataset(action_node_idx, num_nodes, num_benign, num_malic):
#     """Generate Dataset 2"""
#     num_action_nodes = len(action_node_idx)
#     x_benign, y_benign, x_malic, y_malic = load_CICIDS(num_benign, num_malic, action_node_idx)
#     rt_meas_dim = x_benign.shape[2]
#
#     X_benign = torch.zeros(num_benign, num_nodes, rt_meas_dim)
#     X_benign[:, action_node_idx, :] = x_benign
#     Y_benign = y_benign
#
#     X_malic = torch.zeros(num_malic * num_action_nodes, num_nodes, rt_meas_dim)
#     X_malic[:, action_node_idx, :] = x_malic
#     Y_malic = y_malic
#
#     X = torch.cat((X_benign, X_malic), dim=0)
#     Y = torch.cat((Y_benign, Y_malic), dim=0)
#
#     return X, Y


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
    print("Reading and Splitting Raw CIC-IOT2023 Pools...")
    df = pd.read_csv(csv_file, low_memory=False, usecols=range(47),nrows=5000000)

    # 基本清理
    df.columns = df.columns.str.strip()
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).str.strip().str.upper().copy()
    print("真标签：")
    print(df['label'].value_counts())
    target_classes = [
        'DDOS-ICMP_FLOOD',
        'DDOS-UDP_FLOOD',
        'DDOS-TCP_FLOOD',
        'DDOS-PSHACK_FLOOD',
        'DDOS-SYN_FLOOD',
        'DDOS-RSTFINFLOOD',
        'DDOS-SYNONYMOUSIP_FLOOD',
        'DOS-UDP_FLOOD',
        'DOS-TCP_FLOOD',
        'DOS-SYN_FLOOD',
        'BENIGNTRAFFIC',
        'MIRAI-GREETH_FLOOD',
        'MIRAI-UDPPLAIN',
        'MIRAI-GREIP_FLOOD',
        'DDOS-ICMP_FRAGMENTATION',
        'DDOS-UDP_FRAGMENTATION',
        'DDOS-ACK_FRAGMENTATION',
        'DNS_SPOOFING',
    ]
    # print("Original class distribution (top 10):")
    # print(y.value_counts().head(20))
    mask = np.isin(y, target_classes)
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = x.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        x[col] = x[col].fillna(x[col].median())

    # 转换为numpy
    x_values = x.values.astype(np.float32)
    y_values = y.values

    # 按照 8:1:1 划分原始索引池 (确保每一行流量只属于一个池)
    indices = np.arange(len(y_values))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_values)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_values[temp_idx])

    # 返回三个池子的数据
    pools = {
        'train': (x_values[train_idx], y_values[train_idx]),
        'val': (x_values[val_idx], y_values[val_idx]),
        'test': (x_values[test_idx], y_values[test_idx])
    }
    return pools


def load_graphs_from_pool(pool_data, num_benign, num_malic, action_node_idx):
    """第二步：从指定的流量池中抽取流量构造图"""
    x_values, y_values = pool_data
    num_action_nodes = len(action_node_idx)

    # 1. 准备良性池 (无放回打乱)
    benign_pool = np.where(y_values == 'BENIGNTRAFFIC')[0]
    # 计算所需总良性样本
    total_benign_needed = (num_benign * num_action_nodes) + (num_malic * num_action_nodes * num_action_nodes)
    if total_benign_needed > len(benign_pool):
        print(f"正常样本不够")
        all_benign_idx = np.random.choice(benign_pool, size=total_benign_needed, replace=True)
    else:
        all_benign_idx = benign_pool[:total_benign_needed]

    # --- 构造良性图 ---
    x_benign = torch.from_numpy(x_values[all_benign_idx[:num_benign * num_action_nodes]]).reshape(num_benign,
                                                                                                  num_action_nodes, -1).float()
    y_benign = torch.zeros(num_benign, num_action_nodes)

    # --- 构造恶意图背景 ---
    bg_start = num_benign * num_action_nodes
    x_malic = torch.from_numpy(x_values[all_benign_idx[bg_start:]]).reshape(num_malic * num_action_nodes,
                                                                            num_action_nodes, -1).float()
    y_malic = torch.zeros(num_malic * num_action_nodes, num_action_nodes)

    attack_prefixes = {
        'DDOS': [
            'DDOS-ICMP_FLOOD',
            'DDOS-UDP_FLOOD',
            'DDOS-TCP_FLOOD',
            'DDOS-PSHACK_FLOOD',
            'DDOS-SYN_FLOOD',
            'DDOS-RSTFINFLOOD',
            'DDOS-SYNONYMOUSIP_FLOOD',
            'DDOS-ICMP_FRAGMENTATION',
            'DDOS-UDP_FRAGMENTATION',
            'DDOS-ACK_FRAGMENTATION',
            'DDOS-HTTP_FLOOD',
            'DDOS-SLOWLORIS'
        ],
        'MIRAI': [
            'MIRAI-GREETH_FLOOD',
            'MIRAI-UDPPLAIN',
            'MIRAI-GREIP_FLOOD'
        ],
        'DOS': [
            'DOS-UDP_FLOOD',
            'DOS-TCP_FLOOD',
            'DOS-SYN_FLOOD',
            'DOS-HTTP_FLOOD'
        ],
        'BENIGNTRAFFIC': 'BENIGNTRAFFIC',
        'DNS_SPOOFING': 'DNS_SPOOFING'
    }
    # 2. 注入攻击
    attack_types = ['DDOS', 'MIRAI','BENIGNTRAFFIC','DOS','DNS_SPOOFING']

    for i in range(num_malic):
        # --- 【关键改进 1】随机决定注入几个攻击节点 (1 到 全部节点) ---
        num_to_infect = np.random.randint(1, num_action_nodes + 1)
        # --- 【关键改进 2】随机选择注入的具体位置索引 ---
        target_node_indices = np.random.choice(range(num_action_nodes), size=num_to_infect, replace=False)

        for node_idx in target_node_indices:
            target_attack = np.random.choice(attack_types)

            prefix_val = attack_prefixes[target_attack]
            if isinstance(prefix_val, list):
                attack_samples_idx = np.where(np.isin(y_values, prefix_val))[0]
            else:
                attack_samples_idx = np.where(y_values == prefix_val)[0]

            if len(attack_samples_idx) == 0:
                print(f"警告！ No samples found for attack type '{target_attack}'. Background remains.")
                continue

            # 随机抽取该类型的一个样本
            selected_sample_idx = np.random.choice(attack_samples_idx)

            # 注入特征矩阵
            x_malic[i, node_idx, :] = torch.from_numpy(x_values[selected_sample_idx]).float()

            if target_attack =='DDOS':
                y_malic[i, node_idx] = 1.0



    return x_benign, y_benign, x_malic, y_malic


