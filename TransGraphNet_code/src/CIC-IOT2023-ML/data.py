import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
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

def prepare_raw_pools(csv_file):
    """Step 1: Read raw CSV and pre-split into three mutually exclusive traffic pools"""
    print("Reading and Splitting Raw CIC-IOT2023 Pools...")
    df = pd.read_csv(csv_file, low_memory=False, usecols=range(47),nrows=5000000)

    # Basic cleaning
    df.columns = df.columns.str.strip()
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).str.strip().str.upper().copy()
    print("True labels:")
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

    # Convert to numpy
    x_values = x.values.astype(np.float32)
    y_values = y.values

    # Split original index pool by 8:1:1 ratio (ensure each traffic row belongs to only one pool)
    indices = np.arange(len(y_values))
    train_idx, temp_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_values)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y_values[temp_idx])

    # Return data for three pools
    pools = {
        'train': (x_values[train_idx], y_values[train_idx]),
        'val': (x_values[val_idx], y_values[val_idx]),
        'test': (x_values[test_idx], y_values[test_idx])
    }
    return pools


def load_graphs_from_pool(pool_data, num_benign, num_malic, action_node_idx):
    """Step 2: Extract traffic from specified pool to construct graphs"""
    x_values, y_values = pool_data
    num_action_nodes = len(action_node_idx)

    # 1. Prepare benign pool (shuffled without replacement)
    benign_pool = np.where(y_values == 'BENIGNTRAFFIC')[0]
    # Calculate total benign samples needed
    total_benign_needed = (num_benign * num_action_nodes) + (num_malic * num_action_nodes * num_action_nodes)
    if total_benign_needed > len(benign_pool):
        print(f"Insufficient benign samples")
        all_benign_idx = np.random.choice(benign_pool, size=total_benign_needed, replace=True)
    else:
        all_benign_idx = benign_pool[:total_benign_needed]

    # --- Construct benign graphs ---
    x_benign = torch.from_numpy(x_values[all_benign_idx[:num_benign * num_action_nodes]]).reshape(num_benign,
                                                                                                  num_action_nodes, -1).float()
    y_benign = torch.zeros(num_benign, num_action_nodes)

    # --- Construct malicious graph backgrounds ---
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
    # 2. Inject attacks
    attack_types = ['DDOS', 'MIRAI','BENIGNTRAFFIC','DOS','DNS_SPOOFING']

    for i in range(num_malic):
        # --- [Key Improvement 1] Randomly determine number of attack nodes to inject (1 to all nodes) ---
        num_to_infect = np.random.randint(1, num_action_nodes + 1)
        # --- [Key Improvement 2] Randomly select specific position indices for injection ---
        target_node_indices = np.random.choice(range(num_action_nodes), size=num_to_infect, replace=False)

        for node_idx in target_node_indices:
            target_attack = np.random.choice(attack_types)

            prefix_val = attack_prefixes[target_attack]
            if isinstance(prefix_val, list):
                attack_samples_idx = np.where(np.isin(y_values, prefix_val))[0]
            else:
                attack_samples_idx = np.where(y_values == prefix_val)[0]

            if len(attack_samples_idx) == 0:
                print(f"Warning! No samples found for attack type '{target_attack}'. Background remains unchanged.")
                continue

            # Randomly select one sample of this type
            selected_sample_idx = np.random.choice(attack_samples_idx)

            # Inject into feature matrix
            x_malic[i, node_idx, :] = torch.from_numpy(x_values[selected_sample_idx]).float()

            if target_attack =='DDOS':
                y_malic[i, node_idx] = 1.0

    return x_benign, y_benign, x_malic, y_malic