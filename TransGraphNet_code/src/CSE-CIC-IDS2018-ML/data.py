import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random

# >>>>>>>>>> Fix all random seeds <<<<<<<<<<
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
    # Read with float64 to prevent overflow during reading phase
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

    # --- Complete fix for ValueError ---
    # 1. Handle infinity values
    x.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. Identify numeric columns
    numeric_cols = x.select_dtypes(include=[np.number]).columns

    # 3. Imputation and extreme value truncation (for float32 safe upper limit)
    # float32 max is about 3.4e38, we use 1e30 as safe boundary
    MAX_VAL = 1e30
    for col in numeric_cols:
        median_val = x[col].median()
        x[col] = x[col].fillna(median_val)
        x[col] = x[col].clip(lower=-MAX_VAL, upper=MAX_VAL)

    # 4. Double-check no NaNs before converting to float32
    x_values = x.values.astype(np.float32)
    # Force zero if NaNs still exist (e.g., entire column is NaN)
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
    """Ultra speed optimized version: Pre-mapped indices + batch sampling"""
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

    # Pre-classify all indices (avoid using np.where in loops)
    idx_map = {k: np.where(np.isin(y_values, v))[0] for k, v in attack_prefixes.items()}
    attack_types = list(idx_map.keys())

    # --- 1. Batch construct benign graphs ---
    benign_indices = idx_map['BENIGN']
    total_needed_bg = (num_benign + num_malic) * num_action_nodes

    # Uniformly sample all background traffic indices
    sampled_bg_idx = np.random.choice(benign_indices, size=total_needed_bg, replace=True)

    # Allocate to benign graphs and malicious graph backgrounds
    all_bg_feats = x_values[sampled_bg_idx].reshape(-1, num_action_nodes, feat_dim)
    x_benign = torch.from_numpy(all_bg_feats[:num_benign])
    y_benign = torch.zeros(num_benign, num_action_nodes)

    # --- 2. Construct malicious graphs ---
    x_malic = torch.from_numpy(all_bg_feats[num_benign:])
    y_malic = torch.zeros(num_malic, num_action_nodes)

    # Only loop for malicious injection points (complex logic, keep loop but optimize internal lookup)
    for i in range(num_malic):
        num_to_infect = np.random.randint(1, num_action_nodes + 1)
        target_node_indices = np.random.choice(range(num_action_nodes), size=num_to_infect, replace=False)

        for node_idx in target_node_indices:
            target_type = random.choice(attack_types)
            pool = idx_map[target_type]
            if len(pool) > 0:
                # O(1) random sampling
                s_idx = pool[np.random.randint(len(pool))]
                x_malic[i, node_idx] = torch.from_numpy(x_values[s_idx])
                if target_type == 'DDOS':
                    y_malic[i, node_idx] = 1.0

    return x_benign, y_benign, x_malic, y_malic
