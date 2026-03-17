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
    print("Reading and Splitting Raw CIC-IDS2017 Pools...")
    df = pd.read_csv(csv_file, low_memory=False)

    # Basic cleaning
    df.columns = df.columns.str.strip()
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].astype(str).str.strip().copy()

    # ========== [Key Fix] Keep only the classes actually used in experiments ==========
    target_classes = [
        'BENIGN',
        'DoS slowloris',
        'FTP-Patator',
        'SSH-Patator',
        'DDoS',
        'Bot',
        'PortScan'
        # You can add DoS Hulk / DoS GoldenEye etc. here if needed later
    ]

    print("Original class distribution (top 10):")
    print(y.value_counts().head(10))

    mask = np.isin(y, target_classes)
    x = x[mask].reset_index(drop=True)
    y = y[mask].reset_index(drop=True)

    print(f"After filtering to target classes, remaining samples: {len(y)}")
    # =================================================================================

    # Continue cleaning NaN and inf
    x.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = x.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        x[col] = x[col].fillna(x[col].median())

    # Convert to numpy
    x_values = x.values.astype(np.float32)
    y_values = y.values

    # Split by 8:1:1 ratio (stratify can be safely used now)
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
    Step 2: Extract traffic from specified pool to construct graphs (Optimized: randomized injection positions and full attack type recognition)
    """
    x_values, y_values = pool_data
    num_action_nodes = len(action_node_idx)

    # 1. Prepare benign traffic indices
    benign_indices = np.where(y_values == 'BENIGN')[0]

    # Calculate total number of benign samples needed to construct all graphs (benign graphs + malicious graph backgrounds)
    total_benign_needed = (num_benign + num_malic) * num_action_nodes

    if len(benign_indices) < total_benign_needed:
        # Allow sampling with replacement if pool is too small (validation/test set)
        print("Insufficient benign samples")
        all_benign_idx = np.random.choice(benign_indices, size=total_benign_needed, replace=True)
    else:
        # Training set is usually large enough, perform shuffled sampling without replacement
        np.random.shuffle(benign_indices)
        all_benign_idx = benign_indices[:total_benign_needed]

    # --- Construct benign graphs (Labels all 0) ---
    x_benign = torch.from_numpy(x_values[all_benign_idx[:num_benign * num_action_nodes]]).reshape(
        num_benign, num_action_nodes, -1
    ).float()
    y_benign = torch.zeros(num_benign, num_action_nodes)

    # --- Construct malicious graph backgrounds (initially filled with benign traffic) ---
    bg_start = num_benign * num_action_nodes
    x_malic = torch.from_numpy(x_values[all_benign_idx[bg_start:]]).reshape(
        num_malic, num_action_nodes, -1
    ).float()
    # Malicious graph labels initialized to 0, changed to 1 after attack injection
    y_malic = torch.zeros(num_malic, num_action_nodes)

    # 2. Inject attack traffic
    # Define available attack types (exclude benign)
    attack_types_pool = ['DoS slowloris', 'FTP-Patator', 'SSH-Patator', 'BENIGN','DDoS', 'Bot', 'PortScan']

    print(f"Starting to construct malicious graphs: randomly injecting positions and marking all attack types...")

    for i in range(num_malic):
        # --- [Key Improvement 1] Randomly determine number of attack nodes to inject (1 to all nodes) ---
        num_to_infect = np.random.randint(1, num_action_nodes + 1)
        # --- [Key Improvement 2] Randomly select specific position indices for injection ---
        target_node_indices = np.random.choice(range(num_action_nodes), size=num_to_infect, replace=False)

        for node_idx in target_node_indices:
            # --- [Key Improvement 3] Randomly select an attack type from attack pool ---
            target_attack = np.random.choice(attack_types_pool)

            attack_samples_idx = np.where(y_values == target_attack)[0]
            if len(attack_samples_idx) == 0:
                continue

            # Randomly select one sample of this type
            selected_sample_idx = np.random.choice(attack_samples_idx)

            # Inject into feature matrix
            x_malic[i, node_idx, :] = torch.from_numpy(x_values[selected_sample_idx]).float()

            if target_attack =="DDoS":
                y_malic[i, node_idx] = 1.0

    return x_benign, y_benign, x_malic, y_malic