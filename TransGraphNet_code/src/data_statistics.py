import pandas as pd
import numpy as np


def compute_dataset_stats(csv_file, dataset_name="Dataset"):
    """
    Read CSV files of CIC series datasets and calculate key statistical metrics (suitable for paper use)
    Supports common label formats of CICIDS-2017, CSE-CIC-IDS2018, and CIC-IoT-2023

    Returns: dict containing statistical results, which can be directly used for printing or saving to tables
    """
    print(f"\n=== Calculating {dataset_name} Statistics ===")
    try:
        # ==================== Modify only here (resolve low_memory + engine conflict) ====================
        df = pd.read_csv(csv_file, low_memory=False,
                        usecols=range(80))   # Default C engine + skip bad lines (fastest and most stable)
        print(f"Original number of rows read: {len(df):,}")
    except Exception as e:
        print(f"Reading failed: {e}")
        return None

    # 1. Column name cleaning & find label column
    df.columns = df.columns.str.strip()
    possible_label_cols = ['Label', 'label']   # Support both uppercase and lowercase (CIC-IoT-2023 commonly uses lowercase)
    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print("Error: Label column not found (tried Label / label)")
        return None

    # 2. Label standardization (ignore case, remove spaces)
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # 3. Basic cleaning (consistent with your prepare_raw_pools)
    df_numeric = df.drop(columns=[label_col])
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    # Merge back labels
    df_clean = pd.concat([df_numeric, df[label_col]], axis=1)
    df_clean.dropna(inplace=True)  # Discard very few extreme NaNs

    records = len(df_clean)
    features = df_clean.shape[1] - 1  # Minus label column

    # 4. Statistics for benign & malicious samples
    benign_mask = df_clean[label_col].isin(['benign', 'normal', 'benigntraffic'])
    benign_count = benign_mask.sum()
    malicious_count = records - benign_count
    malicious_ratio = (malicious_count / records * 100) if records > 0 else 0.0

    # 5. Number of attack categories (exclude benign/normal)
    all_labels = df_clean[label_col].unique()
    attack_labels = [lbl for lbl in all_labels if lbl not in ['benign', 'normal', 'benigntraffic']]
    attack_classes = len(attack_labels)

    stats = {
        'Dataset': dataset_name,
        'Records': records,
        'Features': features,
        'Attack Classes': attack_classes,
        'Benign': benign_count,
        'Malicious Ratio (%)': round(malicious_ratio, 2)
    }

    # Pretty print
    print("| Metric            | Value          |")
    print("|--------------------|-----------------|")
    print(f"| Records           | {records:,}    |")
    print(f"| Features          | {features}      |")
    print(f"| Attack Classes    | {attack_classes}          |")
    print(f"| Benign            | {benign_count:,}    |")
    print(f"| Malicious Ratio   | {malicious_ratio:.2f}%       |")

    print(f"\nDetected attack categories ({attack_classes} types)：")
    print(", ".join(attack_labels[:15]) + (" ..." if len(attack_labels) > 15 else ""))

    return stats


# ===================== Usage Example =====================
if __name__ == '__main__':
    # Replace with your actual paths
    paths = {
        "CSE-CIC-IDS2018": r"../datasets/CSE-CIC-IDS2018.csv"
    }

    all_stats = []
    for name, path in paths.items():
        stat = compute_dataset_stats(path, name)
        if stat:
            all_stats.append(stat)

    # If you want to output markdown table at once (suitable for paper)
    if all_stats:
        print("\n=== Dataset Summary (Markdown Table) ===")
        print("| Dataset            | Records      | Features | Attack Classes | Benign       | Malicious Ratio |")
        print("|--------------------|--------------|----------|----------------|--------------|-----------------|")
        for s in all_stats:
            print(f"| {s['Dataset']:<18} | {s['Records']:>12,} | {s['Features']:>8} | {s['Attack Classes']:>14} | "
                  f"{s['Benign']:>12,} | {s['Malicious Ratio (%)']:>15.2f}% |")