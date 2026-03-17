import pandas as pd

# ==================== Configuration Area ====================
# Please replace the three paths below with your actual local CSV file paths
file_paths = {
    "CICIDS-2017": r"C:\Datasets\CICIDS-2017.csv",  # Replace with your file path
    "CSE-CIC-IDS2018": r"C:\Datasets\CSE-CIC-IDS2018.csv",  # Replace with your file path
    "CIC-IoT-2023": r"C:\Datasets\CIC-IoT-2023.csv"  # Replace with your file path
}

# Possible names of the label column (the program will match automatically)
possible_label_cols = ["Label", "label", " Label", "class"]
# ============================================================

print("| Dataset          | Records      | Features | Attack Classes | Benign       | Malicious Ratio |")
print("|------------------|--------------|----------|----------------|--------------|-----------------|")

for dataset_name, path in file_paths.items():
    try:
        # Read CSV (add low_memory=False for large files)
        df = pd.read_csv(path, low_memory=False)

        # Automatically find the label column
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            print(f"| {dataset_name:<16} | ERROR: Label column not found |")
            continue

        # Basic statistics
        records = len(df)
        features = df.shape[1] - 1  # Subtract the label column

        # Unify labels to lowercase (ignore case and spaces)
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()

        benign_count = (df[label_col] == "benign").sum()
        malicious_count = records - benign_count
        malicious_ratio = (malicious_count / records * 100) if records > 0 else 0

        # Number of attack classes (excluding Benign)
        unique_labels = df[label_col].unique()
        attack_classes = len(unique_labels) - 1 if "benign" in unique_labels else len(unique_labels)

        # Output a table row
        print(f"| {dataset_name:<16} | {records:>12,} | {features:>8} | {attack_classes:>14} | "
              f"{benign_count:>12,} | {malicious_ratio:>13.2f}% |")

    except FileNotFoundError:
        print(f"| {dataset_name:<16} | File not found, please check the path |")
    except Exception as e:
        print(f"| {dataset_name:<16} | Read error: {str(e)[:50]}... |")