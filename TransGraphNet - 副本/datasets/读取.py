import pandas as pd

# ==================== 配置区 ====================
# 请把下面三个路径替换成你本地实际的CSV文件路径
file_paths = {
    "CICIDS-2017": r"C:\Datasets\CICIDS-2017.csv",  # 替换成你的文件路径
    "CSE-CIC-IDS2018": r"C:\Datasets\CSE-CIC-IDS2018.csv",  # 替换成你的文件路径
    "CIC-IoT-2023": r"C:\Datasets\CIC-IoT-2023.csv"  # 替换成你的文件路径
}

# 标签列可能的名字（程序会自动匹配）
possible_label_cols = ["Label", "label", " Label", "class"]
# ===============================================

print("| Dataset          | Records      | Features | Attack Classes | Benign       | Malicious Ratio |")
print("|------------------|--------------|----------|----------------|--------------|-----------------|")

for dataset_name, path in file_paths.items():
    try:
        # 读取CSV（大文件建议加上 low_memory=False）
        df = pd.read_csv(path, low_memory=False)

        # 自动找到标签列
        label_col = None
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        if label_col is None:
            print(f"| {dataset_name:<16} | ERROR: 未找到标签列 |")
            continue

        # 基础统计
        records = len(df)
        features = df.shape[1] - 1  # 减去标签列

        # 统一标签为小写（忽略大小写和空格）
        df[label_col] = df[label_col].astype(str).str.strip().str.lower()

        benign_count = (df[label_col] == "benign").sum()
        malicious_count = records - benign_count
        malicious_ratio = (malicious_count / records * 100) if records > 0 else 0

        # 攻击类别数（不含 Benign）
        unique_labels = df[label_col].unique()
        attack_classes = len(unique_labels) - 1 if "benign" in unique_labels else len(unique_labels)

        # 输出一行表格
        print(f"| {dataset_name:<16} | {records:>12,} | {features:>8} | {attack_classes:>14} | "
              f"{benign_count:>12,} | {malicious_ratio:>13.2f}% |")

    except FileNotFoundError:
        print(f"| {dataset_name:<16} | 文件不存在，请检查路径 |")
    except Exception as e:
        print(f"| {dataset_name:<16} | 读取错误: {str(e)[:50]}... |")