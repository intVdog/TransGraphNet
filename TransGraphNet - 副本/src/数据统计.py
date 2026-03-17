import pandas as pd
import numpy as np


def compute_dataset_stats(csv_file, dataset_name="Dataset"):
    """
    读取 CIC 系列数据集的 CSV，计算关键统计指标（适合论文使用）
    支持 CICIDS-2017、CSE-CIC-IDS2018、CIC-IoT-2023 的常见标签格式

    返回: dict 包含统计结果，可直接用于打印或保存到表格
    """
    print(f"\n=== 计算 {dataset_name} 统计信息 ===")
    try:
        # ==================== 只改动这里（解决 low_memory + engine 冲突） ====================
        df = pd.read_csv(csv_file, low_memory=False,
                        usecols=range(80))   # 默认C引擎 + 跳过坏行（最快、最稳定）
        print(f"原始读取行数: {len(df):,}")
    except Exception as e:
        print(f"读取失败: {e}")
        return None

    # 1. 列名清理 & 找到标签列
    df.columns = df.columns.str.strip()
    possible_label_cols = ['Label', 'label']   # 同时支持大写和小写（CIC-IoT-2023常用小写）
    label_col = None
    for col in possible_label_cols:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        print("错误：未找到标签列（尝试了 Label / label）")
        return None

    # 2. 标签标准化（忽略大小写、去空格）
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # 3. 基本清理（与你的 prepare_raw_pools 保持一致）
    df_numeric = df.drop(columns=[label_col])
    df_numeric.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_numeric[col] = df_numeric[col].fillna(df_numeric[col].median())

    # 合并回标签
    df_clean = pd.concat([df_numeric, df[label_col]], axis=1)
    df_clean.dropna(inplace=True)  # 极少数极端 NaN 丢弃

    records = len(df_clean)
    features = df_clean.shape[1] - 1  # 减标签列

    # 4. 统计良性 & 攻击
    benign_mask = df_clean[label_col].isin(['benign', 'normal', 'benigntraffic'])
    benign_count = benign_mask.sum()
    malicious_count = records - benign_count
    malicious_ratio = (malicious_count / records * 100) if records > 0 else 0.0

    # 5. 攻击类别数（排除 benign/normal）
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

    # 美观打印
    print("| 指标              | 值              |")
    print("|--------------------|-----------------|")
    print(f"| Records           | {records:,}    |")
    print(f"| Features          | {features}      |")
    print(f"| Attack Classes    | {attack_classes}          |")
    print(f"| Benign            | {benign_count:,}    |")
    print(f"| Malicious Ratio   | {malicious_ratio:.2f}%       |")

    print(f"\n检测到的攻击类别 ({attack_classes} 个)：")
    print(", ".join(attack_labels[:15]) + (" ..." if len(attack_labels) > 15 else ""))

    return stats


# ===================== 使用示例 =====================
if __name__ == '__main__':
    # 替换成你实际的路径
    paths = {
        "CSE-CIC-IDS2018": r"../datasets/CSE-CIC-IDS2018.csv"
    }

    all_stats = []
    for name, path in paths.items():
        stat = compute_dataset_stats(path, name)
        if stat:
            all_stats.append(stat)

    # 如果你想一次性输出 markdown 表格（适合论文）
    if all_stats:
        print("\n=== 数据集汇总（Markdown 表格） ===")
        print("| Dataset            | Records      | Features | Attack Classes | Benign       | Malicious Ratio |")
        print("|--------------------|--------------|----------|----------------|--------------|-----------------|")
        for s in all_stats:
            print(f"| {s['Dataset']:<18} | {s['Records']:>12,} | {s['Features']:>8} | {s['Attack Classes']:>14} | "
                  f"{s['Benign']:>12,} | {s['Malicious Ratio (%)']:>15.2f}% |")