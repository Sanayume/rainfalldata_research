

import os
import csv

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
FEATURES_LIST_CSV_PATH = os.path.join(FEATURES_DIR, "features_list.csv")
OUTPUT_TEXT_PATH = os.path.join(FEATURES_DIR, "all_feature_descriptions.txt")

def export_all_feature_descriptions():
    print("=== 导出所有特征描述到文本文件 ===")

    # 1. 读取 features_list.csv
    print(f"1. 正在读取 features_list.csv: {FEATURES_LIST_CSV_PATH}")
    if not os.path.exists(FEATURES_LIST_CSV_PATH):
        print(f"错误: features_list.csv 不存在: {FEATURES_LIST_CSV_PATH}")
        return
    
    descriptions_to_export = []
    with open(FEATURES_LIST_CSV_PATH, 'r', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader) # 跳过CSV头
        for row in csv_reader:
            # row[1] 是 feature_description 列
            descriptions_to_export.append(row[1])
    print(f"  已从 CSV 加载 {len(descriptions_to_export)} 个特征描述。")

    # 2. 写入文本文件
    print(f"\n2. 正在写入文本文件: {OUTPUT_TEXT_PATH}")
    try:
        with open(OUTPUT_TEXT_PATH, 'w', encoding='utf-8') as f:
            for desc in descriptions_to_export:
                f.write(desc + '\n')
        print("  all_feature_descriptions.txt 创建成功！")
    except Exception as e:
        print(f"错误: 无法写入文本文件。 {e}")

if __name__ == "__main__":
    export_all_feature_descriptions()

