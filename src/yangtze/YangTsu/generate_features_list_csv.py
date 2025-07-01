import numpy as np
import os
import json
import csv

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
DESCRIPTIONS_JSON_PATH = "/mnt/f/rainfalldata/src/yangtze/YangTsu/feature_descriptions.json"
OUTPUT_CSV_PATH = os.path.join(FEATURES_DIR, "features_list.csv")

def generate_features_list_csv():
    print("=== 生成 features_list.csv (包含所有特征) ===")

    # 1. 加载特征描述
    print(f"1. 正在加载特征描述: {DESCRIPTIONS_JSON_PATH}")
    if not os.path.exists(DESCRIPTIONS_JSON_PATH):
        print(f"错误: 描述文件不存在: {DESCRIPTIONS_JSON_PATH}")
        return
    with open(DESCRIPTIONS_JSON_PATH, 'r', encoding='utf-8') as f:
        feature_descriptions = json.load(f)
    print(f"  已加载 {len(feature_descriptions)} 个特征描述。")

    # 2. 准备 CSV 数据
    csv_data = []
    csv_data.append(["feature_file_name", "feature_description", "shape"]) # CSV header

    print("\n2. 正在遍历所有特征文件并收集信息...")
    processed_count = 0
    skipped_count = 0
    for i, fname in enumerate(sorted(os.listdir(FEATURES_DIR))):
        if not fname.endswith('.npy'):
            continue
        
        filepath = os.path.join(FEATURES_DIR, fname)
        
        try:
            data = np.load(filepath)
            current_shape = data.shape
            
            description = feature_descriptions.get(fname, "N/A") # Get description, default to N/A
            csv_data.append([fname, description, str(current_shape)])
            processed_count += 1

        except Exception as e:
            print(f"  错误: 加载文件 {fname} 时出错: {e}。跳过。")
            skipped_count += 1

    print(f"\n3. 成功收集了 {processed_count} 个特征的信息。")
    if skipped_count > 0:
        print(f"  跳过了 {skipped_count} 个文件，因为加载失败。")

    # 4. 写入 CSV 文件
    print(f"\n4. 正在写入 CSV 文件: {OUTPUT_CSV_PATH}")
    try:
        with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(csv_data)
        print("  features_list.csv 创建成功！")
    except Exception as e:
        print(f"错误: 无法写入 CSV 文件。 {e}")

if __name__ == "__main__":
    generate_features_list_csv()