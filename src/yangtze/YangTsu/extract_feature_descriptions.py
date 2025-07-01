

import os
import re
import json

FEATURES_SOURCE_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
SCRIPT_DIR = "/mnt/f/rainfalldata/src/yangtze/YangTsu"

# 关注的特征生成脚本
GENERATE_SCRIPTS = [
    "generate_basic_features.py",
    "generate_temporal_features.py",
    "generate_multi_product_features.py",
    "generate_lag_features.py",
    "generate_spatial_features.py",
    "generate_advanced_features.py",
    "generate_interaction_features.py",
]

def extract_descriptions_from_script(script_path):
    descriptions = {}
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 正则表达式匹配 save_feature(..., "feature_name", "description")
    # 捕获 feature_name 和 description
    # 注意：description 可能包含括号，所以需要更灵活的匹配
    pattern = re.compile(r'save_feature\([^,]+,\s*f?"([^"]+)"\s*,\s*f?"([^"]+)"\s*\)')
    
    for match in pattern.finditer(content):
        feature_name = match.group(1)
        description = match.group(2)
        descriptions[feature_name + '.npy'] = description
        
    return descriptions

def main():
    all_feature_descriptions = {}
    print("=== 提取特征描述 ===")
    for script_name in GENERATE_SCRIPTS:
        script_path = os.path.join(SCRIPT_DIR, script_name)
        if os.path.exists(script_path):
            print(f"处理脚本: {script_name}")
            descriptions = extract_descriptions_from_script(script_path)
            all_feature_descriptions.update(descriptions)
        else:
            print(f"警告: 脚本文件不存在: {script_name}")
    
    print(f"\n总共提取到 {len(all_feature_descriptions)} 个特征描述。")
    
    # 保存到 JSON 文件，方便后续使用
    output_json_path = os.path.join(SCRIPT_DIR, "feature_descriptions.json")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(all_feature_descriptions, f, indent=2, ensure_ascii=False)
    print(f"特征描述已保存到: {output_json_path}")

if __name__ == "__main__":
    main()

