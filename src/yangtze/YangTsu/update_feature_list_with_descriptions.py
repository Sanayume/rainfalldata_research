
import os
import json

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"
SCRIPT_DIR = "/mnt/f/rainfalldata/src/yangtze/YangTsu"
FEATURE_LIST_PATH = os.path.join(FEATURES_DIR, "feature_list.txt")
DESCRIPTIONS_JSON_PATH = os.path.join(SCRIPT_DIR, "feature_descriptions.json")

def update_feature_list_with_descriptions():
    print("=== 更新 feature_list.txt 添加描述 ===")

    # 1. 加载特征描述
    print(f"1. 正在加载特征描述: {DESCRIPTIONS_JSON_PATH}")
    if not os.path.exists(DESCRIPTIONS_JSON_PATH):
        print(f"错误: 描述文件不存在: {DESCRIPTIONS_JSON_PATH}")
        return
    with open(DESCRIPTIONS_JSON_PATH, 'r', encoding='utf-8') as f:
        feature_descriptions = json.load(f)
    print(f"  已加载 {len(feature_descriptions)} 个特征描述。")

    # 2. 读取现有的 feature_list.txt
    print(f"\n2. 正在读取现有的 feature_list.txt: {FEATURE_LIST_PATH}")
    if not os.path.exists(FEATURE_LIST_PATH):
        print(f"错误: feature_list.txt 不存在: {FEATURE_LIST_PATH}")
        return
    with open(FEATURE_LIST_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3. 构建新的内容
    new_lines = []
    for line in lines:
        original_line = line.strip()
        # 尝试从行中提取特征文件名 (不含 .npy)
        match = None
        # 匹配以 .npy 结尾的文件名，并捕获前面的部分
        if ".npy" in original_line:
            parts = original_line.split(".npy")
            feature_name_with_npy = parts[0] + ".npy"
            
            if feature_name_with_npy in feature_descriptions:
                description = feature_descriptions[feature_name_with_npy]
                # 找到 shape 部分
                shape_match = re.search(r'\s*\([^)]+\)$|\s*\([^)]+\)\s*$', original_line)
                shape_str = "" # 默认没有 shape
                if shape_match:
                    shape_str = shape_match.group(0).strip() # 提取 shape 部分
                    # 移除原始行中的 shape 部分，以便插入描述
                    original_line_without_shape = original_line[:shape_match.start()].strip()
                else:
                    original_line_without_shape = original_line.strip()

                # 重新构建行: 文件名 + 描述 + shape
                # 尝试保持对齐，但由于描述长度不一，可能无法完美对齐
                new_line = f"{original_line_without_shape.ljust(60)} # {description} {shape_str}"
                new_lines.append(new_line + "\n")
            else:
                # 如果没有找到描述，保持原样
                new_lines.append(line) # 保持原始换行符
        else:
            # 非特征行，保持原样
            new_lines.append(line)

    # 4. 写回更新后的 feature_list.txt
    print(f"\n3. 正在写回更新后的 feature_list.txt: {FEATURE_LIST_PATH}")
    try:
        with open(FEATURE_LIST_PATH, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print("  feature_list.txt 更新成功！")
    except Exception as e:
        print(f"错误: 无法写入文件。 {e}")

if __name__ == "__main__":
    # 导入 re 模块
    import re
    update_feature_list_with_descriptions()
