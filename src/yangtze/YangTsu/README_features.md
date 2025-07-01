# 长江流域独立特征文件库

本目录包含了长江流域多源降雨数据的独立特征文件，这些文件是经过精细化特征工程后生成的，用于机器学习模型训练和预测。本系统将特征生成过程模块化，按类别分别生成不同类型的特征，每个特征单独保存为`.npy`文件。

## 概述

本库中的特征文件主要分为两类：

1.  **已展平特征 (415个)**：这些特征已统一处理为 `(5247240,)` 的一维数组，可以直接加载并用于模型训练。
2.  **未展平特征 (11个)**：这些特征因其固有的高维度空间结构或不完整的时间维度，未被展平，但仍包含重要的空间信息，可用于可视化或深度学习模型。

## 📊 特征概览

### 数据范围
-   **时间范围**: 2016-2020年 (原始1827天，有效预测1797天)
-   **空间范围**: 长江流域 (统一后2920个有效空间点)
-   **降雨产品**: 6个 (CMORPH, CHIRPS, SM2RAIN, IMERG, GSMAP, PERSIANN)

### 特征类别与数量

| 类别 | 数量 (已展平) | 数量 (未展平) | 描述 |
|---|---|---|---|
| **基础特征** | 28 | 0 | 原始降雨数据和目标变量 |
| **多产品协同** | 36 | 0 | 产品间统计关系和一致性 |
| **时序动态** | 38 | 0 | 周期性、季节性、趋势特征 |
| **滞后特征** | 144 | 0 | 1-30天的时间依赖性特征 |
| **空间特征** | 19 | 11 | 空间梯度、邻域、自相关、连通域等 |
| **高级统计** | 89 | 0 | 分位数、极值、异常检测 |
| **交互特征** | 59 | 0 | 特征间交互和组合 |
| **总计** | **415** | **11** | |

## 🗂️ 特征详细描述

本库中的每个 `.npy` 文件都代表一个独立的特征。其详细含义和原始生成方式请参考 `src/yangtze/YangTsu/generate_*.py` 脚本。

### 1. 已展平特征 (415个)

这些特征已统一处理为 `(5247240,)` 的一维数组，可以直接加载并用于模型训练。它们包含了丰富的时序、多产品协同、高级统计以及部分空间关联信息。

**示例 (部分特征，完整列表请参考 `features_list.csv`):**

*   `anomaly_zscore_CHIRPS.npy`: CHIRPS Z-score异常值
*   `coefficient_of_variation.npy`: 变异系数
*   `correlation_CHIRPS_GSMAP.npy`: CHIRPS与GSMAP相关性
*   `cos_day_of_year.npy`: 年内日周期余弦
*   `lag_1_points_GSMAP.npy`: GSMAP滞后1天点数据
*   `multi_product_mean.npy`: 多产品均值
*   `spatial_avg_gradient_magnitude_GSMAP.npy`: GSMAP平均梯度幅度
*   `spatial_cluster_size_GSMAP_threshold_0.1.npy`: GSMAP阈值0.1连通域大小
*   `target_points_valid.npy`: CHM目标点数据

### 2. 未展平特征 (11个)

这些特征因其固有的高维度空间结构或不完整的时间维度，未被展平。它们仍包含重要的空间信息，可用于可视化或深度学习模型。

*   `spatial_autocorrelation_GSMAP.npy`
*   `spatial_autocorrelation_IMERG.npy`
*   `spatial_gradient_magnitude_GSMAP.npy`
*   `spatial_gradient_magnitude_IMERG.npy`
*   `spatial_gradient_magnitude_SM2RAIN.npy`
*   `spatial_neighbor_3x3_mean_GSMAP_samples.npy`
*   `spatial_neighbor_3x3_mean_IMERG_samples.npy`
*   `spatial_neighbor_3x3_mean_SM2RAIN_samples.npy`
*   `spatial_neighbor_5x5_mean_GSMAP_samples.npy`
*   `spatial_neighbor_5x5_mean_IMERG_samples.npy`
*   `spatial_neighbor_5x5_mean_SM2RAIN_samples.npy`

## 💾 文件存储信息

-   **文件格式**: `.npy` (NumPy二进制格式)
-   **数据类型**: `float32` (节省存储空间)
-   **总存储空间**: 预计约 20 GB (415个展平特征文件)

## 🔧 使用方法

### 1. 加载已展平特征

```python
import numpy as np
import os
import pandas as pd

FEATURES_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"

# 加载特征列表 (包含名称、描述和shape)
features_info_df = pd.read_csv(os.path.join(FEATURES_DIR, "features_list.csv"))

# 筛选出已展平的特征
flattened_features_df = features_info_df[features_info_df['shape'] == '(5247240,)']

# 加载所有已展平特征并构建特征矩阵 X
X_features_list = []
feature_names = []
for index, row in flattened_features_df.iterrows():
    fname = row['feature_file_name']
    fpath = os.path.join(FEATURES_DIR, fname)
    data = np.load(fpath)
    X_features_list.append(data)
    feature_names.append(fname.replace(".npy", ""))

X_matrix = np.stack(X_features_list, axis=1) # 堆叠成 (总样本数, 特征数)

# 加载目标变量 Y
Y_vector = np.load(os.path.join(FEATURES_DIR, "target_points_valid.npy"))

print(f"特征矩阵 X 的 shape: {X_matrix.shape}")
print(f"目标向量 Y 的 shape: {Y_vector.shape}")
print(f"特征数量: {len(feature_names)}")
```

### 2. 使用未展平特征

对于未展平的特征，您需要根据其具体 `shape` 和用途进行处理。例如：

*   **可视化**：直接加载 `spatial_gradient_magnitude_GSMAP.npy` (`(106, 144, 256)`) 进行空间模式的可视化。
*   **深度学习**：将其作为卷积神经网络 (CNN) 的输入。
*   **进一步特征工程**：从这些特征中提取新的统计量（如每个时间步的最大梯度、平均连通域大小），然后将其展平并添加到您的特征集中。

## 📚 相关文档

*   **项目主 README**: `../../README.md` (包含项目整体架构、数据预处理、模型迭代等详细信息)
*   **长江流域核心工作区 README**: `../README.md` (包含特征生成脚本、模型训练脚本等详细信息)
*   **特征列表 CSV**: `features_list.csv` (包含所有特征的名称、描述和shape)

---

**生成时间**: 2024年6月30日  
**数据版本**: v1.0 (展平版)  
**作者**: Claude Code & 降雨预测项目组