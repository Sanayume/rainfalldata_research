# 长江流域单独特征生成系统

## 概述

本系统将特征生成过程模块化，按类别分别生成不同类型的特征，每个特征单独保存为`.npy`文件。这样设计的优势：

- **模块化**: 可以选择性生成需要的特征类别
- **灵活性**: 可以轻松组合不同特征进行实验  
- **效率**: 避免重复计算，一次生成多次使用
- **可扩展**: 容易添加新的特征类别
- **可维护**: 代码结构清晰，便于调试和优化

## 特征类别

### 1. 基础原始特征 (`generate_basic_features.py`)
- **原始空间数据**: `raw_spatial_{product}*.npy` - 保持真实空间结构
- **原始点数据**: `raw_points_{product}*.npy` - 展平的点数据  
- **目标变量**: `target_spatial*.npy`, `target_points*.npy`

### 2. 多产品协同特征 (`generate_multi_product_features.py`)
- **统计量**: `multi_product_mean.npy`, `multi_product_std.npy` 等
- **一致性**: `product_consistency_ratio.npy`, `product_disagreement.npy`
- **相关性**: `correlation_{prod1}_{prod2}.npy`
- **权重**: `weighted_multi_product_mean.npy`

### 3. 时序动态特征 (`generate_temporal_features.py`)
- **周期性**: `sin_day_of_year.npy`, `cos_day_of_year.npy`
- **季节性**: `season_onehot_{0-3}.npy`, `month_onehot_{1-12}.npy`
- **趋势**: `time_index.npy`, `normalized_time.npy`
- **差分**: `diff_1_multi_product_mean.npy` 等

### 4. 滞后特征 (`generate_lag_features.py`)
- **单产品滞后**: `lag_{days}_points_{product}.npy`
- **统计量滞后**: `lag_{days}_multi_product_{stat}.npy`
- **滞后差值**: `lag_diff_{lag1}_{lag2}_multi_product_mean.npy`
- **组合滞后**: `lag_1to7_mean_{product}.npy`

### 5. 真实空间特征 (`generate_spatial_features.py`)
- **梯度**: `spatial_gradient_magnitude_{product}.npy`
- **邻域**: `spatial_neighbor_{size}_mean_{product}.npy`
- **聚集性**: `spatial_variance_{product}.npy`
- **自相关**: `spatial_autocorrelation_{product}.npy`

### 6. 高级统计特征 (`generate_advanced_features.py`)
- **分位数**: `multi_product_quantile_{percentile}.npy`
- **极值**: `extreme_ratio_above_{threshold}.npy`
- **异常检测**: `anomaly_zscore_{product}.npy`
- **强度分箱**: `intensity_bin_{idx}_{type}.npy`

### 7. 交互特征 (`generate_interaction_features.py`)
- **产品交互**: `interaction_multiply_{prod1}_{prod2}.npy`
- **时序交互**: `interaction_{stat}_sin_day.npy`
- **条件交互**: `interaction_low_intensity_std_cv.npy`
- **复合交互**: `interaction_triple_geometric_mean_GSI.npy`

## 使用方法

### 1. 生成特征

```bash
cd /mnt/f/rainfalldata/src/yangtze/YangTsu

# 生成所有特征（推荐）
python generate_all_features.py --all

# 生成特定类别
python generate_all_features.py basic temporal lag

# 静默模式生成
python generate_all_features.py spatial advanced --quiet

# 检查现有特征
python generate_all_features.py --check

# 查看帮助
python generate_all_features.py --help
```

### 2. 单独运行生成器

```bash
# 只生成基础特征
python generate_basic_features.py

# 只生成空间特征  
python generate_spatial_features.py
```

### 3. 使用特征加载器

```python
from feature_loader import IndividualFeatureLoader

# 初始化加载器
loader = IndividualFeatureLoader()

# 查看概要
loader.print_summary()

# 列出特定类别特征
temporal_features = loader.list_features('temporal')

# 加载单个特征
product_mean = loader.load_feature('multi_product_mean')

# 加载多个特征
features = loader.load_multiple_features([
    'multi_product_mean.npy',
    'lag_1_points_GSMAP.npy', 
    'sin_day_of_year.npy'
])

# 创建特征子集
feature_subset = loader.create_feature_subset(
    include_lag=True,
    include_temporal=True,
    max_features=50
)

# 构建特征矩阵
X, feature_names = loader.build_feature_matrix(feature_subset)
```

## 特征命名规范

### 命名模式
- `{type}_{subtype}_{product/stat}_{modifier}.npy`

### 示例
- `raw_points_GSMAP_valid.npy` - GSMAP原始点数据（有效时间）
- `lag_3_multi_product_mean.npy` - 多产品均值滞后3天
- `spatial_gradient_magnitude_IMERG.npy` - IMERG梯度幅度
- `interaction_multiply_GSMAP_IMERG.npy` - GSMAP与IMERG乘性交互

## 数据维度说明

- `(time,)` - 时间序列，如周期性特征
- `(time, points)` - 时间-点数据，如点位特征
- `(time, lat, lon)` - 时间-空间数据，如空间特征
- `(lat, lon)` - 纯空间数据，如平均梯度
- `(lag_i, lag_j)` - 滞后空间，如自相关

## 计算时间估算

- 基础特征: 2-3分钟
- 多产品协同: 3-5分钟  
- 时序动态: 2-3分钟
- 滞后特征: 3-4分钟
- 空间特征: 5-8分钟（计算量最大）
- 高级统计: 4-6分钟
- 交互特征: 3-5分钟

**总计**: 约25-35分钟生成所有特征

## 存储空间

预计每个特征文件大小：
- 点数据特征: 10-50 MB
- 空间数据特征: 50-200 MB  
- 时间序列特征: 1-10 MB

**总存储空间**: 预计5-15 GB

## 注意事项

1. **内存使用**: 空间特征生成需要较多内存
2. **计算时间**: 空间特征计算时间较长，可能需要耐心等待
3. **依赖关系**: 交互特征依赖基础和时序特征
4. **数据类型**: 所有特征统一保存为float32格式
5. **NaN处理**: 已进行安全的NaN值处理

## 故障排除

### 常见问题

1. **内存不足**
   ```bash
   # 单独运行计算量小的特征
   python generate_temporal_features.py
   python generate_lag_features.py
   ```

2. **计算超时**
   ```bash
   # 跳过计算量大的空间特征
   python generate_all_features.py basic multi_product temporal lag advanced interaction
   ```

3. **特征文件损坏**
   ```bash
   # 检查特征文件
   python generate_all_features.py --check
   
   # 重新生成特定类别
   python generate_basic_features.py
   ```

### 调试模式

```python
# 在Python中测试单个特征生成
import numpy as np
from loaddata import mydata

# 加载数据测试
ALL_DATA = mydata()
X_points, Y_points = ALL_DATA.get_basin_point_data(basin_mask_value=2)
print(f"数据形状: {X_points.shape}, {Y_points.shape}")
```

## 扩展指南

### 添加新特征类别

1. 创建新的生成器脚本
2. 在`generate_all_features.py`中添加配置
3. 更新`feature_loader.py`的分类逻辑
4. 更新此README文档

### 优化计算性能

1. 减少计算的时间步数
2. 限制处理的空间点数
3. 使用更高效的算法
4. 并行化计算过程

这个系统为您的特征工程提供了极大的灵活性，您可以根据具体需求选择和组合不同的特征进行实验！