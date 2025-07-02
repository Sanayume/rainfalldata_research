
# 降水数据分析工具函数参考文档

本文档详细介绍了降水数据分析工具包中各个模块的功能和用法。工具包主要包含三个辅助模块：字体处理、GPU加速和数据可视化。这些模块旨在解决中文显示问题、提高大规模数据处理效率，以及提供专业的数据可视化功能。

## 目录

- [1. 字体处理模块 (font_helper.py)](#1-字体处理模块-font_helperpy)
- [2. GPU工具模块 (gpu_utils.py)](#2-gpu工具模块-gpu_utilspy)
- [3. 可视化辅助模块 (visualization_helper.py)](#3-可视化辅助模块-visualization_helperpy)
- [4. 使用示例](#4-使用示例)

## 1. 字体处理模块 (font_helper.py)

该模块用于解决matplotlib等可视化库中文显示乱码的问题。

### 主要函数

#### `get_system_fonts()`
获取系统中安装的所有字体。
- **返回值**：包含(字体名称，字体路径)的列表。

#### `find_chinese_fonts()`
查找系统中可能支持中文的字体。
- **返回值**：支持中文的字体名称和路径的列表。

#### `setup_chinese_font(font_name=None, test=True, save_path=None)`
配置matplotlib以正确显示中文。
- **参数**:
  - `font_name`：字体名称，如果为None会自动查找。
  - `test`：是否测试字体是否正确显示中文。
  - `save_path`：测试图像的保存路径。
- **返回值**：成功设置的字体名称。

#### `list_available_chinese_fonts()`
列出系统中所有可用的中文字体。
- **返回值**：中文字体名称列表。

#### `create_font_config_file(font_name=None)`
创建matplotlib字体配置文件，使中文显示配置持久化。
- **参数**:
  - `font_name`：字体名称，如为None则自动查找。
- **返回值**：布尔值，表示配置文件是否创建成功。

#### `test_all_chinese_fonts(output_dir='f:/rainfalldata/figures/font_tests')`
测试所有中文字体并生成对比图像。
- **参数**:
  - `output_dir`：输出目录路径。
- **返回值**：布尔值，表示测试是否成功。

#### `setup_chinese_matplotlib(test=True)`
快速设置matplotlib支持中文显示，是最常用的入口函数。
- **参数**:
  - `test`：是否进行字体测试。
- **返回值**：布尔值，表示设置是否成功。

## 2. GPU工具模块 (gpu_utils.py)

该模块提供了GPU相关的工具函数，包括检测GPU可用性、数据传输和内存管理。

### 主要函数

#### `check_gpu_availability()`
检查系统是否有可用的GPU以及相关库是否已安装。
- **返回值**：包含GPU可用性和支持信息的字典。

#### `to_device(data, device='cuda', dtype=None)`
将数据转移到指定设备（GPU或CPU）。
- **参数**:
  - `data`：要转移的数据（numpy数组或其他）。
  - `device`：目标设备，'cuda'或'cpu'。
  - `dtype`：可选，数据类型。
- **返回值**：在目标设备上的数据。

#### `copy_if_needed(data, device='cuda')`
仅在必要时复制数据到指定设备，避免不必要的复制。
- **参数**:
  - `data`：要转移的数据。
  - `device`：目标设备。
- **返回值**：在目标设备上的数据。

#### `gpu_timer(func)`
装饰器，用于测量GPU函数执行时间。
- **参数**:
  - `func`：要测量的函数。
- **返回值**：装饰后的函数。

#### `get_optimal_batch_size(array_shape, dtype=np.float32, max_memory_gb=4)`
根据可用GPU内存计算最佳批处理大小。
- **参数**:
  - `array_shape`：数组形状，第一维是样本数。
  - `dtype`：数据类型。
  - `max_memory_gb`：最大允许使用的GPU内存（GB）。
- **返回值**：最佳批处理大小。

#### `batch_process(data, func, batch_size=None, use_gpu=True, *args, **kwargs)`
使用批处理方式处理大型数组，避免GPU内存不足。
- **参数**:
  - `data`：要处理的输入数据。
  - `func`：处理函数。
  - `batch_size`：批处理大小，如果为None则自动计算。
  - `use_gpu`：是否使用GPU。
  - `*args, **kwargs`：传递给处理函数的额外参数。
- **返回值**：处理后的数据。

#### `memory_limit_decorator(max_memory_gb=4)`
装饰器，限制函数使用的最大GPU内存。
- **参数**:
  - `max_memory_gb`：最大内存限制（GB）。
- **返回值**：装饰器函数。

#### `print_gpu_status()`
打印当前GPU状态信息，包括可用性、内存和设备信息。
- **返回值**：无。

## 3. 可视化辅助模块 (visualization_helper.py)

该模块提供了多种数据可视化函数，使创建专业图表变得简单。

### 主要函数

#### `set_style(style='whitegrid', context='paper', font_scale=1.2)`
设置可视化样式，使图表更专业美观。
- **参数**:
  - `style`：seaborn样式名称。
  - `context`：上下文，影响字体大小和线条粗细。
  - `font_scale`：字体缩放比例。

#### `ensure_dir(filepath)`
确保文件所在目录存在，若不存在则创建。
- **参数**:
  - `filepath`：文件路径。
- **返回值**：原始文件路径。

#### `plot_confusion_matrix(y_true, y_pred, labels=None, title='混淆矩阵', filepath=None, normalize=False, figsize=(8, 6))`
绘制混淆矩阵，显示分类模型的预测性能。
- **参数**:
  - `y_true`：真实标签。
  - `y_pred`：预测标签。
  - `labels`：标签名称列表。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `normalize`：是否归一化。
  - `figsize`：图表大小。
- **返回值**：混淆矩阵数组。

#### `plot_roc_curve(y_true, y_scores, title='ROC曲线', filepath=None, figsize=(8, 6))`
绘制ROC曲线，评估分类器的性能。
- **参数**:
  - `y_true`：真实标签。
  - `y_scores`：预测概率。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：fpr, tpr, roc_auc三元组。

#### `plot_pr_curve(y_true, y_scores, title='PR曲线', filepath=None, figsize=(8, 6))`
绘制精确率-召回率曲线，对不平衡数据集特别有用。
- **参数**:
  - `y_true`：真实标签。
  - `y_scores`：预测概率。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：precision, recall, pr_auc三元组。

#### `plot_feature_importance(importance_dict, feature_names=None, title='特征重要性', filepath=None, figsize=(10, 8), top_n=None, importance_type='gain', sort=True, error_bars=None)`
绘制特征重要性条形图，展示模型中各特征的贡献度。
- **参数**:
  - `importance_dict`：特征重要性字典或数组。
  - `feature_names`：特征名称列表。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
  - `top_n`：只显示前N个特征。
  - `importance_type`：重要性类型。
  - `sort`：是否按重要性排序。
  - `error_bars`：误差棒数据。
- **返回值**：特征重要性DataFrame。

#### `plot_learning_curve(train_scores, val_scores=None, metric_name='错误率', title=None, filepath=None, figsize=(10, 6))`
绘制学习曲线，展示模型在训练过程中的性能变化。
- **参数**:
  - `train_scores`：训练集上的评估分数。
  - `val_scores`：验证集上的评估分数。
  - `metric_name`：指标名称。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：布尔值，表示操作是否成功。

#### `plot_multiple_learning_curves(result_dict, title='学习曲线', filepath=None, figsize=(12, 6), cols=2, share_y=False)`
绘制多个学习曲线，方便比较不同指标的变化趋势。
- **参数**:
  - `result_dict`：评估结果字典，格式为 {metric_name: {'train': [...], 'val': [...], ...}, ...}。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
  - `cols`：列数。
  - `share_y`：是否共享y轴。
- **返回值**：figure对象。

#### `plot_spatial_distribution(data_array, mask=None, title='空间分布', filepath=None, figsize=(10, 8), cmap='viridis', vmin=None, vmax=None, colorbar_label='值')`
绘制空间分布图，适用于地理数据可视化。
- **参数**:
  - `data_array`：二维数据数组。
  - `mask`：掩膜数组，用于屏蔽部分区域。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
  - `cmap`：颜色映射。
  - `vmin, vmax`：颜色范围。
  - `colorbar_label`：颜色条标签。
- **返回值**：布尔值，表示操作是否成功。

#### `plot_distributions(data, hue=None, title='数据分布', filepath=None, figsize=(12, 6))`
绘制数据分布(直方图和密度图)。
- **参数**:
  - `data`：DataFrame或类似数据结构。
  - `hue`：用于分组的列。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：布尔值，表示操作是否成功。

#### `plot_prediction_scatter(y_true, y_pred, title='预测值与实际值对比', filepath=None, figsize=(8, 8), alpha=0.5)`
绘制预测值与实际值的散点图，评估回归模型性能。
- **参数**:
  - `y_true`：真实值。
  - `y_pred`：预测值。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
  - `alpha`：点的透明度。
- **返回值**：mse, mae, r2三元组。

#### `plot_correlation_matrix(data, method='pearson', title='相关性矩阵', filepath=None, figsize=(10, 8), cmap='coolwarm', annot=True)`
绘制特征间相关性矩阵，帮助分析特征关系。
- **参数**:
  - `data`：DataFrame格式的数据。
  - `method`：相关系数方法，'pearson', 'kendall', 'spearman'。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
  - `cmap`：颜色映射。
  - `annot`：是否在方格中显示数值。
- **返回值**：相关矩阵。

#### `plot_radar_chart(values, categories, title='模型评估雷达图', filepath=None, figsize=(8, 8), color=COLORS['blue'], alpha=0.25)`
绘制雷达图，用于比较多个指标的表现。
- **参数**:
  - `values`：评估指标值数组。
  - `categories`：指标名称数组。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
  - `color`：填充颜色。
  - `alpha`：透明度。
- **返回值**：matplotlib轴对象。

#### `plot_residuals(y_true, y_pred, title='残差分析', filepath=None, figsize=(12, 5))`
绘制残差分析图，包括残差散点图和分布图。
- **参数**:
  - `y_true`：真实值。
  - `y_pred`：预测值。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：figure对象。

#### `plot_feature_interactions(X, y, feature1_idx, feature2_idx, feature_names=None, title='特征交互图', filepath=None, figsize=(10, 8))`
绘制两个特征之间的交互关系图。
- **参数**:
  - `X`：特征矩阵。
  - `y`：目标变量。
  - `feature1_idx, feature2_idx`：要分析的特征索引。
  - `feature_names`：特征名称列表。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：散点图对象。

#### `plot_model_comparison(models_data, title='模型性能比较', filepath=None, figsize=(12, 6))`
比较多个模型性能的条形图。
- **参数**:
  - `models_data`：字典，格式为 {model_name: {metric_name: value, ...}, ...}。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：比较数据的DataFrame。

#### `plot_class_distribution(y, title='类别分布', filepath=None, figsize=(10, 6))`
绘制目标变量类别分布图。
- **参数**:
  - `y`：目标变量数组。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：类别分布的DataFrame。

#### `plot_missing_values(data, title='缺失值分析', filepath=None, figsize=(12, 6))`
绘制数据集中的缺失值分析图。
- **参数**:
  - `data`：DataFrame格式的数据。
  - `title`：图表标题。
  - `filepath`：保存路径。
  - `figsize`：图表大小。
- **返回值**：缺失值统计的DataFrame。

## 4. 使用示例

### 基本设置

```python
# 导入必要的模块
from utils.font_helper import setup_chinese_matplotlib
from utils.gpu_utils import check_gpu_availability, print_gpu_status
from utils.visualization_helper import plot_confusion_matrix, plot_roc_curve

# 设置中文字体支持
setup_chinese_matplotlib()

# 检查GPU可用性
gpu_info = check_gpu_availability()
print_gpu_status()

# 设置绘图风格
from utils.visualization_helper import set_style
set_style(style='whitegrid', context='paper', font_scale=1.3)
```

### 训练和评估模型

```python
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utils.gpu_utils import to_device

# 假设X和y是已加载的数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 使用GPU加速训练(如果可用)
if gpu_info['cuda_available']:
    dtrain = xgb.DMatrix(data=to_device(X_train, 'cuda'), 
                        label=to_device(y_train, 'cuda'))
    dtest = xgb.DMatrix(data=to_device(X_test, 'cuda'),
                       label=to_device(y_test, 'cuda'))
else:
    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)

# 训练模型...
# model = xgb.train(params, dtrain, ...)

# 模型评估和可视化
y_pred = model.predict(dtest) > 0.5
y_pred_proba = model.predict(dtest)

# 绘制混淆矩阵
from utils.visualization_helper import plot_confusion_matrix
plot_confusion_matrix(
    y_test, y_pred, 
    title='降水预测混淆矩阵',
    filepath='f:/rainfalldata/figures/confusion_matrix.png'
)

# 绘制ROC曲线
from utils.visualization_helper import plot_roc_curve
plot_roc_curve(
    y_test, y_pred_proba,
    title='ROC曲线',
    filepath='f:/rainfalldata/figures/roc_curve.png'
)

# 绘制特征重要性
from utils.visualization_helper import plot_feature_importance
importance = model.get_score(importance_type='gain')
plot_feature_importance(
    importance,
    title='特征重要性',
    filepath='f:/rainfalldata/figures/feature_importance.png'
)
```

### 其他可视化示例

```python
# 绘制相关性矩阵
import pandas as pd
from utils.visualization_helper import plot_correlation_matrix

df = pd.DataFrame(X, columns=['特征1', '特征2', '特征3', '特征4'])
plot_correlation_matrix(
    df, 
    title='特征相关性矩阵',
    filepath='f:/rainfalldata/figures/correlation_matrix.png'
)

# 绘制雷达图比较不同模型
from utils.visualization_helper import plot_radar_chart

metrics = ['精确率', '召回率', 'F1分数', 'AUC', '准确率']
values = [0.85, 0.76, 0.80, 0.92, 0.88]
plot_radar_chart(
    values, metrics,
    title='模型性能雷达图',
    filepath='f:/rainfalldata/figures/radar_chart.png'
)

# 绘制学习曲线
from utils.visualization_helper import plot_learning_curve

# 假设有训练过程中的评估结果
train_loss = [0.8, 0.6, 0.5, 0.4, 0.38, 0.35]
val_loss = [0.83, 0.7, 0.6, 0.55, 0.54, 0.53]
plot_learning_curve(
    train_loss, val_loss,
    metric_name='损失值',
    title='XGBoost模型学习曲线',
    filepath='f:/rainfalldata/figures/learning_curve.png'
)
```
