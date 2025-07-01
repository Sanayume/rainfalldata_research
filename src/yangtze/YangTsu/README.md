# 长江流域降雨预测核心工作区 (`src/yangtze/YangTsu`)

本目录是长江流域降雨预测研究的核心工作区，包含了从数据加载、特征工程、模型训练、超参数优化到误差分析和集成学习的**完整流程**。这里是项目中最活跃的开发区域，承载了大量实验和迭代。

## 目录概览

本目录结构清晰，主要包含以下几类脚本和子目录：

*   **数据加载与管理**：负责原始数据的加载和特征文件的管理。
*   **特征工程**：生成各种类型特征的核心脚本，包括基础、时序、多产品协同、滞后、空间、高级统计和交互特征。
*   **特征处理工具**：用于统一特征格式，使其可以直接用于模型训练。
*   **模型训练与优化**：XGBoost 模型训练、超参数优化及相关工具。
*   **模型评估与诊断**：分析模型性能，特别是误报/漏报 (FP/FN) 特性。
*   **集成学习**：构建多层次集成模型。
*   **历史迭代脚本**：记录了特征工程和模型训练的不同探索版本。

## 主要脚本与功能

以下是本目录中一些关键脚本及其核心功能：

*   `loaddata.py`: **核心数据加载模块**。负责从 `.mat` 文件加载原始产品数据和掩码，并提供按流域提取空间格栅数据 (`get_basin_spatial_data`) 和离散点数据 (`get_basin_point_data`) 的功能。

*   `loadfeatures.py`: **特征加载器**。提供扫描、分类、加载和管理 `.npy` 特征文件的功能，是后续特征处理和模型构建的基础。

*   `generate_basic_features.py`: 生成**基础原始特征**（各产品原始值、目标变量）。
*   `generate_temporal_features.py`: 生成**时序动态特征**（周期性、趋势、差分、累积）。
*   `generate_multi_product_features.py`: 生成**多产品协同特征**（均值、标准差、相关性、一致性）。
*   `generate_lag_features.py`: 生成**滞后特征**（单产品滞后、多产品统计量滞后、滞后差值/比值）。
*   `generate_spatial_features.py`: 生成**空间特征**（梯度、邻域统计、空间自相关、连通域）。
*   `generate_advanced_features.py`: 生成**高级统计特征**（分位数、极值、异常值、弱信号增强）。
*   `generate_interaction_features.py`: 生成**高阶交互特征**（产品间乘性、统计量与时序交互、条件交互）。

*   `flatten_individual_features.py`: **特征展平工具**。将 `generate_*.py` 脚本生成的独立特征文件统一展平为 `(总样本数,)` 的一维格式，并覆盖保存回原文件，使其可以直接用于模型训练。

*   `build_model_matrix.py`: **模型矩阵构建工具**。将所有已展平的独立特征文件加载，并堆叠成一个最终的 `(总样本数, 特征数)` 的二维矩阵 (`X_matrix.npy`)，以及对应的目标变量 (`y_vector.npy`)。

*   `turn1.py` - `turn6.py`: **历史特征工程迭代脚本**。这些脚本代表了不同阶段的特征工程策略，其中 `turn1.py` 到 `turn5.py` 主要处理格栅数据并计算完整空间特征，而 `turn6.py` 处理点数据。

*   `xgboost*.py`: XGBoost 模型训练脚本，对应不同特征集版本。

*   `xgboost_optimization_main.py`: **核心优化脚本**，负责 Optuna 超参数寻优、训练最终优化模型。

*   `xgboostv6_for_Ensemble.py`: 为集成学习生成折外预测的脚本。

*   `loadoptunadb.py`: Optuna 优化日志导入数据库的工具。

*   `feature_of_FP_FN_Yangtsu.py` / `feature_of_FP_FN_Yangtsu_Mean.py`: 误报/漏报 (FP/FN) 深度诊断脚本，用于分析导致预测错误的特征分布。

*   `Spatial_characteristics_plot_fp_fn.py` / `Statistical_spatial_characteristics_fp_fn_.py` / `value_characteristics_fp_fn_.py`: FP/FN 空间和统计特性分析及绘图脚本。

*   `model_compare_plot.py`: 模型性能对比绘图脚本。

*   `Ensemble1/`: **子目录**，包含长江流域 Level 1 专家模型相关代码，用于构建多层次集成模型。

## 工作流程概述

本目录下的脚本协同工作，形成以下主要工作流程：

1.  **数据准备**：通过 `loaddata.py` 加载原始降雨数据。
2.  **特征工程**：
    *   使用 `generate_*.py` 系列脚本生成各种类型的独立特征文件。
    *   （历史迭代：`turn*.py` 脚本代表了特征工程的不同探索版本，它们会生成包含完整空间特征的扁平化矩阵。）
    *   使用 `flatten_individual_features.py` 将 `generate_*.py` 生成的独立特征文件统一展平为 `(总样本数,)` 格式，使其可以直接用于模型训练。
3.  **模型训练与优化**：
    *   使用 `xgboost*.py` 脚本进行基础模型训练。
    *   通过 `xgboost_optimization_main.py` 和 `loadoptunadb.py` 进行自动化超参数优化，以提升模型性能。
4.  **模型评估与诊断**：
    *   通过 `feature_of_FP_FN_Yangtsu.py` 等脚本进行误报/漏报分析，深入理解模型误差。
    *   通过 `model_compare_plot.py` 等脚本进行结果可视化和性能对比。
5.  **集成学习**：通过 `xgboostv6_for_Ensemble.py` 和 `Ensemble1/` 子目录下的脚本构建多层次集成模型，进一步提升预测精度和鲁棒性。

## 重要说明

*   **数据一致性**：本目录下的脚本已解决了原始数据源中空间格点数据与离散点数据之间的不一致问题，确保了所有特征在空间维度上的完美对齐。
*   **特征格式**：经过 `flatten_individual_features.py` 处理后，绝大多数 `.npy` 特征文件都已展平为 `(总样本数,)` 的一维数组，可以直接加载使用。
*   **未展平特征**：有少量特殊空间特征（如 `spatial_autocorrelation_*.npy` 和 `spatial_neighbor_*_samples.npy`）因其固有的高维度结构或不完整的时间维度，未被展平。这些特征更适合用于可视化或深度学习模型。

---