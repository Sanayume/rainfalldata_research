# rainfalldata_research
# 高分辨率降雨融合与机器学习预测研究项目

## 1. 项目概述与研究意义

**背景:** 精准的降雨预测对于洪水预警、农业生产、水资源管理等领域至关重要。然而，单一数据源往往存在时空覆盖不完整、精度不足等问题。本项目旨在应对这一挑战，通过 **融合多种先进的卫星遥感降雨产品** (CMORPH, CHIRPS, GSMAP, IMERG, PERSIANN, SM2RAIN) 与 **地面观测融合数据** (CHM)，结合 **前沿的机器学习技术**，探索和构建高性能、高分辨率的降雨预测模型。

**目标:**
*   开发一套稳健的数据处理与融合流程，有效整合多源异构降雨数据。
*   设计并实现大规模、多维度的特征工程体系，深度挖掘数据中蕴含的降雨相关信息。
*   系统性地评估和优化多种机器学习模型（特别是梯度提升树如 XGBoost, LightGBM，并辅以贝叶斯方法等）在降雨预测任务上的性能。
*   深入分析模型误差，理解预测偏差来源，并驱动特征和模型的迭代优化。
*   针对重点区域（如长江流域）进行精细化建模，探索区域适应性预测策略。
*   重点提升在于降雨的命中率误报率以及临界成功指数, 主要看误报率是否有突破

**核心价值:** 本项目不仅致力于提升降雨预测精度，也深入探索了数据融合、特征工程和机器学习在复杂气象问题中的应用潜力，为相关领域的研究和应用提供方法论参考和技术积累。

## 2. 项目技术架构与实施细节

### 2.1 数据体系 (`data/`)

*   **多源数据整合:**
    *   **原始数据 (`raw/`)**: 涵盖 CMORPH, CHIRPS, GSMAP, IMERG, PERSIANN, SM2RAIN, CHM 等 **7 种主流降雨产品** 的原始数据（如 `.nc`, `.zip` 格式），时间跨度覆盖 **多年** (例如 2016-2020 年数据被重点处理)，空间分辨率各异，总数据量达到 **数十 GB 甚至 TB 级别**。相关的初步读取脚本位于 `src/readdata/` 目录下，例如 `CHM.py`, `CMORPH.py` 等。
    *   **挑战**: 处理不同产品的格式差异、坐标系统、时间分辨率不一致等问题。
*   **精细化预处理 (`intermediate/`, `processed/`)**:
    *   **NaN 值处理**: 实施了系统的缺失值处理流程，结合了 **时空插值方法** 与 **阈值替换策略**，确保数据完整性。处理后的数据按产品和年份存储 (`intermediate/`)，并最终生成用于模型输入的合并时间序列 `.mat` 文件 (`processed/`)。
    *   **空间掩码**: 生成并应用了中国大陆及主要流域（如长江流域）的高精度地理掩码 (`processed/.../masks/`)，以精确提取研究区域数据。
    *   **标准化与对齐**: 进行了必要的时间和空间分辨率对齐操作。

### 2.2 特征工程 (主要代码位于 `src/nationwide/project_all_for_deepling_learning/turn*.py` 和 `src/yangtze/YangTsu/turn*.py`)

本项目构建了一个大规模、多维度的特征库，旨在深度挖掘多源降雨数据中蕴含的与降雨生消、强度、时空分布相关的物理机制和数据驱动模式。特征工程经历了从基础到复杂、从单一到协同、从时序到空间、再到高阶交互和弱信号增强的系统性构建与迭代优化过程。全国范围和长江流域的特征工程遵循了相似的迭代思路和核心特征类别，均迭代了六个主要版本（V1至V6）。

以下是特征体系的主要构成类别：

*   **基础信息**: 各降雨产品在目标点的原始降雨量。
*   **多产品协同**: 利用多个降雨产品间的统计关系（如均值、标准差、中位数、极差、一致指示降雨的产品数量等）来量化不同产品间的一致性与不确定性，这是单一产品分析无法实现的。
*   **时序动态捕捉**:
    *   **周期性**: 采用正余弦函数编码年内日周期 (`sin_day`, `cos_day`)、季节性虚拟变量 (`season_onehot`) 等捕捉降雨的季节性变化规律。
    *   **记忆性 (滞后项)**: 引入多尺度时间滞后项 (如 t-1, t-2, t-3 天的原始降雨量、产品间均值/标准差、降雨产品数等)，以捕捉降雨事件的持续性和短期依赖关系。
    *   **变化率 (差分特征)**: 计算不同时间点（如 t 时刻与 t-1 时刻）的原始值、产品间均值/标准差的差分，以捕捉降雨强度和产品不确定性的变化趋势。
    *   **累积效应 (滑动窗口统计)**: 设计了基于不同时间窗口（如 3, 7, 15 天）的滑动统计量（如窗口内的均值、标准差、最值、范围、特定分位数等），应用于各产品原始值或其统计量（如产品均值），以捕捉不同时间尺度下的降雨累积效应和波动特征。
*   **空间关联 (针对格点数据或名义点位数据)**:
    *   **邻域统计**: 针对格点数据，实际计算中心格点周围不同大小邻域（如 3x3, 5x5）内各降雨产品的统计量（均值、标准差、最大值等）及其与中心点的差异，以刻画降雨事件的空间展布和局部梯度。对于点位数据，这些特征作为结构占位符，通常填充为NaN，但在后续可能通过插值或与其他空间代理变量结合来赋予意义。
    *   **空间梯度**: 针对格点数据，计算特定产品（如GSMAP, PERSIANN）或产品均值的空间梯度幅度和方向，以表征降雨场的空间变化剧烈程度。点位数据则为名义特征。
*   **弱信号增强 / 模糊性特征**:
    *   特别设计了针对毛毛雨、小雨等低强度降雨事件的探测特征，例如：距特定降雨阈值（如0.1mm）的距离、低强度降雨下的产品间标准差（条件不确定性）、特定低强度区间的降雨产品比例、变异系数 (CV) 等，旨在提升模型对这类易被忽略但对径流累积和干旱缓解有重要意义的降雨事件的敏感度。
    *   引入特定强度分箱特征，例如基于产品均值或降雨产品数量划分的降雨强度等级的独热编码。
*   **高阶交互特征**:
    *   探索性地构建了部分交互特征，例如将产品间的标准差与季节性周期因子相乘 (`product_std * sin_day`)，或将低强度不确定性与总体变异系数结合 (`low_intensity_std * coef_of_variation`)，试图捕捉非线性、多因素耦合的复杂降雨模式。

**特征集迭代优化 (V1 至 V6):**

特征集的构建是一个持续迭代和优化的过程，每一版本的调整都基于前一轮的模型评估、误差分析结果以及对降雨物理过程理解的深化。全国范围和长江流域的特征工程脚本 (`src/nationwide/.../turn*.py` 和 `src/yangtze/YangTsu/turn*.py`) 体现了这一迭代。以下简述长江流域特征工程的主要迭代思路（全国范围类似）：

*   **V1 (`turn1.py`): 早期探索 - 格点数据与简化特征**
    *   **数据处理重大变化**: 通过 `ALL_DATA.yangtsu()` 返回的是**格点数据** (`(prod, time, lat, lon)`)，并使用 `memmap` 处理。脚本中明确处理了经纬度维度，并在展平时应用了有效格点掩码 (`valid_mask`)。
    *   **特征集核心**: 基础原始降雨量、简化的多产品协同（仅当前时刻降雨产品数）、简化的时序动态（季节虚拟变量、原始值滞后项、原始值差分、基于各产品滑动窗口均值的累积统计），以及**针对格点数据真实计算的5x5邻域统计量**。弱信号和交互特征被移除。此版本为后续复杂模型和格点数据处理奠定基础。

*   **V2 (`turn2.py`): 转向点位数据 - 增补与调整**
    *   **数据处理**: 输入转为流域点位数据 (`(prod, time, points)`)。
    *   **特征集调整**: 在V1格点特征简化的基础上，针对点位数据特性进行调整和补充。
        *   多产品协同增强: 引入了当前时刻的变异系数 (`coef_of_variation`) 和产品极差 (`product_range`)，保留 `rain_product_count`。
        *   时序滞后项: 包含原始值 (`lag_X_values`)、产品间标准差 (`lag_X_std`) 和降雨产品数 (`lag_X_rain_count`)。
        *   时序周期性: 仅保留季节虚拟变量 (`season_onehot`)，尚未引入年内周期正余弦。
        *   空间关联: 采用名义上的5x5邻域和梯度特征 (均为NaN占位符)。
        *   弱信号增强: 引入距0.1mm阈值距离 (`threshold_proximity`)，新增(0, 0.5mm]区间产品比例 (`fraction_products_low_range`)，保留基于降雨产品数量的强度分箱 (`intensity_bins_count`)。

*   **V3 (`turn3.py`): 点位数据 - 特征体系简化探索**
    *   **核心思路**: 在V2的基础上进一步简化特征集，旨在测试核心特征的有效性，并探索特定产品滑动窗口特征。
    *   **特征集调整**:
        *   多产品协同简化: 仅保留当前时刻降雨产品数 (`rain_product_count`)。
        *   时序周期性简化: 依然仅保留季节虚拟变量 (`season_onehot`)。
        *   时序累积效应: 在基于产品均值的滑动统计基础上，**新增了针对特定产品（GSMAP均值，PERSIANN标准差）的7天滑动窗口统计** (`window_7_mean_GSMAP`, `window_7_std_PERSIANN`)。
        *   弱信号增强大幅简化: 仅保留基于降雨产品数量的特定强度分箱 (`intensity_bins_count`)。
        *   移除了V2中引入的部分多产品协同特征（如变异系数、极差）和弱信号特征。

*   **V4 (`turn4.py`): 点位数据 - 特征体系深化与扩展**
    *   **核心思路**: 在之前版本（特别是V2和V3的经验）基础上，进行特征的全面深化和扩展，构建更丰富的特征集。
    *   **特征集调整**:
        *   多产品协同 (当前时刻, 全面): 包括均值、标准差、中位数、最值、极差、降雨产品数。
        *   时序动态 (全面深化): **重新引入年内周期正余弦编码 (`sin_day`, `cos_day`)**；滞后项扩展至包含产品间均值 (`lag_X_mean`)，并**新增了滞后项之间的均值差分 (`lag_1_2_mean_diff`, `lag_2_3_mean_diff`)**；变化率扩展至包含均值差分 (`diff_1_mean`) 和标准差差分 (`diff_1_std`)。
        *   空间关联 (扩展): 名义上的3x3邻域统计量及其与中心点的差异被加入，保留5x5名义特征和梯度特征 (均为NaN)。
        *   弱信号增强 (强化): 重新引入并扩展了弱信号特征，如距阈值距离、**当前及滞后一天 (t-1) 的变异系数**、条件不确定性 (`low_intensity_std`)，以及基于**产品均值 (`intensity_bins_mean`)** 和降雨产品数量的强度分箱。
        *   交互特征 (强化): 构建了多项有物理意义或数据驱动的交互项，如 `product_std * sin_day`、`low_intensity_std * coef_of_variation` 等。

*   **V5 (`turn5.py`): 点位数据 - V3简化基础上重点恢复周期特征**
    *   **核心思路**: 以V3的简化特征集为主要框架，关键在于**恢复了年内周期正余弦编码 (`sin_day`, `cos_day`)**，这在V3中被移除了，但在V4中证明了其价值。
    *   其他大部分特征（如多产品协同、滞后项、累积效应、空间关联、弱信号增强）与V3的简化版本保持一致或相似。交互特征依然移除。

*   **V6 (`turn6.py`): 点位数据 - 当前主流特征体系**
    *   **数据处理**: 输入为特定流域（如长江）的点位数据，通过 `ALL_DATA.get_basin_point_data(basin_mask_value=2)` 获取。
    *   **核心思路**: 在V4的全面性基础上进行提炼和整合，构建一套相对完整且被验证为当前性能最优的、针对点位数据的基础特征体系，作为当前模型优化和集成学习的基石。
    *   **特征集构成**:
        *   基础信息: 各产品原始降雨量。
        *   多产品协同 (当前时刻): 均值、标准差、中位数、最值、极差、降雨产品数。
        *   时序动态:
            *   周期性: 年内周期正余弦编码 (`sin_day`, `cos_day`)，季节虚拟变量 (`season_onehot`)。
            *   记忆性 (滞后项): t-1, t-2, t-3 天的原始降雨量、产品间均值 (`lag_X_mean`) 和标准差 (`lag_X_std`)。（与V4相比，此处可能略有简化，例如移除了滞后降雨产品数或滞后均值差分，具体需参照`turn6.py`的实现细节进行确认）。
            *   变化率: t 时刻与 t-1 时刻的原始值、均值、标准差的差分。
            *   累积效应 (滑动窗口): 基于各产品**均值**序列计算 3, 7, 15 天窗口的统计量。
        *   空间关联 (点位数据的名义空间特征): 名义上的3x3邻域统计量及其与中心点的差异 (实际填充为NaN，作为结构占位符)。V6可能简化了空间特征的种类（例如只保留3x3，移除5x5的名义特征）。
        *   弱信号增强: 距0.1mm阈值距离、变异系数CV、条件不确定性 (`low_intensity_std`)、基于产品均值的特定强度分箱。
        *   交互特征: `product_std * sin_day`, `low_intensity_std * coef_of_variation`, `rain_count_std_interaction`。

这一系列迭代清晰地展示了从早期对格点数据的探索（V1），到后续聚焦于点位数据，并逐步从初步构建（V2）、简化测试（V3, V5部分）、到构建全面特征体系（V4），并最终提炼出当前稳定且表现优异的V6特征集的演进过程。**V6特征集是当前长江流域机器学习模型（特别是XGBoost）超参数优化和集成学习工作的基础。** 特征工程的结果（如各版本生成的特征矩阵和名称列表）主要存储于 `results/yangtze/features/` 和 `results/nationwide/features/` (如果全国范围也有对应版本产出) 目录下。
这一系列迭代清晰地展示了长江流域特征工程的演进：从 V1 版本探索格点数据与简化特征开始，后续版本 (V2-V6) 则聚焦于点位数据，经历了从简化测试 (V3, V5部分特征)、逐步增补关键信号 (V2, V5)，到构建全面特征体系 (V4, V6) 的不同阶段，反映了特征集在不同策略下的探索和优化过程。

### 2.3 机器学习建模 (主要代码位于 `src/nationwide/project_all_for_deepling_learning/`, `src/yangtze/YangTsu/`, `src/ensemble_learning/ensemble_learning/`)

系统性地探索和评估了多种先进的机器学习模型：

*   **主力模型**:
    *   **XGBoost / LightGBM**: 作为业界领先的梯度提升树模型，重点用于追求高预测性能，并针对性地调整了目标函数和评估指标。相关脚本包括 `src/nationwide/project_all_for_deepling_learning/xgboost*.py`, `lightGBM1.py` 以及长江流域对应的 `src/yangtze/YangTsu/xgboost*.py`。
*   **辅助与对比模型**:
    *   **Naive Bayes**: 作为基准模型之一，用于快速评估特征效果。相关脚本如 `src/nationwide/project_all_for_deepling_learning/naive_bayes*.py`。
    *   **Bayesian Network**: 探索结合 **专家先验知识** 与数据驱动学习的可能性，特别用于 FP/FN 样本的分析和建模，如 `src/nationwide/project_all_for_deepling_learning/bayesian_network_fp_expert.py` 和 `src/ensemble_learning/ensemble_learning/` 中的相关脚本。
*   **自动化超参数优化**: 广泛应用 **Optuna** 框架，对 XGBoost 和 Naive Bayes 等模型的 **关键超参数** 进行了系统的优化。相关脚本如 `src/nationwide/project_all_for_deepling_learning/xgboost3_optuna.py`, `naive_bayes1_optuna.py`。
*   **性能加速**: 积极探索 **GPU 加速** 技术 (相关探索代码位于 `src/legacy/` 目录，如 `cuda_train.py`, `xgboost_cuda_training.py`)，利用 CUDA 环境优化训练过程，以应对大规模数据和复杂模型的计算挑战。
*   **集成学习探索**: 进行了集成学习的实践，包括训练基模型、FP/FN 专家模型及元学习器等步骤，具体脚本位于 `src/ensemble_learning/ensemble_learning/` 目录下 (例如 `1_generate_base_predictions.py` 至 `5_train_evaluate_meta_learner.py`)，旨在结合多个模型的优势，进一步提升预测稳定性和准确性。
*   **版本化管理**: 严格保存了 **不同版本、不同参数配置** 下训练得到的模型文件 (`.joblib`, `.pkl` 存储于 `results/.../models/`)，便于追溯和比较。

#### 2.3.1 基础模型选择与性能对比

在基础模型的选择阶段，我们首先在v1特征集上对部分候选模型进行了初步训练与评估。基于这些对比结果，XGBoost模型因其表现突出而被选为主要模型。

值得注意的是，除XGBoost和贝叶斯网络模型外，其他所有参与比较的模型均在v1特征集上进行了超参数寻优。因此，下表中展示的XGBoost性能是基于其默认参数配置的。

各模型性能对比如下：

| 模型名称                               | 准确率 (Accuracy) | 命中率 (POD) | 空报率 (FAR) | 临界成功指数 (CSI) | TP     | FN     | FP     | TN     |
|----------------------------------------|-----------------|------------|------------|-----------------|--------|--------|--------|--------|
| K-Nearest Neighbors (Tuned on Subset)  | 0.7917          | 0.7839     | 0.1308     | 0.7012          | 516971 | 142520 | 77795  | 320429 |
| Support Vector Machine (Tuned on Subset)| 0.8021          | 0.7496     | 0.0819     | 0.7026          | 494322 | 165169 | 44111  | 354113 |
| Random Forest                          | 0.8408          | 0.8378     | 0.1001     | 0.7665          | 552553 | 106938 | 61430  | 336794 |
| LightGBM                               | 0.8366          | 0.8221     | 0.0929     | 0.7582          | 542149 | 117342 | 55529  | 342695 |
| Gaussian Naive Bayes (Default)         | 0.7019          | 0.5799     | 0.0909     | 0.5481          | 382432 | 277059 | 38237  | 359987 |
| XGBoost (Default)                      | **0.8819**      | **0.8880** | **0.0819** | **0.8228**      | -      | **73133**  | **51763**  | -      |

#### 2.3.2 长江流域 XGBoost 模型性能迭代与评估实例

针对长江流域，基于不同版本的特征工程（详见 2.2 节中 V1-V6 的迭代描述），我们使用 XGBoost 作为主力分类模型进行了训练和评估。以下表格汇总了部分关键版本的模型在**测试集**上的性能表现，均采用二分类（有雨/无雨，阈值 > 0.1mm/d 判为有雨），并展示了不同预测概率阈值下的核心气象评估指标。

**1. 特征集 V1 (`turn1.py`, 模型 `xgboost1.py`):** (*此版本特征工程转向格点数据，结果代表了不同数据处理方式下的性能*)
    *   在预测概率阈值为 0.50 时：POD: 0.8410, **FAR: 0.0823**, CSI: 0.7820
    *   **不同预测阈值下的性能 (Test Set):**

        | 预测阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN     |
        |----------|----------|--------|--------|--------|-------|--------|
        | 0.30     | 0.8559   | 0.9032 | 0.1280 | 0.7975 | 87458 | 63813  |
        | 0.35     | 0.8576   | 0.8889 | 0.1149 | 0.7969 | 76109 | 73296  |
        | 0.40     | 0.8576   | 0.8738 | 0.1030 | 0.7941 | 66168 | 83234  |
        | 0.45     | 0.8558   | 0.8578 | 0.0924 | 0.7889 | 57590 | 93759  |
        | **0.50** | **0.8526** | **0.8410** | **0.0823** | **0.7820** | **49753** | **104886** |
        | 0.55     | 0.8480   | 0.8230 | 0.0731 | 0.7728 | 42821 | 116730 |
        | 0.60     | 0.8418   | 0.8037 | 0.0646 | 0.7615 | 36601 | 129435 |
        | 0.65     | 0.8343   | 0.7829 | 0.0561 | 0.7481 | 30679 | 143174 |
        | 0.70     | 0.8252   | 0.7601 | 0.0479 | 0.7321 | 25198 | 158214 |

**2. 特征集 V2 (`turn2.py`, 模型 `xgboost2.py`):**
    *   在预测概率阈值为 0.5 时：POD: 0.8322, **FAR: 0.0687**, CSI: 0.7841

    | 预测阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN     |
    |----------|----------|--------|--------|--------|-------|--------|
    | 0.45     | 0.8617   | 0.8495 | 0.0797 | 0.7913 | 48033 | 98275  |
    | **0.50** | **0.8585** | **0.8322** | **0.0687** | **0.7841** | **40072** | **109563** |
    | 0.55     | 0.8539   | 0.8140 | 0.0586 | 0.7747 | 33064 | 121487 |

**3. 特征集 V3 (`turn3.py`, 模型 `xgboost3.py`):**
    *   在预测概率阈值为 0.5 时：POD: 0.8337, **FAR: 0.0681**, CSI: 0.7859

    | 预测阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN     |
    |----------|----------|--------|--------|--------|-------|--------|
    | 0.45     | 0.8629   | 0.8506 | 0.0786 | 0.7930 | 47404 | 97565  |
    | **0.50** | **0.8597** | **0.8337** | **0.0681** | **0.7859** | **39779** | **108584** |
    | 0.55     | 0.8551   | 0.8156 | 0.0581 | 0.7765 | 32884 | 120423 |

**4. 特征集 V4 (`turn4.py`, 模型 `xgboost4.py`):**
    *   在预测概率阈值为 0.5 时：POD: 0.8520, **FAR: 0.0572**, CSI: 0.8101

    | 预测阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN     |
    |----------|----------|--------|--------|--------|-------|--------|
    | 0.45     | 0.8685   | 0.8675 | 0.0675 | 0.8171 | 41073 | 85895  |
    | **0.50** | **0.8767** | **0.8520** | **0.0572** | **0.8101** | **33775** | **96659**  |
    | 0.55     | 0.8720   | 0.8349 | 0.0482 | 0.8011 | 27602 | 107801 |

**5. 特征集 V5 (`turn5.py`, 模型 `xgboost5.py`):**
    *   在预测概率阈值为 0.5 时：POD: 0.8529, **FAR: 0.0564**, CSI: 0.8115

    | 预测阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN     |
    |----------|----------|--------|--------|--------|-------|--------|
    | 0.45     | 0.8810   | 0.8695 | 0.0667 | 0.8186 | 40579 | 85237  |
    | **0.50** | **0.8777** | **0.8529** | **0.0564** | **0.8115** | **33320** | **96033**  |
    | 0.55     | 0.8729   | 0.8357 | 0.0474 | 0.8023 | 27185 | 107285 |

**6. 特征集 V6 (基于 `turn6.py`):**

*   **默认参数模型 (`xgboost6 - 副本.py`):**
    *   在预测概率阈值为 0.5 时：
        *   Accuracy: 0.8819, POD: 0.8880, **FAR: 0.0819**, CSI: 0.8228
        *   FP: 51763, FN: 73133
    *   模型在不同阈值下的表现：

        | 预测阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN     |
        |----------|----------|--------|--------|--------|-------|--------|
        | 0.30     | 0.8685   | 0.9213 | 0.1241 | 0.8100 | 83500 | 50972  |
        | 0.35     | 0.8759   | 0.9115 | 0.1084 | 0.8184 | 70624 | 57114  |
        | 0.40     | 0.8801   | 0.9019 | 0.0956 | 0.8227 | 60500 | 63220  |
        | 0.45     | 0.8825   | 0.8920 | 0.0849 | 0.8242 | 53240 | 70103  |
        | **0.50** | **0.8819** | **0.8880** | **0.0819** | **0.8228** | **51763** | **73133**  |
        | 0.55     | 0.8806   | 0.8722 | 0.0700 | 0.8185 | 42860 | 83457  |
        | 0.60     | 0.8775   | 0.8555 | 0.0593 | 0.8117 | 35205 | 94373  |
        | 0.65     | 0.8723   | 0.8369 | 0.0496 | 0.8019 | 28511 | 106507 |
        | 0.70     | 0.8655   | 0.8164 | 0.0403 | 0.7894 | 22368 | 119904 |

    *   与原始卫星产品相比 (在测试集，概率阈值 0.5，降雨阈值 0.1mm/d)：

        | 模型/产品                    | Accuracy | POD    | FAR    | CSI    | FP      | FN      |
        |------------------------------|----------|--------|--------|--------|---------|---------|
        | XGBoost_Yangtsu (Thr 0.5)    | 0.8819   | 0.8880 | 0.0819 | 0.8228 | 51763   | 73133   |
        | Baseline_CMORPH_Yangtsu      | 0.6334   | 0.5168 | 0.1763 | 0.4654 | 361158  | 1577665 |
        | Baseline_CHIRPS_Yangtsu      | 0.5906   | 0.4413 | 0.1913 | 0.3996 | 340810  | 1824393 |
        | Baseline_SM2RAIN_Yangtsu     | 0.6930   | 0.9033 | 0.3072 | 0.6450 | 1307610 | 315866  |
        | Baseline_IMERG_Yangtsu       | 0.7145   | 0.7073 | 0.1935 | 0.6047 | 554123  | 955706  |
        | Baseline_GSMAP_Yangtsu       | 0.7353   | 0.6108 | 0.0606 | 0.5876 | 128730  | 1270999 |
        | Baseline_PERSIANN_Yangtsu    | 0.6498   | 0.6387 | 0.2437 | 0.5297 | 672194  | 1179721 |

*   **特征集 V6 + Optuna 超参数优化 + 最终模型评估 (基于Trial 322参数):**
    *   **优化过程与选定参数:** 在长江流域V6特征集上，通过Optuna进行了大规模超参数寻优。其中，第322次试验 (`Trial 322`) 取得了 AUC 为 0.988785 的优异表现，其参数组合被选定用于后续的交叉验证和最终模型训练：
        ```python
        # Trial 322 Parameters
        {
            'n_estimators': 2960,
            'learning_rate': 0.026319051020408163,
            'max_depth': 18,
            'subsample': 0.8985668163265306,
            'colsample_bytree': 0.846647612244898,
            'gamma': 0.09964387755102041,
            'lambda': 7.34496612e-06,
            'alpha': 1.1915502e-06
        }
        ```
        *(注: 完整的Optuna寻优日志包含超过334次试验，记录于 `my_optimization_history.db` 和相关日志文件。)*
        **寻优记录**
        *334次*
        ```python
        log_data = 
        """
        [I 2025-05-18 00:45:25,251] A new study created in memory with name: no-name-45d5b364-2724-400b-9b4b-2c15d566afe3
        [I 2025-05-18 00:48:13,722] Trial 0 finished with value: 0.9519780839154898 and parameters: {'n_estimators': 718, 'learning_rate': 0.007012079925836971, 'max_depth': 12, 'subsample': 0.7443934338339511, 'colsample_bytree': 0.8525985730244483, 'gamma': 0.9713678815251202, 'lambda': 0.2785976772582166, 'alpha': 1.1726802977165098e-08}. Best is trial 0 with value: 0.9519780839154898.
        [I 2025-05-18 00:58:28,209] Trial 1 finished with value: 0.9488917677988358 and parameters: {'n_estimators': 2035, 'learning_rate': 0.001656057871843877, 'max_depth': 13, 'subsample': 0.7948707721490318, 'colsample_bytree': 0.9031624019345381, 'gamma': 0.46754170910914106, 'lambda': 0.217478680370962, 'alpha': 5.951911498230732}. Best is trial 0 with value: 0.9519780839154898.
        [I 2025-05-18 00:59:39,938] Trial 2 finished with value: 0.9818246828286973 and parameters: {'n_estimators': 718, 'learning_rate': 0.27355203968828584, 'max_depth': 13, 'subsample': 0.8821287099851913, 'colsample_bytree': 0.6359762085388919, 'gamma': 0.7963653900622107, 'lambda': 3.7397726925165233e-07, 'alpha': 9.674960248200315e-05}. Best is trial 2 with value: 0.9818246828286973.
        [I 2025-05-18 01:04:36,275] Trial 4 finished with value: 0.976023765159064 and parameters: {'n_estimators': 1369, 'learning_rate': 0.4683799100760631, 'max_depth': 10, 'subsample': 0.6115044926819482, 'colsample_bytree': 0.86473010722483, 'gamma': 0.6825997504261366, 'lambda': 5.411953884562014e-07, 'alpha': 1.908931930406256e-08}. Best is trial 2 with value: 0.9818246828286973.
        [I 2025-05-18 01:07:21,113] Trial 5 finished with value: 0.9268188718676637 and parameters: {'n_estimators': 1976, 'learning_rate': 0.0021726562297615204, 'max_depth': 6, 'subsample': 0.5513319698606656, 'colsample_bytree': 0.9016926055512742, 'gamma': 0.42873901570036166, 'lambda': 1.958742558625567e-06, 'alpha': 0.0014698109993754756}. Best is trial 2 with value: 0.9818246828286973.
        [I 2025-05-18 01:15:24,060] Trial 6 finished with value: 0.9614927845774889 and parameters: {'n_estimators': 2366, 'learning_rate': 0.004174049695441833, 'max_depth': 12, 'subsample': 0.8757666251420504, 'colsample_bytree': 0.680003990923393, 'gamma': 0.10974431620689629, 'lambda': 1.234422889727366e-07, 'alpha': 1.0189456751150507e-06}. Best is trial 2 with value: 0.9818246828286973.
        [I 2025-05-18 01:22:48,758] Trial 7 finished with value: 0.9676414393882465 and parameters: {'n_estimators': 1154, 'learning_rate': 0.00698877845261341, 'max_depth': 14, 'subsample': 0.7791386657395465, 'colsample_bytree': 0.5415142097122159, 'gamma': 0.13546079954929335, 'lambda': 1.9519092487552014e-05, 'alpha': 0.2661818170838475}. Best is trial 2 with value: 0.9818246828286973.
        [I 2025-05-18 01:27:23,114] Trial 8 finished with value: 0.9591231780755392 and parameters: {'n_estimators': 774, 'learning_rate': 0.007482349914381423, 'max_depth': 13, 'subsample': 0.8027909578436907, 'colsample_bytree': 0.8805932158728492, 'gamma': 0.10812224619120758, 'lambda': 2.319012546392842e-07, 'alpha': 7.325602125176082e-07}. Best is trial 2 with value: 0.9818246828286973.
        [I 2025-05-18 01:30:46,466] Trial 9 finished with value: 0.983136935929579 and parameters: {'n_estimators': 2044, 'learning_rate': 0.3946720626285836, 'max_depth': 14, 'subsample': 0.8555637291044709, 'colsample_bytree': 0.6145647723034193, 'gamma': 0.3095463631756792, 'lambda': 0.07153008639139238, 'alpha': 5.308893710688652}. Best is trial 9 with value: 0.983136935929579.
        [I 2025-05-18 01:32:44,862] Trial 10 finished with value: 0.9352929878854478 and parameters: {'n_estimators': 2494, 'learning_rate': 0.04392582935458374, 'max_depth': 3, 'subsample': 0.9910150362882679, 'colsample_bytree': 0.5178795330317768, 'gamma': 0.3003155853310795, 'lambda': 0.0012757222867200181, 'alpha': 0.05183073698893637}. Best is trial 9 with value: 0.983136935929579.
        [I 2025-05-18 01:33:41,445] Trial 11 finished with value: 0.9798199957207181 and parameters: {'n_estimators': 1616, 'learning_rate': 0.4726405719755079, 'max_depth': 15, 'subsample': 0.9395323961864944, 'colsample_bytree': 0.674339760209314, 'gamma': 0.6592906252823639, 'lambda': 0.0010011983664979488, 'alpha': 0.0007815311115024722}. Best is trial 9 with value: 0.983136935929579.
        [I 2025-05-18 01:35:54,246] Trial 12 finished with value: 0.982068896357805 and parameters: {'n_estimators': 1070, 'learning_rate': 0.13544462747895863, 'max_depth': 11, 'subsample': 0.8841228073908317, 'colsample_bytree': 0.6245627741386445, 'gamma': 0.3080334564940874, 'lambda': 1.208078649820873e-08, 'alpha': 0.019202886076042494}. Best is trial 9 with value: 0.983136935929579.
        [I 2025-05-18 01:37:59,742] Trial 13 finished with value: 0.9770416398266238 and parameters: {'n_estimators': 1129, 'learning_rate': 0.11669301907304759, 'max_depth': 10, 'subsample': 0.8742173184374223, 'colsample_bytree': 0.6022414345448315, 'gamma': 0.30040077868731196, 'lambda': 0.01690719076826415, 'alpha': 9.918804538282368}. Best is trial 9 with value: 0.983136935929579.
        [I 2025-05-18 01:39:28,003] Trial 14 finished with value: 0.9716895681483267 and parameters: {'n_estimators': 1069, 'learning_rate': 0.1384225490354471, 'max_depth': 8, 'subsample': 0.6992598673115924, 'colsample_bytree': 0.7672854866727796, 'gamma': 0.3200783788908013, 'lambda': 1.1528056201587347e-08, 'alpha': 0.044910842233665045}. Best is trial 9 with value: 0.983136935929579.
        [I 2025-05-18 01:48:02,637] Trial 15 finished with value: 0.9856468531176371 and parameters: {'n_estimators': 1567, 'learning_rate': 0.03190108086337776, 'max_depth': 15, 'subsample': 0.9308058747736159, 'colsample_bytree': 0.7517500226375462, 'gamma': 0.23710286659023017, 'lambda': 4.8645995490385064e-05, 'alpha': 0.6400568254751905}. Best is trial 15 with value: 0.9856468531176371.
        [I 2025-05-18 01:58:10,965] Trial 16 finished with value: 0.9841986455051804 and parameters: {'n_estimators': 1693, 'learning_rate': 0.02331527704739685, 'max_depth': 15, 'subsample': 0.9859876592992003, 'colsample_bytree': 0.7614659157203131, 'gamma': 0.0031905338786306636, 'lambda': 5.6015157993678665e-05, 'alpha': 0.7219421386865369}. Best is trial 15 with value: 0.9856468531176371.
        [I 2025-05-18 02:07:56,245] Trial 17 finished with value: 0.9836339684436517 and parameters: {'n_estimators': 1614, 'learning_rate': 0.02112124903325072, 'max_depth': 15, 'subsample': 0.9566021751460724, 'colsample_bytree': 0.7740919505111992, 'gamma': 0.016539489367668813, 'lambda': 3.4138972236405394e-05, 'alpha': 0.4133996571234476}. Best is trial 15 with value: 0.9856468531176371.
        [I 2025-05-18 02:09:45,857] Trial 18 finished with value: 0.9487472267840599 and parameters: {'n_estimators': 1447, 'learning_rate': 0.023165700493995932, 'max_depth': 7, 'subsample': 0.9957156154380726, 'colsample_bytree': 0.8046855035515929, 'gamma': 0.006686800008247451, 'lambda': 7.209623537546983e-05, 'alpha': 0.0061384370826199726}. Best is trial 15 with value: 0.9856468531176371.
        [I 2025-05-18 02:11:25,577] Trial 19 finished with value: 0.9349356596216689 and parameters: {'n_estimators': 1782, 'learning_rate': 0.024442893241835142, 'max_depth': 4, 'subsample': 0.9321511480467881, 'colsample_bytree': 0.7094856194760204, 'gamma': 0.19063076932811246, 'lambda': 8.253569906977359e-06, 'alpha': 0.39425787270286206}. Best is trial 15 with value: 0.9856468531176371.
        [I 2025-05-18 02:19:02,576] Trial 20 finished with value: 0.9860914182020476 and parameters: {'n_estimators': 1391, 'learning_rate': 0.05288611610019165, 'max_depth': 15, 'subsample': 0.7048686035069371, 'colsample_bytree': 0.8152532666208882, 'gamma': 0.20427993318953397, 'lambda': 0.0002010902827279487, 'alpha': 1.1209918586545113}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:24:57,272] Trial 21 finished with value: 0.9846273776516076 and parameters: {'n_estimators': 1315, 'learning_rate': 0.04580016999955667, 'max_depth': 14, 'subsample': 0.6800448432465434, 'colsample_bytree': 0.8182550095150586, 'gamma': 0.20802883156697297, 'lambda': 0.00027077130165657174, 'alpha': 0.5461679939648083}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:32:07,316] Trial 22 finished with value: 0.9856813148481677 and parameters: {'n_estimators': 1293, 'learning_rate': 0.053752348597253335, 'max_depth': 15, 'subsample': 0.6746370644966275, 'colsample_bytree': 0.8225993089018886, 'gamma': 0.19388835193825055, 'lambda': 0.0004832635781178572, 'alpha': 1.300423595044934}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:38:23,449] Trial 23 finished with value: 0.9854341991445774 and parameters: {'n_estimators': 1266, 'learning_rate': 0.05768549236205113, 'max_depth': 15, 'subsample': 0.6699334438503222, 'colsample_bytree': 0.9632873090106544, 'gamma': 0.5350723838007486, 'lambda': 0.005071928654577362, 'alpha': 2.1163026350546246}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:41:28,999] Trial 24 finished with value: 0.9642811992558139 and parameters: {'n_estimators': 915, 'learning_rate': 0.013266233398539846, 'max_depth': 12, 'subsample': 0.6258619534013005, 'colsample_bytree': 0.7208355452479855, 'gamma': 0.4222612081912654, 'lambda': 0.00029706615557860064, 'alpha': 0.004379104927321376}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:47:29,316] Trial 25 finished with value: 0.9860722893276616 and parameters: {'n_estimators': 1481, 'learning_rate': 0.06548055296297502, 'max_depth': 14, 'subsample': 0.7328088499312534, 'colsample_bytree': 0.8329882204591967, 'gamma': 0.21885783724751395, 'lambda': 0.004036940080919897, 'alpha': 0.07311784828670682}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:51:21,633] Trial 26 finished with value: 0.9845314446475769 and parameters: {'n_estimators': 895, 'learning_rate': 0.07120285368271209, 'max_depth': 14, 'subsample': 0.7299778751623316, 'colsample_bytree': 0.8114789595842812, 'gamma': 0.14574591018771493, 'lambda': 0.004949278413659157, 'alpha': 0.07695795329644642}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:54:28,190] Trial 27 finished with value: 0.9816839263436404 and parameters: {'n_estimators': 1430, 'learning_rate': 0.17084064811619634, 'max_depth': 11, 'subsample': 0.5196710398085865, 'colsample_bytree': 0.9346272283422461, 'gamma': 0.5423420566198147, 'lambda': 0.029922818999577102, 'alpha': 0.008527191862821545}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:57:31,472] Trial 28 finished with value: 0.9841475053366557 and parameters: {'n_estimators': 1281, 'learning_rate': 0.23418138708969846, 'max_depth': 13, 'subsample': 0.7095635975646728, 'colsample_bytree': 0.8295701745794688, 'gamma': 0.36535548777676663, 'lambda': 0.0012716340183634618, 'alpha': 0.07829728491799169}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 02:59:34,313] Trial 29 finished with value: 0.9778370371097015 and parameters: {'n_estimators': 947, 'learning_rate': 0.08237361310533693, 'max_depth': 11, 'subsample': 0.7541035337762033, 'colsample_bytree': 0.842339851937068, 'gamma': 0.977187511117946, 'lambda': 1.9352716599747124, 'alpha': 3.442445080795034}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 03:01:48,234] Trial 30 finished with value: 0.9589033422773919 and parameters: {'n_estimators': 578, 'learning_rate': 0.013977473378430536, 'max_depth': 12, 'subsample': 0.6400826720324237, 'colsample_bytree': 0.7903886604546726, 'gamma': 0.22897108246226475, 'lambda': 4.228778131195981e-06, 'alpha': 0.1353802260047189}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 03:09:45,131] Trial 31 finished with value: 0.9853664348720345 and parameters: {'n_estimators': 1540, 'learning_rate': 0.0372859871393372, 'max_depth': 15, 'subsample': 0.8306917162244325, 'colsample_bytree': 0.7183540029220103, 'gamma': 0.22889552867299376, 'lambda': 0.00014244804467837782, 'alpha': 1.7152189473728123}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 03:16:43,488] Trial 32 finished with value: 0.9839293588735267 and parameters: {'n_estimators': 1521, 'learning_rate': 0.03538846269068019, 'max_depth': 14, 'subsample': 0.7522031272629728, 'colsample_bytree': 0.7447900016549424, 'gamma': 0.07517427559839324, 'lambda': 0.003418412249941654, 'alpha': 1.4785493400943501}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 03:20:47,262] Trial 33 finished with value: 0.9793296333743371 and parameters: {'n_estimators': 1216, 'learning_rate': 0.056590185360466234, 'max_depth': 13, 'subsample': 0.5767362722494374, 'colsample_bytree': 0.8690682685598177, 'gamma': 0.18059901562492775, 'lambda': 0.0005533923636171783, 'alpha': 9.76071186433849}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 03:31:48,837] Trial 34 finished with value: 0.9819599759820545 and parameters: {'n_estimators': 1824, 'learning_rate': 0.01433118092150991, 'max_depth': 15, 'subsample': 0.6640551984685292, 'colsample_bytree': 0.9097218304533068, 'gamma': 0.26152500490813685, 'lambda': 0.3816171037721248, 'alpha': 6.364788333992061e-05}. Best is trial 20 with value: 0.9860914182020476.
        [I 2025-05-18 03:38:45,013] Trial 35 finished with value: 0.9867239343080104 and parameters: {'n_estimators': 1961, 'learning_rate': 0.08148819575292222, 'max_depth': 14, 'subsample': 0.7211670065391603, 'colsample_bytree': 0.84674190405471, 'gamma': 0.3677495786830702, 'lambda': 0.012339402904032905, 'alpha': 1.1481405582782804}. Best is trial 35 with value: 0.9867239343080104.
        [I 2025-05-18 03:44:52,583] Trial 36 finished with value: 0.986220323456852 and parameters: {'n_estimators': 2163, 'learning_rate': 0.08656757013193256, 'max_depth': 13, 'subsample': 0.709837912631171, 'colsample_bytree': 0.8559118625356164, 'gamma': 0.3852395829770314, 'lambda': 0.23715037291485583, 'alpha': 0.022705744833505775}. Best is trial 35 with value: 0.9867239343080104.
        [I 2025-05-18 03:48:47,338] Trial 37 finished with value: 0.9850671695160074 and parameters: {'n_estimators': 2266, 'learning_rate': 0.20665753838493744, 'max_depth': 12, 'subsample': 0.7289066209821345, 'colsample_bytree': 0.9425782459841792, 'gamma': 0.38481037544123997, 'lambda': 0.9126959327257776, 'alpha': 0.025951867068802866}. Best is trial 35 with value: 0.9867239343080104.
        [I 2025-05-18 03:53:46,030] Trial 38 finished with value: 0.9862428298698291 and parameters: {'n_estimators': 2115, 'learning_rate': 0.0950258130502709, 'max_depth': 13, 'subsample': 0.7852822206946048, 'colsample_bytree': 0.8924682169629599, 'gamma': 0.4818544711608178, 'lambda': 0.10495809455620367, 'alpha': 0.0004739461129161449}. Best is trial 35 with value: 0.9867239343080104.
        [I 2025-05-18 03:57:25,971] Trial 39 finished with value: 0.9827956547194193 and parameters: {'n_estimators': 2207, 'learning_rate': 0.09523154861061227, 'max_depth': 10, 'subsample': 0.7845799465439514, 'colsample_bytree': 0.994149398346107, 'gamma': 0.6323060802410889, 'lambda': 0.13712621329404526, 'alpha': 0.00026827114063725986}. Best is trial 35 with value: 0.9867239343080104.
        [I 2025-05-18 04:02:41,702] Trial 40 finished with value: 0.9856678434997315 and parameters: {'n_estimators': 1992, 'learning_rate': 0.10326200254541172, 'max_depth': 13, 'subsample': 0.8092273362671708, 'colsample_bytree': 0.8996196749465734, 'gamma': 0.4792249358292987, 'lambda': 7.138925566158392, 'alpha': 3.772474346200693e-06}. Best is trial 35 with value: 0.9867239343080104.
        [I 2025-05-18 04:09:37,679] Trial 41 finished with value: 0.9868991283507776 and parameters: {'n_estimators': 2104, 'learning_rate': 0.0704168538408898, 'max_depth': 14, 'subsample': 0.7606516160412129, 'colsample_bytree': 0.8589145057140283, 'gamma': 0.3814671703583167, 'lambda': 0.02826908505045723, 'alpha': 0.16220644839732679}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:11:57,218] Trial 42 finished with value: 0.9826218448846854 and parameters: {'n_estimators': 2109, 'learning_rate': 0.27449481974989043, 'max_depth': 13, 'subsample': 0.6968408096822988, 'colsample_bytree': 0.8530067856476531, 'gamma': 0.5916609354152665, 'lambda': 0.02924180680229433, 'alpha': 0.0019942116671117335}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:17:24,623] Trial 43 finished with value: 0.9865694551176777 and parameters: {'n_estimators': 2144, 'learning_rate': 0.08909198902872933, 'max_depth': 14, 'subsample': 0.8261164293585093, 'colsample_bytree': 0.8832696160733673, 'gamma': 0.4335628568401465, 'lambda': 0.39444690341986166, 'alpha': 0.2048367468137038}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:21:04,130] Trial 44 finished with value: 0.9857744092865167 and parameters: {'n_estimators': 1911, 'learning_rate': 0.16660654541057654, 'max_depth': 14, 'subsample': 0.7716073024965986, 'colsample_bytree': 0.8845720480405255, 'gamma': 0.4165092533433429, 'lambda': 0.3760959567786662, 'alpha': 0.20293470960787363}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:23:09,772] Trial 45 finished with value: 0.9832146891186533 and parameters: {'n_estimators': 2386, 'learning_rate': 0.31178344595632135, 'max_depth': 12, 'subsample': 0.8265569887385398, 'colsample_bytree': 0.9054294837113531, 'gamma': 0.7665343936999394, 'lambda': 1.865352359853347, 'alpha': 0.017859497738657273}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:37:03,257] Trial 46 finished with value: 0.9503591413509217 and parameters: {'n_estimators': 2139, 'learning_rate': 0.0012915449363217432, 'max_depth': 13, 'subsample': 0.7685465147226331, 'colsample_bytree': 0.9420714712486465, 'gamma': 0.4572720368238271, 'lambda': 0.11927577849562578, 'alpha': 0.00039275121638362005}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:41:17,598] Trial 47 finished with value: 0.9854088998683661 and parameters: {'n_estimators': 2331, 'learning_rate': 0.13220312965308356, 'max_depth': 11, 'subsample': 0.8432117262905029, 'colsample_bytree': 0.9247021700127378, 'gamma': 0.36598021719575496, 'lambda': 0.052982973444946395, 'alpha': 8.024069797870954e-06}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:44:14,137] Trial 48 finished with value: 0.9787891432325024 and parameters: {'n_estimators': 2086, 'learning_rate': 0.08501156343319827, 'max_depth': 9, 'subsample': 0.8015827667311771, 'colsample_bytree': 0.8561183973814745, 'gamma': 0.5197270329058616, 'lambda': 0.011572571287486367, 'alpha': 1.1267210050503135e-07}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:46:24,660] Trial 49 finished with value: 0.98452380674593 and parameters: {'n_estimators': 2495, 'learning_rate': 0.1867270151929091, 'max_depth': 14, 'subsample': 0.9079888536892778, 'colsample_bytree': 0.9711717752076867, 'gamma': 0.5719712404949467, 'lambda': 1.5601257890507785, 'alpha': 0.0017768938256628802}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:50:26,998] Trial 50 finished with value: 0.9856794504304878 and parameters: {'n_estimators': 1900, 'learning_rate': 0.1205261816652673, 'max_depth': 12, 'subsample': 0.8184201080448814, 'colsample_bytree': 0.8814585601758557, 'gamma': 0.4586789221420917, 'lambda': 0.24137440737294402, 'alpha': 0.15201217466359354}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 04:57:35,853] Trial 51 finished with value: 0.9867423992926169 and parameters: {'n_estimators': 2196, 'learning_rate': 0.07578730125314163, 'max_depth': 14, 'subsample': 0.7112156295688871, 'colsample_bytree': 0.8646857768472801, 'gamma': 0.3763606250884941, 'lambda': 0.06371131994591023, 'alpha': 0.25475321181012733}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:04:59,796] Trial 52 finished with value: 0.9863840976281957 and parameters: {'n_estimators': 2206, 'learning_rate': 0.08634680129413391, 'max_depth': 14, 'subsample': 0.6474588490673616, 'colsample_bytree': 0.8704765055899375, 'gamma': 0.34495532487747327, 'lambda': 0.594811268962626, 'alpha': 0.009384325372262267}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:13:30,889] Trial 53 finished with value: 0.9861757862939499 and parameters: {'n_estimators': 2287, 'learning_rate': 0.06995107815762094, 'max_depth': 14, 'subsample': 0.5988497395296642, 'colsample_bytree': 0.8845583442049176, 'gamma': 0.33593111048982316, 'lambda': 0.6956341821747972, 'alpha': 0.009988985067124166}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:15:25,180] Trial 54 finished with value: 0.943769175787887 and parameters: {'n_estimators': 2018, 'learning_rate': 0.03052350874874676, 'max_depth': 5, 'subsample': 0.6531253660289966, 'colsample_bytree': 0.9180044359676799, 'gamma': 0.2763611717137594, 'lambda': 0.09111541060201951, 'alpha': 0.00015207813475690994}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:19:17,282] Trial 55 finished with value: 0.9862042934127546 and parameters: {'n_estimators': 2395, 'learning_rate': 0.14862965144603973, 'max_depth': 14, 'subsample': 0.7889378695675593, 'colsample_bytree': 0.8694564195056887, 'gamma': 0.34129554133299794, 'lambda': 0.020504260523071492, 'alpha': 0.0008019871948761676}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:30:23,514] Trial 56 finished with value: 0.9632384055004022 and parameters: {'n_estimators': 2202, 'learning_rate': 0.003533653058137867, 'max_depth': 13, 'subsample': 0.850113229748425, 'colsample_bytree': 0.7802834985530666, 'gamma': 0.4930323942683901, 'lambda': 0.010112750040283777, 'alpha': 0.0035531487476770955}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:37:29,835] Trial 57 finished with value: 0.985787731208478 and parameters: {'n_estimators': 1731, 'learning_rate': 0.043582056645884, 'max_depth': 14, 'subsample': 0.7621513509548348, 'colsample_bytree': 0.8982737703341084, 'gamma': 0.42010422851289553, 'lambda': 0.06486506907488991, 'alpha': 0.2768412968242799}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:44:39,140] Trial 58 finished with value: 0.9854896009022058 and parameters: {'n_estimators': 1922, 'learning_rate': 0.11228549224562959, 'max_depth': 15, 'subsample': 0.7419443991633283, 'colsample_bytree': 0.584255260741197, 'gamma': 0.2825738702560525, 'lambda': 4.716263053864094, 'alpha': 3.237738388632324}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:47:46,692] Trial 59 finished with value: 0.9823555598357774 and parameters: {'n_estimators': 2049, 'learning_rate': 0.33857403565644223, 'max_depth': 13, 'subsample': 0.6393417788029663, 'colsample_bytree': 0.8392575973452983, 'gamma': 0.40124849363579507, 'lambda': 0.711204392693664, 'alpha': 0.730132008642274}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:53:10,633] Trial 60 finished with value: 0.9865354265831559 and parameters: {'n_estimators': 2435, 'learning_rate': 0.06879248161224488, 'max_depth': 15, 'subsample': 0.8629566512294817, 'colsample_bytree': 0.7977747173265597, 'gamma': 0.4477273884187639, 'lambda': 0.04542380771897262, 'alpha': 0.04309404963033434}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 05:58:13,429] Trial 61 finished with value: 0.9864817604608809 and parameters: {'n_estimators': 2420, 'learning_rate': 0.07917304039028436, 'max_depth': 15, 'subsample': 0.8676290829026923, 'colsample_bytree': 0.7965855789967763, 'gamma': 0.43312875120917216, 'lambda': 0.0545155170664313, 'alpha': 0.03430342832669055}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:03:33,518] Trial 62 finished with value: 0.9864625029536402 and parameters: {'n_estimators': 2442, 'learning_rate': 0.0700907696055756, 'max_depth': 15, 'subsample': 0.8686926170629071, 'colsample_bytree': 0.789996735617481, 'gamma': 0.4508382777239839, 'lambda': 0.008267752898160551, 'alpha': 0.04050091728780911}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:10:42,326] Trial 63 finished with value: 0.9863303724418219 and parameters: {'n_estimators': 2427, 'learning_rate': 0.04616951093813746, 'max_depth': 15, 'subsample': 0.8953437549027068, 'colsample_bytree': 0.786340861075336, 'gamma': 0.5143675716330861, 'lambda': 0.010064228681701867, 'alpha': 0.03290534015877739}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:16:19,468] Trial 64 finished with value: 0.9865128880469679 and parameters: {'n_estimators': 2304, 'learning_rate': 0.06552442428047707, 'max_depth': 15, 'subsample': 0.8634578723014344, 'colsample_bytree': 0.8047851714376449, 'gamma': 0.44329427019129913, 'lambda': 0.03858736333847899, 'alpha': 0.11383076808396017}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:22:21,160] Trial 65 finished with value: 0.9864921752776795 and parameters: {'n_estimators': 2299, 'learning_rate': 0.0634708230465487, 'max_depth': 15, 'subsample': 0.865703036086721, 'colsample_bytree': 0.738244160717777, 'gamma': 0.44599315148434554, 'lambda': 0.05158322955023464, 'alpha': 0.15964565357556346}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:28:49,082] Trial 66 finished with value: 0.9821569427351894 and parameters: {'n_estimators': 2285, 'learning_rate': 0.026897343007744975, 'max_depth': 14, 'subsample': 0.9161272355498197, 'colsample_bytree': 0.6730892820777106, 'gamma': 0.9144981262672452, 'lambda': 0.03575587308701628, 'alpha': 0.11127060576724915}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:34:08,626] Trial 67 finished with value: 0.9859677447337502 and parameters: {'n_estimators': 2332, 'learning_rate': 0.05858596172722831, 'max_depth': 15, 'subsample': 0.8885256446280885, 'colsample_bytree': 0.8241007203654488, 'gamma': 0.5443761324344004, 'lambda': 0.019784027618300434, 'alpha': 0.296099643686057}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:43:09,951] Trial 68 finished with value: 0.9820508640980871 and parameters: {'n_estimators': 2215, 'learning_rate': 0.018393442300420716, 'max_depth': 14, 'subsample': 0.9652963818846587, 'colsample_bytree': 0.7400374350218889, 'gamma': 0.580048205309525, 'lambda': 0.0021950253563051904, 'alpha': 0.8357557236076552}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:45:43,721] Trial 69 finished with value: 0.9634808567162632 and parameters: {'n_estimators': 2313, 'learning_rate': 0.046691558057740265, 'max_depth': 7, 'subsample': 0.8377067086307813, 'colsample_bytree': 0.762038258927524, 'gamma': 0.3796662125361834, 'lambda': 0.17659342576439643, 'alpha': 0.4679618023913234}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:54:12,138] Trial 70 finished with value: 0.9852300411704388 and parameters: {'n_estimators': 1955, 'learning_rate': 0.03828975649020561, 'max_depth': 15, 'subsample': 0.8535685144994577, 'colsample_bytree': 0.8058419114955419, 'gamma': 0.4398960905058707, 'lambda': 6.504896024597212e-08, 'alpha': 4.053254039115158}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 06:58:59,772] Trial 71 finished with value: 0.9862916946581349 and parameters: {'n_estimators': 2439, 'learning_rate': 0.07530470476489713, 'max_depth': 15, 'subsample': 0.862305790951109, 'colsample_bytree': 0.700806636705095, 'gamma': 0.43024523875809884, 'lambda': 0.05316186329964265, 'alpha': 0.08222386400474643}. Best is trial 41 with value: 0.9868991283507776.
        [I 2025-05-18 07:06:21,336] Trial 72 finished with value: 0.9870821735461613 and parameters: {'n_estimators': 2360, 'learning_rate': 0.06129925300575542, 'max_depth': 14, 'subsample': 0.8808035408597943, 'colsample_bytree': 0.7343224953487896, 'gamma': 0.3057738154991831, 'lambda': 0.039938266045508757, 'alpha': 0.19727209733436937}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:13:13,044] Trial 73 finished with value: 0.9869526252595275 and parameters: {'n_estimators': 2249, 'learning_rate': 0.063149852981033, 'max_depth': 14, 'subsample': 0.8969793012334497, 'colsample_bytree': 0.732335542550544, 'gamma': 0.3195092460729692, 'lambda': 0.02773988561500493, 'alpha': 0.2111960186576006}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:17:19,097] Trial 74 finished with value: 0.9864262240744941 and parameters: {'n_estimators': 2251, 'learning_rate': 0.11081027597045391, 'max_depth': 14, 'subsample': 0.9091516211533601, 'colsample_bytree': 0.6572134563089824, 'gamma': 0.29928163197967167, 'lambda': 0.0031843732622556732, 'alpha': 0.3651901770049153}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:25:17,537] Trial 75 finished with value: 0.9857242204630171 and parameters: {'n_estimators': 2069, 'learning_rate': 0.0584038725028168, 'max_depth': 14, 'subsample': 0.6896471689593658, 'colsample_bytree': 0.70034594358523, 'gamma': 0.35010698127108414, 'lambda': 0.017443013472179948, 'alpha': 2.4684734798644583}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:28:00,003] Trial 76 finished with value: 0.9694240180329501 and parameters: {'n_estimators': 2181, 'learning_rate': 0.05046200476801789, 'max_depth': 8, 'subsample': 0.947770584634526, 'colsample_bytree': 0.7271316631282245, 'gamma': 0.24622413799638548, 'lambda': 0.006584304554285708, 'alpha': 1.1356133751777437}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:37:11,563] Trial 77 finished with value: 0.9863076743320993 and parameters: {'n_estimators': 2346, 'learning_rate': 0.03971902566615107, 'max_depth': 14, 'subsample': 0.7241619726563794, 'colsample_bytree': 0.8387658704245442, 'gamma': 0.3117775166352686, 'lambda': 0.0019069424959695742, 'alpha': 0.05806951352887347}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:44:13,011] Trial 78 finished with value: 0.9839620928358443 and parameters: {'n_estimators': 2247, 'learning_rate': 0.031015105062451538, 'max_depth': 13, 'subsample': 0.9240332463861916, 'colsample_bytree': 0.7689816797623702, 'gamma': 0.3935338077793531, 'lambda': 0.18603981364333774, 'alpha': 0.17119131406275984}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:49:32,328] Trial 79 finished with value: 0.9857107917405565 and parameters: {'n_estimators': 2159, 'learning_rate': 0.133804087184111, 'max_depth': 14, 'subsample': 0.897915692615578, 'colsample_bytree': 0.8047883594177991, 'gamma': 0.3264442593015109, 'lambda': 0.028309068724896597, 'alpha': 6.657917766911489}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:51:22,491] Trial 80 finished with value: 0.941219856074277 and parameters: {'n_estimators': 2472, 'learning_rate': 0.10104114255582479, 'max_depth': 3, 'subsample': 0.8805099149931838, 'colsample_bytree': 0.7513645650494196, 'gamma': 0.4050328883261925, 'lambda': 0.09816899851873297, 'alpha': 0.590403702955011}. Best is trial 72 with value: 0.9870821735461613.
        [I 2025-05-18 07:58:56,316] Trial 81 finished with value: 0.9871777819748337 and parameters: {'n_estimators': 2365, 'learning_rate': 0.06484530828688133, 'max_depth': 15, 'subsample': 0.8193131469026914, 'colsample_bytree': 0.7382062699145382, 'gamma': 0.36607924388692653, 'lambda': 0.36731062425223354, 'alpha': 0.22413598365642257}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:06:47,291] Trial 82 finished with value: 0.9871619961923918 and parameters: {'n_estimators': 2366, 'learning_rate': 0.062409305349043674, 'max_depth': 15, 'subsample': 0.8060741238338825, 'colsample_bytree': 0.7304286033599195, 'gamma': 0.3668588654272414, 'lambda': 0.3592184801933573, 'alpha': 0.2602286199043307}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:11:54,788] Trial 83 finished with value: 0.9865944085348696 and parameters: {'n_estimators': 2375, 'learning_rate': 0.08846469811529402, 'max_depth': 15, 'subsample': 0.8138602018798915, 'colsample_bytree': 0.6890906363018433, 'gamma': 0.3568722150636854, 'lambda': 0.2690708541872698, 'alpha': 0.2623409614488584}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:16:35,119] Trial 84 finished with value: 0.9860149024547529 and parameters: {'n_estimators': 2379, 'learning_rate': 0.15156280938080421, 'max_depth': 14, 'subsample': 0.815247516756876, 'colsample_bytree': 0.6882191703667089, 'gamma': 0.36562155232629684, 'lambda': 0.334934251388936, 'alpha': 1.724065603333951}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:22:32,107] Trial 85 finished with value: 0.9867508906143925 and parameters: {'n_estimators': 2122, 'learning_rate': 0.09851175979550397, 'max_depth': 13, 'subsample': 0.7995862597343153, 'colsample_bytree': 0.7301081348145928, 'gamma': 0.2695671269906868, 'lambda': 0.3720546968311421, 'alpha': 0.252990744927445}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:29:38,060] Trial 86 finished with value: 0.9850007414596706 and parameters: {'n_estimators': 2357, 'learning_rate': 0.052640332329345126, 'max_depth': 13, 'subsample': 0.7159835383926354, 'colsample_bytree': 0.6515126390102405, 'gamma': 0.2586658442245525, 'lambda': 1.224146965761701, 'alpha': 0.0152991723934936}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:32:26,650] Trial 87 finished with value: 0.9853250888392555 and parameters: {'n_estimators': 1852, 'learning_rate': 0.22868307188872694, 'max_depth': 15, 'subsample': 0.7928813618429722, 'colsample_bytree': 0.7302438800404568, 'gamma': 0.28877478054592326, 'lambda': 0.16318880978550343, 'alpha': 0.3691891908375184}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:38:02,898] Trial 88 finished with value: 0.9853110954252612 and parameters: {'n_estimators': 2240, 'learning_rate': 0.123350010915782, 'max_depth': 12, 'subsample': 0.7365709790831979, 'colsample_bytree': 0.7183800389927706, 'gamma': 0.31510189981606296, 'lambda': 3.748814109511486, 'alpha': 1.1214759646867503}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:46:07,850] Trial 89 finished with value: 0.9720448335222522 and parameters: {'n_estimators': 2129, 'learning_rate': 0.008953237296373618, 'max_depth': 13, 'subsample': 0.8018906690213159, 'colsample_bytree': 0.70385126700732, 'gamma': 0.17439528286726358, 'lambda': 2.089343452610327, 'alpha': 0.2216582281615626}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 08:52:57,663] Trial 90 finished with value: 0.9866298855257383 and parameters: {'n_estimators': 2044, 'learning_rate': 0.10479012164573305, 'max_depth': 14, 'subsample': 0.7775291559656712, 'colsample_bytree': 0.6875738504840576, 'gamma': 0.26462756506868323, 'lambda': 3.2238438502608573, 'alpha': 0.07981999978952944}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:00:04,568] Trial 91 finished with value: 0.9866135750723863 and parameters: {'n_estimators': 2085, 'learning_rate': 0.10171044025175913, 'max_depth': 14, 'subsample': 0.7737899700569111, 'colsample_bytree': 0.6856646025664306, 'gamma': 0.2596775465810603, 'lambda': 2.5211976532928575, 'alpha': 0.6979994531633891}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:07:00,396] Trial 92 finished with value: 0.9864392125095455 and parameters: {'n_estimators': 2002, 'learning_rate': 0.10064060255462089, 'max_depth': 14, 'subsample': 0.7758102415046687, 'colsample_bytree': 0.6674209712093528, 'gamma': 0.2648523129379107, 'lambda': 3.1511099131218483, 'alpha': 0.5699559733468282}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:12:57,083] Trial 93 finished with value: 0.9859806393042425 and parameters: {'n_estimators': 1962, 'learning_rate': 0.08003386239775406, 'max_depth': 13, 'subsample': 0.7573323201778254, 'colsample_bytree': 0.6876878819818795, 'gamma': 0.15225637411703574, 'lambda': 0.5016558885290245, 'alpha': 0.09353510744153831}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:19:32,090] Trial 94 finished with value: 0.985744196069138 and parameters: {'n_estimators': 2034, 'learning_rate': 0.17575482131880465, 'max_depth': 14, 'subsample': 0.7498337421969538, 'colsample_bytree': 0.7141556593742027, 'gamma': 0.23197917796179074, 'lambda': 9.679395197169871, 'alpha': 0.9691703141811848}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:27:53,180] Trial 95 finished with value: 0.9861426313431246 and parameters: {'n_estimators': 2098, 'learning_rate': 0.06154797184801452, 'max_depth': 14, 'subsample': 0.7698640626837148, 'colsample_bytree': 0.7327717728709374, 'gamma': 0.21084028710479824, 'lambda': 1.1591352156904058, 'alpha': 2.354031040179216}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:33:28,111] Trial 96 finished with value: 0.9852043358169594 and parameters: {'n_estimators': 2170, 'learning_rate': 0.1115831491564787, 'max_depth': 12, 'subsample': 0.7797786968120014, 'colsample_bytree': 0.7542019403997962, 'gamma': 0.10337816788784493, 'lambda': 6.350334520442132, 'alpha': 0.4373557159444159}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:40:53,746] Trial 97 finished with value: 0.9858185537659667 and parameters: {'n_estimators': 2077, 'learning_rate': 0.07515126537049598, 'max_depth': 14, 'subsample': 0.7169396890905966, 'colsample_bytree': 0.6196667400608978, 'gamma': 0.32023409514294693, 'lambda': 3.26956217115763, 'alpha': 0.06064777155003413}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:45:29,710] Trial 98 finished with value: 0.9859637088074584 and parameters: {'n_estimators': 1853, 'learning_rate': 0.14209817266815367, 'max_depth': 13, 'subsample': 0.7958990412861797, 'colsample_bytree': 0.6303277083448157, 'gamma': 0.28394924595160875, 'lambda': 0.9288435360883431, 'alpha': 0.14144557197854313}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 09:53:33,386] Trial 99 finished with value: 0.9855511701119548 and parameters: {'n_estimators': 2234, 'learning_rate': 0.09268786708402073, 'max_depth': 15, 'subsample': 0.7469914595025045, 'colsample_bytree': 0.7108035106237939, 'gamma': 0.3322798520143788, 'lambda': 0.5050832888043156, 'alpha': 6.242454704332096}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:01:50,074] Trial 100 finished with value: 0.986289858753636 and parameters: {'n_estimators': 2125, 'learning_rate': 0.04200907837352452, 'max_depth': 14, 'subsample': 0.8329071312294485, 'colsample_bytree': 0.6536454025236577, 'gamma': 0.24901575030953532, 'lambda': 0.09180035536520285, 'alpha': 0.7675060895748556}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:08:06,795] Trial 101 finished with value: 0.9868783493294956 and parameters: {'n_estimators': 2382, 'learning_rate': 0.09055103705019289, 'max_depth': 15, 'subsample': 0.8121790993635442, 'colsample_bytree': 0.698974993317169, 'gamma': 0.357114046146373, 'lambda': 2.4875769395997898, 'alpha': 0.31047181752204}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:16:32,296] Trial 102 finished with value: 0.9868741352932636 and parameters: {'n_estimators': 2487, 'learning_rate': 0.05516904307959775, 'max_depth': 15, 'subsample': 0.824296265029483, 'colsample_bytree': 0.7246029515922061, 'gamma': 0.37562722915432856, 'lambda': 2.0316648530275634, 'alpha': 0.2776614476738798}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:25:26,365] Trial 103 finished with value: 0.9870612380183564 and parameters: {'n_estimators': 2487, 'learning_rate': 0.05029316195629177, 'max_depth': 15, 'subsample': 0.8184560873262151, 'colsample_bytree': 0.7253303357644955, 'gamma': 0.3709629946782507, 'lambda': 1.1469720279329116, 'alpha': 0.20262206280947828}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:33:45,254] Trial 104 finished with value: 0.9870047789250006 and parameters: {'n_estimators': 2494, 'learning_rate': 0.050218387667782215, 'max_depth': 15, 'subsample': 0.8429979194390597, 'colsample_bytree': 0.7468030306797847, 'gamma': 0.3856942814240154, 'lambda': 1.3032521778109414, 'alpha': 0.23496442424779107}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:43:29,442] Trial 105 finished with value: 0.9866689890923854 and parameters: {'n_estimators': 2484, 'learning_rate': 0.035729585455759105, 'max_depth': 15, 'subsample': 0.8422132227340449, 'colsample_bytree': 0.7494699094072396, 'gamma': 0.3787698585645067, 'lambda': 1.2017806659712291, 'alpha': 0.2009115559771003}. Best is trial 81 with value: 0.9871777819748337.
        [I 2025-05-18 10:52:06,206] Trial 106 finished with value: 0.987247534514811 and parameters: {'n_estimators': 2487, 'learning_rate': 0.04967551548387531, 'max_depth': 15, 'subsample': 0.8252641200539966, 'colsample_bytree': 0.7737740714037915, 'gamma': 0.34952521528481467, 'lambda': 0.7330641062318338, 'alpha': 0.31425756893058554}. Best is trial 106 with value: 0.987247534514811.
        [I 2025-05-18 11:00:32,331] Trial 107 finished with value: 0.9871953862038371 and parameters: {'n_estimators': 2500, 'learning_rate': 0.05135727332977585, 'max_depth': 15, 'subsample': 0.8080345676703272, 'colsample_bytree': 0.7609074018362313, 'gamma': 0.3515416753330241, 'lambda': 0.8120460332508996, 'alpha': 1.794766097403972e-08}. Best is trial 106 with value: 0.987247534514811.
        [I 2025-05-18 11:10:53,776] Trial 108 finished with value: 0.9858443021030191 and parameters: {'n_estimators': 2489, 'learning_rate': 0.02735849299039239, 'max_depth': 15, 'subsample': 0.8244352904002, 'colsample_bytree': 0.7700649698213049, 'gamma': 0.4039536538433286, 'lambda': 1.5585783260062744, 'alpha': 5.32728899315881e-08}. Best is trial 106 with value: 0.987247534514811.
        [I 2025-05-18 11:19:00,579] Trial 109 finished with value: 0.9872050350737241 and parameters: {'n_estimators': 2407, 'learning_rate': 0.04932215623463175, 'max_depth': 15, 'subsample': 0.850383727884743, 'colsample_bytree': 0.7781178560883784, 'gamma': 0.35055445877383673, 'lambda': 0.6408887944589198, 'alpha': 1.0326888638598633e-08}. Best is trial 106 with value: 0.987247534514811.
        [I 2025-05-18 11:27:53,520] Trial 110 finished with value: 0.9871878553789286 and parameters: {'n_estimators': 2400, 'learning_rate': 0.04662812740277815, 'max_depth': 15, 'subsample': 0.8481553315115857, 'colsample_bytree': 0.7607239430434414, 'gamma': 0.33901596798477257, 'lambda': 0.7893076133458468, 'alpha': 1.3464579385839153e-08}. Best is trial 106 with value: 0.987247534514811.
        [I 2025-05-18 11:37:12,962] Trial 111 finished with value: 0.9873415889410657 and parameters: {'n_estimators': 2397, 'learning_rate': 0.04899742663656813, 'max_depth': 15, 'subsample': 0.8424042214542694, 'colsample_bytree': 0.7799914949111625, 'gamma': 0.2997531858054152, 'lambda': 0.8283606956491497, 'alpha': 1.1699720328135459e-08}. Best is trial 111 with value: 0.9873415889410657.
        [I 2025-05-18 11:48:59,038] Trial 112 finished with value: 0.9868264208405779 and parameters: {'n_estimators': 2413, 'learning_rate': 0.03356974649920845, 'max_depth': 15, 'subsample': 0.849949695604873, 'colsample_bytree': 0.7772133685186248, 'gamma': 0.34185944842244376, 'lambda': 0.6640403292618899, 'alpha': 1.0090354816077498e-08}. Best is trial 111 with value: 0.9873415889410657.
        [I 2025-05-18 11:58:42,328] Trial 113 finished with value: 0.9871991341602457 and parameters: {'n_estimators': 2437, 'learning_rate': 0.043183629590133284, 'max_depth': 15, 'subsample': 0.8788554150206154, 'colsample_bytree': 0.75958519636032, 'gamma': 0.3220913457586469, 'lambda': 0.8413233467785995, 'alpha': 2.0828629244109084e-08}. Best is trial 111 with value: 0.9873415889410657.
        [I 2025-05-18 12:07:05,939] Trial 114 finished with value: 0.9872743516761133 and parameters: {'n_estimators': 2451, 'learning_rate': 0.050059232587786905, 'max_depth': 15, 'subsample': 0.8770100404036086, 'colsample_bytree': 0.7580336938894974, 'gamma': 0.31738144388435063, 'lambda': 0.9348650951762247, 'alpha': 1.9265353325280373e-08}. Best is trial 111 with value: 0.9873415889410657.
        [I 2025-05-18 12:15:47,643] Trial 115 finished with value: 0.9873733959733606 and parameters: {'n_estimators': 2448, 'learning_rate': 0.048462394509859195, 'max_depth': 15, 'subsample': 0.8765098230090178, 'colsample_bytree': 0.7589625687982171, 'gamma': 0.2988190688855731, 'lambda': 0.9010994701453632, 'alpha': 2.3652859971952107e-08}. Best is trial 115 with value: 0.9873733959733606.
        [I 2025-05-18 12:24:57,278] Trial 116 finished with value: 0.9873346757320903 and parameters: {'n_estimators': 2441, 'learning_rate': 0.04325291599150339, 'max_depth': 15, 'subsample': 0.8736062650689151, 'colsample_bytree': 0.7816737097361934, 'gamma': 0.2955845393875145, 'lambda': 0.6516765617783813, 'alpha': 1.858896060740952e-08}. Best is trial 115 with value: 0.9873733959733606.
        [I 2025-05-18 12:33:59,384] Trial 117 finished with value: 0.9873214788272776 and parameters: {'n_estimators': 2446, 'learning_rate': 0.043988544355184685, 'max_depth': 15, 'subsample': 0.874450176207771, 'colsample_bytree': 0.7812882237256196, 'gamma': 0.30058427640145713, 'lambda': 0.7610416035228766, 'alpha': 2.1063198587268283e-08}. Best is trial 115 with value: 0.9873733959733606.
        [I 2025-05-18 12:45:00,921] Trial 118 finished with value: 0.985429426511102 and parameters: {'n_estimators': 2442, 'learning_rate': 0.02156364889453994, 'max_depth': 15, 'subsample': 0.8565247638961231, 'colsample_bytree': 0.7571342497046983, 'gamma': 0.3003312737174191, 'lambda': 0.6598140598422675, 'alpha': 2.3888825242441383e-08}. Best is trial 115 with value: 0.9873733959733606.
        [I 2025-05-18 12:54:12,811] Trial 119 finished with value: 0.9873873053765879 and parameters: {'n_estimators': 2404, 'learning_rate': 0.04164054688849172, 'max_depth': 15, 'subsample': 0.8742551277110467, 'colsample_bytree': 0.7820257997337541, 'gamma': 0.28577573432282644, 'lambda': 0.2644763213229494, 'alpha': 2.7619820968811404e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 13:04:52,284] Trial 120 finished with value: 0.9861844829957703 and parameters: {'n_estimators': 2454, 'learning_rate': 0.02660343124111994, 'max_depth': 15, 'subsample': 0.8745176506511457, 'colsample_bytree': 0.78033845176588, 'gamma': 0.2951653145833996, 'lambda': 0.8152263255514167, 'alpha': 2.0930564594980576e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 13:13:17,394] Trial 121 finished with value: 0.9871726342823443 and parameters: {'n_estimators': 2400, 'learning_rate': 0.042377412681157106, 'max_depth': 15, 'subsample': 0.8889698437255684, 'colsample_bytree': 0.7597865888184572, 'gamma': 0.3338505716605857, 'lambda': 0.2508176397692741, 'alpha': 4.2377954374123146e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 13:21:42,377] Trial 122 finished with value: 0.98725536544899 and parameters: {'n_estimators': 2399, 'learning_rate': 0.043094084364884475, 'max_depth': 15, 'subsample': 0.8860625780260715, 'colsample_bytree': 0.765942856264843, 'gamma': 0.32983991277763913, 'lambda': 0.27658361029269923, 'alpha': 3.7467761736554514e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 13:31:01,214] Trial 123 finished with value: 0.987196820725242 and parameters: {'n_estimators': 2318, 'learning_rate': 0.039479087873652216, 'max_depth': 15, 'subsample': 0.8797727723556815, 'colsample_bytree': 0.7920245966512272, 'gamma': 0.2851553876640029, 'lambda': 0.5689572948607535, 'alpha': 1.8007343253339606e-07}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 13:40:46,120] Trial 124 finished with value: 0.9869741129837736 and parameters: {'n_estimators': 2314, 'learning_rate': 0.03565857030117601, 'max_depth': 15, 'subsample': 0.8752470557234667, 'colsample_bytree': 0.7886881559655927, 'gamma': 0.23844103394396693, 'lambda': 0.5276146074590543, 'alpha': 1.7229335377280579e-07}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 13:52:21,480] Trial 125 finished with value: 0.9852533740872532 and parameters: {'n_estimators': 2414, 'learning_rate': 0.01885347704420024, 'max_depth': 15, 'subsample': 0.8864878712790922, 'colsample_bytree': 0.8126554967534008, 'gamma': 0.22066058935118793, 'lambda': 0.14743579779074314, 'alpha': 1.5336025535430775e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 14:02:35,418] Trial 126 finished with value: 0.9866357420763399 and parameters: {'n_estimators': 2437, 'learning_rate': 0.030546179168307697, 'max_depth': 15, 'subsample': 0.9135029164254239, 'colsample_bytree': 0.7697766941823476, 'gamma': 0.2818028055596106, 'lambda': 0.9257378942307876, 'alpha': 3.5809665715540006e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 14:11:26,991] Trial 127 finished with value: 0.9863188494499348 and parameters: {'n_estimators': 2323, 'learning_rate': 0.04445279912065863, 'max_depth': 15, 'subsample': 0.9027934537347162, 'colsample_bytree': 0.7958504632808088, 'gamma': 0.309689356833891, 'lambda': 5.294548811792661, 'alpha': 1.0704266210760984e-07}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 14:20:24,704] Trial 128 finished with value: 0.9872263527973527 and parameters: {'n_estimators': 2455, 'learning_rate': 0.0401657511498772, 'max_depth': 15, 'subsample': 0.8509223156803895, 'colsample_bytree': 0.7843777964527353, 'gamma': 0.34188504740671266, 'lambda': 7.299764791985171e-07, 'alpha': 6.406158356333908e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 14:29:17,490] Trial 129 finished with value: 0.9873385669204049 and parameters: {'n_estimators': 2464, 'learning_rate': 0.04004312901592535, 'max_depth': 15, 'subsample': 0.9274226188665118, 'colsample_bytree': 0.7809816478306798, 'gamma': 0.27797006305499655, 'lambda': 1.3824496012094897e-05, 'alpha': 6.898161489242479e-08}. Best is trial 119 with value: 0.9873873053765879.
        [I 2025-05-18 14:38:53,265] Trial 130 finished with value: 0.9875640292867937 and parameters: {'n_estimators': 2276, 'learning_rate': 0.03844051055554991, 'max_depth': 15, 'subsample': 0.9240462730295915, 'colsample_bytree': 0.7843854088875306, 'gamma': 0.17870057468056816, 'lambda': 1.401713227770493e-05, 'alpha': 2.535993244900807e-07}. Best is trial 130 with value: 0.9875640292867937.
        [I 2025-05-18 14:48:35,788] Trial 131 finished with value: 0.9878166629720316 and parameters: {'n_estimators': 2459, 'learning_rate': 0.03999996560758967, 'max_depth': 15, 'subsample': 0.9376778881209966, 'colsample_bytree': 0.7855003885822004, 'gamma': 0.19449049002132435, 'lambda': 4.458692856679572e-06, 'alpha': 2.922184313953772e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 14:58:59,175] Trial 132 finished with value: 0.9875111561866537 and parameters: {'n_estimators': 2456, 'learning_rate': 0.03397302051426409, 'max_depth': 15, 'subsample': 0.9423348083005482, 'colsample_bytree': 0.8199401568918144, 'gamma': 0.1952428836784086, 'lambda': 8.120407419111717e-07, 'alpha': 6.843337305113291e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 15:09:51,059] Trial 133 finished with value: 0.9868937991559037 and parameters: {'n_estimators': 2462, 'learning_rate': 0.027448120944972912, 'max_depth': 15, 'subsample': 0.9670784248908393, 'colsample_bytree': 0.8198078670515636, 'gamma': 0.190893305581402, 'lambda': 1.2438977017701964e-06, 'alpha': 4.820237965511705e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 15:20:18,843] Trial 134 finished with value: 0.9873553402200681 and parameters: {'n_estimators': 2409, 'learning_rate': 0.03207817730781709, 'max_depth': 15, 'subsample': 0.9224313355139376, 'colsample_bytree': 0.7827022504159105, 'gamma': 0.15859669542320942, 'lambda': 6.238342490258195e-06, 'alpha': 7.193770682403063e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 15:30:25,654] Trial 135 finished with value: 0.9873432603905432 and parameters: {'n_estimators': 2350, 'learning_rate': 0.033597663880353054, 'max_depth': 15, 'subsample': 0.9387053613651503, 'colsample_bytree': 0.786700172840062, 'gamma': 0.16370615842369135, 'lambda': 5.930080868194264e-06, 'alpha': 7.279312291687852e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 15:40:59,107] Trial 136 finished with value: 0.9871683544633468 and parameters: {'n_estimators': 2333, 'learning_rate': 0.0319070474093166, 'max_depth': 15, 'subsample': 0.9346240078249235, 'colsample_bytree': 0.801308852046712, 'gamma': 0.16568741005452628, 'lambda': 8.990258104580535e-06, 'alpha': 8.616595317729572e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 15:52:32,246] Trial 137 finished with value: 0.9860217330757822 and parameters: {'n_estimators': 2287, 'learning_rate': 0.023252582106261088, 'max_depth': 15, 'subsample': 0.9470215001458838, 'colsample_bytree': 0.8123115479832995, 'gamma': 0.11730561606480666, 'lambda': 3.683507707819755e-06, 'alpha': 3.094561789021736e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 16:02:53,993] Trial 138 finished with value: 0.9874592647043257 and parameters: {'n_estimators': 2361, 'learning_rate': 0.035034276682463554, 'max_depth': 15, 'subsample': 0.9687638741546695, 'colsample_bytree': 0.7735869715391734, 'gamma': 0.1344862817615808, 'lambda': 7.484834801922105e-06, 'alpha': 3.327642083293211e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 16:11:33,322] Trial 139 finished with value: 0.9862154752549328 and parameters: {'n_estimators': 2356, 'learning_rate': 0.034823076986971915, 'max_depth': 14, 'subsample': 0.9809107396163881, 'colsample_bytree': 0.7782857204115096, 'gamma': 0.09364964615004176, 'lambda': 1.7386924904069253e-05, 'alpha': 4.478044362754981e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 16:22:24,155] Trial 140 finished with value: 0.9867319016816388 and parameters: {'n_estimators': 2288, 'learning_rate': 0.02820617049724375, 'max_depth': 15, 'subsample': 0.9220738947037868, 'colsample_bytree': 0.7878299970146798, 'gamma': 0.04970241955997198, 'lambda': 4.0242764284087495e-06, 'alpha': 2.902801022161066e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 16:32:54,008] Trial 141 finished with value: 0.9877612947396224 and parameters: {'n_estimators': 2390, 'learning_rate': 0.03755940684099901, 'max_depth': 15, 'subsample': 0.943702196280772, 'colsample_bytree': 0.8315195128733719, 'gamma': 0.12442280137428627, 'lambda': 8.258780162522e-06, 'alpha': 7.092090163423286e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 16:43:25,935] Trial 142 finished with value: 0.987756125892131 and parameters: {'n_estimators': 2389, 'learning_rate': 0.0373441056465176, 'max_depth': 15, 'subsample': 0.9426169397849775, 'colsample_bytree': 0.8348890289993507, 'gamma': 0.13003327838511325, 'lambda': 2.6456584450698526e-05, 'alpha': 6.437586239609914e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 16:54:44,233] Trial 143 finished with value: 0.9864147891328989 and parameters: {'n_estimators': 2350, 'learning_rate': 0.024295310879278432, 'max_depth': 15, 'subsample': 0.9423729118947062, 'colsample_bytree': 0.8313856736590636, 'gamma': 0.13482278982907792, 'lambda': 1.2196559610239222e-05, 'alpha': 7.112853117320481e-08}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 17:05:18,918] Trial 144 finished with value: 0.9877997156643907 and parameters: {'n_estimators': 2423, 'learning_rate': 0.0373655505938673, 'max_depth': 15, 'subsample': 0.9741381538960191, 'colsample_bytree': 0.8247127012134795, 'gamma': 0.13160653854705914, 'lambda': 3.365644638739895e-05, 'alpha': 2.049957466102124e-06}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 17:14:04,470] Trial 145 finished with value: 0.9864192109672146 and parameters: {'n_estimators': 2393, 'learning_rate': 0.03590065225353421, 'max_depth': 14, 'subsample': 0.9637983217359463, 'colsample_bytree': 0.845786532851908, 'gamma': 0.12089783852998465, 'lambda': 3.357615068180455e-05, 'alpha': 1.0876425549147485e-06}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 17:16:25,270] Trial 146 finished with value: 0.9520480480773821 and parameters: {'n_estimators': 2372, 'learning_rate': 0.03220458428477149, 'max_depth': 6, 'subsample': 0.9778409316628633, 'colsample_bytree': 0.8161226635529553, 'gamma': 0.15455675202524669, 'lambda': 8.236254883764219e-05, 'alpha': 1.7245142136126883e-06}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 17:20:17,142] Trial 147 finished with value: 0.9694623854411079 and parameters: {'n_estimators': 2277, 'learning_rate': 0.01977209319746182, 'max_depth': 10, 'subsample': 0.9985021036329198, 'colsample_bytree': 0.8288023992883803, 'gamma': 0.19651606837289368, 'lambda': 5.3554289492149485e-06, 'alpha': 1.6253714908559552e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 17:30:46,254] Trial 148 finished with value: 0.9877720095900095 and parameters: {'n_estimators': 2419, 'learning_rate': 0.03812689415014961, 'max_depth': 15, 'subsample': 0.9568994265390411, 'colsample_bytree': 0.8039188561598449, 'gamma': 0.13754273020677527, 'lambda': 2.3811273297010285e-06, 'alpha': 2.6723250419185224e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-18 17:39:23,468] Trial 149 finished with value: 0.9866988449395176 and parameters: {'n_estimators': 2342, 'learning_rate': 0.03784112700523976, 'max_depth': 14, 'subsample': 0.9536701957054081, 'colsample_bytree': 0.8038416880006265, 'gamma': 0.05066443447540406, 'lambda': 2.360053962726179e-06, 'alpha': 2.8855325077793323e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-20 23:45:52,480] Trial 150 finished with value: 0.987522258844853 and parameters: {'n_estimators': 2469, 'learning_rate': 0.03404562016592237, 'max_depth': 15, 'subsample': 0.9284020593798544, 'colsample_bytree': 0.7839310860551069, 'gamma': 0.1781564539824054, 'lambda': 6.533121966029519e-06, 'alpha': 1.1981150531580227e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-20 23:58:51,234] Trial 151 finished with value: 0.9877907575646197 and parameters: {'n_estimators': 2468, 'learning_rate': 0.03684359407333554, 'max_depth': 15, 'subsample': 0.9356194218658826, 'colsample_bytree': 0.7940955610817298, 'gamma': 0.1350135835943232, 'lambda': 6.158525799307771e-06, 'alpha': 1.1641362093557454e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-21 00:11:19,330] Trial 152 finished with value: 0.9877569974205562 and parameters: {'n_estimators': 2472, 'learning_rate': 0.03706222187652726, 'max_depth': 15, 'subsample': 0.9268761230058863, 'colsample_bytree': 0.774069739487561, 'gamma': 0.13785901308323604, 'lambda': 6.680553748283453e-06, 'alpha': 1.253037233857321e-07}. Best is trial 131 with value: 0.9878166629720316.
        [I 2025-05-21 03:45:51,600] Trial 153 finished with value: 0.9882006764516362 and parameters: {'n_estimators': 2526, 'learning_rate': 0.03330372658823485, 'max_depth': 16, 'subsample': 0.9381111666999238, 'colsample_bytree': 0.8260975614946654, 'gamma': 0.13841949514732158, 'lambda': 7.229430932599723e-06, 'alpha': 1.2016573033100657e-07}. Best is trial 155 with value: 0.9883980340337895.
        [I 2025-05-21 03:59:04,863] Trial 154 finished with value: 0.9883980340337895 and parameters: {'n_estimators': 2533, 'learning_rate': 0.03370939109765275, 'max_depth': 16, 'subsample': 0.9346392945761361, 'colsample_bytree': 0.8248079147571348, 'gamma': 0.13687740266008272, 'lambda': 5.88382025732101e-06, 'alpha': 1.343270997105374e-07}. Best is trial 155 with value: 0.9883980340337895.
        [I 2025-05-21 04:11:41,750] Trial 155 finished with value: 0.9884165518428587 and parameters: {'n_estimators': 2532, 'learning_rate': 0.03716084041777274, 'max_depth': 16, 'subsample': 0.9346368307374464, 'colsample_bytree': 0.8374239126447817, 'gamma': 0.14113797686524304, 'lambda': 8.377899478363726e-06, 'alpha': 1.187703398188169e-07}. Best is trial 155 with value: 0.9884165518428587.
        [I 2025-05-21 04:24:25,623] Trial 156 finished with value: 0.9883959275086884 and parameters: {'n_estimators': 2549, 'learning_rate': 0.0369253372297127, 'max_depth': 16, 'subsample': 0.9345890009695679, 'colsample_bytree': 0.8265341258607147, 'gamma': 0.13967593259838048, 'lambda': 2.863761376839359e-05, 'alpha': 2.890393798939634e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 04:36:58,484] Trial 157 finished with value: 0.9884163901967265 and parameters: {'n_estimators': 2544, 'learning_rate': 0.0376429037645163, 'max_depth': 16, 'subsample': 0.9335178496468752, 'colsample_bytree': 0.8259102875153288, 'gamma': 0.14247481308362678, 'lambda': 8.751819616335967e-06, 'alpha': 2.60694892224097e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 04:49:37,095] Trial 158 finished with value: 0.9883460613271705 and parameters: {'n_estimators': 2538, 'learning_rate': 0.03743215286548398, 'max_depth': 16, 'subsample': 0.9331320708687799, 'colsample_bytree': 0.8366202456488346, 'gamma': 0.1386325492225345, 'lambda': 3.321165415707447e-05, 'alpha': 3.114432135017255e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 05:02:14,198] Trial 159 finished with value: 0.9883639893630656 and parameters: {'n_estimators': 2551, 'learning_rate': 0.03685567848694086, 'max_depth': 16, 'subsample': 0.9376490332851897, 'colsample_bytree': 0.8361311704201737, 'gamma': 0.14537652756854152, 'lambda': 2.3280058863645053e-05, 'alpha': 1.2547137535243876e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 05:14:53,652] Trial 160 finished with value: 0.9883873491417537 and parameters: {'n_estimators': 2539, 'learning_rate': 0.03660285227702813, 'max_depth': 16, 'subsample': 0.9335492751433246, 'colsample_bytree': 0.8363729350352516, 'gamma': 0.14406680459997193, 'lambda': 2.65055896350756e-05, 'alpha': 1.2950599553641322e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 05:27:36,315] Trial 161 finished with value: 0.9883612800889254 and parameters: {'n_estimators': 2543, 'learning_rate': 0.036951566810141675, 'max_depth': 16, 'subsample': 0.9346590288863543, 'colsample_bytree': 0.8377172379361311, 'gamma': 0.1410887372793616, 'lambda': 3.190660341762145e-05, 'alpha': 2.199745339233868e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 05:40:19,848] Trial 162 finished with value: 0.9883607062400977 and parameters: {'n_estimators': 2535, 'learning_rate': 0.03656531580214643, 'max_depth': 16, 'subsample': 0.9347868840251115, 'colsample_bytree': 0.8344079017620601, 'gamma': 0.1425296065582372, 'lambda': 3.118782352220719e-05, 'alpha': 2.105370966468774e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 05:52:56,845] Trial 163 finished with value: 0.9883471018697334 and parameters: {'n_estimators': 2547, 'learning_rate': 0.03738744577884789, 'max_depth': 16, 'subsample': 0.9349001397793165, 'colsample_bytree': 0.836867946399997, 'gamma': 0.1404751410118029, 'lambda': 2.809274996455116e-05, 'alpha': 6.211012975878841e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 06:05:32,688] Trial 164 finished with value: 0.9883380299696956 and parameters: {'n_estimators': 2541, 'learning_rate': 0.03710684203175373, 'max_depth': 16, 'subsample': 0.9343157252865913, 'colsample_bytree': 0.8254954932021162, 'gamma': 0.1435829188040081, 'lambda': 4.383508101664103e-05, 'alpha': 6.857399955403061e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 06:18:10,280] Trial 165 finished with value: 0.9883701385419999 and parameters: {'n_estimators': 2548, 'learning_rate': 0.03658206006451079, 'max_depth': 16, 'subsample': 0.9333319080614562, 'colsample_bytree': 0.8259986341235384, 'gamma': 0.1452180026850402, 'lambda': 5.653609200452372e-05, 'alpha': 7.195969062325026e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 06:30:52,637] Trial 166 finished with value: 0.9883754911762104 and parameters: {'n_estimators': 2545, 'learning_rate': 0.03662022987544079, 'max_depth': 16, 'subsample': 0.9350658462002302, 'colsample_bytree': 0.8243910300192534, 'gamma': 0.14404710189999088, 'lambda': 4.066261314643093e-05, 'alpha': 8.074274092652418e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 06:43:29,372] Trial 167 finished with value: 0.9883348637748443 and parameters: {'n_estimators': 2537, 'learning_rate': 0.03649923891465063, 'max_depth': 16, 'subsample': 0.936715340665977, 'colsample_bytree': 0.8241284163997193, 'gamma': 0.14869213899203656, 'lambda': 5.75489725944517e-05, 'alpha': 7.91668228343759e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 06:56:15,251] Trial 168 finished with value: 0.9883904944930104 and parameters: {'n_estimators': 2535, 'learning_rate': 0.03622786315264629, 'max_depth': 16, 'subsample': 0.9302545722396115, 'colsample_bytree': 0.8258234347702816, 'gamma': 0.14415721832063857, 'lambda': 5.860252876610051e-05, 'alpha': 6.410037190014798e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 07:09:06,401] Trial 169 finished with value: 0.9883430588899818 and parameters: {'n_estimators': 2546, 'learning_rate': 0.03610538743048593, 'max_depth': 16, 'subsample': 0.9322520448106173, 'colsample_bytree': 0.8371665578768541, 'gamma': 0.14517300388675973, 'lambda': 5.44349156488398e-05, 'alpha': 7.258819076046908e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 07:21:58,966] Trial 170 finished with value: 0.9883679808381987 and parameters: {'n_estimators': 2551, 'learning_rate': 0.03636328249826372, 'max_depth': 16, 'subsample': 0.9323319409161272, 'colsample_bytree': 0.838049767226578, 'gamma': 0.14230878566367735, 'lambda': 8.437187123953493e-05, 'alpha': 6.976895180638596e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 07:34:52,289] Trial 171 finished with value: 0.9883842106208581 and parameters: {'n_estimators': 2616, 'learning_rate': 0.03635501869877421, 'max_depth': 16, 'subsample': 0.931914510007871, 'colsample_bytree': 0.8375631103233596, 'gamma': 0.14776896263590514, 'lambda': 5.623393963842426e-05, 'alpha': 7.982578028789237e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 07:47:45,165] Trial 172 finished with value: 0.9883414902142277 and parameters: {'n_estimators': 2619, 'learning_rate': 0.0363790847997781, 'max_depth': 16, 'subsample': 0.9316089305147575, 'colsample_bytree': 0.8375679901594951, 'gamma': 0.1484227092147585, 'lambda': 6.453941328455114e-05, 'alpha': 7.655047814982635e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 08:00:42,074] Trial 173 finished with value: 0.9884059085880475 and parameters: {'n_estimators': 2602, 'learning_rate': 0.03616397753239335, 'max_depth': 16, 'subsample': 0.9309340058913978, 'colsample_bytree': 0.8373238128359002, 'gamma': 0.1469514705008168, 'lambda': 5.972583307409247e-05, 'alpha': 6.683163779860265e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 08:13:03,997] Trial 174 finished with value: 0.9883822159020491 and parameters: {'n_estimators': 2608, 'learning_rate': 0.03885668962650058, 'max_depth': 16, 'subsample': 0.9309295697793163, 'colsample_bytree': 0.8376843793081074, 'gamma': 0.1475692921503799, 'lambda': 0.00010197926120531584, 'alpha': 9.912448834982635e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 08:25:21,832] Trial 175 finished with value: 0.9884000490916942 and parameters: {'n_estimators': 2610, 'learning_rate': 0.03902813204961556, 'max_depth': 16, 'subsample': 0.9324384074813589, 'colsample_bytree': 0.8374343890250625, 'gamma': 0.1514897615024097, 'lambda': 0.00013873564551770058, 'alpha': 9.655995085818559e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 08:37:33,202] Trial 176 finished with value: 0.9882974955071477 and parameters: {'n_estimators': 2578, 'learning_rate': 0.03866259779555627, 'max_depth': 16, 'subsample': 0.9186834079893475, 'colsample_bytree': 0.8422547568581691, 'gamma': 0.1694793010762193, 'lambda': 0.0001045279188043905, 'alpha': 1.1179562719601334e-06}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 08:49:31,372] Trial 177 finished with value: 0.9884001309858607 and parameters: {'n_estimators': 2632, 'learning_rate': 0.04118064560155209, 'max_depth': 17, 'subsample': 0.9483491321045391, 'colsample_bytree': 0.8483459142718712, 'gamma': 0.1512194098997576, 'lambda': 2.4424859873499874e-05, 'alpha': 3.1261867748464633e-06}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 09:00:20,048] Trial 178 finished with value: 0.9882142289650085 and parameters: {'n_estimators': 2684, 'learning_rate': 0.04125149301559869, 'max_depth': 17, 'subsample': 0.9494874015699195, 'colsample_bytree': 0.8483974868018318, 'gamma': 0.15283522253677464, 'lambda': 2.552865225010667e-05, 'alpha': 3.3382792621021465e-06}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 09:12:23,923] Trial 179 finished with value: 0.9883049103306915 and parameters: {'n_estimators': 2628, 'learning_rate': 0.0388607839352723, 'max_depth': 16, 'subsample': 0.9461463333333334, 'colsample_bytree': 0.8424218949392176, 'gamma': 0.1583476057209798, 'lambda': 0.0001460683074091694, 'alpha': 4.325695033878474e-07}. Best is trial 156 with value: 0.9884165518428587.
        [I 2025-05-21 09:26:23,960] Trial 180 finished with value: 0.9884344933924165 and parameters: {'n_estimators': 2579, 'learning_rate': 0.03477749008985923, 'max_depth': 17, 'subsample': 0.9392136009890209, 'colsample_bytree': 0.8500881023773173, 'gamma': 0.1473887019154217, 'lambda': 2.440314782010839e-05, 'alpha': 1.3443242095085442e-06}. Best is trial 180 with value: 0.9884344933924165.
        [I 2025-05-21 09:41:06,913] Trial 181 finished with value: 0.9886074213766779 and parameters: {'n_estimators': 2582, 'learning_rate': 0.03480881515907921, 'max_depth': 17, 'subsample': 0.9273230489991666, 'colsample_bytree': 0.8502024622942182, 'gamma': 0.1252469612089993, 'lambda': 4.346100346338725e-05, 'alpha': 1.3767827827827828e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 09:55:16,769] Trial 182 finished with value: 0.9884643560606061 and parameters: {'n_estimators': 2585, 'learning_rate': 0.03440935626577583, 'max_depth': 17, 'subsample': 0.9285103131333333, 'colsample_bytree': 0.8512015093751096, 'gamma': 0.15024907963665242, 'lambda': 4.277761899178051e-05, 'alpha': 1.4406443424105294e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 10:09:58,059] Trial 183 finished with value: 0.988603417721519 and parameters: {'n_estimators': 2576, 'learning_rate': 0.03519395636043063, 'max_depth': 17, 'subsample': 0.9263520625471698, 'colsample_bytree': 0.847967300762283, 'gamma': 0.1255931215444839, 'lambda': 4.170596323485075e-05, 'alpha': 2.0520300438063264e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 10:23:58,637] Trial 184 finished with value: 0.9884282363102432 and parameters: {'n_estimators': 2579, 'learning_rate': 0.03483167520023023, 'max_depth': 17, 'subsample': 0.9282312014603953, 'colsample_bytree': 0.850634790074251, 'gamma': 0.15429307775902014, 'lambda': 4.295825316335194e-05, 'alpha': 1.527364121404179e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 10:38:03,595] Trial 185 finished with value: 0.9884815984249117 and parameters: {'n_estimators': 2579, 'learning_rate': 0.03477909772322046, 'max_depth': 17, 'subsample': 0.9258582963073738, 'colsample_bytree': 0.8505927357002008, 'gamma': 0.1524436798039604, 'lambda': 3.9626304561085024e-05, 'alpha': 1.797070381656209e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 10:51:58,460] Trial 186 finished with value: 0.9884151743603417 and parameters: {'n_estimators': 2620, 'learning_rate': 0.03505266453982701, 'max_depth': 17, 'subsample': 0.9253818671607447, 'colsample_bytree': 0.8598416805166299, 'gamma': 0.1608522961811559, 'lambda': 4.480098319696357e-05, 'alpha': 1.759411267882562e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 11:05:57,234] Trial 187 finished with value: 0.9883935272633005 and parameters: {'n_estimators': 2578, 'learning_rate': 0.03478323207436033, 'max_depth': 17, 'subsample': 0.9182900010908179, 'colsample_bytree': 0.8613483259992015, 'gamma': 0.16015760810482404, 'lambda': 4.1791404176211115e-05, 'alpha': 7.856670155008544e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 11:20:04,219] Trial 188 finished with value: 0.9884821639090909 and parameters: {'n_estimators': 2581, 'learning_rate': 0.03470866030999516, 'max_depth': 17, 'subsample': 0.9173381676669784, 'colsample_bytree': 0.8617046096002015, 'gamma': 0.1615061409249969, 'lambda': 1.981682399238961e-05, 'alpha': 7.121173214046182e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 11:34:05,006] Trial 189 finished with value: 0.9883900742785536 and parameters: {'n_estimators': 2582, 'learning_rate': 0.03483526569107954, 'max_depth': 17, 'subsample': 0.9182408103953538, 'colsample_bytree': 0.8611556100140224, 'gamma': 0.1621467297920792, 'lambda': 2.053468087910006e-05, 'alpha': 1.1290781702812224e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 11:47:56,762] Trial 190 finished with value: 0.9883482705900095 and parameters: {'n_estimators': 2583, 'learning_rate': 0.03463457106889255, 'max_depth': 17, 'subsample': 0.9175152014603952, 'colsample_bytree': 0.8609468600079857, 'gamma': 0.16994665485459242, 'lambda': 1.823967396660133e-05, 'alpha': 1.1124383188581691e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 12:02:24,715] Trial 191 finished with value: 0.9884342371987595 and parameters: {'n_estimators': 2635, 'learning_rate': 0.0325620541759604, 'max_depth': 17, 'subsample': 0.9254792994463453, 'colsample_bytree': 0.8522574768392556, 'gamma': 0.16037302485541655, 'lambda': 4.33797866504222e-05, 'alpha': 6.712988180104618e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 12:29:37,804] Trial 192 finished with value: 0.9884360677561962 and parameters: {'n_estimators': 2577, 'learning_rate': 0.03538399436363636, 'max_depth': 17, 'subsample': 0.9256003264636364, 'colsample_bytree': 0.8506588960000001, 'gamma': 0.15741160350000002, 'lambda': 3.91562024e-05, 'alpha': 8.58531726e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 12:44:07,934] Trial 193 finished with value: 0.9883852037740259 and parameters: {'n_estimators': 2575, 'learning_rate': 0.03525838965939257, 'max_depth': 17, 'subsample': 0.9256963782782783, 'colsample_bytree': 0.8506238466185223, 'gamma': 0.1586761376510309, 'lambda': 4.76531524e-05, 'alpha': 9.33608976e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 12:59:27,870] Trial 194 finished with value: 0.9884271816041793 and parameters: {'n_estimators': 2648, 'learning_rate': 0.03461618210352136, 'max_depth': 17, 'subsample': 0.9287278210135136, 'colsample_bytree': 0.8528878297495914, 'gamma': 0.1560075027599818, 'lambda': 3.79122067e-05, 'alpha': 1.35792742e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 13:14:59,032] Trial 195 finished with value: 0.9884102283626896 and parameters: {'n_estimators': 2652, 'learning_rate': 0.03448782006121966, 'max_depth': 17, 'subsample': 0.9289889608552632, 'colsample_bytree': 0.8547015112195123, 'gamma': 0.1563397127733718, 'lambda': 3.90228642e-05, 'alpha': 1.37947926e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 13:30:21,013] Trial 196 finished with value: 0.988409440625 and parameters: {'n_estimators': 2648, 'learning_rate': 0.03433878891516246, 'max_depth': 17, 'subsample': 0.9208478260869565, 'colsample_bytree': 0.8534861937989932, 'gamma': 0.1552718147493994, 'lambda': 4.40423277e-05, 'alpha': 1.53426693e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 13:46:16,472] Trial 197 finished with value: 0.9884302677561962 and parameters: {'n_estimators': 2641, 'learning_rate': 0.03444359007629578, 'max_depth': 17, 'subsample': 0.928621743606132, 'colsample_bytree': 0.8539862175960098, 'gamma': 0.1551634351338676, 'lambda': 4.02329188e-05, 'alpha': 5.68717325e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 14:01:50,823] Trial 198 finished with value: 0.9884472396369046 and parameters: {'n_estimators': 2654, 'learning_rate': 0.03441913410292723, 'max_depth': 17, 'subsample': 0.929263434606132, 'colsample_bytree': 0.8532752538058169, 'gamma': 0.15588763636363637, 'lambda': 3.89190412e-05, 'alpha': 2.01174987e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 14:17:41,502] Trial 199 finished with value: 0.9884368523581789 and parameters: {'n_estimators': 2651, 'learning_rate': 0.0342709605417855, 'max_depth': 17, 'subsample': 0.9289948577900516, 'colsample_bytree': 0.8551622744749399, 'gamma': 0.1545763073715103, 'lambda': 3.89540708e-05, 'alpha': 1.52933742e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 14:33:30,634] Trial 200 finished with value: 0.9884560114942528 and parameters: {'n_estimators': 2657, 'learning_rate': 0.03433491474950942, 'max_depth': 17, 'subsample': 0.9283508119293103, 'colsample_bytree': 0.8549537157812151, 'gamma': 0.1558261309854497, 'lambda': 3.75729759e-05, 'alpha': 2.13559639e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 14:49:49,287] Trial 201 finished with value: 0.9884275726207869 and parameters: {'n_estimators': 2662, 'learning_rate': 0.03293816999741009, 'max_depth': 17, 'subsample': 0.9282575235940059, 'colsample_bytree': 0.8549860472480287, 'gamma': 0.15563769974257424, 'lambda': 3.78433777e-05, 'alpha': 1.93943438e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 15:06:13,189] Trial 202 finished with value: 0.9884501254714652 and parameters: {'n_estimators': 2656, 'learning_rate': 0.03279714088924322, 'max_depth': 17, 'subsample': 0.9245885235940059, 'colsample_bytree': 0.8543526972480286, 'gamma': 0.15601109974257426, 'lambda': 3.87334105e-05, 'alpha': 2.02443603e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 15:22:27,585] Trial 203 finished with value: 0.9884568019999999 and parameters: {'n_estimators': 2666, 'learning_rate': 0.03299335438883733, 'max_depth': 17, 'subsample': 0.9285531234794276, 'colsample_bytree': 0.8540024500000001, 'gamma': 0.1568967236510309, 'lambda': 3.83097312e-05, 'alpha': 1.83598305e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 15:38:30,982] Trial 204 finished with value: 0.9884090280800001 and parameters: {'n_estimators': 2678, 'learning_rate': 0.03304869857921612, 'max_depth': 17, 'subsample': 0.9252029307218659, 'colsample_bytree': 0.8548115982888043, 'gamma': 0.1564914107218659, 'lambda': 3.83263721e-05, 'alpha': 2.17437212e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 15:54:52,522] Trial 205 finished with value: 0.9884119934794276 and parameters: {'n_estimators': 2664, 'learning_rate': 0.0326926066224101, 'max_depth': 17, 'subsample': 0.9290405107218659, 'colsample_bytree': 0.8574386762288043, 'gamma': 0.15632043657742574, 'lambda': 3.71347231e-05, 'alpha': 2.15471712e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 16:11:37,049] Trial 206 finished with value: 0.9884674395000001 and parameters: {'n_estimators': 2715, 'learning_rate': 0.03232966147986064, 'max_depth': 17, 'subsample': 0.926919047986064, 'colsample_bytree': 0.8514529685792162, 'gamma': 0.15366524107013977, 'lambda': 3.66301389e-05, 'alpha': 2.39685712e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 16:26:34,149] Trial 207 finished with value: 0.9883630656754021 and parameters: {'n_estimators': 2712, 'learning_rate': 0.03539106886363636, 'max_depth': 17, 'subsample': 0.9263558397959184, 'colsample_bytree': 0.8510998188163266, 'gamma': 0.16161973650000002, 'lambda': 4.86371020e-05, 'alpha': 2.18396316e-05}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 16:43:20,726] Trial 208 finished with value: 0.9884639999999999 and parameters: {'n_estimators': 2763, 'learning_rate': 0.03268251020408163, 'max_depth': 17, 'subsample': 0.9247568163265306, 'colsample_bytree': 0.8515448979591836, 'gamma': 0.1537366224489796, 'lambda': 3.67116326e-05, 'alpha': 5.68858775e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 17:00:00,220] Trial 209 finished with value: 0.988485293877551 and parameters: {'n_estimators': 2783, 'learning_rate': 0.03264567346938775, 'max_depth': 17, 'subsample': 0.9234532367346939, 'colsample_bytree': 0.8514586734693878, 'gamma': 0.1536280620408163, 'lambda': 3.00136612e-05, 'alpha': 5.59984387e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 17:16:47,631] Trial 210 finished with value: 0.9885068214285714 and parameters: {'n_estimators': 2799, 'learning_rate': 0.03272862244897959, 'max_depth': 17, 'subsample': 0.9232107142857143, 'colsample_bytree': 0.851088887755102, 'gamma': 0.1536841020408163, 'lambda': 3.72447102e-05, 'alpha': 4.93274693e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 17:33:34,605] Trial 211 finished with value: 0.9884950346938776 and parameters: {'n_estimators': 2798, 'learning_rate': 0.03279741632653061, 'max_depth': 17, 'subsample': 0.9227255102040816, 'colsample_bytree': 0.8515955612244898, 'gamma': 0.15299795918367346, 'lambda': 3.58442245e-05, 'alpha': 5.53050102e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 17:50:21,833] Trial 212 finished with value: 0.9884594612244898 and parameters: {'n_estimators': 2793, 'learning_rate': 0.03253739591836734, 'max_depth': 17, 'subsample': 0.9232392408163266, 'colsample_bytree': 0.8514798979591836, 'gamma': 0.1534165387755102, 'lambda': 3.61781142e-05, 'alpha': 5.77508898e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 18:06:57,138] Trial 213 finished with value: 0.9884860163265306 and parameters: {'n_estimators': 2795, 'learning_rate': 0.03268467346938775, 'max_depth': 17, 'subsample': 0.9228173673469387, 'colsample_bytree': 0.8509421428571429, 'gamma': 0.15371740204081633, 'lambda': 3.55342653e-05, 'alpha': 6.31464816e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 18:58:51,239] Trial 214 finished with value: 0.988562145398773 and parameters: {'n_estimators': 2818, 'learning_rate': 0.032455510204081635, 'max_depth': 17, 'subsample': 0.9223525306122449, 'colsample_bytree': 0.8514750102040816, 'gamma': 0.15100281632653062, 'lambda': 3.50891224e-05, 'alpha': 5.00010502e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 19:15:15,435] Trial 215 finished with value: 0.9885078571428571 and parameters: {'n_estimators': 2803, 'learning_rate': 0.03252462244897959, 'max_depth': 17, 'subsample': 0.9207225306122449, 'colsample_bytree': 0.8518643469387755, 'gamma': 0.15087857142857144, 'lambda': 2.99014408e-05, 'alpha': 5.40252612e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 19:29:54,079] Trial 216 finished with value: 0.9885015510204082 and parameters: {'n_estimators': 2831, 'learning_rate': 0.03135367346938775, 'max_depth': 17, 'subsample': 0.9230504081632653, 'colsample_bytree': 0.8515669387755102, 'gamma': 0.1515384081632653, 'lambda': 3.14366938e-05, 'alpha': 5.61945612e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 19:44:17,257] Trial 217 finished with value: 0.9885009653061224 and parameters: {'n_estimators': 2798, 'learning_rate': 0.03252462244897959, 'max_depth': 17, 'subsample': 0.922797387755102, 'colsample_bytree': 0.8516220408163265, 'gamma': 0.1521206530612245, 'lambda': 2.99283469e-05, 'alpha': 5.74941734e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 19:58:54,872] Trial 218 finished with value: 0.9885020142857143 and parameters: {'n_estimators': 2804, 'learning_rate': 0.03171554081632653, 'max_depth': 17, 'subsample': 0.9208619591836735, 'colsample_bytree': 0.8485582857142857, 'gamma': 0.15005670408163264, 'lambda': 2.9819851e-05, 'alpha': 4.89096612e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 20:13:36,113] Trial 219 finished with value: 0.9884727397959184 and parameters: {'n_estimators': 2808, 'learning_rate': 0.03175275510204082, 'max_depth': 17, 'subsample': 0.921243306122449, 'colsample_bytree': 0.8483551020408163, 'gamma': 0.15096990204081632, 'lambda': 3.4429253e-05, 'alpha': 4.70245214e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 20:28:27,277] Trial 220 finished with value: 0.9885085387755102 and parameters: {'n_estimators': 2805, 'learning_rate': 0.03100081224489796, 'max_depth': 17, 'subsample': 0.9209911836734694, 'colsample_bytree': 0.8481678775510204, 'gamma': 0.1510489306122449, 'lambda': 3.13331918e-05, 'alpha': 5.07853877e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 20:43:09,212] Trial 221 finished with value: 0.9885012581632653 and parameters: {'n_estimators': 2805, 'learning_rate': 0.0314274306122449, 'max_depth': 17, 'subsample': 0.9207438469387756, 'colsample_bytree': 0.8479838081632653, 'gamma': 0.1502041632653061, 'lambda': 3.18130041e-05, 'alpha': 4.7721551e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 20:57:59,525] Trial 222 finished with value: 0.9885022132653061 and parameters: {'n_estimators': 2807, 'learning_rate': 0.03139124081632653, 'max_depth': 17, 'subsample': 0.9209721020408163, 'colsample_bytree': 0.8465553367346939, 'gamma': 0.1502877142857143, 'lambda': 3.17894612e-05, 'alpha': 4.82488898e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 21:12:43,574] Trial 223 finished with value: 0.9884732153061224 and parameters: {'n_estimators': 2807, 'learning_rate': 0.0313529693877551, 'max_depth': 17, 'subsample': 0.9209373673469388, 'colsample_bytree': 0.8463325612244898, 'gamma': 0.15153818367346938, 'lambda': 3.09484898e-05, 'alpha': 4.67122714e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 21:27:29,481] Trial 224 finished with value: 0.9885094244897959 and parameters: {'n_estimators': 2806, 'learning_rate': 0.0313341306122449, 'max_depth': 17, 'subsample': 0.9212728469387755, 'colsample_bytree': 0.8467996734693877, 'gamma': 0.1522055612244898, 'lambda': 3.26782346e-05, 'alpha': 4.47816938e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 21:42:20,887] Trial 225 finished with value: 0.9885055204081632 and parameters: {'n_estimators': 2809, 'learning_rate': 0.031222142857142856, 'max_depth': 17, 'subsample': 0.9208542244897959, 'colsample_bytree': 0.846659081632653, 'gamma': 0.15062020408163264, 'lambda': 3.08140102e-05, 'alpha': 4.72289183e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 21:57:08,984] Trial 226 finished with value: 0.9885304734693878 and parameters: {'n_estimators': 2808, 'learning_rate': 0.03138387346938775, 'max_depth': 17, 'subsample': 0.9207672244897959, 'colsample_bytree': 0.8459833265306123, 'gamma': 0.15055581632653062, 'lambda': 3.02571816e-05, 'alpha': 4.61682449e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 22:11:52,605] Trial 227 finished with value: 0.9884430755102041 and parameters: {'n_estimators': 2804, 'learning_rate': 0.031566989795918366, 'max_depth': 17, 'subsample': 0.9208986530612245, 'colsample_bytree': 0.8456400040816327, 'gamma': 0.14937291836734694, 'lambda': 3.06898571e-05, 'alpha': 4.44670612e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 22:28:11,591] Trial 228 finished with value: 0.9885414612244898 and parameters: {'n_estimators': 2786, 'learning_rate': 0.03080768163265306, 'max_depth': 17, 'subsample': 0.9147796734693877, 'colsample_bytree': 0.8481557346938775, 'gamma': 0.15018289795918367, 'lambda': 2.89142122e-05, 'alpha': 4.40531734e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 22:44:26,894] Trial 229 finished with value: 0.9884914102040816 and parameters: {'n_estimators': 2778, 'learning_rate': 0.03081819387755102, 'max_depth': 18, 'subsample': 0.9144640816326531, 'colsample_bytree': 0.8481610204081633, 'gamma': 0.15018728571428572, 'lambda': 2.8213102e-05, 'alpha': 4.38818775e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 23:00:13,267] Trial 230 finished with value: 0.9884700693877551 and parameters: {'n_estimators': 2785, 'learning_rate': 0.03070757142857143, 'max_depth': 17, 'subsample': 0.9149483469387756, 'colsample_bytree': 0.8477896734693878, 'gamma': 0.15062632653061225, 'lambda': 2.19054489e-05, 'alpha': 4.28018346e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 23:16:03,725] Trial 231 finished with value: 0.9884854581632653 and parameters: {'n_estimators': 2794, 'learning_rate': 0.030645736734693877, 'max_depth': 17, 'subsample': 0.9147340000000001, 'colsample_bytree': 0.8480617346938776, 'gamma': 0.150195693877551, 'lambda': 2.13743326e-05, 'alpha': 4.39165234e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 23:33:39,928] Trial 232 finished with value: 0.9884650632653062 and parameters: {'n_estimators': 2803, 'learning_rate': 0.030507142857142857, 'max_depth': 18, 'subsample': 0.9143855102040816, 'colsample_bytree': 0.8476146938775511, 'gamma': 0.15111538775510204, 'lambda': 2.22937755e-05, 'alpha': 3.8684449e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-21 23:51:06,874] Trial 233 finished with value: 0.9884934653061224 and parameters: {'n_estimators': 2776, 'learning_rate': 0.03075541632653061, 'max_depth': 18, 'subsample': 0.9123883673469388, 'colsample_bytree': 0.8439622448979592, 'gamma': 0.14991827551020408, 'lambda': 2.99319673e-05, 'alpha': 4.6706551e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 00:08:05,268] Trial 234 finished with value: 0.9884271816326531 and parameters: {'n_estimators': 2786, 'learning_rate': 0.02994151020408163, 'max_depth': 18, 'subsample': 0.911899530612245, 'colsample_bytree': 0.8445387040816326, 'gamma': 0.14991449591836735, 'lambda': 2.86584102e-05, 'alpha': 4.64664938e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 00:23:19,930] Trial 235 finished with value: 0.9884889653061224 and parameters: {'n_estimators': 2828, 'learning_rate': 0.03097277551020408, 'max_depth': 17, 'subsample': 0.9203051020408163, 'colsample_bytree': 0.8490420408163265, 'gamma': 0.14981190204081633, 'lambda': 2.06222122e-05, 'alpha': 2.6827151e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 00:38:27,362] Trial 236 finished with value: 0.9883193265306122 and parameters: {'n_estimators': 2824, 'learning_rate': 0.030049632653061224, 'max_depth': 18, 'subsample': 0.9200965306122449, 'colsample_bytree': 0.8493771020408164, 'gamma': 0.15230981632653062, 'lambda': 2.91398367e-05, 'alpha': 2.6560449e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 00:53:10,657] Trial 237 finished with value: 0.9885227755102041 and parameters: {'n_estimators': 2868, 'learning_rate': 0.031492061224489795, 'max_depth': 17, 'subsample': 0.9199093469387756, 'colsample_bytree': 0.843950612244898, 'gamma': 0.15025611428571428, 'lambda': 2.6596351e-05, 'alpha': 5.01584898e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 01:08:58,325] Trial 238 finished with value: 0.9884816040816326 and parameters: {'n_estimators': 2884, 'learning_rate': 0.03098381632653061, 'max_depth': 18, 'subsample': 0.9115058265306122, 'colsample_bytree': 0.8439838081632653, 'gamma': 0.14898089795918367, 'lambda': 2.71035918e-05, 'alpha': 2.50867428e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 01:24:09,871] Trial 239 finished with value: 0.9884540489795918 and parameters: {'n_estimators': 2889, 'learning_rate': 0.03169310204081633, 'max_depth': 18, 'subsample': 0.9101984489795919, 'colsample_bytree': 0.8424032244897959, 'gamma': 0.14888698979591836, 'lambda': 2.70399418e-05, 'alpha': 2.61608693e-06}. Best is trial 181 with value: 0.9886074213766779.
        [I 2025-05-22 01:40:57,783] Trial 240 finished with value: 0.9886490326530612 and parameters: {'n_estimators': 2849, 'learning_rate': 0.03010042857142857, 'max_depth': 18, 'subsample': 0.9199243469387755, 'colsample_bytree': 0.8443373469387756, 'gamma': 0.12670251020408164, 'lambda': 3.22723142e-05, 'alpha': 3.63126816e-06}. Best is trial 240 with value: 0.9886490326530612.
        [I 2025-05-22 01:57:23,656] Trial 241 finished with value: 0.9886423959183674 and parameters: {'n_estimators': 2849, 'learning_rate': 0.030986102040816327, 'max_depth': 18, 'subsample': 0.9196908163265306, 'colsample_bytree': 0.8439158163265306, 'gamma': 0.12469524489795919, 'lambda': 2.50566122e-05, 'alpha': 3.48841346e-06}. Best is trial 240 with value: 0.9886490326530612.
        [I 2025-05-22 02:13:27,034] Trial 242 finished with value: 0.9885241775510204 and parameters: {'n_estimators': 2851, 'learning_rate': 0.030040191836734694, 'max_depth': 18, 'subsample': 0.919704612244898, 'colsample_bytree': 0.8444175918367347, 'gamma': 0.14684751020408162, 'lambda': 2.41086122e-05, 'alpha': 3.4544653e-06}. Best is trial 240 with value: 0.9886490326530612.
        [I 2025-05-22 02:30:42,851] Trial 243 finished with value: 0.9886492306122449 and parameters: {'n_estimators': 2856, 'learning_rate': 0.029398367346938775, 'max_depth': 18, 'subsample': 0.9201646938775511, 'colsample_bytree': 0.8440221428571428, 'gamma': 0.12223502040816327, 'lambda': 3.29425816e-05, 'alpha': 5.10663775e-06}. Best is trial 240 with value: 0.9886490326530612.
        [I 2025-05-22 02:47:36,685] Trial 244 finished with value: 0.9886142714285714 and parameters: {'n_estimators': 2843, 'learning_rate': 0.029565555102040817, 'max_depth': 18, 'subsample': 0.9197770408163265, 'colsample_bytree': 0.8444495102040816, 'gamma': 0.12559714285714285, 'lambda': 3.20742816e-05, 'alpha': 3.65025234e-06}. Best is trial 240 with value: 0.9886490326530612.
        [I 2025-05-22 03:04:55,387] Trial 245 finished with value: 0.9886384244897959 and parameters: {'n_estimators': 2858, 'learning_rate': 0.029284918367346938, 'max_depth': 18, 'subsample': 0.9198861224489796, 'colsample_bytree': 0.8439182040816327, 'gamma': 0.12430323673469387, 'lambda': 3.30412245e-05, 'alpha': 3.37540041e-06}. Best is trial 240 with value: 0.9886490326530612.
        [I 2025-05-22 03:22:11,098] Trial 246 finished with value: 0.9886580551020408 and parameters: {'n_estimators': 2864, 'learning_rate': 0.029277461224489797, 'max_depth': 18, 'subsample': 0.9203648979591837, 'colsample_bytree': 0.8440193469387755, 'gamma': 0.12491522857142857, 'lambda': 2.46029224e-05, 'alpha': 3.24394234e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 03:39:23,250] Trial 247 finished with value: 0.9886210000000001 and parameters: {'n_estimators': 2858, 'learning_rate': 0.0292062693877551, 'max_depth': 18, 'subsample': 0.9200942040816327, 'colsample_bytree': 0.8441781224489796, 'gamma': 0.12664583673469387, 'lambda': 2.44774816e-05, 'alpha': 3.40096571e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 03:56:18,458] Trial 248 finished with value: 0.9886259071428572 and parameters: {'n_estimators': 2865, 'learning_rate': 0.029197832653061224, 'max_depth': 18, 'subsample': 0.9191176224489796, 'colsample_bytree': 0.8441434693877551, 'gamma': 0.12770870408163266, 'lambda': 2.39176938e-05, 'alpha': 3.33359306e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 04:13:29,520] Trial 249 finished with value: 0.9886242040816327 and parameters: {'n_estimators': 2860, 'learning_rate': 0.029046642857142856, 'max_depth': 18, 'subsample': 0.9193384693877551, 'colsample_bytree': 0.8403362040816327, 'gamma': 0.12671991020408164, 'lambda': 2.43661142e-05, 'alpha': 3.33057551e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 04:30:48,004] Trial 250 finished with value: 0.9886014285714285 and parameters: {'n_estimators': 2863, 'learning_rate': 0.029118004081632653, 'max_depth': 18, 'subsample': 0.9190842857142857, 'colsample_bytree': 0.8407795102040816, 'gamma': 0.12683010204081634, 'lambda': 2.47108693e-05, 'alpha': 3.32168775e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 04:48:11,010] Trial 251 finished with value: 0.9886454795918367 and parameters: {'n_estimators': 2863, 'learning_rate': 0.02909589387755102, 'max_depth': 18, 'subsample': 0.9190297959183673, 'colsample_bytree': 0.8399777959183673, 'gamma': 0.12572528571428572, 'lambda': 2.42200612e-05, 'alpha': 3.34441428e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 05:05:16,635] Trial 252 finished with value: 0.9886071428571428 and parameters: {'n_estimators': 2860, 'learning_rate': 0.02918461224489796, 'max_depth': 18, 'subsample': 0.9167017346938776, 'colsample_bytree': 0.8405303673469388, 'gamma': 0.12575010204081632, 'lambda': 2.50322245e-05, 'alpha': 3.50708734e-06}. Best is trial 246 with value: 0.9886580551020408.
        [I 2025-05-22 05:22:40,576] Trial 253 finished with value: 0.9886860000000001 and parameters: {'n_estimators': 2867, 'learning_rate': 0.028979738775510203, 'max_depth': 18, 'subsample': 0.9171222653061225, 'colsample_bytree': 0.8428135102040816, 'gamma': 0.12497280408163265, 'lambda': 2.44214816e-05, 'alpha': 3.37956041e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 07:54:06,882] Trial 254 finished with value: 0.988593306122449 and parameters: {'n_estimators': 2861, 'learning_rate': 0.02929165306122449, 'max_depth': 18, 'subsample': 0.9164904081632653, 'colsample_bytree': 0.8410402040816327, 'gamma': 0.1259092448979592, 'lambda': 2.34354285e-05, 'alpha': 3.37493469e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 08:11:27,578] Trial 255 finished with value: 0.9886708612244898 and parameters: {'n_estimators': 2860, 'learning_rate': 0.02908065306122449, 'max_depth': 18, 'subsample': 0.9172696734693877, 'colsample_bytree': 0.8404438367346939, 'gamma': 0.12559360408163265, 'lambda': 1.70423326e-05, 'alpha': 3.43449734e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 08:28:29,086] Trial 256 finished with value: 0.9885830000000001 and parameters: {'n_estimators': 2865, 'learning_rate': 0.029094738775510203, 'max_depth': 18, 'subsample': 0.9166380612244898, 'colsample_bytree': 0.8411901020408163, 'gamma': 0.12607271428571428, 'lambda': 2.44048041e-05, 'alpha': 3.31087346e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 08:45:48,447] Trial 257 finished with value: 0.9886329428571428 and parameters: {'n_estimators': 2863, 'learning_rate': 0.029013342857142857, 'max_depth': 18, 'subsample': 0.9164735918367347, 'colsample_bytree': 0.8408183265306123, 'gamma': 0.12592391020408164, 'lambda': 2.39327041e-05, 'alpha': 3.36199183e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 09:02:31,803] Trial 258 finished with value: 0.9885679591836735 and parameters: {'n_estimators': 2865, 'learning_rate': 0.02911262244897959, 'max_depth': 18, 'subsample': 0.9165357142857142, 'colsample_bytree': 0.8405877551020408, 'gamma': 0.1268927142857143, 'lambda': 2.40830816e-05, 'alpha': 3.27111918e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 09:19:54,818] Trial 259 finished with value: 0.9886690408163265 and parameters: {'n_estimators': 2869, 'learning_rate': 0.0290403306122449, 'max_depth': 18, 'subsample': 0.9169248367346939, 'colsample_bytree': 0.8406397142857143, 'gamma': 0.12500150204081633, 'lambda': 1.6495751e-05, 'alpha': 3.35513265e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 09:37:19,475] Trial 260 finished with value: 0.9886001020408163 and parameters: {'n_estimators': 2858, 'learning_rate': 0.02887969387755102, 'max_depth': 18, 'subsample': 0.9160283265306123, 'colsample_bytree': 0.8409802040816327, 'gamma': 0.12614478571428572, 'lambda': 1.7195302e-05, 'alpha': 3.2813702e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 09:54:08,271] Trial 261 finished with value: 0.9884959224489796 and parameters: {'n_estimators': 2916, 'learning_rate': 0.028938424489795918, 'max_depth': 18, 'subsample': 0.9168985102040816, 'colsample_bytree': 0.8394600408163266, 'gamma': 0.1253682142857143, 'lambda': 1.42516041e-05, 'alpha': 3.18905306e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 10:11:11,725] Trial 262 finished with value: 0.9886134387755102 and parameters: {'n_estimators': 2854, 'learning_rate': 0.02902369387755102, 'max_depth': 18, 'subsample': 0.9161007346938776, 'colsample_bytree': 0.8403218469387756, 'gamma': 0.12613073469387756, 'lambda': 1.60182041e-05, 'alpha': 3.64685428e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 10:28:43,118] Trial 263 finished with value: 0.9886411510204081 and parameters: {'n_estimators': 2855, 'learning_rate': 0.028867151020408163, 'max_depth': 18, 'subsample': 0.916176918367347, 'colsample_bytree': 0.8412703265306122, 'gamma': 0.12526982653061225, 'lambda': 1.69474816e-05, 'alpha': 3.02983224e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 10:45:53,479] Trial 264 finished with value: 0.988658406122449 and parameters: {'n_estimators': 2857, 'learning_rate': 0.028897938775510203, 'max_depth': 18, 'subsample': 0.9165080408163266, 'colsample_bytree': 0.8408432367346939, 'gamma': 0.12645501020408164, 'lambda': 1.73712714e-05, 'alpha': 2.25655428e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 11:02:53,971] Trial 265 finished with value: 0.9885822367346939 and parameters: {'n_estimators': 2851, 'learning_rate': 0.029164604081632652, 'max_depth': 18, 'subsample': 0.9162007346938775, 'colsample_bytree': 0.8407854081632653, 'gamma': 0.1265494081632653, 'lambda': 1.71252326e-05, 'alpha': 3.03300612e-06}. Best is trial 253 with value: 0.9886860000000001.
        [I 2025-05-22 11:20:21,665] Trial 266 finished with value: 0.9887151020408163 and parameters: {'n_estimators': 2848, 'learning_rate': 0.02905174081632653, 'max_depth': 18, 'subsample': 0.9162676040816326, 'colsample_bytree': 0.8411663020408163, 'gamma': 0.12182691020408163, 'lambda': 1.68407234e-05, 'alpha': 2.31146122e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 11:38:00,276] Trial 267 finished with value: 0.9886214040816326 and parameters: {'n_estimators': 2845, 'learning_rate': 0.028672755102040816, 'max_depth': 18, 'subsample': 0.9175890000000001, 'colsample_bytree': 0.8410671428571428, 'gamma': 0.12161561632653061, 'lambda': 1.6179951e-05, 'alpha': 2.30841632e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 11:55:45,090] Trial 268 finished with value: 0.9886624979591837 and parameters: {'n_estimators': 2847, 'learning_rate': 0.0285162693877551, 'max_depth': 18, 'subsample': 0.9181729081632653, 'colsample_bytree': 0.8419654163265306, 'gamma': 0.1219099224489796, 'lambda': 1.6536953e-05, 'alpha': 2.66587551e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 12:13:24,988] Trial 269 finished with value: 0.9886872581632653 and parameters: {'n_estimators': 2842, 'learning_rate': 0.0285864306122449, 'max_depth': 18, 'subsample': 0.9182980612244898, 'colsample_bytree': 0.8416644979591836, 'gamma': 0.1217561224489796, 'lambda': 1.56721142e-05, 'alpha': 2.23432653e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 12:31:16,245] Trial 270 finished with value: 0.9886616428571428 and parameters: {'n_estimators': 2886, 'learning_rate': 0.0283826306122449, 'max_depth': 18, 'subsample': 0.9187607755102041, 'colsample_bytree': 0.8395334795918367, 'gamma': 0.11985354285714285, 'lambda': 1.63749591e-05, 'alpha': 2.20515234e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 12:49:06,981] Trial 271 finished with value: 0.9886650959183674 and parameters: {'n_estimators': 2840, 'learning_rate': 0.028276800000000002, 'max_depth': 18, 'subsample': 0.917719387755102, 'colsample_bytree': 0.8425160408163265, 'gamma': 0.12136091836734694, 'lambda': 1.23651428e-05, 'alpha': 2.17581428e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 13:06:53,367] Trial 272 finished with value: 0.9886694653061224 and parameters: {'n_estimators': 2840, 'learning_rate': 0.02834832448979592, 'max_depth': 18, 'subsample': 0.918326612244898, 'colsample_bytree': 0.8426708163265306, 'gamma': 0.12078580408163265, 'lambda': 1.58770938e-05, 'alpha': 2.18442122e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 13:24:42,303] Trial 273 finished with value: 0.9886821428571428 and parameters: {'n_estimators': 2840, 'learning_rate': 0.028364285714285714, 'max_depth': 18, 'subsample': 0.9185936734693877, 'colsample_bytree': 0.8426722040816327, 'gamma': 0.12040163265306123, 'lambda': 1.5197753e-05, 'alpha': 2.21584898e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 13:42:28,682] Trial 274 finished with value: 0.9886532448979592 and parameters: {'n_estimators': 2838, 'learning_rate': 0.02827657142857143, 'max_depth': 18, 'subsample': 0.918185612244898, 'colsample_bytree': 0.8425999387755102, 'gamma': 0.1206975918367347, 'lambda': 1.54722857e-05, 'alpha': 2.20038367e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 14:00:28,401] Trial 275 finished with value: 0.9886952734693878 and parameters: {'n_estimators': 2842, 'learning_rate': 0.028230224489795918, 'max_depth': 18, 'subsample': 0.9182430204081633, 'colsample_bytree': 0.8429200204081633, 'gamma': 0.11741234693877551, 'lambda': 1.15139306e-05, 'alpha': 2.26901102e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 14:18:18,580] Trial 276 finished with value: 0.9886832265306122 and parameters: {'n_estimators': 2886, 'learning_rate': 0.028380932653061224, 'max_depth': 18, 'subsample': 0.9177438163265306, 'colsample_bytree': 0.8428054081632653, 'gamma': 0.12007142857142857, 'lambda': 1.06028306e-05, 'alpha': 2.18443102e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 14:36:19,526] Trial 277 finished with value: 0.9886940081632653 and parameters: {'n_estimators': 2891, 'learning_rate': 0.02824177551020408, 'max_depth': 18, 'subsample': 0.9184334693877551, 'colsample_bytree': 0.8428574285714286, 'gamma': 0.11782834693877551, 'lambda': 1.09080734e-05, 'alpha': 2.18621632e-06}. Best is trial 266 with value: 0.9887151020408163.
        [I 2025-05-22 14:54:35,157] Trial 278 finished with value: 0.9887152040816327 and parameters: {'n_estimators': 2883, 'learning_rate': 0.02810091836734694, 'max_depth': 18, 'subsample': 0.918160081632653, 'colsample_bytree': 0.8426782857142857, 'gamma': 0.11627461224489796, 'lambda': 1.07398775e-05, 'alpha': 2.09440938e-06}. Best is trial 271 with value: 0.9887151020408163.
        [I 2025-05-22 15:12:39,138] Trial 279 finished with value: 0.9886937530612245 and parameters: {'n_estimators': 2883, 'learning_rate': 0.028193873469387755, 'max_depth': 18, 'subsample': 0.9182090204081633, 'colsample_bytree': 0.8427882857142857, 'gamma': 0.11700516326530612, 'lambda': 1.02822041e-05, 'alpha': 2.13882449e-06}. Best is trial 271 with value: 0.9887151020408163.
        [I 2025-05-22 15:30:48,457] Trial 280 finished with value: 0.9886915387755102 and parameters: {'n_estimators': 2887, 'learning_rate': 0.02828308163265306, 'max_depth': 18, 'subsample': 0.9135897142857143, 'colsample_bytree': 0.8427578775510204, 'gamma': 0.11585852653061225, 'lambda': 1.2699951e-05, 'alpha': 2.11877734e-06}. Best is trial 271 with value: 0.9887151020408163.
        [I 2025-05-22 15:49:00,554] Trial 281 finished with value: 0.9887670979591837 and parameters: {'n_estimators': 2889, 'learning_rate': 0.028274979591836736, 'max_depth': 18, 'subsample': 0.913674693877551, 'colsample_bytree': 0.8425945102040816, 'gamma': 0.11477967346938775, 'lambda': 1.04696244e-05, 'alpha': 2.11668122e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 16:07:14,514] Trial 282 finished with value: 0.9887077653061224 and parameters: {'n_estimators': 2886, 'learning_rate': 0.028178214285714286, 'max_depth': 18, 'subsample': 0.912105612244898, 'colsample_bytree': 0.8426575102040816, 'gamma': 0.11585477551020408, 'lambda': 1.04364938e-05, 'alpha': 2.12511346e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 16:25:33,823] Trial 283 finished with value: 0.9886663591836734 and parameters: {'n_estimators': 2889, 'learning_rate': 0.028165867346938775, 'max_depth': 18, 'subsample': 0.9075265306122449, 'colsample_bytree': 0.8424526530612245, 'gamma': 0.11455461224489796, 'lambda': 1.03702653e-05, 'alpha': 2.06035306e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 16:44:39,021] Trial 284 finished with value: 0.9886228367346939 and parameters: {'n_estimators': 2898, 'learning_rate': 0.028170061224489796, 'max_depth': 19, 'subsample': 0.9060703469387756, 'colsample_bytree': 0.8426235102040816, 'gamma': 0.11506183673469388, 'lambda': 1.13614612e-05, 'alpha': 2.10983938e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 17:02:47,925] Trial 285 finished with value: 0.9886890000000001 and parameters: {'n_estimators': 2882, 'learning_rate': 0.028224604081632653, 'max_depth': 18, 'subsample': 0.9129986530612245, 'colsample_bytree': 0.8427598367346939, 'gamma': 0.11584636734693878, 'lambda': 1.04297102e-05, 'alpha': 1.95964081e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 17:20:42,990] Trial 286 finished with value: 0.9887038673469387 and parameters: {'n_estimators': 2891, 'learning_rate': 0.02809339387755102, 'max_depth': 18, 'subsample': 0.9124880612244898, 'colsample_bytree': 0.8428792653061224, 'gamma': 0.11461367346938776, 'lambda': 1.00814408e-05, 'alpha': 1.93035204e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 17:38:59,191] Trial 287 finished with value: 0.9886700857142857 and parameters: {'n_estimators': 2905, 'learning_rate': 0.028291232653061224, 'max_depth': 18, 'subsample': 0.9073795918367347, 'colsample_bytree': 0.8428860408163265, 'gamma': 0.11563028571428572, 'lambda': 1.03920204e-05, 'alpha': 1.94524081e-06}. Best is trial 281 with value: 0.9887670979591837.
        [I 2025-05-22 17:57:19,593] Trial 288 finished with value: 0.9887243959183674 and parameters: {'n_estimators': 2955, 'learning_rate': 0.02807601224489796, 'max_depth': 18, 'subsample': 0.9067462448979591, 'colsample_bytree': 0.8426480816326531, 'gamma': 0.11620271428571428, 'lambda': 1.0061151e-05, 'alpha': 1.89685428e-06}. Best is trial 288 with value: 0.9887243959183674.
        [I 2025-05-22 18:15:48,805] Trial 289 finished with value: 0.9886932408163266 and parameters: {'n_estimators': 2911, 'learning_rate': 0.027730791836734694, 'max_depth': 18, 'subsample': 0.9072804081632653, 'colsample_bytree': 0.8424016326530613, 'gamma': 0.11485561224489796, 'lambda': 1.03200244e-05, 'alpha': 1.82147142e-06}. Best is trial 288 with value: 0.9887243959183674.
        [I 2025-05-22 18:34:25,101] Trial 290 finished with value: 0.988690387755102 and parameters: {'n_estimators': 2920, 'learning_rate': 0.02776785306122449, 'max_depth': 18, 'subsample': 0.9067270408163265, 'colsample_bytree': 0.8425881632653061, 'gamma': 0.11398816326530612, 'lambda': 1.05382612e-05, 'alpha': 1.8449151e-06}. Best is trial 288 with value: 0.9887243959183674.
        [I 2025-05-22 18:52:48,309] Trial 291 finished with value: 0.9886408265306123 and parameters: {'n_estimators': 2969, 'learning_rate': 0.027783861224489795, 'max_depth': 18, 'subsample': 0.9067840408163265, 'colsample_bytree': 0.8388092653061224, 'gamma': 0.11362404081632653, 'lambda': 1.03971224e-05, 'alpha': 1.7778551e-06}. Best is trial 288 with value: 0.9887243959183674.
        [I 2025-05-22 19:13:49,369] Trial 292 finished with value: 0.9887038673469387 and parameters: {'n_estimators': 2920, 'learning_rate': 0.02772049387755102, 'max_depth': 18, 'subsample': 0.9102984897959184, 'colsample_bytree': 0.842572693877551, 'gamma': 0.11675891836734694, 'lambda': 9.79912448e-06, 'alpha': 1.90506734e-06}. Best is trial 288 with value: 0.9887243959183674.
        [I 2025-05-22 19:34:21,868] Trial 293 finished with value: 0.9887180408163265 and parameters: {'n_estimators': 2916, 'learning_rate': 0.027758448979591837, 'max_depth': 18, 'subsample': 0.9041992653061224, 'colsample_bytree': 0.8427054081632653, 'gamma': 0.11760191836734694, 'lambda': 1.00840326e-05, 'alpha': 1.63535836e-06}. Best is trial 288 with value: 0.9887243959183674.
        [I 2025-05-22 19:54:33,072] Trial 294 finished with value: 0.9887431428571428 and parameters: {'n_estimators': 2919, 'learning_rate': 0.027740902040816326, 'max_depth': 18, 'subsample': 0.9041552346938776, 'colsample_bytree': 0.8429519183673469, 'gamma': 0.11326312244897959, 'lambda': 1.01687918e-05, 'alpha': 1.63676816e-06}. Best is trial 294 with value: 0.9887431428571428.
        [I 2025-05-22 20:15:42,851] Trial 295 finished with value: 0.988721387755102 and parameters: {'n_estimators': 2922, 'learning_rate': 0.027727755102040816, 'max_depth': 18, 'subsample': 0.9032773673469387, 'colsample_bytree': 0.8425608163265306, 'gamma': 0.11076191836734694, 'lambda': 1.00603673e-05, 'alpha': 1.61032612e-06}. Best is trial 294 with value: 0.9887431428571428.
        [I 2025-05-22 20:36:36,135] Trial 296 finished with value: 0.9887220020408163 and parameters: {'n_estimators': 2923, 'learning_rate': 0.027640051020408164, 'max_depth': 18, 'subsample': 0.9006166326530612, 'colsample_bytree': 0.8431175204081633, 'gamma': 0.1102465306122449, 'lambda': 9.82820734e-06, 'alpha': 1.60891734e-06}. Best is trial 294 with value: 0.9887431428571428.
        [I 2025-05-22 20:57:20,423] Trial 297 finished with value: 0.9887464816326531 and parameters: {'n_estimators': 2926, 'learning_rate': 0.027593351020408163, 'max_depth': 18, 'subsample': 0.9028893469387756, 'colsample_bytree': 0.8425975918367347, 'gamma': 0.11003902040816326, 'lambda': 1.0496102e-05, 'alpha': 1.63730408e-06}. Best is trial 294 with value: 0.9887431428571428.
        [I 2025-05-22 21:17:21,549] Trial 298 finished with value: 0.9887652367346939 and parameters: {'n_estimators': 2918, 'learning_rate': 0.02758113469387755, 'max_depth': 18, 'subsample': 0.9036606530612245, 'colsample_bytree': 0.8453116326530612, 'gamma': 0.11214321428571429, 'lambda': 9.83365102e-06, 'alpha': 1.66707755e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-22 21:36:19,040] Trial 299 finished with value: 0.9887630795918367 and parameters: {'n_estimators': 2930, 'learning_rate': 0.027515502040816326, 'max_depth': 18, 'subsample': 0.9021591224489796, 'colsample_bytree': 0.8456481224489796, 'gamma': 0.11069624489795918, 'lambda': 9.41799795e-06, 'alpha': 1.54711836e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-22 21:55:27,545] Trial 300 finished with value: 0.9887570408163265 and parameters: {'n_estimators': 2932, 'learning_rate': 0.027195302040816326, 'max_depth': 18, 'subsample': 0.9022233265306122, 'colsample_bytree': 0.8452794693877551, 'gamma': 0.11083792244897959, 'lambda': 9.4908151e-06, 'alpha': 1.56524244e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-22 22:14:32,412] Trial 301 finished with value: 0.9887421428571428 and parameters: {'n_estimators': 2933, 'learning_rate': 0.027127326530612245, 'max_depth': 18, 'subsample': 0.902029387755102, 'colsample_bytree': 0.8458148163265306, 'gamma': 0.11026481632653062, 'lambda': 9.48000306e-06, 'alpha': 1.58394408e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-22 22:33:41,205] Trial 302 finished with value: 0.9886981836734694 and parameters: {'n_estimators': 2934, 'learning_rate': 0.026914961224489796, 'max_depth': 18, 'subsample': 0.901598693877551, 'colsample_bytree': 0.845522693877551, 'gamma': 0.11029832653061224, 'lambda': 9.43718571e-06, 'alpha': 1.54531734e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-23 00:58:45,851] Trial 303 finished with value: 0.9887292244897959 and parameters: {'n_estimators': 2924, 'learning_rate': 0.027095961224489795, 'max_depth': 18, 'subsample': 0.9017605102040816, 'colsample_bytree': 0.8458208163265306, 'gamma': 0.10838900408163265, 'lambda': 9.37973061e-06, 'alpha': 1.5909302e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-23 01:19:17,129] Trial 304 finished with value: 0.9887232265306122 and parameters: {'n_estimators': 2932, 'learning_rate': 0.026883261224489796, 'max_depth': 18, 'subsample': 0.902597306122449, 'colsample_bytree': 0.8460664285714286, 'gamma': 0.10927391836734694, 'lambda': 9.26468122e-06, 'alpha': 1.51957959e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-23 01:38:32,539] Trial 305 finished with value: 0.9887124979591837 and parameters: {'n_estimators': 2933, 'learning_rate': 0.02683732244897959, 'max_depth': 18, 'subsample': 0.9018152244897959, 'colsample_bytree': 0.8454430918367347, 'gamma': 0.10905791836734694, 'lambda': 9.34085306e-06, 'alpha': 1.54208122e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-23 01:57:49,676] Trial 306 finished with value: 0.9886833918367347 and parameters: {'n_estimators': 2943, 'learning_rate': 0.02693452244897959, 'max_depth': 18, 'subsample': 0.9017437959183673, 'colsample_bytree': 0.8457582040816327, 'gamma': 0.10919030612244898, 'lambda': 9.09576041e-06, 'alpha': 1.2921102e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-23 02:17:09,309] Trial 307 finished with value: 0.9887413469387755 and parameters: {'n_estimators': 2938, 'learning_rate': 0.027137551020408165, 'max_depth': 18, 'subsample': 0.901849387755102, 'colsample_bytree': 0.8457580408163265, 'gamma': 0.10605804081632653, 'lambda': 9.27683469e-06, 'alpha': 1.47635918e-06}. Best is trial 298 with value: 0.9887652367346939.
        [I 2025-05-23 02:36:31,788] Trial 308 finished with value: 0.9887732040816327 and parameters: {'n_estimators': 2939, 'learning_rate': 0.02703161224489796, 'max_depth': 18, 'subsample': 0.901620306122449, 'colsample_bytree': 0.8454857755102041, 'gamma': 0.10526832653061224, 'lambda': 7.6791853e-06, 'alpha': 1.6142102e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 02:55:57,703] Trial 309 finished with value: 0.9887640306122449 and parameters: {'n_estimators': 2938, 'learning_rate': 0.026917173469387755, 'max_depth': 18, 'subsample': 0.9018213265306122, 'colsample_bytree': 0.8457611020408163, 'gamma': 0.10508873469387755, 'lambda': 9.20322346e-06, 'alpha': 1.47943469e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 03:15:33,456] Trial 310 finished with value: 0.9887560795918367 and parameters: {'n_estimators': 2940, 'learning_rate': 0.0265235306122449, 'max_depth': 18, 'subsample': 0.9017754387755102, 'colsample_bytree': 0.8452367755102041, 'gamma': 0.10545734693877551, 'lambda': 7.6249902e-06, 'alpha': 1.54438469e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 03:35:11,424] Trial 311 finished with value: 0.9887123959183673 and parameters: {'n_estimators': 2935, 'learning_rate': 0.026461604081632653, 'max_depth': 18, 'subsample': 0.901274693877551, 'colsample_bytree': 0.8458334081632653, 'gamma': 0.1055490612244898, 'lambda': 7.71788122e-06, 'alpha': 1.49264285e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 03:54:49,640] Trial 312 finished with value: 0.9887580795918367 and parameters: {'n_estimators': 2931, 'learning_rate': 0.026610351020408163, 'max_depth': 18, 'subsample': 0.9012435714285714, 'colsample_bytree': 0.8461388571428571, 'gamma': 0.10446050204081633, 'lambda': 7.64661142e-06, 'alpha': 1.54024285e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 04:14:40,967] Trial 313 finished with value: 0.9887700632653062 and parameters: {'n_estimators': 2948, 'learning_rate': 0.026376214285714285, 'max_depth': 18, 'subsample': 0.9038843265306122, 'colsample_bytree': 0.8456291836734694, 'gamma': 0.10341021428571428, 'lambda': 7.4154502e-06, 'alpha': 1.22621346e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 04:34:36,553] Trial 314 finished with value: 0.9887562040816326 and parameters: {'n_estimators': 2954, 'learning_rate': 0.026082032653061224, 'max_depth': 18, 'subsample': 0.8986877040816326, 'colsample_bytree': 0.8459068571428572, 'gamma': 0.1041646224489796, 'lambda': 7.63209183e-06, 'alpha': 1.21343877e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 04:54:29,452] Trial 315 finished with value: 0.9887293836734694 and parameters: {'n_estimators': 2954, 'learning_rate': 0.026229679591836734, 'max_depth': 18, 'subsample': 0.8986493265306122, 'colsample_bytree': 0.8458457346938776, 'gamma': 0.10381928571428572, 'lambda': 7.59844204e-06, 'alpha': 1.18428775e-06}. Best is trial 308 with value: 0.9887732040816327.
        [I 2025-05-23 05:14:23,346] Trial 316 finished with value: 0.9887852693877551 and parameters: {'n_estimators': 2956, 'learning_rate': 0.02621041020408163, 'max_depth': 18, 'subsample': 0.8990884693877551, 'colsample_bytree': 0.8460454387755102, 'gamma': 0.10382344897959183, 'lambda': 7.15345918e-06, 'alpha': 1.2002251e-06}. Best is trial 316 with value: 0.9887852693877551.
        [I 2025-05-23 05:34:18,087] Trial 317 finished with value: 0.9887834571428571 and parameters: {'n_estimators': 2956, 'learning_rate': 0.0261096306122449, 'max_depth': 18, 'subsample': 0.899348918367347, 'colsample_bytree': 0.846044493877551, 'gamma': 0.10115214285714286, 'lambda': 7.78086734e-06, 'alpha': 1.10508122e-06}. Best is trial 316 with value: 0.9887852693877551.
        [I 2025-05-23 05:54:25,023] Trial 318 finished with value: 0.9887684612244898 and parameters: {'n_estimators': 2961, 'learning_rate': 0.02600261224489796, 'max_depth': 18, 'subsample': 0.8981331224489796, 'colsample_bytree': 0.8461470408163265, 'gamma': 0.1009634081632653, 'lambda': 7.60602653e-06, 'alpha': 1.11735234e-06}. Best is trial 316 with value: 0.9887852693877551.
        [I 2025-05-23 06:14:20,151] Trial 319 finished with value: 0.9887590306122449 and parameters: {'n_estimators': 2953, 'learning_rate': 0.0263163306122449, 'max_depth': 18, 'subsample': 0.8987301224489796, 'colsample_bytree': 0.8465127040816326, 'gamma': 0.10099516326530612, 'lambda': 7.47792612e-06, 'alpha': 1.18830041e-06}. Best is trial 316 with value: 0.9887852693877551.
        [I 2025-05-23 06:34:25,529] Trial 320 finished with value: 0.9887560795918367 and parameters: {'n_estimators': 2957, 'learning_rate': 0.02612628163265306, 'max_depth': 18, 'subsample': 0.8988928367346939, 'colsample_bytree': 0.846601918367347, 'gamma': 0.10000870408163265, 'lambda': 7.60275214e-06, 'alpha': 1.1724902e-06}. Best is trial 316 with value: 0.9887852693877551.
        [I 2025-05-23 06:54:28,678] Trial 321 finished with value: 0.9887320000000001 and parameters: {'n_estimators': 2961, 'learning_rate': 0.026193951020408163, 'max_depth': 18, 'subsample': 0.8985832244897959, 'colsample_bytree': 0.8466461224489796, 'gamma': 0.10085302040816326, 'lambda': 7.33724041e-06, 'alpha': 1.14483469e-06}. Best is trial 316 with value: 0.9887852693877551.
        [I 2025-05-23 07:14:24,911] Trial 322 finished with value: 0.9887853102040816 and parameters: {'n_estimators': 2960, 'learning_rate': 0.026319051020408163, 'max_depth': 18, 'subsample': 0.8985668163265306, 'colsample_bytree': 0.846647612244898, 'gamma': 0.09964387755102041, 'lambda': 7.34496612e-06, 'alpha': 1.1915502e-06}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 07:35:59,915] Trial 323 finished with value: 0.9887121775510204 and parameters: {'n_estimators': 2956, 'learning_rate': 0.02607908163265306, 'max_depth': 19, 'subsample': 0.8984592040816326, 'colsample_bytree': 0.8465752448979592, 'gamma': 0.09903932653061224, 'lambda': 7.1501653e-06, 'alpha': 1.1530351e-06}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 07:56:09,017] Trial 324 finished with value: 0.9887541632653061 and parameters: {'n_estimators': 2996, 'learning_rate': 0.02609520408163265, 'max_depth': 18, 'subsample': 0.894733612244898, 'colsample_bytree': 0.8467265306122449, 'gamma': 0.10176114285714286, 'lambda': 8.0590653e-06, 'alpha': 1.19014612e-06}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 08:16:21,659] Trial 325 finished with value: 0.9887593959183674 and parameters: {'n_estimators': 2997, 'learning_rate': 0.025961861224489795, 'max_depth': 18, 'subsample': 0.8937715102040817, 'colsample_bytree': 0.8470054081632653, 'gamma': 0.10064041632653061, 'lambda': 7.73943877e-06, 'alpha': 9.3854302e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 08:36:32,284] Trial 326 finished with value: 0.9887570408163265 and parameters: {'n_estimators': 2998, 'learning_rate': 0.02615628163265306, 'max_depth': 18, 'subsample': 0.8934745714285714, 'colsample_bytree': 0.8468112346938776, 'gamma': 0.10086510204081633, 'lambda': 7.72193469e-06, 'alpha': 9.41728571e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 08:56:44,745] Trial 327 finished with value: 0.9887320000000001 and parameters: {'n_estimators': 2989, 'learning_rate': 0.02608609387755102, 'max_depth': 18, 'subsample': 0.8932554081632653, 'colsample_bytree': 0.8469291836734694, 'gamma': 0.10056489795918367, 'lambda': 5.10877551e-06, 'alpha': 9.35064693e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 09:17:08,327] Trial 328 finished with value: 0.9887233306122449 and parameters: {'n_estimators': 2998, 'learning_rate': 0.025854877551020408, 'max_depth': 19, 'subsample': 0.893650081632653, 'colsample_bytree': 0.8467851020408163, 'gamma': 0.09833877551020408, 'lambda': 4.92363673e-06, 'alpha': 9.44251632e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 09:37:30,011] Trial 329 finished with value: 0.9887773265306123 and parameters: {'n_estimators': 2983, 'learning_rate': 0.02567312244897959, 'max_depth': 19, 'subsample': 0.8938940408163265, 'colsample_bytree': 0.8468040816326531, 'gamma': 0.09888642857142857, 'lambda': 7.99838857e-06, 'alpha': 9.2630051e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 09:57:49,440] Trial 330 finished with value: 0.9887171428571428 and parameters: {'n_estimators': 2977, 'learning_rate': 0.02565258163265306, 'max_depth': 19, 'subsample': 0.8962851428571428, 'colsample_bytree': 0.8492612244897959, 'gamma': 0.10157285714285714, 'lambda': 6.80777551e-06, 'alpha': 1.23115224e-06}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 10:18:17,705] Trial 331 finished with value: 0.9887760040816327 and parameters: {'n_estimators': 2972, 'learning_rate': 0.025668979591836734, 'max_depth': 19, 'subsample': 0.894384612244898, 'colsample_bytree': 0.8472928163265306, 'gamma': 0.09653404081632653, 'lambda': 8.14690102e-06, 'alpha': 1.05996041e-06}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 10:39:31,895] Trial 332 finished with value: 0.9887570408163265 and parameters: {'n_estimators': 2980, 'learning_rate': 0.025467857142857142, 'max_depth': 19, 'subsample': 0.8946463265306122, 'colsample_bytree': 0.8451604591836735, 'gamma': 0.09498532244897959, 'lambda': 8.03178367e-06, 'alpha': 9.43677346e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 11:00:55,865] Trial 333 finished with value: 0.9887722448979592 and parameters: {'n_estimators': 2984, 'learning_rate': 0.025541873469387753, 'max_depth': 19, 'subsample': 0.8941354285714286, 'colsample_bytree': 0.8475336734693877, 'gamma': 0.09581234285714286, 'lambda': 8.09132041e-06, 'alpha': 9.22212346e-07}. Best is trial 322 with value: 0.9887852693877551.
        [I 2025-05-23 11:45:40,354] Trial 334 finished with value: 0.9887243959183674 and parameters: {'n_estimators': 2969, 'learning_rate': 0.026487738775510203, 'max_depth': 18, 'subsample': 0.8930605306122449, 'colsample_bytree': 0.844949306122449, 'gamma': 0.0937485387755102, 'lambda': 8.14514204e-06, 'alpha': 1.25230122e-06}. Best is trial 322 with value: 0.9887852693877551.
        """
        ```
    *   **K-Fold 交叉验证:** 使用上述优化参数，在除去最终保持测试集后的训练数据池 (`Training/CV Pool`) 上进行了 5 折 `StratifiedKFold` 交叉验证。
     *   **5-折交叉验证性能 (使用上述Trial 322参数):**
        在除去最终保持测试集后的训练数据池 (`Training/CV Pool`) 上，使用选定的Trial 322参数进行了5折 `StratifiedKFold` 交叉验证。各折在各自验证集上得到的AUC和早停时的最佳迭代次数如下 (详细训练日志见 `xgboost6_opt_log_main.txt`，时间戳 `20250524_121350`):

        | 折数 (Fold) | 最佳迭代次数 (Best Iteration) | 验证集LogLoss (Validation LogLoss) | 验证集AUC (Validation AUC) |
        |-------------|---------------------------|------------------------------------|----------------------------|
        | 1           | 2955                      | 0.147121                           | 0.986765                   |
        | 2           | 2937                      | 0.146035                           | 0.986906                   |
        | 3           | 2957                      | 0.146042                           | 0.986947                   |
        | 4           | 2958                      | 0.146933                           | 0.986793                   |
        | 5           | 2958                      | 0.147808                           | 0.986667                   |
        | **平均值**  | **~2953**                 | **~0.1468**                        | **~0.9868**                |

        K-Fold交叉验证的平均AUC约为 **0.9868**。折外预测概率已保存用于集成学习。

    *   **最终模型评估 (在独立的 Hold-out Test Set 上，使用Trial 322参数):**
        使用上述Trial 322的参数，在完整的 `Training/CV Pool` 上重新训练了一个最终模型 (`xgboost_yangtsu_v1_opt_xgboost_yangtsu_cv_20250524_121350_final_model.joblib`)。该模型在早停时的最佳迭代次数为2957。其在独立的**保持测试集 (Hold-out Test Set)** 上的性能表现随不同预测概率阈值变化如下（降雨定义阈值 > 0.1mm/d），数据来自 `xgboost6_opt_log_main.txt` (运行时间戳 `20250524_121350` 的评估部分)：

            | 预测概率阈值 | Accuracy | POD    | FAR    | CSI    | FP    | FN    |
            |--------------|----------|--------|--------|--------|-------|-------|
            | 0.10         | 0.9228   | 0.9876 | 0.1024 | 0.8876 | 73571 | 8073  |
            | 0.15         | 0.9344   | 0.9819 | 0.0824 | 0.9024 | 57552 | 11811 |
            | 0.20         | 0.9404   | 0.9764 | 0.0695 | 0.9100 | 47621 | 15424 |
            | 0.25         | 0.9439   | 0.9711 | 0.0600 | 0.9144 | 40491 | 18871 |
            | 0.30         | 0.9459   | 0.9660 | 0.0526 | 0.9168 | 35019 | 22198 |
            | 0.35         | 0.9468   | 0.9609 | 0.0467 | 0.9177 | 30729 | 25525 |
            | 0.40         | 0.9471   | 0.9558 | 0.0416 | 0.9177 | 27111 | 28850 |
            | 0.45         | 0.9466   | 0.9505 | 0.0374 | 0.9166 | 24099 | 32345 |
            | **0.50**     | **0.9456** | **0.9447** | **0.0335** | **0.9147** | **21408** | **36132** |
            | 0.55         | 0.9440   | 0.9385 | 0.0301 | 0.9119 | 19029 | 40166 |
            | 0.60         | 0.9420   | 0.9319 | 0.0270 | 0.9084 | 16881 | 44456 |
            | 0.65         | 0.9395   | 0.9248 | 0.0240 | 0.9042 | 14824 | 49140 |
            | 0.70         | 0.9366   | 0.9169 | 0.0210 | 0.8992 | 12864 | 54238 |

    *   **关键观察 (基于 Hold-out Test Set 最新结果):**
        *   基于V6特征集和Optuna Trial 322的优化参数，最终模型在独立的保持测试集上表现出色。
        *   在预测概率阈值为 **0.50** 时，达到了 **FAR: 0.0335** 和 **CSI: 0.9147** 的当前最优水平。
        *   通过调整预测概率阈值，可以在 POD 和 FAR 之间进行权衡。例如，在阈值 0.40 时，POD高达0.9558，但FAR也相应增至0.0416；而在阈值 0.70 时，FAR 可以降低到约 **0.0210**，此时POD为0.9169。
    *   **Top 10 特征重要性 (来自基于Trial 322参数训练的最终模型，引自 `xgboost6_opt_log_main.txt`，时间戳 `20250524_121350`):**
        1.  `lag_1_values_GSMAP` (0.366157)
        2.  `diff_1_values_GSMAP` (0.170459)
        3.  `season_onehot_2` (夏季) (0.098154)
        4.  `raw_values_GSMAP` (0.078974)
        5.  `lag_1_mean` (0.009979)
        6.  `sin_day` (0.008308)
        7.  `cos_day` (0.007852)
        8.  `season_onehot_1` (春季) (0.007400)
        9.  `season_onehot_0` (冬季) (0.006025)
        10. `season_onehot_3` (秋季) (0.005991)
        *   观察可见，GSMAP 相关的特征（滞后一天值、差分值、当前值）以及多产品一致性（滞后均值）和季节性特征在模型决策中占据主导地位。
    *   **产出文件 (用于集成学习和最终评估):**
        *   折外预测概率 (Training/CV Pool): `results/yangtze/features/kfold_optimization_v6/Train_L0_Probs_v6_Opt.npy` (由 `xgboostv6_for_Ensemble.py` 生成)
        *   最终模型在保持测试集上的预测概率: `results/yangtze/predictions/v1_opt_xgboost_yangtsu_cv_20250524_121350/xgboost_yangtsu_v1_opt_xgboost_yangtsu_cv_20250524_121350_holdout_test_proba_predictions.npy`
        *   最终优化模型: `results/yangtze/models/v1_opt_xgboost_yangtsu_cv_20250524_121350/xgboost_yangtsu_v1_opt_xgboost_yangtsu_cv_20250524_121350_final_model.joblib`
        *   各折模型位于 `results/yangtze/features/kfold_optimization_v6/` (或相应模型保存路径，如日志中所示的 `xgboost_yangtsu_v1_opt_xgboost_yangtsu_cv_20250524_121350_fold_X_model.joblib`) 目录下。

**总结与洞察:**

*   **特征集 V6 + Optuna 深度优化展现最佳性能**: 基于 `turn6.py` 构建的 V6 特征集，在经过大规模 Optuna 超参数寻优（例如，以 Trial 322 的参数为代表，该次试验 AUC 达到约 0.98878），并结合规范的 K-Fold 交叉验证流程后，其最终训练的 XGBoost 模型在独立的**保持测试集 (Hold-out Test Set)** 上展现了当前项目中最优的综合性能。尤其在误报率 (FAR) 控制方面取得了显著成效：在预测概率阈值为 **0.50** 时，FAR 降低至约 **0.0335**，CSI 达到约 **0.9147**。当预测概率阈值提升至 0.70 时，FAR 可进一步降低至约 **0.0210**，同时保持较高的命中率。

*   **Optuna 优化的显著且持续的效果**: 对比特征集 V6 的默认参数模型（在0.5阈值下，FAR: 0.0819, CSI: 0.8228）与经过持续 Optuna 优化后的模型（例如，早期优化FAR约0.0357, CSI约0.9043；最新基于Trial 322参数的最终模型FAR约0.0335, CSI约0.9147），超参数优化能显著且持续地提升模型性能，特别是在降低误报率 (FAR) 和提高临界成功指数 (CSI) 方面效果尤为突出。这证明了精细化调参对于挖掘模型潜力的重要性。

*   **关键特征与后续工作基础**: 从最新的V6优化模型（基于Trial 322参数）的特征重要性分析来看，GSMAP 相关的特征（如 `lag_1_values_GSMAP`, `diff_1_values_GSMAP`, `raw_values_GSMAP`）、多产品一致性特征（如 `lag_1_mean`）以及季节性特征（如 `season_onehot_2` (夏季)）在模型决策中起到了决定性作用。通过 `xgboostv6_for_Ensemble.py` 生成的基于此优化模型的折外预测 (`Train_L0_Probs_v6_Opt.npy`) 为后续的集成学习（特别是 Level 1 层 FP/FN 专家模型的构建）提供了高质量、高可靠性的基础输入。

*   **特征集迭代观察**:
    *   V1 (`turn1.py`) 特征集基于格点数据，在默认参数下，预测概率阈值 0.50 时 CSI 为 0.7820，FAR 为 0.0823，为早期格点数据处理提供了性能基准。
    *   V2 (`turn2.py`) 至 V5 (`turn5.py`) 特征集均基于点位数据，采用默认或预设参数进行评估，展示了不同特征简化或调整策略下的性能。例如，在预测概率阈值 0.5 时：
        *   V2: CSI 0.7841, FAR 0.0687
        *   V3: CSI 0.7859, FAR 0.0681
        *   V4: CSI 0.8101, FAR 0.0572
        *   V5: CSI 0.8115, FAR 0.0564
    *   这些结果清晰地反映了不同特征工程策略对模型性能的直接影响，并与最终 V6 版本（默认参数下 CSI 0.8228, FAR 0.0819；Optuna优化后在Hold-out测试集上 CSI 0.9147, FAR 0.0335）形成了鲜明对比，凸显了特征工程与超参数优化协同作用的重要性。

*   这些详尽的迭代和评估结果，为后续的模型选择、特征优化方向（例如，是否需要针对特定误差类型进一步精炼V6特征、或探索其他类型的特征）以及集成学习策略（如FP/FN专家模型的设计和元学习器的选择）的设计提供了宝贵的经验积累和坚实的数据支持。
### 2.4 全方位评估与诊断 (主要分析脚本位于 `src/nationwide/analysis/`, `src/nationwide/project_all_for_deepling_learning/feature_of_FP_FN.py`, `src/yangtze/YangTsu/feature_of_FP_FN_Yangtsu.py`)

建立了全面的模型评估和诊断体系，远超标准指标评估：

*   **标准指标**: Accuracy, Precision, Recall, F1-score, ROC AUC 等。
*   **气象相关指标**: 可能探索了如 Heidke Skill Score (HSS), Equitable Threat Score (ETS) 等专业评分。
*   **训练过程监控**: 可视化并分析 **训练/验证损失曲线**，判断模型拟合状态。
*   **特征重要性分析**: 利用模型内置方法（如 SHAP 值或 Gini 不纯度）量化和排序 **特征贡献度**，理解模型决策依据。
*   **特征相关性研究**: 计算并可视化 **特征间的相关系数矩阵**，识别冗余特征，并对比不同特征集 (`v1` vs `v2`) 的差异。相关脚本如 `src/nationwide/analysis/analyze_feature_sets_v1_v2.py`, `analyze_correlations_v1.py`。
*   **误差空间分布**: 绘制 **预测误差（如偏差、均方根误差）的空间分布图**，识别模型表现的地理差异和薄弱区域。
*   **误差时间演变**: 分析误差的 **季节性变化和月度趋势**。
*   **误报/漏报 (FP/FN) 深度诊断**:
    *   识别 **FP/FN 事件高发的热点区域**。
    *   **专门分析 FP/FN 样本对应的特征分布** (通过 `src/nationwide/project_all_for_deepling_learning/feature_of_FP_FN.py` 和长江流域的 `src/yangtze/YangTsu/feature_of_FP_FN_Yangtsu.py` 及 `feature_of_FP_FN_Yangtsu_Mean.py` 等脚本)，反向推断导致预测错误的关键因素，为特征工程优化提供直接依据。
*   **预测阈值敏感性分析**: 系统评估不同 **分类阈值** 对 Precision, Recall 等指标的影响，为实际应用选择最佳阈值提供依据。

## 3. 项目研发历程与关键节点

本项目从零起步，经历了从理论学习到复杂系统实现、从全国宏观到区域精细的完整研发周期，充分展现了研究者在跨学科知识融合、大规模数据处理、前沿算法应用和系统性问题解决方面的综合能力。关键节点与活动深度剖析如下：

-   **阶段一：理论奠基与技术预研 (2024年底)**
    *   **跨学科知识储备**: 面对降雨预测这一复杂交叉领域，进行了 **广泛而深入的文献调研**，系统学习了大气科学基础、卫星遥感原理、机器学习/深度学习核心算法（特别是梯度提升、时间序列分析、集成学习等），以及 Python 数据科学生态（Numpy, Pandas, Scipy, Matplotlib 等）。**目标是建立坚实的理论基础，理解问题本质**。
    *   **多源数据理解与挑战识别**: 获取部分早期数据后，投入大量时间 **解析多种卫星产品**（CMORPH, CHIRPS 等）的 **数据格式 (NetCDF, GRIB, HDF等)、投影方式、时空分辨率差异以及各自的优缺点**。识别出 **数据融合、时空对齐、缺失值处理** 等核心技术挑战。
    *   **技术可行性探索与环境搭建**: 搭建了基于 Python 的数据分析与机器学习环境。针对小规模数据子集，**编写了原型脚本** 进行数据读取 (如使用 `xarray`, `netCDF4`)、基础可视化和预处理流程的 **初步验证**，为后续大规模处理积累了宝贵经验，并验证了技术路线的可行性。

-   **阶段二：数据壁垒攻克与基线建立 (2025年1月中旬 - 2月底)**
    *   **全量数据获取与管理**: 成功获取 **覆盖多年、多源的 TB 级完整数据集**，并建立了初步的数据管理方案。
    *   **大规模并行预处理流程设计与实现**: 面对海量数据，设计并实现了 **高效的预处理流程** (`src/readdata/` 下的各产品处理脚本及 `scale_utils.py` 等)，关键环节包括：
        *   **地理空间处理**: 利用 `geopandas`, `rasterio` 等库，实现了从全球数据中 **精确、高效地裁剪中国区域** 数据，并生成和应用了研究区掩码 (`data/processed/nationwide/masks/`)。
        *   **时空对齐**: 开发算法（如 `src/readdata/scale_utils.py`）将不同来源、不同分辨率的数据 **插值/重采样到统一的0.25°时空网格**，确保后续特征工程和模型输入的有效性。
        *   **智能 NaN 值处理**: 针对降雨数据特性，设计了 **结合时空插值和阈值替换的多阶段 NaN 处理策略** (如 `src/readdata/datafix.py` 中的逻辑)，最大程度保留有效信息，并将处理结果固化为标准化的 `.mat` 文件，存放于 `data/intermediate/` 各产品目录下。
    *   **数据驱动的理解深化**: 对预处理后的各产品数据进行了 **统计分析和可视化对比** (如通过 `src/yangtze/YangTsu/Spatial_characteristics_plot_fp_fn.py` 和 `src/yangtze/YangTsu/value_characteristics_fp_fn_.py` 等脚本进行的早期探索)，量化了产品间的差异、一致性及其性能的时空变异性，为后续特征工程和模型选择提供了关键洞见。
    *   **模型技术栈拓展**: 在掌握基础 ML 理论后，开始深入学习 **LSTM 等深度学习模型** 在处理时空序列数据方面的潜力，并调研其在气象领域的应用。
    *   **基线模型快速迭代 (2月底)**: **快速实现了** 包括 XGBoost、LSTM 及基础 MLP 在内的 **多种基线模型**，进行了初步训练和评估（如 `src/legacy/` 中的部分早期脚本）。**目的不仅是测试模型效果，更是为了打通整个 "数据-特征-模型-评估" 的流程**，识别瓶颈。

-   **阶段三：核心算法精通与特征体系构建 (2025年3月中旬 - 4月初)**
    *   **核心模型聚焦**: **超越库调用层面**，深入研究了 XGBoost 的 **原理（梯度提升、正则化、分裂算法等）、核心参数含义、调优技巧以及如何针对不平衡数据调整目标函数/评估指标**。
    *   **关键库能力强化**: **系统性地掌握了 scikit-learn** 的核心模块（`Pipeline`, `Preprocessing`, `Metrics`, `Model Selection` 等）和 **PyTorch** 的基础（张量操作、自动微分、模型构建），为复杂模型实现和高效实验奠定基础。
    *   **文献与数据驱动的特征工程设计**: 结合第二阶段的数据分析结果和 **对领域文献的深入研读**，**系统性地设计了多维度特征体系** (详见 2.2 节)，从理论层面确保了特征的科学性和有效性。特征工程脚本 (`turn*.py`) 开始迭代。
    *   **问题形式化与基准确立**: **清晰地将任务解构** 为降雨分类和回归两个子问题，并 **严谨地选择 CHM 数据作为参照真值 (Y)**，为后续模型评估提供了统一、可靠的基准。

-   **阶段四：实验迭代、性能优化与区域聚焦 (2025年4月)**
    *   **特征工程"炼金术" - 高速迭代与验证**: 进入 **高强度实验迭代周期**，基于 **模型性能反馈和误差分析**，快速实现、测试并优化了 **至少六个核心版本的特征集 (`v1` 至 `v6`)**。长江流域的特征工程 (`src/yangtze/YangTsu/turn*.py`) 从V1（格点数据探索）逐步演进到V5，全国范围的特征工程 (`src/nationwide/.../turn*.py`) 也同步进行。**证明了研究者具备快速试错、数据驱动决策和高效实现复杂特征的能力**。特征维度和复杂度显著提升。
    *   **自动化超参数寻优**: 引入并 **熟练运用 Optuna** 对 XGBoost, LightGBM 等模型进行 **系统的超参数优化** (如 `src/nationwide/.../xgboost3_optuna.py`)，**显著提升了调参效率和模型性能上限**，避免了手动调参的低效和盲目性。
    *   **创新性误差修正 - FP/FN 专家集成模型初步探索**: 针对降雨分类中 **误报 (FP) 和漏报 (FN) 的不对称性及关键影响**，**创新性地设计并初步实现了基于误差分析的集成策略**：利用贝叶斯网络等模型 (如 `src/nationwide/.../bayesian_network_fp_expert.py`) **专门训练识别 FP 和 FN 样本的"专家"**，旨在 **靶向性地提升分类器的 POD, FAR 等关键业务指标**。
    *   **空间异质性洞察与分区建模探索**: 通过误差分析敏锐地 **识别出模型性能的空间异质性**，认识到单一全局模型的局限性，**前瞻性地启动了分区建模** 的探索。
    *   **长江流域精细化研究快速启动 (4月底)**: **高效地将全国范围的分析框架和代码复用并适配到长江流域**，快速完成了区域掩码生成、数据提取、特征工程 (至V5版本) 和初步建模 (`src/yangtze/YangTsu/xgboost1.py` 至 `xgboost5.py`)，**展现了研究流程的模块化和可扩展性**。

-   **阶段五：长江流域深度优化与集成学习体系构建 (2025年5月初至今)**
    *   **工程化重构与文档化**: 持续投入精力对项目代码和文件进行**结构化重构和规范化**，建立了更清晰、模块化的项目结构（如引入 `src/visualization/` 目录），并**系统性地更新和丰富 README 文档**，确保项目知识的有效沉淀与清晰展示。
    *   **长江流域 V6 特征集最终确立与深度优化**:
        *   完成了长江流域特征工程的 **V6 版本 (`src/yangtze/YangTsu/turn6.py`)**，该版本在V4的全面性基础上进行提炼，成为当前模型优化的基石。
        *   使用 `src/yangtze/YangTsu/xgboost_optimization_main.py` 脚本，对基于V6特征集的XGBoost模型进行了**大规模（超过250次试验，记录于 `my_optimization_history.db`）的 Optuna 超参数寻优**。
        *   通过持续优化，在独立的**保持测试集 (Hold-out Test Set)** 上，模型性能得到进一步提升，例如在0.5概率阈值下，FAR 降低至约 **0.0336**，CSI 提升至约 **0.9145** (详见 2.3.2 节和 `xgboost6_opt_log_main.txt` 日志)。
    *   **Optuna 优化过程管理与分析**:
        *   通过 `src/yangtze/YangTsu/loadoptunadb.py` 脚本，将历史的Optuna试验日志（如早期150次试验）统一导入SQLite数据库 (`my_optimization_history.db`)，实现了优化历史的集中管理。
        *   开发了 `src/visualization/optimization_history.py` 等脚本，对Optuna的优化过程、参数敏感度等进行了可视化分析，增强了对超参数空间的理解。
    *   **集成学习框架搭建 (长江流域)**:
        *   **Level 0 层输出准备**: 利用 `src/yangtze/YangTsu/xgboostv6_for_Ensemble.py` 脚本，基于V6优化后的最佳参数，在 `Training/CV Pool` 上进行了K-Fold交叉验证，生成了用于集成学习的**折外预测概率 (`Train_L0_Probs_v6_Opt.npy`)** 和在Hold-out测试集上的预测 (`Test_L0_Probs_v6_Opt_from_final_model.npy`)。各折模型也进行了保存。
        *   **Level 1 层专家模型构建**: 在 `src/yangtze/YangTsu/Ensemble1/` 目录下启动了针对FP和FN的专家模型研究。`indentify_fp_fn_v1.py` 脚本用于识别L0模型的FP/FN样本并生成专家模型的目标标签；`fp_v1.py` 和 `fn_v1.py` 则分别基于原始特征训练FP和FN专家模型，并生成其折外预测。这些输出将作为更高层次元学习器的输入。
    *   **FP/FN 深度诊断与可视化**:
        *   除了分析导致FP/FN的输入特征 (`src/yangtze/YangTsu/feature_of_FP_FN_Yangtsu.py`)，还通过新脚本如 `src/yangtze/YangTsu/Spatial_characteristics_plot_fp_fn.py` 和 `src/yangtze/YangTsu/value_characteristics_fp_fn_.py`，从原始产品性能指标（POD, FAR, CSI）的空间分布、以及FP/FN事件发生时原始观测值的统计特征等多个角度，对误差成因进行了更细致的探究。
    *   **成果可视化体系完善**: 开发并使用了 `src/visualization/` 目录下的系列脚本 (`evolution.py`, `plot_3d_performance.py`, `thresholds_plot.py`) 以及 `src/yangtze/YangTsu/model_compare_plot.py`，系统性地生成了模型性能演进图、多模型/多指标对比图（热力图、雷达图、3D图、棒棒糖图等），用于全面展示项目成果和迭代进展。
## 4. 项目可持续性与未来展望

本项目已在多源降雨数据融合、大规模特征工程、机器学习建模与深度诊断方面奠定了坚实的基础，取得了显著的阶段性成果，并展现出强大的可持续发展潜力。当前构建的研究框架不仅在降雨预测精度（尤其是在误报率控制方面）取得了突破，也为后续更深层次的探索和应用拓展铺平了道路。

### 4.1 近期工作与代码库完善

1.  **代码库规范化与模块化**:
    *   **【进行中】** 持续优化 `src/` 目录下的脚本，确保文件路径的准确性和代码在新结构下的无缝运行。
    *   逐步将核心功能（如数据加载、特征工程、模型训练与评估、特定可视化模块）**封装为可复用的 Python 模块或包**，以提升代码质量、可维护性和易用性。
    *   **文档完善**: 持续补充和更新 `README.md`，确保其准确反映项目最新进展。逐步完善代码内注释和必要的技术文档，包括环境配置、数据说明和关键脚本的运行指令。
2.  **Git 版本控制的严格执行**: 确保所有重要代码和实验配置都纳入 Git 版本控制，规范管理代码历史、分支和实验追溯。

### 4.2 研究方向深化与拓展

1.  **区域化研究深化**:
    *   **对比分析**: 深入对比长江流域与全国范围的模型表现、关键特征差异以及误差模式，探索区域适应性特征和模型参数的迁移规律。
    *   **区域扩展**: 可将当前成熟的分析框架和模型体系**扩展到其他重要流域（如黄河流域、珠江流域）或不同气候特征区域**，进行模型迁移性和区域适应性研究，探索普适性与区域特异性相结合的建模策略。
    *   **引入高分辨率地理空间因子**: 考虑引入更高分辨率的 **地形地貌数据 (DEM)**、**土地利用/土地覆盖变化 (LULCC)**、**土壤类型**、**植被指数 (NDVI)** 等地理因子，通过精细化的空间特征工程，增强区域模型的物理基础和预测精度。

2.  **特征工程前沿探索**:
    *   **多模态数据融合**: 在现有降雨产品基础上，**引入其他相关的多模态气象与环境数据源**，例如：
        *   **地理高程数据 (DEM)**: 已在区域化研究中提及，可更细致地刻画地形对降雨的影响。
        *   **地面气象观测**: 如温度、湿度、风速风向、气压等站点或格点化数据。
        *   **卫星云图产品**: 如云顶亮温、云分类、云光学厚度等，提供降雨系统上游信息。
        *   **大气再分析资料**: 如NCEP/NCAR, ERA5等，提供大尺度环流背景、水汽输送、不稳定能量等物理量场。
    *   **基于深度学习的特征提取**: 利用 **卷积神经网络 (CNN)** 或 **ConvLSTM** 等深度学习模型，在原始降雨产品数据（尤其是格点数据）上进行预训练，以自动提取更复杂、更抽象的时空特征，作为下游机器学习模型的辅助输入。
    *   **误差驱动的特征迭代**:
        *   **FP/FN 样本特征深度挖掘**: 持续分析导致误报 (FP) 和漏报 (FN) 的样本在各类特征（包括新引入的多模态数据）上的分布差异。
        *   **物理意义引导的特征交互**: 基于对FP/FN成因的理解，设计针对性的高阶交互特征。
        *   **降雨阈值敏感特征**: 探索构建能够提醒模型在特定条件下（例如，当多种产品指示的原始降雨强度普遍低于某个动态阈值时）误报或漏报概率增大的警示性特征。
    *   **特征重要性反馈优化**: 根据每一轮模型训练和评估得到的特征重要性排序，有选择性地增强或减弱某类特征的权重，调整特征组合，或对高重要性特征进行更细致的衍生和交叉。

3.  **机器学习与深度学习模型前沿探索**:
    *   **集成学习策略深度优化**:
        *   **完善Level 1专家模型**: 持续优化针对FP和FN的专家模型，可尝试不同的模型结构（如更轻量级的模型、或结合领域知识的规则模型），并探索更有效的专家模型训练样本筛选策略。
        *   **Level 2元学习器 (Meta-learner)**: 系统比较不同类型的元学习器（如逻辑回归、梯度提升树、神经网络、贝叶斯模型等）性能，并对元特征进行选择和优化。探索将元学习器的输出与基础模型预测进行更智能的融合方式。
        *   **使用朴树贝叶斯模型的可行性分析**:
        *   可行性与优势：
            信息互补与多样性利用 (Stacking的核心思想):
            不同的主模型（算法、特征子集）可能会从不同角度捕捉数据中的信息，犯不同类型的错误。Stacking 的核心优势就在于元学习器可以学习如何结合这些不同模型的“智慧”和“偏见”，从而做出比任何单个模型都更好的最终判断。
            FP/FN 专家模型的加入，直接为元学习器提供了关于基础模型可能犯特定类型错误的信息，这对于有针对性地改善误报率和漏报率非常有价值。
            贝叶斯模型作为元学习器的优势:
            概率性输出: 贝叶斯模型天然地提供概率性输出，这对于降雨预测（通常需要预测降雨概率）是非常合适的。
            不确定性量化: 许多贝叶斯模型能够提供预测的不确定性估计，这在气象预报中是非常有价值的附加信息。例如，高斯过程分类器可以直接输出预测类别的概率和置信区间。
            处理小样本元特征: 元特征的数量通常远小于原始特征数量，样本量也可能相对较少（取决于K-Fold的折数和原始数据量）。一些贝叶斯模型（如朴素贝叶斯）在小样本情况下依然能有不错的表现，并且不容易过拟合。
            先验知识的融入: 理论上，贝叶斯框架允许融入先验知识。虽然在纯数据驱动的Stacking中不常用，但如果对元特征的可靠性或某些主模型的表现有先验判断，理论上可以体现在模型中。
            对噪声的鲁棒性: 某些贝叶斯方法（如通过正则化或合适的先验）可能对元特征中的噪声具有一定的鲁棒性。
            针对性解决误报/漏报:
            通过引入FP/FN专家模型的预测作为元特征，元学习器（贝叶斯模型）可以直接学习在什么情况下基础模型的“有雨”预测更可能是FP，或者“无雨”预测更可能是FN。这比单一模型试图同时优化所有指标可能更有效。
            避免过拟合:
            使用折外预测（OOF Predictions）作为元学习器的训练数据，是 Stacking 防止信息泄露和过拟合的关键机制。只要严格执行，元学习器就是在学习基础模型在未见过数据上的泛化行为。
        *   **多模型融合**: 训练多个不同架构或不同参数配置的强主模型，研究如何结合它们的优劣，例如通过加权平均、投票或更复杂的元学习策略进行融合。
    *   **深度学习在降雨分类与回归任务中的应用**:
        *   **降雨分类**: 继续评估和优化现有基于梯度提升树的模型，同时可探索Transformer等序列模型在处理长时序依赖和特征交互方面的潜力。
        *   **降雨量回归 (后续重点)**: 计划使用深度学习模型。
            *   **ConvLSTM / PredRNN**: 利用其结合CNN的空间特征提取能力和RNN的时序信息捕捉能力，处理高分辨率格点降雨数据，进行时空序列预测。
            *   **Transformer架构**: 探索基于Transformer的模型（如Swin Transformer, VideoMAE等变体）在气象时空数据建模方面的潜力，特别是其捕捉长距离依赖关系的能力。
            *   **注意力机制增强**: 在ConvLSTM等模型中引入自注意力 (Self-Attention)、多头注意力 (Multi-Head Attention) 或空洞卷积与注意力结合的机制，以捕捉更长、更广的时空特征和复杂依赖。
    *   **借鉴大型预训练模型与生成式AI的思路与技术**:
        *   **预训练与微调范式**: 探索将在大规模、多样化气象数据（甚至跨领域时空数据）上预训练过的基础模型（如基于Transformer或CNN-RNN的编码器），针对本项目的特定降雨预测任务（分类或回归）进行微调 (Fine-tuning)。这有望利用预训练模型学习到的通用时空表征能力，提升模型在数据相对不足或特定区域的性能。
        *   **生成式模型用于数据增强或场景模拟**: 研究使用生成对抗网络 (GANs) 或扩散模型 (Diffusion Models) 等生成式AI技术，根据已有的降雨数据特征，生成更多样化、更逼真的降雨场景样本，特别是针对稀有的极端降雨事件，以增强模型的训练数据和鲁棒性。
        *   **Prompt Engineering / 指令微调的启发**: 虽然直接应用LLM的Prompt机制到数值预测任务有难度，但其“通过指令引导模型行为”的思路，可能启发我们设计更灵活的模型输入方式或条件化预测框架，例如，通过输入特定的气象条件描述或预设的风险等级，来引导模型生成对应的降雨预测。
        *   **多模态融合的新视角**: 借鉴大型多模态模型（如CLIP, DALL-E等）处理不同类型数据（文本、图像等）的思路，探索如何更有效地融合本项目中已有的多源降雨产品与未来可能引入的地理、云图、再分析资料等多模态数据，让模型学习到更深层次的跨模态关联。
    *   **可解释性 AI (XAI)**: 深入应用SHAP、LIME等XAI技术，不仅分析全局特征重要性，更要分析模型对具体预测个例（尤其是FP/FN样本）的决策依据，增强模型透明度和可信度，并反过来指导特征工程。
    *   **多任务学习与迁移学习**:
        *   **多任务学习**: 例如，同时预测降雨有无（分类）和降雨量（回归），或同时预测不同提前期的降雨。
        *   **迁移学习**: 将在数据丰富的区域或全国范围训练得到的模型参数作为初始权重，在数据相对稀疏的特定小区域进行微调。

### 4.3 应用潜力与拓展 (详细展望)

本项目不仅致力于提升降雨预测的科学精度，更着眼于其在实际业务中的应用潜力和社会经济价值。

1.  **构建业务化准实时降雨监测与预警系统**:
    *   **目标**: 将当前离线模型框架逐步改造为能够 **接近实时** (例如，每小时或每日更新) 处理最新获取的卫星遥感数据、地面观测数据及其他辅助数据，并快速输出高分辨率降雨分析与短临预测结果的业务化系统。
    *   **技术挑战**: 解决 **多源异构数据获取与实时接入的延迟**、**大规模数据流的在线处理与特征计算效率**、**模型推理速度** (需要适配高频更新需求) 等工程难题。可能需要引入流处理框架 (如 Apache Kafka, Flink)、高效的并行计算与分布式存储架构 (如 Spark, Dask, Hadoop HDFS)，以及模型压缩与量化技术。
    *   **潜在应用**: 为 **城市内涝预警**、**中小流域山洪灾害风险实时评估**、**交通气象保障**、**精细化农业灌溉决策支持**、**水库安全调度辅助** 等提供关键技术支撑，具有显著的社会经济效益。

2.  **极端降雨事件预测与风险评估能力强化**:
    *   **目标**: 针对 **暴雨、持续性强降水、极端短时强降水** 等致灾性强的极端降雨事件，显著提升模型的 **预报预警能力，特别是事件的命中率、提前量和空间定位准确率**。
    *   **技术路径**:
        *   **数据层面**: 重点收集和标注历史极端降雨事件样本，可能需要采用 **过采样 (e.g., SMOTE, ADASYN)、欠采样或代价敏感采样** 技术来处理极端事件样本的稀疏性和不平衡问题；探索 **数据增强 (Data Augmentation)** 方法，如对现有极端事件进行合理的扰动以生成更多训练样本。
        *   **特征层面**: 挖掘与极端降雨事件（如梅雨锋暴雨、台风暴雨、暖区暴雨等不同类型）更相关的 **大尺度环流背景特征 (如副热带高压位置强度、切变线、低涡等)**、**热动力不稳定能量指标 (如CAPE, K指数等)**、**水汽输送特征 (如水汽通量散度)**，以及**高时空分辨率的前兆信号**。
        *   **模型层面**: 探索 **代价敏感学习 (Cost-Sensitive Learning)** 算法，赋予极端事件样本（特别是漏报）更高的错分代价；研究 **异常检测 (Anomaly Detection)** 或 **离群点检测 (Outlier Detection)** 算法用于识别极端降雨信号；设计 **针对极端事件优化的损失函数** (如 Focal Loss 的变体，或结合物理约束的损失)；发展**概率性极端事件预报模型**。
    *   **价值**: 直接服务于国家和地方的防灾减灾需求，为应对气候变化背景下日益频发的极端天气事件提供科学支撑，最大限度减少生命财产损失。

3.  **与下游应用模型深度耦合，构建智慧应用生态**:
    *   **目标**: 将本项目输出的 **高精度、高分辨率、概率性的降雨监测与预测产品** 作为高质量的 **驱动数据或输入场**，与水利、农业、交通、能源、环境等相关领域的专业应用模型进行深度耦合，提升下游模型的模拟精度和决策支持能力。
    *   **耦合实例**:
        *   **智慧水利**: 驱动 **分布式水文模型** (如 SWAT, HEC-HMS, MIKE SHE, VIC 等)，进行更精准的 **流域洪水演进模拟、中小河流水位预测、水库优化调度、城市内涝积水深度与范围预估**。
        *   **精准农业**: 耦合 **作物生长模型** (如 DSSAT, WOFOST, APSIM)，进行更准确的 **作物产量预估、旱涝灾害风险评估、病虫害发生发展气象风险预测、变量灌溉决策**。
        *   **智慧交通**: 为交通管理部门提供精细化的路面降雨和积水风险预报，支持 **交通拥堵疏导、事故风险预警、路网规划**。
        *   **地质灾害防治**: 为 **滑坡、泥石流、崩塌** 等降雨诱发型地质灾害的风险评估和预警模型提供更可靠的实时和预报降雨输入。
    *   **价值**: 打破领域壁垒，通过提供高质量的基础气象数据输入，**赋能** 其他相关领域的模型预测能力和应用水平，促进形成跨行业的气象服务生态。

4.  **不确定性量化与概率预报产品研发**:
    *   **目标**: 不仅提供确定性的降雨预测值（或分类），更要提供 **预测结果的不确定性信息**，即输出 **概率预报产品** (例如，未来 X 小时某地降雨量超过 Y mm 的概率；降雨强度的概率密度函数等)。
    *   **技术路径**: 深入探索 **贝叶斯深度学习 (Bayesian Deep Learning)** 方法 (如利用 Monte Carlo Dropout, 变分推断等)；进一步发展和优化 **模型集成 (Ensemble Forecasting)** 技术，通过多个模型或多次运行的扰动结果来估计不确定性；研究 **分位数回归 (Quantile Regression)** 或 **生成模型 (如 GANs, VAEs)** 在概率预报中的应用。
    *   **价值**: 为各类决策者（从政府应急管理到企业风险控制，再到个人出行规划）提供更全面的风险信息，支持基于概率的风险管理和鲁棒决策制定，提升预报产品的实用价值。

5.  **机器学习与数值天气预报 (NWP) 的双向融合探索**:
    *   **目标**: 探索将本项目的机器学习模型与传统的数值天气预报 (NWP) 模型进行更深层次的双向融合，以期实现优势互补，提升整体天气预报能力。
    *   **技术路径**:
        *   **机器学习改进NWP后处理**: 将本项目的高分辨率降雨分析与预测结果用于订正NWP模式的降雨预报偏差。
        *   **机器学习辅助NWP参数化**: 利用机器学习模型从观测和高分辨率模拟数据中学习复杂的物理过程（如对流触发、微物理过程），以改进NWP模式中不完美的物理参数化方案。
        *   **数据同化应用**: 将机器学习模型提取的关键特征或预测结果作为新的“观测”信息，通过数据同化技术（如集合卡尔曼滤波 EnKF, 3DVar/4DVar）融入NWP模式的初始场分析或模式状态更新中。
    *   **价值**: 探索机器学习与传统数值预报结合的新范式，有望从更根本的层面提升天气预报，特别是定量降水预报 (QPF) 的能力。

### 4.4 工程化与可持续发展实践

1.  **构建自动化训练与评估流水线 (Pipeline)**: 逐步利用 Airflow, Kubeflow, MLflow 等工具，实现从数据获取、预处理、特征工程、模型训练、超参数优化、评估到模型部署的自动化流程，提高研发效率和实验可重复性。
2.  **模型版本管理与部署**: 建立规范的模型版本控制机制，探索将优化后的模型部署为可调用的服务接口，为业务化应用奠定基础。
3.  **持续集成/持续部署 (CI/CD)**: 在代码库达到一定成熟度后，引入CI/CD流程，确保代码质量和快速迭代。

**结论:** 本项目在多源降雨数据融合、大规模特征工程、机器学习建模与深度诊断方面开展了系统性的研究和实践，取得了丰富的阶段性成果，并已构建了一个可持续发展、潜力巨大的研究框架。项目不仅在降雨预测精度上取得了显著进展，也为未来的智能化气象服务和跨领域应用展现了广阔前景。研究者在处理复杂气象数据、应用先进机器学习技术以及系统性解决科学问题方面的综合能力得到了充分体现。后续工作将围绕区域化深化、特征与模型前沿探索、以及推动研究成果向业务化应用转化等方向持续推进。
## 5. 项目目录结构说明

本项目采用以下目录结构进行组织和管理：

*   **`.git/`**: Git 版本控制系统目录，包含所有版本历史和分支信息。
*   **`data/`**: 存放项目所需的所有数据。
    *   `raw/`: 存储从原始数据源下载的未处理数据，如 CMORPH, CHIRPS, GSMAP, IMERG, PERSIANN, SM2RAIN, CHM 等产品的 `.nc`, `.zip` 文件。
    *   `intermediate/`: 存储经过完整预处理流程（如 NaN 值填充、时空对齐、掩码应用）后，可直接用于模型输入的最终数据每一年或者连续5年的数据。主要用于训练或者特征工程处理的适合从这里加载，这里的数据就经过空间对齐操作的中国地区0.25分辨率144*256矩阵的数据，只有在中国mask区域上的点有值不在中国区域mask上的点为nan。文件为：
        ```
        CMORPHdata/
            -a----          1/3/2025  10:53 pm       53969088 CMORPH_2016.mat
            -a----          5/3/2025   5:18 pm      269402304 CMORPH_2016_2020.mat
            -a----          1/3/2025  10:53 pm       53821632 CMORPH_2017.mat
            -a----          1/3/2025  10:53 pm       53821632 CMORPH_2018.mat
            -a----          1/3/2025  10:53 pm       53821632 CMORPH_2019.mat
            -a----          1/3/2025  10:53 pm       53969088 CMORPH_2020.mat
        CHIRPSdata/
            -a----          2/3/2025  11:19 am      107937984 chirps_2016.mat
            -a----          5/3/2025   4:30 pm      538804416 chirps_2016_2020.mat
            -a----          2/3/2025  11:19 am      107643072 chirps_2017.mat
            -a----          2/3/2025  11:19 am      107643072 chirps_2018.mat
            -a----          2/3/2025  11:19 am      107643072 chirps_2019.mat
            -a----          2/3/2025  11:20 am      107937984 chirps_2020.mat
        CHMdata/
            -a----          4/3/2025   8:44 pm      107937984 CHM_2016.mat
            -a----          4/3/2025   8:54 pm      538804416 CHM_2016_2020.mat
            -a----          4/3/2025   8:47 pm      107643072 CHM_2017.mat
            -a----          4/3/2025   8:49 pm      107643072 CHM_2018.mat
            -a----          4/3/2025   8:51 pm      107643072 CHM_2019.mat
            -a----          4/3/2025   8:53 pm      107937984 CHM_2020.mat
        IMERGdata/
            -a----          4/3/2025   9:04 pm      107937984 IMERG_2016.mat
            -a----          4/3/2025   9:16 pm      538804416 IMERG_2016_2020.mat
            -a----          4/3/2025   9:07 pm      107643072 IMERG_2017.mat
            -a----          4/3/2025   9:10 pm      107643072 IMERG_2018.mat
            -a----          4/3/2025   9:13 pm      107643072 IMERG_2019.mat
            -a----          4/3/2025   9:16 pm      107937984 IMERG_2020.mat
        sm2raindata/
            -a----          2/3/2025  10:44 am      107937984 sm2rain_2016.mat
            -a----          5/3/2025   5:18 pm      538804416 sm2rain_2016_2020.mat
            -a----          5/3/2025   5:12 pm      538804416 sm2rain_2016_2020processed.mat
            -a----          2/3/2025  10:46 am      107643072 sm2rain_2017.mat
            -a----          2/3/2025  10:49 am      107643072 sm2rain_2018.mat
            -a----          2/3/2025  10:52 am      107643072 sm2rain_2019.mat
            -a----          2/3/2025  10:54 am      107937984 sm2rain_2020.mat
        PERSIANNdata/
            -a----          1/3/2025  10:14 pm       53969088 PERSIANN_2016.mat
            -a----          1/3/2025  10:15 pm      269402304 PERSIANN_2016_2020.mat
            -a----          1/3/2025  10:14 pm       53821632 PERSIANN_2017.mat
            -a----          1/3/2025  10:14 pm       53821632 PERSIANN_2018.mat
            -a----          1/3/2025  10:15 pm       53821632 PERSIANN_2019.mat
            -a----          1/3/2025  10:15 pm       53969088 PERSIANN_2020.mat
        GSMAPdata/
            -a----          2/3/2025   8:47 pm      107937984 GSMAP_2016.mat
            -a----          2/3/2025   8:57 pm      538804416 GSMAP_2016_2020.mat
            -a----          2/3/2025   8:50 pm      107643072 GSMAP_2017.mat
            -a----          2/3/2025   8:52 pm      107643072 GSMAP_2018.mat
            -a----          2/3/2025   8:55 pm      107643072 GSMAP_2019.mat
            -a----          2/3/2025   8:57 pm      107937984 GSMAP_2020.mat
        ```
    *   `processed/`: 初步数据处理尝试生成的文件，基本已经弃用。
    *   `mask/` : 储存从全球数据中选取出中国区域以及从中国区域中选出长江流域的掩码矩阵，为`.mat`文件，数据键值为`"data"`，大小为144*256大小矩阵。目前矩阵中有三类值(0, 1, 2)：大于等于1的值掩码出中国版图，大于等于2的值掩码出长江流域。
*   **`docs/`**: 存放项目相关的各类文档。
    *   `functions_reference.md`: 项目中重要函数的参考说明。
    *   `降雨预测模型优化方法_.pdf`: 关于模型优化的详细 PDF 文档。
    *   (可能包含其他设计文档、研究报告等)
*   **`logs/`**: 存放项目运行过程中产生的日志文件。
    *   `feature_names_v5_1.txt`, `feature_names_v5.txt`, `feature_names_v4.txt`, `feature_names_v3.txt`, `feature_names_v2.txt`, `feature_names.txt`: 不同版本特征工程所使用的特征列表。
    *   `training_log.txt`: 主要模型训练过程的详细日志。
    *   `continued_training_log.txt`: 继续训练过程的日志 (当前为空)。
    *   `fine_tuned_train_log.txt`: 微调训练过程的日志 (当前为空)。
    *   `fine_tuning_log.txt`: 微调过程的日志 (当前为空)。
    *   `distributed_training_log.txt`: 分布式训练过程的日志 (当前为空)。
*   **`models/`**: 存放训练完成的机器学习模型文件的主目录（**注意：`results/`目录下也可能有`models/`子目录，需明确区分用途或版本**）。
    *   `xgboost_model_YYYYMMDD_HHMMSS.json` 等: XGBoost 模型早期可能保存为JSON格式。
    *   `.joblib`, `.pkl`: 后期更常用的模型持久化格式，例如 `xgboost_optimized_yangtsu_model_from_db.joblib`。
    *   (其他如PyTorch的 `.pth` 文件，根据实际使用的模型框架添加说明)
*   **`notebooks/`**: 存放用于探索性数据分析、原型验证、可视化等的 Jupyter Notebooks 或相关学习材料。
    *   `book.md`, `book.pdf`: 可能为项目方法论或背景知识的文档。
    *   `ml_tutorial.md`, `ml_tutorial.pdf`: 机器学习相关的教程材料。
    *   `wordpress.md`, `wordpress.pdf`: (用途待确认，如与项目博客或文档发布平台相关可说明)
*   **`results/`**: 存放模型训练、评估和分析产生的结果文件的主要聚合目录。
    *   `features/`: 存储不同版本迭代生成的特征集文件（如 `X_Yangtsu_flat_features_v6.npy`, `feature_names_yangtsu_v6.txt`）以及K-Fold交叉验证的产出（如 `kfold_optimization_v6/` 子目录下的折外预测和每折模型）。
    *   `models/`: (可能与顶层 `models/` 分工) 存储特定实验或区域（如长江流域）训练好的模型文件。
    *   `plots/`: 存储评估过程中生成的各类图表，如特征重要性图、误差分布图、训练历史曲线等。
    *   `predictions/`: 存储模型的原始预测概率或类别输出。
    *   `statistical_analysis/`: **(新增)** 存放由 `src/yangtze/YangTsu/value_characteristics_fp_fn_.py` 等脚本生成的FP/FN相关的统计特征分析结果和图表。
    *   `model_predict/`: **(新增，主要用于长江流域)** 存放最终优化模型的预测结果、日志 (`xgboost6_opt_log_main.txt`) 和Optuna相关的可视化图表。
*   **`scripts/`**: 存放一些辅助性的脚本。
    *   `utils/`: 通用工具脚本 (大部分已废弃)。
        *   `scale_utils.py`: 关键工具，用于处理原始多源降雨数据空间分辨率不一致的问题，将它们统一插值/重采样到0.25°分辨率，是 `data/intermediate/` 数据生成的重要环节。
    *   `mask_creation/`: (推测) 创建地理掩码的脚本。
    *   `ReadGsmapG01d25y0820.m`: 读取特定格式 GSMAP 数据的 MATLAB 脚本（可能是早期数据处理遗留）。
*   **`src/`**: 核心源代码目录。
    *   `common/`: 通用模块。
        *   `feature_engineering/`: (预期存放) 通用的特征工程函数和类 (目前为空)。
    *   `ensemble_learning/`: 全国范围的早期集成学习探索代码。
        *   `ensemble_learning/`: 具体的集成学习流程脚本 (如 `1_generate_base_predictions.py` 至 `5_train_evaluate_meta_learner.py`)，目前可能已暂停维护，重心转向长江流域的集成。
    *   `legacy/`: 存放旧的、废弃的或实验性和学习性质的代码，包含大量早期的模型训练尝试和工具脚本。
    *   `nationwide/`: 全国范围的降雨预测研究代码。
        *   `project_all_for_deepling_learning/`: 核心处理流程和实验脚本。
            *   `loaddata.py`: 加载全国范围的预处理数据。
            *   `turn1.py` 至 `turn5_1.py`: 全国范围特征工程 V1 到 V5.1 的迭代脚本。
            *   `xgboost1.py` 至 `xgboost5_1.py`, `lightGBM1.py`, `naive_bayes1.py`, `bayesian_network_fp_expert.py` 等: 基于不同特征集和模型的训练脚本。
            *   包含大量由 `homeworkplot.py` 等脚本生成的针对全国范围原始产品性能分析的 `.png` 图表。
        *   `analysis/`: 针对全国范围研究的分析脚本 (如特征集对比、相关性分析、PCA分析)。
    *   `readdata/`: 各类降雨原始数据的读取、初步解析和预处理脚本。
        *   `CHM.py`, `sm2rain.py`, `PERSIANN.py`, `gsmap.py`, `CMORPH.py`, `IMERG.py`, `chirps.py`: 分别对应7种降雨产品，从原始格式（如 `.nc`, `.zip`）读取数据，进行初步处理（如提取变量、时间筛选）并保存为 `data/intermediate/` 下的 `.mat` 格式的脚本。
        *   `datafix.py`: （可能）用于处理空间对齐后数据中的NaN值或进行产品间修正的脚本。
        *   `scale_utils.py`: (此处的 `scale_utils.py` 可能与 `scripts/utils/` 下的功能重复或为特定版本，需确认。通常负责空间分辨率对齐。)
        *   `academic_plotter.py`, `plot.py`, `overword.py`: 新增的绘图或文本处理相关工具。
    *   `temp/`: 临时测试或开发中的脚本。
        *   `temp.py`: 通用临时脚本。
    *   `visualization/`: **(新增)** 存放用于生成项目总结性和对比性高级图表的脚本。
        *   `evolution.py`: 绘制模型/特征迭代性能演进图、多模型雷达图和热力图。
        *   `optimization_history.py`: 解析Optuna日志并可视化超参数优化过程。
        *   `plot_3d_performance.py`: 创建模型在POD, 1-FAR, CSI三个维度上的3D性能可视化图。
        *   `thresholds_plot.py`: 绘制不同概率阈值下各项性能指标的棒棒糖图。
    *   `yangtze/`: 长江流域的降雨预测研究代码，是当前项目的核心工作区域。
        *   `YangTsu/` (目录名可能为 `Yangtze`): 长江流域核心处理流程。
            *   `loaddata.py`: 加载长江流域的预处理数据，调用顶层 `loaddata.py` 或有特定实现。
            *   `turn1.py` 至 `turn6.py`: 针对长江流域的特征工程 V1 (格点数据) 到 V6 (点位数据，当前优化基础) 的迭代脚本。
            *   `xgboost1.py` 至 `xgboost6.py`: 对应各版本特征集的XGBoost模型训练脚本。
            *   `xgboost6_opt.py`: (可能) 早期用于V6特征集Optuna优化的脚本，其功能已整合或被 `xgboost_optimization_main.py` 取代。
            *   `xgboost_optimization_main.py`: **核心脚本**，负责加载长江V6特征、执行Optuna超参数寻优、训练最终优化模型、并在Hold-out测试集上评估。
            *   `xgboost_best.py`: 用于测试特定（如V1）特征集在当前获取的Optuna最优参数下的性能。
            *   `xgboostv6_for_Ensemble.py`: **核心脚本**，基于V6优化参数进行K-Fold交叉验证，生成集成学习Level 0层的OOF预测和测试集预测。
            *   `loadoptunadb.py`: 用于将外部Optuna试验日志导入SQLite数据库 (`my_optimization_history.db`)。
            *   `model_compare_plot.py`: 比较多种机器学习模型在早期特征集上的性能，使用 `AcademicStylePlotter`。
            *   `feature_of_FP_FN_Yangtsu.py`, `feature_of_FP_FN_Yangtsu_Mean.py`: 分析导致FP/FN的输入特征。
            *   `Spatial_characteristics_plot_fp_fn.py`: **新增**，分析原始降雨产品FP/FN指标的空间分布特征。
            *   `Statistical_spatial_characteristics_fp_fn_.py`: **新增**，对FP/FN的空间统计特征进行分析。
            *   `value_characteristics_fp_fn_.py`: **新增**，分析FP/FN事件发生时原始观测值的统计特征。
            *   `product_pod_far.py`: 计算长江流域各降雨产品POD/FAR等基础指标。
            *   `Ensemble1/`: **新增子目录**，长江流域Level 1专家模型相关代码。
                *   `indentify_fp_fn_v1.py`: 识别L0模型的FP/FN，并为专家模型生成目标标签。
                *   `fp_v1.py`: 训练FP专家模型并生成OOF预测。
                *   `fn_v1.py`: 训练FN专家模型并生成OOF预测。
                *   `utils.py`: (可能包含) Ensemble1 模块的通用工具。
            *   `start.py`, `startxgb1-5.py`: 实验流程的启动脚本。
        *   `analysis/`: (预期存放) 针对长江流域的特定分析脚本 (目前为空或内容较少)。
*   **`tests/`**: 存放单元测试、集成测试等相关代码 (当前为空或较少)。
*   **`.gitignore`**: Git 配置文件，指定哪些文件或目录不纳入版本控制。
*   **`README.md`**: 项目的入口文档，提供项目概述、技术架构、研究历程、未来展望和本目录结构说明。
*   **`my_optimization_history.db`**: **(新增)** Optuna 优化过程的 SQLite 数据库文件。
*   **`optuna_optimization_history.png`, `performance_evolution.png`, `model_metrics_heatmap.png`, `flashy_vertical_lollipop_*.png`**: **(新增)** 由 `src/visualization/` 下脚本生成的各类可视化图表，用于展示优化过程和模型性能。
*   **`populate_optuna_db_log.txt`**: **(新增)** 记录了 `loadoptunadb.py` 导入历史 Optuna 试验到数据库的日志。
## 6. 长江流域多源降雨产品性能评估 (基于不同分类阈值)

以下表格展示了长江流域几种主要降雨产品 (CMORPH, CHIRPS, GSMAP, IMERG, PERSIANN, SM2RAIN) 在不同降雨量分类阈值下的关键性能指标：命中率 (POD - Probability of Detection)、空报率 (FAR - False Alarm Ratio) 和临界成功指数 (CSI - Critical Success Index)。这些指标基于 TP (命中数), FP (空报数), FN (漏报数), TN (正确负例数) 计算得出。

#### 表1: 各产品在不同阈值下的 POD (命中率)

| 阈值   | CMORPH | CHIRPS | GSMAP  | IMERG  | PERSIANN | SM2RAIN |
|------|--------|--------|--------|--------|----------|---------|
| 0.00 | 0.4919 | 0.4113 | 0.5557 | 0.7515 | 0.6004   | 0.9153  |
| 0.05 | 0.5243 | 0.4280 | 0.5887 | 0.7227 | 0.6235   | 0.9074  |
| 0.10 | 0.5460 | 0.4383 | 0.6085 | 0.7053 | 0.6368   | 0.9030  |
| 0.15 | 0.5281 | 0.4441 | 0.6216 | 0.6933 | 0.6449   | 0.8973  |
| 0.20 | 0.5401 | 0.4470 | 0.6317 | 0.6841 | 0.6507   | 0.8946  |
| 0.25 | 0.5285 | 0.4487 | 0.6403 | 0.6767 | 0.6552   | 0.8914  |
| 0.30 | 0.5372 | 0.4497 | 0.6471 | 0.6700 | 0.6584   | 0.8894  |
| 0.35 | 0.5256 | 0.4509 | 0.6533 | 0.6645 | 0.6606   | 0.8870  |
| 0.40 | 0.5327 | 0.4517 | 0.6585 | 0.6593 | 0.6617   | 0.8852  |
| 0.45 | 0.5231 | 0.4522 | 0.6630 | 0.6548 | 0.6619   | 0.8831  |
| 0.50 | 0.5148 | 0.4525 | 0.6669 | 0.6506 | 0.6612   | 0.8809  |
| 0.55 | 0.5204 | 0.4528 | 0.6704 | 0.6466 | 0.6598   | 0.8793  |
| 0.60 | 0.5258 | 0.4530 | 0.6735 | 0.6429 | 0.6580   | 0.8776  |
| 0.65 | 0.5166 | 0.4532 | 0.6764 | 0.6394 | 0.6559   | 0.8758  |
| 0.70 | 0.5109 | 0.4534 | 0.6790 | 0.6361 | 0.6538   | 0.8739  |
| 0.75 | 0.5156 | 0.4536 | 0.6813 | 0.6329 | 0.6515   | 0.8724  |
| 0.80 | 0.5199 | 0.4537 | 0.6833 | 0.6300 | 0.6491   | 0.8710  |
| 0.85 | 0.5122 | 0.4538 | 0.6853 | 0.6273 | 0.6466   | 0.8691  |
| 0.90 | 0.5162 | 0.4538 | 0.6870 | 0.6245 | 0.6441   | 0.8674  |
| 0.95 | 0.5109 | 0.4539 | 0.6885 | 0.6218 | 0.6416   | 0.8659  |

#### 表2: 各产品在不同阈值下的 FAR (空报率)

| 阈值   | CMORPH | CHIRPS | GSMAP  | IMERG  | PERSIANN | SM2RAIN |
|------|--------|--------|--------|--------|----------|---------|
| 0.00 | 0.1173 | 0.1246 | 0.0250 | 0.1425 | 0.1602   | 0.2076  |
| 0.05 | 0.1607 | 0.1672 | 0.0454 | 0.1766 | 0.2145   | 0.2716  |
| 0.10 | 0.1905 | 0.1935 | 0.0613 | 0.1947 | 0.2467   | 0.3088  |
| 0.15 | 0.1977 | 0.2105 | 0.0734 | 0.2061 | 0.2676   | 0.3316  |
| 0.20 | 0.2144 | 0.2240 | 0.0837 | 0.2145 | 0.2831   | 0.3494  |
| 0.25 | 0.2187 | 0.2352 | 0.0931 | 0.2215 | 0.2960   | 0.3631  |
| 0.30 | 0.2311 | 0.2452 | 0.1014 | 0.2273 | 0.3067   | 0.3747  |
| 0.35 | 0.2332 | 0.2543 | 0.1091 | 0.2326 | 0.3165   | 0.3845  |
| 0.40 | 0.2435 | 0.2625 | 0.1162 | 0.2374 | 0.3249   | 0.3932  |
| 0.45 | 0.2449 | 0.2703 | 0.1231 | 0.2419 | 0.3327   | 0.4006  |
| 0.50 | 0.2464 | 0.2774 | 0.1296 | 0.2462 | 0.3397   | 0.4071  |
| 0.55 | 0.2546 | 0.2841 | 0.1357 | 0.2501 | 0.3460   | 0.4132  |
| 0.60 | 0.2625 | 0.2905 | 0.1416 | 0.2540 | 0.3519   | 0.4188  |
| 0.65 | 0.2628 | 0.2967 | 0.1472 | 0.2576 | 0.3574   | 0.4235  |
| 0.70 | 0.2646 | 0.3028 | 0.1528 | 0.2613 | 0.3627   | 0.4280  |
| 0.75 | 0.2714 | 0.3083 | 0.1580 | 0.2648 | 0.3675   | 0.4322  |
| 0.80 | 0.2779 | 0.3136 | 0.1630 | 0.2681 | 0.3719   | 0.4361  |
| 0.85 | 0.2778 | 0.3189 | 0.1678 | 0.2712 | 0.3763   | 0.4394  |
| 0.90 | 0.2842 | 0.3241 | 0.1726 | 0.2746 | 0.3805   | 0.4426  |
| 0.95 | 0.2856 | 0.3290 | 0.1773 | 0.2777 | 0.3845   | 0.4457  |

#### 表3: 各产品在不同阈值下的 CSI (临界成功指数)

| 阈值   | CMORPH | CHIRPS | GSMAP  | IMERG  | PERSIANN | SM2RAIN |
|------|--------|--------|--------|--------|----------|---------|
| 0.00 | 0.4617 | 0.3885 | 0.5479 | 0.6681 | 0.5387   | 0.7382  |
| 0.05 | 0.4765 | 0.3941 | 0.5727 | 0.6257 | 0.5328   | 0.6780  |
| 0.10 | 0.4838 | 0.3966 | 0.5852 | 0.6025 | 0.5269   | 0.6434  |
| 0.15 | 0.4673 | 0.3971 | 0.5924 | 0.5875 | 0.5219   | 0.6208  |
| 0.20 | 0.4707 | 0.3960 | 0.5973 | 0.5764 | 0.5176   | 0.6043  |
| 0.25 | 0.4604 | 0.3943 | 0.6008 | 0.5674 | 0.5137   | 0.5910  |
| 0.30 | 0.4625 | 0.3924 | 0.6031 | 0.5597 | 0.5099   | 0.5802  |
| 0.35 | 0.4532 | 0.3908 | 0.6049 | 0.5531 | 0.5058   | 0.5708  |
| 0.40 | 0.4548 | 0.3892 | 0.6061 | 0.5470 | 0.5019   | 0.5625  |
| 0.45 | 0.4472 | 0.3873 | 0.6065 | 0.5416 | 0.4976   | 0.5553  |
| 0.50 | 0.4406 | 0.3856 | 0.6067 | 0.5366 | 0.4934   | 0.5489  |
| 0.55 | 0.4419 | 0.3839 | 0.6065 | 0.5319 | 0.4891   | 0.5431  |
| 0.60 | 0.4429 | 0.3821 | 0.6062 | 0.5274 | 0.4848   | 0.5376  |
| 0.65 | 0.4363 | 0.3804 | 0.6057 | 0.5233 | 0.4806   | 0.5329  |
| 0.70 | 0.4316 | 0.3788 | 0.6049 | 0.5192 | 0.4765   | 0.5284  |
| 0.75 | 0.4325 | 0.3773 | 0.6040 | 0.5154 | 0.4726   | 0.5242  |
| 0.80 | 0.4332 | 0.3758 | 0.6031 | 0.5119 | 0.4689   | 0.5204  |
| 0.85 | 0.4279 | 0.3743 | 0.6021 | 0.5086 | 0.4652   | 0.5170  |
| 0.90 | 0.4284 | 0.3727 | 0.6009 | 0.5051 | 0.4615   | 0.5137  |
| 0.95 | 0.4243 | 0.3713 | 0.5995 | 0.5019 | 0.4580   | 0.5105  |

## 7. 长江流域各降雨产品基本统计特征

以下数据基于长江流域范围（掩码值为2的区域），时间跨度为2016-2020年。

| 指标                                   | CMORPH                 | CHIRPS                 | SM2RAIN                | IMERG                  | GSMAP                  | PERSIANN               |
|----------------------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| 平均值                                 | 3.3629                 | 3.5047                 | 3.9431                 | 3.5663                 | 3.2589                 | 3.5872                 |
| 标准差                                 | 9.9921                 | 9.8690                 | 5.4609                 | 9.7228                 | 7.9307                 | 8.1056                 |
| 最大值                                 | 379.8000               | 330.5150               | 228.4960               | 357.1993               | 280.7053               | 230.6858               |
| 最小值                                 | 0.0000                 | 0.0000                 | 0.0000                 | 0.0000                 | 0.0000                 | 0.0000                 |
| 中位数                                 | 0.0000                 | 0.0000                 | 1.9240                 | 0.1765                 | 0.0000                 | 0.2830                 |
| 众数 (count)                           | 0.0000 (3103372)       | 0.0000 (3453450)       | 0.0000 (709395)        | 0.0000 (1825289)       | 0.0000 (3052396)       | 0.0000 (2472006)       |
| 方差                                   | 99.8425                | 97.3975                | 29.8218                | 94.5334                | 62.8952                | 65.7014                |
| 偏度 (Skewness)                        | 6.1082                 | 5.5473                 | 3.0639                 | 5.9726                 | 4.9681                 | 4.7259                 |
| 峰度 (Kurtosis)                        | 57.8855                | 50.6663                | 21.6656                | 55.5745                | 40.8976                | 34.8649                |
| 和地面观测站(CHM)之间的相关系数        | 0.4456                 | 0.3900                 | 0.5695                 | 0.5014                 | 0.6454                 | 0.4133                 |
| 和 CMORPH 之间的相关系数               | 1.0000 (self-correlation) | 0.6147                 | 0.5000                 | 0.8259                 | 0.6593                 | 0.6752                 |
| 和 CHIRPS 之间的相关系数               | 0.6147                 | 1.0000 (self-correlation) | 0.4278                 | 0.6541                 | 0.5802                 | 0.7533                 |
| 和 SM2RAIN 之间的相关系数              | 0.5000                 | 0.4278                 | 1.0000 (self-correlation) | 0.5359                 | 0.6233                 | 0.4636                 |
| 和 IMERG 之间的相关系数                | 0.8259                 | 0.6541                 | 0.5359                 | 1.0000 (self-correlation) | 0.7261                 | 0.7134                 |
| 和 GSMAP 之间的相关系数                | 0.6593                 | 0.5802                 | 0.6233                 | 0.7261                 | 1.0000 (self-correlation) | 0.5920                 |
| 和 PERSIANN 之间的相关系数             | 0.6752                 | 0.7533                 | 0.4636                 | 0.7134                 | 0.5920                 | 1.0000 (self-correlation) |

// ... existing code ...
| 和 PERSIANN 之间的相关系数         | 0.6551                 | 0.7431                 | 0.5092                 | 0.7137                 | 0.5990                 | 1.0000 (self-correlation) |

## 8. 全国范围各降雨产品性能评估 (基于不同分类阈值)

以下表格展示了全国范围几种主要降雨产品 (CMORPH, CHIRPS, GSMAP, IMERG, PERSIANN, SM2RAIN) 在不同降雨量分类阈值下的关键性能指标：命中率 (POD - Probability of Detection)、空报率 (FAR - False Alarm Ratio) 和临界成功指数 (CSI - Critical Success Index)。这些指标基于 TP (命中数), FP (空报数), FN (漏报数), TN (正确负例数) 计算得出，并以 CHM 产品作为地面真值。

#### 表4: 全国范围各产品在不同阈值下的 POD (命中率)

| 阈值   | CMORPH | CHIRPS | GSMAP  | IMERG  | PERSIANN | SM2RAIN |
|------|--------|--------|--------|--------|----------|---------|
| 0.00 | 0.4474 | 0.3849 | 0.4636 | 0.7002 | 0.6594   | 0.7931  |
| 0.05 | 0.4871 | 0.3995 | 0.5097 | 0.6694 | 0.6740   | 0.7845  |
| 0.10 | 0.5124 | 0.4079 | 0.5366 | 0.6532 | 0.6787   | 0.7780  |
| 0.15 | 0.4917 | 0.4125 | 0.5539 | 0.6424 | 0.6798   | 0.7693  |
| 0.20 | 0.5052 | 0.4148 | 0.5670 | 0.6344 | 0.6793   | 0.7665  |
| 0.25 | 0.4925 | 0.4162 | 0.5776 | 0.6278 | 0.6781   | 0.7630  |
| 0.30 | 0.5021 | 0.4170 | 0.5861 | 0.6221 | 0.6762   | 0.7616  |
| 0.35 | 0.4896 | 0.4179 | 0.5936 | 0.6173 | 0.6738   | 0.7595  |
| 0.40 | 0.4973 | 0.4185 | 0.5999 | 0.6128 | 0.6708   | 0.7585  |
| 0.45 | 0.4869 | 0.4189 | 0.6055 | 0.6089 | 0.6674   | 0.7569  |
| 0.50 | 0.4780 | 0.4191 | 0.6103 | 0.6052 | 0.6636   | 0.7554  |
| 0.55 | 0.4840 | 0.4193 | 0.6146 | 0.6017 | 0.6595   | 0.7547  |
| 0.60 | 0.4897 | 0.4195 | 0.6185 | 0.5985 | 0.6553   | 0.7541  |
| 0.65 | 0.4803 | 0.4197 | 0.6221 | 0.5955 | 0.6512   | 0.7529  |
| 0.70 | 0.4743 | 0.4199 | 0.6254 | 0.5927 | 0.6472   | 0.7518  |
| 0.75 | 0.4792 | 0.4201 | 0.6283 | 0.5900 | 0.6432   | 0.7513  |
| 0.80 | 0.4838 | 0.4201 | 0.6309 | 0.5875 | 0.6393   | 0.7507  |
| 0.85 | 0.4757 | 0.4202 | 0.6334 | 0.5851 | 0.6354   | 0.7496  |
| 0.90 | 0.4800 | 0.4203 | 0.6357 | 0.5828 | 0.6316   | 0.7488  |
| 0.95 | 0.4746 | 0.4203 | 0.6377 | 0.5805 | 0.6279   | 0.7481  |

#### 表5: 全国范围各产品在不同阈值下的 FAR (空报率)

| 阈值   | CMORPH | CHIRPS | GSMAP  | IMERG  | PERSIANN | SM2RAIN |
|------|--------|--------|--------|--------|----------|---------|
| 0.00 | 0.2750 | 0.3085 | 0.0615 | 0.2850 | 0.3816   | 0.3519  |
| 0.05 | 0.3360 | 0.3551 | 0.0885 | 0.3116 | 0.4382   | 0.4127  |
| 0.10 | 0.3742 | 0.3766 | 0.1080 | 0.3244 | 0.4601   | 0.4387  |
| 0.15 | 0.3747 | 0.3881 | 0.1222 | 0.3318 | 0.4706   | 0.4456  |
| 0.20 | 0.3943 | 0.3962 | 0.1340 | 0.3371 | 0.4769   | 0.4537  |
| 0.25 | 0.3943 | 0.4026 | 0.1443 | 0.3418 | 0.4812   | 0.4577  |
| 0.30 | 0.4082 | 0.4080 | 0.1533 | 0.3456 | 0.4842   | 0.4619  |
| 0.35 | 0.4059 | 0.4127 | 0.1615 | 0.3491 | 0.4867   | 0.4643  |
| 0.40 | 0.4170 | 0.4168 | 0.1689 | 0.3522 | 0.4887   | 0.4672  |
| 0.45 | 0.4145 | 0.4208 | 0.1758 | 0.3553 | 0.4905   | 0.4689  |
| 0.50 | 0.4127 | 0.4245 | 0.1823 | 0.3581 | 0.4921   | 0.4703  |
| 0.55 | 0.4214 | 0.4278 | 0.1882 | 0.3607 | 0.4935   | 0.4723  |
| 0.60 | 0.4297 | 0.4312 | 0.1940 | 0.3634 | 0.4949   | 0.4741  |
| 0.65 | 0.4265 | 0.4343 | 0.1995 | 0.3661 | 0.4962   | 0.4751  |
| 0.70 | 0.4258 | 0.4375 | 0.2047 | 0.3686 | 0.4976   | 0.4762  |
| 0.75 | 0.4329 | 0.4405 | 0.2098 | 0.3711 | 0.4989   | 0.4777  |
| 0.80 | 0.4396 | 0.4433 | 0.2145 | 0.3734 | 0.5000   | 0.4790  |
| 0.85 | 0.4366 | 0.4462 | 0.2191 | 0.3757 | 0.5011   | 0.4798  |
| 0.90 | 0.4429 | 0.4490 | 0.2236 | 0.3779 | 0.5024   | 0.4807  |
| 0.95 | 0.4421 | 0.4517 | 0.2279 | 0.3802 | 0.5035   | 0.4817  |

#### 表6: 全国范围各产品在不同阈值下的 CSI (临界成功指数)

| 阈值   | CMORPH | CHIRPS | GSMAP  | IMERG  | PERSIANN | SM2RAIN |
|------|--------|--------|--------|--------|----------|---------|
| 0.00 | 0.3825 | 0.3285 | 0.4499 | 0.5474 | 0.4687   | 0.5544  |
| 0.05 | 0.3908 | 0.3275 | 0.4856 | 0.5138 | 0.4417   | 0.5057  |
| 0.10 | 0.3922 | 0.3273 | 0.5039 | 0.4973 | 0.4300   | 0.4838  |
| 0.15 | 0.3798 | 0.3270 | 0.5142 | 0.4871 | 0.4237   | 0.4753  |
| 0.20 | 0.3802 | 0.3261 | 0.5213 | 0.4796 | 0.4195   | 0.4684  |
| 0.25 | 0.3729 | 0.3250 | 0.5264 | 0.4735 | 0.4163   | 0.4641  |
| 0.30 | 0.3730 | 0.3239 | 0.5299 | 0.4683 | 0.4136   | 0.4605  |
| 0.35 | 0.3669 | 0.3231 | 0.5327 | 0.4638 | 0.4111   | 0.4580  |
| 0.40 | 0.3668 | 0.3221 | 0.5348 | 0.4596 | 0.4088   | 0.4555  |
| 0.45 | 0.3621 | 0.3211 | 0.5362 | 0.4559 | 0.4063   | 0.4537  |
| 0.50 | 0.3578 | 0.3201 | 0.5372 | 0.4524 | 0.4039   | 0.4521  |
| 0.55 | 0.3579 | 0.3192 | 0.5379 | 0.4492 | 0.4015   | 0.4505  |
| 0.60 | 0.3577 | 0.3183 | 0.5384 | 0.4461 | 0.3991   | 0.4489  |
| 0.65 | 0.3539 | 0.3174 | 0.5386 | 0.4431 | 0.3967   | 0.4478  |
| 0.70 | 0.3509 | 0.3165 | 0.5387 | 0.4403 | 0.3944   | 0.4466  |
| 0.75 | 0.3508 | 0.3157 | 0.5385 | 0.4377 | 0.3921   | 0.4453  |
| 0.80 | 0.3507 | 0.3148 | 0.5382 | 0.4352 | 0.3900   | 0.4442  |
| 0.85 | 0.3476 | 0.3140 | 0.5378 | 0.4327 | 0.3878   | 0.4432  |
| 0.90 | 0.3474 | 0.3130 | 0.5373 | 0.4304 | 0.3857   | 0.4423  |
| 0.95 | 0.3449 | 0.3122 | 0.5367 | 0.4281 | 0.3836   | 0.4412  |

| 和 PERSIANN 之间的相关系数             | 0.6752                 | 0.7533                 | 0.4636                 | 0.7134                 | 0.5920                 | 1.0000 (self-correlation) |

## 9. 全国范围各降雨产品基本统计特征

以下数据基于全国范围（掩码值为1的区域），时间跨度为2016-2020年 (1827天)。目标变量 Y 为 CHM 产品。

| 指标                               | CMORPH                 | CHIRPS                 | SM2RAIN                | IMERG                  | GSMAP                  | PERSIANN               |
|------------------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|------------------------|
| 平均值                             | 1.8652                 | 1.8549                 | 1.9931                 | 1.8931                 | 1.6654                 | 1.9591                 |
| 标准差                             | 6.9623                 | 6.9233                 | 4.2815                 | 6.7717                 | 5.7024                 | 5.5107                 |
| 最大值                             | 379.8000               | 330.5150               | 301.2768               | 408.9030               | 297.9254               | 281.0631               |
| 最小值                             | 0.0000                 | 0.0000                 | 0.0000                 | 0.0000                 | 0.0000                 | 0.0000                 |
| 中位数                             | 0.0000                 | 0.0000                 | 0.2000                 | 0.0194                 | 0.0000                 | 0.1256                 |
| 众数 (count)                       | 0.0000 (18647303)      | 0.0000 (19567263)      | 0.0000 (9411016)       | 0.0000 (13131758)      | 0.0000 (20520991)      | 0.0000 (11810182)      |
| 方差                               | 48.4740                | 47.9320                | 18.3316                | 45.8562                | 32.5170                | 30.3680                |
| 偏度 (Skewness)                    | 8.2594                 | 7.7101                 | 4.6727                 | 8.5174                 | 7.4365                 | 6.6837                 |
| 峰度 (Kurtosis)                    | 108.5944               | 96.0344                | 46.5308                | 119.3158               | 96.1135                | 72.6359                |
| 和 Y (CHM) 之间的相关系数         | 0.4407                 | 0.4124                 | 0.5743                 | 0.5204                 | 0.6591                 | 0.4416                 |
| 和 CMORPH 之间的相关系数           | 1.0000 (self-correlation) | 0.6000                 | 0.5021                 | 0.7943                 | 0.6313                 | 0.6551                 |
| 和 CHIRPS 之间的相关系数           | 0.6000                 | 1.0000 (self-correlation) | 0.4650                 | 0.6605                 | 0.5851                 | 0.7431                 |
| 和 SM2RAIN 之间的相关系数          | 0.5021                 | 0.4650                 | 1.0000 (self-correlation) | 0.5690                 | 0.6298                 | 0.5092                 |
| 和 IMERG 之间的相关系数            | 0.7943                 | 0.6605                 | 0.5690                 | 1.0000 (self-correlation) | 0.7326                 | 0.7137                 |
| 和 GSMAP 之间的相关系数            | 0.6313                 | 0.5851                 | 0.6298                 | 0.7326                 | 1.0000 (self-correlation) | 0.5990                 |
| 和 PERSIANN 之间的相关系数         | 0.6551                 | 0.7431                 | 0.5092                 | 0.7137                 | 0.5990                 | 1.0000 (self-correlation) |

## 10. 各降雨产品逐年性能指标的空间统计特征详析

**10.2 各降雨产品逐年性能指标的空间统计特征详析**

**10.2.1 CMORPH 产品性能分析**

CMORPH 产品作为一种常用的卫星降雨产品，其在 2016 年至 2020 年间的性能表现出一定的时空变异性。

*   **10.2.1.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** 在大多数年份（如 2016, 2017, 2018, 2019），CMORPH 的平均空间 POD 在非常小的阈值（0.0mm/d 至 0.1mm/d 或 0.2mm/d）下通常会略有上升，表明在探测微小降雨事件时，略微提高阈值有助于排除一些噪声，反而提升名义上的命中率。然而，随着阈值的进一步增高（超过 0.2mm/d 或 0.3mm/d），平均空间 POD 整体呈现下降趋势或在高位波动后缓慢下降。这符合预期，因为更高强度的降雨事件本身发生频率较低，探测难度也可能增加。
        *   **2016年：** 从 0.0mm/d 的 0.5134 上升到 0.1mm/d 的 0.5696，之后在 0.51-0.56 之间波动，整体略有下降趋势。
        *   **2017年：** 从 0.0mm/d 的 0.4975 上升到 0.1mm/d 的 0.5493，之后在 0.50-0.54 之间波动，整体趋势不明显，但高阈值下略有下降。
        *   **2018年：** 从 0.0mm/d 的 0.5052 上升到 0.1mm/d 的 0.5551，之后在 0.50-0.54 之间波动，高阈值下有下降。
        *   **2019年：** 从 0.0mm/d 的 0.4797 上升到 0.1mm/d 的 0.5340，之后在 0.49-0.53 之间波动，高阈值下略有下降。
        *   **2020年：** 从 0.0mm/d 的 0.4787 上升到 0.1mm/d 的 0.5384，之后在 0.52-0.54 之间波动，整体趋势相对平稳，但在极高阈值下略有下降。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** CMORPH 的 POD 空间标准差在各年份通常维持在 0.07 至 0.10 之间，表明其命中率在全国范围内存在一定的空间差异。
        *   **随阈值变化：** 标准差随阈值的变化没有非常一致的模式，有时略增，有时略减，但整体波动不大。例如，2016年标准差在0.078-0.088之间；2017年在0.091-0.103之间，显示出更大的空间不一致性。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** 在大多数年份和阈值下，POD 的空间偏度为正（如 2016 年在 0.0 到 0.65 之间波动，2018 年在 0.1 到 0.78 之间），表明 POD 空间分布略微右偏，即存在一些 POD 表现显著高于平均值的区域。随着阈值增加，偏度值有时会减小，甚至变为负值（如 2016 年高阈值下 POD 偏度接近 -0.5），说明高强度降雨的命中率空间差异模式可能发生改变。
        *   **峰度：** POD 的空间峰度值多数情况下接近 0 或为正（如 2016 年在 -0.02 到 1.7 之间，2017 年在 0.18 到 2.15 之间），表明其分布形态与正态分布相似或略尖峭。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 POD 在 2016 年 (0.5696), 2017 年 (0.5493), 2018 年 (0.5551), 2019 年 (0.5340), 2020 年 (0.5384)。年际间存在一定波动，但幅度不算特别剧烈。2016年表现相对较好。
        *   2017年和2020年在较低阈值下的POD均值相对其他年份偏低，且2017年POD的空间标准差较大，说明这两年CMORPH在探测小雨和空间一致性上可能面临更大挑战。

*   **10.2.1.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** CMORPH 的平均空间 FAR 在所有年份和阈值下都保持在非常低的水平，通常在 0.008 到 0.023 之间。这是一个积极的信号，表明 CMORPH 整体的误报情况控制得较好。
        *   **随阈值变化：** 平均空间 FAR 随着阈值的增加呈现缓慢上升的趋势。例如，2016 年从 0.0mm/d 的 0.0088 上升到 0.95mm/d 的 0.0213。这可能是因为随着阈值提高，实际有雨的样本减少，使得少数误报事件在比例上更为显著。
    *   **空间 FAR 中位数：**
        *   在所有年份和所有阈值下，CMORPH 的 FAR 空间中位数几乎**始终为 0.0000**。这表明在全国大部分格点上，CMORPH 产品的误报率极低，误报可能主要集中在少数区域。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差也相对较低，通常在 0.03 到 0.09 之间波动，并随阈值增加而略有增大。这与中位数为0相呼应，说明误报主要由少数离群的高值贡献。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** FAR 的空间偏度值在所有年份和阈值下均表现出**极高的正偏** (通常在 4.0 到 6.5 之间)。这强烈指示 FAR 的空间分布是高度不对称的，存在少数格点的 FAR 值远高于平均水平，即误报具有显著的局部集中性。
        *   **峰度：** FAR 的空间峰度值同样表现出**极高的正峰度** (通常在 18 到 50+ 之间)。这进一步证实了 FAR 分布的尖峰厚尾特征，即大部分区域 FAR 极低，但存在少数区域误报非常严重。
        *   **对误报研究的意义：** FAR 的高偏度和高峰度特性是本项目需要重点关注的，它提示误报并非随机均匀分布，而是有其特定的空间聚集模式，这为后续分析误报成因、改进模型提供了重要线索。例如，可以针对这些高 FAR 区域进行更细致的特征分析和建模。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0142), 2017 年 (0.0147), 2018 年 (0.0136), 2019 年 (0.0141), 2020 年 (0.0138)。年际间波动非常小，表现出较好的一致性。

*   **10.2.1.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** CMORPH 的平均空间 CSI 通常在 0.42 到 0.50 之间波动。
        *   **随阈值变化：** CSI 随阈值的变化趋势通常是先略微上升（在0.0mm/d到0.1mm/d或0.2mm/d之间达到峰值），然后逐渐下降。例如，2016年从0.0mm/d的0.4759上升到0.1mm/d的0.4989，之后下降至0.95mm/d的0.4352。这反映了在极小阈值时，虽然POD可能增加，但FAR的轻微增加也可能影响CSI；而在高阈值时，POD的下降成为影响CSI的主要因素。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.06 到 0.09 之间，与 POD 的标准差水平相似，表明其综合评分在空间上也存在一定的变异。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在多数情况下为负值（尤其是在中高阈值下，如 2016 年多数阈值下在 -0.6 到 -0.2 之间），表明存在一些 CSI 表现显著低于平均值的区域。
        *   **峰度：** CSI 的空间峰度值多在 -0.5 到 1.0 之间，分布形态相对接近正态或略平坦/略尖。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.4989), 2017 年 (0.4800), 2018 年 (0.4914), 2019 年 (0.4719), 2020 年 (0.4782)。2016年和2018年表现相对较好。

*   **10.2.1.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **与 CHIRPS：** POD 相关性中等 (0.52-0.64)，FAR 相关性非常高 (0.91-0.93)，CSI 相关性中高 (0.50-0.66)。表明两者在误报的空间分布上高度一致，在命中和综合评分上也有一定相似性。
    *   **与 SM2RAIN：** POD 相关性较低 (0.31-0.41)，FAR 相关性高 (0.88-0.90)，CSI 相关性较低至中等 (0.13-0.44)。表明两者在误报空间分布上有一定相似性，但在命中和综合评分的空间模式上差异较大。2017年CSI相关性仅0.1307，差异尤为显著。
    *   **与 IMERG：** POD 相关性中高 (0.55-0.67)，FAR 相关性非常高 (0.93-0.95)，CSI 相关性高 (0.58-0.76)。表明两者在各类性能指标的空间分布上都表现出较强的一致性，尤其是在误报和综合评分上。
    *   **与 GSMAP：** POD 相关性中等 (0.37-0.56)，FAR 相关性高 (0.89-0.93)，CSI 相关性中高 (0.56-0.67)。表明两者在误报和综合评分的空间分布上有较好的一致性。
    *   **与 PERSIANN：** POD 相关性中高 (0.56-0.60)，FAR 相关性非常高 (0.93-0.96)，CSI 相关性中高 (0.51-0.67)。表明两者在各类性能指标的空间分布上也表现出较强的一致性。
    *   **总结：** CMORPH 在 FAR 的空间分布模式上与所有其他产品都表现出高度一致性。在 POD 和 CSI 方面，与 IMERG、PERSIANN、GSMAP 和 CHIRPS 的空间分布模式有中高程度的相似性，而与 SM2RAIN 的相似性较低。

*   **10.2.1.5 CMORPH 年度表现小结**
    *   **整体性能：** CMORPH 在全国范围内的平均 CSI 大约在 0.45-0.50 之间（针对中低阈值）。其 FAR 控制得非常好，空间中位数常年为0，但误报呈现高度局部化的特点（高偏度和峰度）。
    *   **年际差异：** 2016年和2018年在POD和CSI的平均水平上略优于其他年份。2017年POD的空间变异性较大。FAR的年际稳定性非常好。
    *   **空间特征：** POD 和 CSI 的空间分布存在一定变异，而 FAR 的误报高度集中在少数区域。
    *   **与其他产品关系：** 其误报高发区的空间位置可能与其他产品相似；在综合性能的空间好坏区域分布上，与 IMERG, PERSIANN 等产品有较强的同步性。

**10.2 各降雨产品逐年性能指标的空间统计特征详析**

**10.2.1 CMORPH 产品性能分析**

CMORPH 产品作为一种常用的卫星降雨产品，其在 2016 年至 2020 年间的性能表现出一定的时空变异性。

*   **10.2.1.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** 在大多数年份（如 2016, 2017, 2018, 2019），CMORPH 的平均空间 POD 在非常小的阈值（0.0mm/d 至 0.1mm/d 或 0.2mm/d）下通常会略有上升，表明在探测微小降雨事件时，略微提高阈值有助于排除一些噪声，反而提升名义上的命中率。然而，随着阈值的进一步增高（超过 0.2mm/d 或 0.3mm/d），平均空间 POD 整体呈现下降趋势或在高位波动后缓慢下降。这符合预期，因为更高强度的降雨事件本身发生频率较低，探测难度也可能增加。
        *   **2016年：** 从 0.0mm/d 的 0.5134 上升到 0.1mm/d 的 0.5696，之后在 0.51-0.56 之间波动，整体略有下降趋势。
        *   **2017年：** 从 0.0mm/d 的 0.4975 上升到 0.1mm/d 的 0.5493，之后在 0.50-0.54 之间波动，整体趋势不明显，但高阈值下略有下降。
        *   **2018年：** 从 0.0mm/d 的 0.5052 上升到 0.1mm/d 的 0.5551，之后在 0.50-0.54 之间波动，高阈值下有下降。
        *   **2019年：** 从 0.0mm/d 的 0.4797 上升到 0.1mm/d 的 0.5340，之后在 0.49-0.53 之间波动，高阈值下略有下降。
        *   **2020年：** 从 0.0mm/d 的 0.4787 上升到 0.1mm/d 的 0.5384，之后在 0.52-0.54 之间波动，整体趋势相对平稳，但在极高阈值下略有下降。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** CMORPH 的 POD 空间标准差在各年份通常维持在 0.07 至 0.10 之间，表明其命中率在全国范围内存在一定的空间差异。
        *   **随阈值变化：** 标准差随阈值的变化没有非常一致的模式，有时略增，有时略减，但整体波动不大。例如，2016年标准差在0.078-0.088之间；2017年在0.091-0.103之间，显示出更大的空间不一致性。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** 在大多数年份和阈值下，POD 的空间偏度为正（如 2016 年在 0.0 到 0.65 之间波动，2018 年在 0.1 到 0.78 之间），表明 POD 空间分布略微右偏，即存在一些 POD 表现显著高于平均值的区域。随着阈值增加，偏度值有时会减小，甚至变为负值（如 2016 年高阈值下 POD 偏度接近 -0.5），说明高强度降雨的命中率空间差异模式可能发生改变。
        *   **峰度：** POD 的空间峰度值多数情况下接近 0 或为正（如 2016 年在 -0.02 到 1.7 之间，2017 年在 0.18 到 2.15 之间），表明其分布形态与正态分布相似或略尖峭。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 POD 在 2016 年 (0.5696), 2017 年 (0.5493), 2018 年 (0.5551), 2019 年 (0.5340), 2020 年 (0.5384)。年际间存在一定波动，但幅度不算特别剧烈。2016年表现相对较好。
        *   2017年和2020年在较低阈值下的POD均值相对其他年份偏低，且2017年POD的空间标准差较大，说明这两年CMORPH在探测小雨和空间一致性上可能面临更大挑战。

*   **10.2.1.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** CMORPH 的平均空间 FAR 在所有年份和阈值下都保持在非常低的水平，通常在 0.008 到 0.023 之间。这是一个积极的信号，表明 CMORPH 整体的误报情况控制得较好。
        *   **随阈值变化：** 平均空间 FAR 随着阈值的增加呈现缓慢上升的趋势。例如，2016 年从 0.0mm/d 的 0.0088 上升到 0.95mm/d 的 0.0213。这可能是因为随着阈值提高，实际有雨的样本减少，使得少数误报事件在比例上更为显著。
    *   **空间 FAR 中位数：**
        *   在所有年份和所有阈值下，CMORPH 的 FAR 空间中位数几乎**始终为 0.0000**。这表明在全国大部分格点上，CMORPH 产品的误报率极低，误报可能主要集中在少数区域。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差也相对较低，通常在 0.03 到 0.09 之间波动，并随阈值增加而略有增大。这与中位数为0相呼应，说明误报主要由少数离群的高值贡献。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** FAR 的空间偏度值在所有年份和阈值下均表现出**极高的正偏** (通常在 4.0 到 6.5 之间)。这强烈指示 FAR 的空间分布是高度不对称的，存在少数格点的 FAR 值远高于平均水平，即误报具有显著的局部集中性。
        *   **峰度：** FAR 的空间峰度值同样表现出**极高的正峰度** (通常在 18 到 50+ 之间)。这进一步证实了 FAR 分布的尖峰厚尾特征，即大部分区域 FAR 极低，但存在少数区域误报非常严重。
        *   **对误报研究的意义：** FAR 的高偏度和高峰度特性是本项目需要重点关注的，它提示误报并非随机均匀分布，而是有其特定的空间聚集模式，这为后续分析误报成因、改进模型提供了重要线索。例如，可以针对这些高 FAR 区域进行更细致的特征分析和建模。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0142), 2017 年 (0.0147), 2018 年 (0.0136), 2019 年 (0.0141), 2020 年 (0.0138)。年际间波动非常小，表现出较好的一致性。

*   **10.2.1.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** CMORPH 的平均空间 CSI 通常在 0.42 到 0.50 之间波动。
        *   **随阈值变化：** CSI 随阈值的变化趋势通常是先略微上升（在0.0mm/d到0.1mm/d或0.2mm/d之间达到峰值），然后逐渐下降。例如，2016年从0.0mm/d的0.4759上升到0.1mm/d的0.4989，之后下降至0.95mm/d的0.4352。这反映了在极小阈值时，虽然POD可能增加，但FAR的轻微增加也可能影响CSI；而在高阈值时，POD的下降成为影响CSI的主要因素。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.06 到 0.09 之间，与 POD 的标准差水平相似，表明其综合评分在空间上也存在一定的变异。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在多数情况下为负值（尤其是在中高阈值下，如 2016 年多数阈值下在 -0.6 到 -0.2 之间），表明存在一些 CSI 表现显著低于平均值的区域。
        *   **峰度：** CSI 的空间峰度值多在 -0.5 到 1.0 之间，分布形态相对接近正态或略平坦/略尖。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.4989), 2017 年 (0.4800), 2018 年 (0.4914), 2019 年 (0.4719), 2020 年 (0.4782)。2016年和2018年表现相对较好。

*   **10.2.1.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **与 CHIRPS：** POD 相关性中等 (0.52-0.64)，FAR 相关性非常高 (0.91-0.93)，CSI 相关性中高 (0.50-0.66)。表明两者在误报的空间分布上高度一致，在命中和综合评分上也有一定相似性。
    *   **与 SM2RAIN：** POD 相关性较低 (0.31-0.41)，FAR 相关性高 (0.88-0.90)，CSI 相关性较低至中等 (0.13-0.44)。表明两者在误报空间分布上有一定相似性，但在命中和综合评分的空间模式上差异较大。2017年CSI相关性仅0.1307，差异尤为显著。
    *   **与 IMERG：** POD 相关性中高 (0.55-0.67)，FAR 相关性非常高 (0.93-0.95)，CSI 相关性高 (0.58-0.76)。表明两者在各类性能指标的空间分布上都表现出较强的一致性，尤其是在误报和综合评分上。
    *   **与 GSMAP：** POD 相关性中等 (0.37-0.56)，FAR 相关性高 (0.89-0.93)，CSI 相关性中高 (0.56-0.67)。表明两者在误报和综合评分的空间分布上有较好的一致性。
    *   **与 PERSIANN：** POD 相关性中高 (0.56-0.60)，FAR 相关性非常高 (0.93-0.96)，CSI 相关性中高 (0.51-0.67)。表明两者在各类性能指标的空间分布上也表现出较强的一致性。
    *   **总结：** CMORPH 在 FAR 的空间分布模式上与所有其他产品都表现出高度一致性。在 POD 和 CSI 方面，与 IMERG、PERSIANN、GSMAP 和 CHIRPS 的空间分布模式有中高程度的相似性，而与 SM2RAIN 的相似性较低。

*   **10.2.1.5 CMORPH 年度表现小结**
    *   **整体性能：** CMORPH 在全国范围内的平均 CSI 大约在 0.45-0.50 之间（针对中低阈值）。其 FAR 控制得非常好，空间中位数常年为0，但误报呈现高度局部化的特点（高偏度和峰度）。
    *   **年际差异：** 2016年和2018年在POD和CSI的平均水平上略优于其他年份。2017年POD的空间变异性较大。FAR的年际稳定性非常好。
    *   **空间特征：** POD 和 CSI 的空间分布存在一定变异，而 FAR 的误报高度集中在少数区域。
    *   **与其他产品关系：** 其误报高发区的空间位置可能与其他产品相似；在综合性能的空间好坏区域分布上，与 IMERG, PERSIANN 等产品有较强的同步性。

**10.2.2 CHIRPS 产品性能分析**

CHIRPS 产品作为另一种广泛应用的降雨数据集，其在 2016 年至 2020 年的性能表现如下：

*   **10.2.2.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** CHIRPS 的平均空间 POD 整体水平相较于 CMORPH 偏低，通常在 0.38 到 0.46 之间波动 (2020年略高，达到0.44-0.50)。与 CMORPH 类似，在极低阈值下，POD 随阈值略微增加后趋于平稳或缓慢下降。
        *   **2016年：** 从 0.0mm/d 的 0.4234 缓慢上升至 0.25-0.45mm/d 区间的约 0.46，之后略有下降。
        *   **2017年：** 从 0.0mm/d 的 0.3838 缓慢上升至 0.4-0.55mm/d 区间的约 0.42-0.423，之后变化不大。
        *   **2018年：** 从 0.0mm/d 的 0.4284 上升至 0.2-0.65mm/d 区间的约 0.46，之后略有下降。
        *   **2019年：** 从 0.0mm/d 的 0.3876 上升至 0.2-0.55mm/d 区间的约 0.416-0.418，之后变化不大。
        *   **2020年：** 从 0.0mm/d 的 0.4424 上升至 0.5-0.7mm/d 区间的约 0.498-0.499，是历年中POD表现最好的。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** CHIRPS 的 POD 空间标准差相对较小，通常在 0.05 至 0.07 之间，这表明其命中率在全国范围内的空间一致性相对较好，优于CMORPH。
        *   **随阈值变化：** 标准差随阈值的变化不明显，整体保持在一个较低的稳定水平。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** POD 的空间偏度值在不同年份和阈值下正负皆有，但绝对值通常较小（例如，2016年多数为负，在-0.3到0.1之间；2017年多数为正，在0.1到0.3之间）。这表明其空间分布相对接近对称，或轻微偏斜。
        *   **峰度：** POD 的空间峰度值多数情况下为负值（例如，2016年普遍在-0.2到-0.7之间，2019年也多为负），表明其分布形态通常比正态分布更平坦（扁平峰）。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 POD 在 2016 年 (0.4498), 2017 年 (0.4082), 2018 年 (0.4517), 2019 年 (0.4094), 2020 年 (0.4783)。2017年和2019年表现相对较差，2020年表现最好。CHIRPS 的POD年际波动较CMORPH更为明显。

*   **10.2.2.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** CHIRPS 的平均空间 FAR 整体水平与 CMORPH 相当，甚至略优，通常在 0.007 到 0.027 之间。
        *   **随阈值变化：** 与 CMORPH 类似，平均空间 FAR 随着阈值的增加呈现缓慢上升的趋势。例如，2016 年从 0.0mm/d 的 0.0107 上升到 0.95mm/d 的 0.0269。
    *   **空间 FAR 中位数：**
        *   与 CMORPH 一致，在所有年份和所有阈值下，CHIRPS 的 FAR 空间中位数几乎**始终为 0.0000**。表明大部分区域误报控制良好。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差也与 CMORPH 相似，通常在 0.03 到 0.10 之间波动，并随阈值增加而略有增大。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** CHIRPS 的 FAR 空间偏度值同样表现出**极高的正偏** (通常在 3.4 到 5.0 之间)，指示误报具有显著的局部集中性。
        *   **峰度：** CHIRPS 的 FAR 空间峰度值同样表现出**极高的正峰度** (通常在 10 到 30+ 之间)，进一步证实了误报分布的尖峰厚尾特征。
        *   **与CMORPH比较：** CHIRPS的FAR偏度和峰度值通常略低于CMORPH在相应条件下的值，但依然属于高度偏斜和尖峰的分布。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0165), 2017 年 (0.0165), 2018 年 (0.0148), 2019 年 (0.0158), 2020 年 (0.0133)。年际间波动非常小，表现出较好的一致性。2020年FAR控制略好。

*   **10.2.2.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** CHIRPS 的平均空间 CSI 通常在 0.34 到 0.43 之间波动，整体水平低于 CMORPH。
        *   **随阈值变化：** CSI 随阈值的变化趋势与 CMORPH 类似，通常是先略微上升或保持平稳，然后在较高阈值下逐渐下降。例如，2016年从0.0mm/d的0.3942，在0.05-0.2mm/d附近达到约0.40的峰值，之后缓慢下降至0.95mm/d的0.3717。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.045 到 0.065 之间，相对较小，表明其综合评分在空间上的差异性也较小，表现出比CMORPH更好的空间均一性。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在不同年份和阈值下表现不一，有时为正，有时为负，但绝对值通常不大（多数在-0.3到0.3之间）。
        *   **峰度：** CSI 的空间峰度值多数情况下为负值（通常在-0.7到-0.1之间），表明其空间分布通常比正态分布更平坦。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.4011), 2017 年 (0.3680), 2018 年 (0.4082), 2019 年 (0.3701), 2020 年 (0.4347)。2017年和2019年表现较差，2020年表现最好。

*   **10.2.2.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **与 SM2RAIN：** POD 相关性较低 (0.25-0.50)，FAR 相关性非常高 (0.96-0.98)，CSI 相关性较低至中等 (0.16-0.49)。
    *   **与 IMERG：** POD 相关性中高 (0.56-0.61)，FAR 相关性非常高 (0.97-0.98)，CSI 相关性高 (0.65-0.73)。
    *   **与 GSMAP：** POD 相关性中等 (0.42-0.61)，FAR 相关性非常高 (0.96-0.97)，CSI 相关性中高 (0.53-0.74)。
    *   **与 PERSIANN：** POD 相关性中等 (0.40-0.67)，FAR 相关性非常高 (0.98-0.99)，CSI 相关性高 (0.65-0.78)。
    *   **总结：** CHIRPS 在 FAR 的空间分布模式上与所有其他产品都表现出极高的一致性。在 POD 和 CSI 方面，与 IMERG, PERSIANN, GSMAP 的空间分布模式有中高至较高程度的相似性，与 SM2RAIN 的相似性相对较低。

*   **10.2.2.5 CHIRPS 年度表现小结**
    *   **整体性能：** CHIRPS 的平均 CSI 略低于 CMORPH，但其 POD 和 CSI 的空间一致性（标准差较小）通常优于 CMORPH。FAR 控制良好，与 CMORPH 类似，误报也呈现高度局部化特征。
    *   **年际差异：** 2020 年是 CHIRPS 表现最好的年份，在 POD 和 CSI 上均有明显提升。2017 年和 2019 年相对表现较弱。
    *   **空间特征：** POD 和 CSI 的空间分布相对均匀，峰度多为负，表明空间上极端好或差的区域较少。FAR 的误报同样高度集中在少数区域。
    *   **与其他产品关系：** 其误报高发区的空间位置与其他产品高度相似；在综合性能的空间好坏区域分布上，与IMERG, PERSIANN等产品有较强的同步性。

**10.2.3 SM2RAIN 产品性能分析**

SM2RAIN 产品基于土壤湿度反演降雨，其性能特征与其他直接遥感降雨的产品有所不同。

*   **10.2.3.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** SM2RAIN 的平均空间 POD 在所有产品中表现**最高**，通常在 0.85 到 0.92 之间。这是一个非常显著的特点。随阈值增加，POD 整体呈现非常缓慢的下降趋势，但在所有测试阈值下均保持较高水平。
        *   **2016年：** 从 0.0mm/d 的 0.9070 缓慢下降至 0.95mm/d 的 0.8494。
        *   **2017年：** 从 0.0mm/d 的 0.9168 缓慢下降至 0.95mm/d 的 0.8744。
        *   **2018年：** 从 0.0mm/d 的 0.9120 缓慢下降至 0.95mm/d 的 0.8765。
        *   **2019年：** 从 0.0mm/d 的 0.9128 缓慢下降至 0.95mm/d 的 0.8429。
        *   **2020年：** 从 0.0mm/d 的 0.9146 缓慢下降至 0.95mm/d 的 0.8573。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** SM2RAIN 的 POD 空间标准差相对较小，通常在 0.07 至 0.11 之间，表明其高命中率在全国范围内也具有较好的一致性。
        *   **随阈值变化：** 标准差随阈值的变化不大，整体维持在一个相对稳定的水平。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** POD 的空间偏度值在所有年份和阈值下均表现出显著的**负偏** (通常在 -1.6 到 -1.0 之间)。这表明 POD 空间分布是左偏的，即大部分区域的 POD 都非常高，接近其最大值，而少数区域的 POD 表现相对较差，拉低了平均值。
        *   **峰度：** POD 的空间峰度值通常为正（多数在 0.5 到 3.5 之间），表明其分布比正态分布更尖峭，进一步印证了大部分区域 POD 集中在高值区。
    *   **年际变化：**
        *   SM2RAIN 的平均空间 POD 在各年份间表现非常稳定，例如在0.1mm/d阈值下，均在0.89-0.91之间，波动很小。

*   **10.2.3.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** SM2RAIN 的平均空间 FAR 相比 CMORPH 和 CHIRPS 略高，通常在 0.015 到 0.035 之间。
        *   **随阈值变化：** 与其他产品类似，平均空间 FAR 随着阈值的增加呈现缓慢上升的趋势。例如，2016 年从 0.0mm/d 的 0.0168 上升到 0.95mm/d 的 0.0341。
    *   **空间 FAR 中位数：**
        *   与 CMORPH 和 CHIRPS 一致，在所有年份和所有阈值下，SM2RAIN 的 FAR 空间中位数几乎**始终为 0.0000**。表明其误报也主要集中在少数区域。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差在 0.05 到 0.12 之间波动，略高于 CMORPH 和 CHIRPS，表明其误报的空间差异性可能稍大一些。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** SM2RAIN 的 FAR 空间偏度值同样表现出**极高的正偏** (通常在 3.3 到 3.8 之间)，指示误报具有显著的局部集中性。
        *   **峰度：** SM2RAIN 的 FAR 空间峰度值同样表现出**极高的正峰度** (通常在 9 到 15+ 之间，个别年份在低阈值下可达20-30)，证实了误报分布的尖峰厚尾特征。其峰度值相比CMORPH和CHIRPS可能略低一些，但依然显著。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0240), 2017 年 (0.0244), 2018 年 (0.0224), 2019 年 (0.0244), 2020 年 (0.0241)。年际间波动非常小，表现出良好的一致性。

*   **10.2.3.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** SM2RAIN 的平均空间 CSI 表现优异，通常在 0.50 到 0.75 之间波动，在低阈值下表现尤为突出，显著高于其他产品。
        *   **随阈值变化：** CSI 随阈值的增加呈现清晰的下降趋势，尤其是在从极低阈值（如0.0mm/d）向中低阈值过渡时下降较快，之后趋于平缓下降。例如，2016年从0.0mm/d的0.7260下降至0.1mm/d的0.6396，再缓慢下降至0.95mm/d的0.5118。这表明SM2RAIN在探测微小降雨事件方面的综合技巧非常高，但对稍大降雨事件的区分能力相对减弱（尽管CSI绝对值仍然较高）。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.05 到 0.08 之间，表明其较高的综合评分在空间上也具有较好的一致性。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在低阈值下多为负或接近零（如2016年0.0mm/d时为-0.31），随着阈值增加，偏度值趋向于正值或在零附近波动（如2016年0.95mm/d时为0.1279）。这表明在探测微小降雨时，大部分区域CSI表现优异；而在探测稍大降雨时，空间差异模式有所改变。
        *   **峰度：** CSI 的空间峰度值变化较大，低阈值下可能为正（如2016年0.0mm/d时为0.46），高阈值下多接近零或略为负。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.6396), 2017 年 (0.6402), 2018 年 (0.6583), 2019 年 (0.6343), 2020 年 (0.6399)。年际间表现非常稳定。

*   **10.2.3.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **与 IMERG：** POD 相关性较低 (0.11-0.35)，FAR 相关性高 (0.95-0.97)，CSI 相关性中等 (0.33-0.52)。
    *   **与 GSMAP：** POD 相关性中等 (0.30-0.50)，FAR 相关性高 (0.95-0.96)，CSI 相关性较低至中等 (0.10-0.46)。
    *   **与 PERSIANN：** POD 相关性极低甚至为负 (-0.13 至 0.015)，FAR 相关性高 (0.96-0.98)，CSI 相关性中等 (0.51-0.63)。
    *   **回顾与CMORPH/CHIRPS的关系：** 与CMORPH的POD相关性低，FAR相关性高，CSI相关性较低。与CHIRPS的POD相关性低，FAR相关性高，CSI相关性较低。
    *   **总结：** SM2RAIN 在 FAR 的空间分布模式上与所有其他产品都表现出高度一致性。然而，在 POD 和 CSI 的空间分布模式上，SM2RAIN 与其他所有直接遥感降雨产品（CMORPH, CHIRPS, IMERG, GSMAP, PERSIANN）均表现出较低的相似性。这突显了 SM2RAIN 作为一种基于不同物理原理（土壤湿度反演）的降雨产品的独特性。

*   **10.2.3.5 SM2RAIN 年度表现小结**
    *   **整体性能：** SM2RAIN 的最大特点是其**极高的 POD** 和在**低阈值下非常优异的 CSI**。这意味着它在探测“是否有雨”以及小雨量级事件方面能力很强。其 FAR 水平略高于 CMORPH/CHIRPS，但误报也呈现高度局部化特征。
    *   **年际差异：** SM2RAIN 的各项性能指标在年际间都表现出高度的稳定性。
    *   **空间特征：** POD 空间分布高度左偏，大部分区域表现优异。CSI 在低阈值下也有类似趋势。FAR 的误报同样高度集中在少数区域。
    *   **与其他产品关系：** 其误报高发区的空间位置可能与其他产品相似；但在命中能力和综合评分的空间好坏区域分布上，与传统卫星遥感产品差异较大，这使其在数据融合中可能提供独特的补充信息。

**10.2.4 IMERG 产品性能分析**

IMERG 作为 GPM 时代的核心降雨产品之一，其在 2016 年至 2020 年间的性能表现备受关注。

*   **10.2.4.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** IMERG 的平均空间 POD 整体水平较高，通常在 0.61 到 0.77 之间波动。与其他产品类似，POD 随阈值增加呈现下降趋势，但在高阈值下仍能维持一个相对不错的水平。
        *   **2016年：** 从 0.0mm/d 的 0.7567 持续下降至 0.95mm/d 的 0.6147。
        *   **2017年：** 从 0.0mm/d 的 0.7510 持续下降至 0.95mm/d 的 0.6160。
        *   **2018年：** 从 0.0mm/d 的 0.7688 持续下降至 0.95mm/d 的 0.6286。
        *   **2019年：** 从 0.0mm/d 的 0.7480 持续下降至 0.95mm/d 的 0.6026。
        *   **2020年：** 从 0.0mm/d 的 0.7501 持续下降至 0.95mm/d 的 0.6493 (2020年在高阈值下POD表现相对突出)。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** IMERG 的 POD 空间标准差通常在 0.068 至 0.095 之间，表明其命中率在全国范围内存在一定的空间差异，与CMORPH的变异程度相似或略好。
        *   **随阈值变化：** 标准差随阈值的变化没有非常一致的模式，但整体波动不大。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** POD 的空间偏度值在所有年份和阈值下几乎都为**负偏** (通常在 -0.6 到 -0.2 之间)。这表明 POD 空间分布是左偏的，即大部分区域的 POD 表现较好，接近其最大值，而少数区域的 POD 表现相对较差。
        *   **峰度：** POD 的空间峰度值多数情况下接近 0 或略为正（例如，2016年多数在0.0到0.3之间），表明其分布形态与正态分布相似或略尖峭。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 POD 在 2016 年 (0.7100), 2017 年 (0.6994), 2018 年 (0.7206), 2019 年 (0.6975), 2020 年 (0.7148)。年际间波动相对较小，2018年表现略优。

*   **10.2.4.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** IMERG 的平均空间 FAR 水平与 CMORPH 和 CHIRPS 相当，甚至在某些情况下略优，通常在 0.010 到 0.023 之间。
        *   **随阈值变化：** 与其他产品类似，平均空间 FAR 随着阈值的增加呈现缓慢上升的趋势。例如，2016 年从 0.0mm/d 的 0.0119 上升到 0.95mm/d 的 0.0212。
    *   **空间 FAR 中位数：**
        *   与前面分析的产品一致，在所有年份和所有阈值下，IMERG 的 FAR 空间中位数几乎**始终为 0.0000**。表明大部分区域误报控制良好。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差也与 CMORPH 和 CHIRPS 相似，通常在 0.035 到 0.08 之间波动，并随阈值增加而略有增大。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** IMERG 的 FAR 空间偏度值同样表现出**极高的正偏** (通常在 3.5 到 4.2 之间)，指示误报具有显著的局部集中性。
        *   **峰度：** IMERG 的 FAR 空间峰度值同样表现出**极高的正峰度** (通常在 11 到 20+ 之间)，证实了误报分布的尖峰厚尾特征。其偏度和峰度值与CMORPH和CHIRPS在同一量级。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0153), 2017 年 (0.0156), 2018 年 (0.0145), 2019 年 (0.0151), 2020 年 (0.0151)。年际间波动非常小，表现出良好的一致性。

*   **10.2.4.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** IMERG 的平均空间 CSI 表现较好，通常在 0.50 到 0.68 之间波动，整体优于 CMORPH 和 CHIRPS，但通常略低于 SM2RAIN 在极低阈值下的表现。
        *   **随阈值变化：** CSI 随阈值的增加呈现清晰的下降趋势。例如，2016年从0.0mm/d的0.6636下降至0.95mm/d的0.5017。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.05 到 0.07 之间，表明其综合评分在空间上也具有较好的一致性，与CHIRPS类似，优于CMORPH。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在不同年份和阈值下表现不一，但多数情况下绝对值较小（例如，2016年多数在0.0到0.3之间，2017年多数在-0.4到-0.1之间）。
        *   **峰度：** CSI 的空间峰度值多数情况下接近0或为负值（例如，2016年多数在-0.1到0.2之间），表明其空间分布形态相对接近正态或略平坦。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.6044), 2017 年 (0.5933), 2018 年 (0.6166), 2019 年 (0.5941), 2020 年 (0.6050)。年际间表现非常稳定。

*   **10.2.4.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **回顾与CMORPH/CHIRPS/SM2RAIN的关系：** 与CMORPH的POD、FAR、CSI空间相关性均较高。与CHIRPS的FAR空间相关性极高，POD、CSI相关性较高。与SM2RAIN的FAR空间相关性高，但POD、CSI相关性较低。
    *   **与 GSMAP：** POD 相关性中高 (0.42-0.54)，FAR 相关性非常高 (0.96-0.98)，CSI 相关性高 (0.58-0.71)。
    *   **与 PERSIANN：** POD 相关性中高 (0.55-0.65)，FAR 相关性非常高 (0.98-0.99)，CSI 相关性高 (0.72-0.78)。
    *   **总结：** IMERG 在 FAR 的空间分布模式上与所有其他产品都表现出高度一致性。在 POD 和 CSI 方面，与 CMORPH, CHIRPS, GSMAP, PERSIANN 的空间分布模式都有较高程度的相似性，而与 SM2RAIN 的相似性较低。

*   **10.2.4.5 IMERG 年度表现小结**
    *   **整体性能：** IMERG 是一款综合性能较强的产品，其平均 POD 和 CSI 均处于较高水平，尤其是在中低阈值下表现稳健。FAR 控制良好，误报同样呈现局部化特征。
    *   **年际差异：** IMERG 的各项性能指标在年际间均表现出高度的稳定性。2020年在高阈值下的POD保持能力略有优势。
    *   **空间特征：** POD 空间分布通常左偏，大部分区域表现良好。CSI 空间分布相对均匀。FAR 的误报高度集中。
    *   **与其他产品关系：** 在各项性能指标的空间分布上，IMERG 与 CMORPH、PERSIANN、GSMAP、CHIRPS 等传统卫星遥感产品展现出较强的一致性，而与基于土壤湿度反演的 SM2RAIN 在命中和综合评分模式上差异较大。这使得 IMERG 成为一个可靠的基准产品，并且在融合时可以与其他传统产品形成良好的协同。

**10.2.5 GSMAP 产品性能分析**

GSMAP 是另一款重要的全球降雨产品，其在 2016 年至 2020 年间的性能表现如下：

*   **10.2.5.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** GSMAP 的平均空间 POD 整体水平属于中等偏上，通常在 0.53 到 0.75 之间波动。与其他产品类似，POD 在极低阈值下可能略有上升，之后随阈值增加整体呈现缓慢上升或在高位平稳波动的趋势，这与其他产品在高阈值下POD下降的趋势有所不同，显示出其在探测较高强度降雨方面可能具有一定优势或不同特性。
        *   **2016年：** 从 0.0mm/d 的 0.5554 持续缓慢上升至 0.95mm/d 的 0.6690。
        *   **2017年：** 从 0.0mm/d 的 0.5444 持续缓慢上升至 0.95mm/d 的 0.6690。
        *   **2018年：** 从 0.0mm/d 的 0.5662 持续缓慢上升至 0.95mm/d 的 0.6779。
        *   **2019年：** 从 0.0mm/d 的 0.5389 持续缓慢上升至 0.95mm/d 的 0.6675。
        *   **2020年：** 从 0.0mm/d 的 0.5698 持续缓慢上升至 0.95mm/d 的 0.7450 (2020年在高阈值下POD表现非常突出)。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** GSMAP 的 POD 空间标准差非常小，通常在 0.05 至 0.07 之间，这表明其命中率在全国范围内的空间一致性非常好，是所有分析产品中表现最佳的之一。
        *   **随阈值变化：** 标准差随阈值的变化非常小，整体保持在一个极低的稳定水平。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** POD 的空间偏度值在所有年份和阈值下几乎都为**负偏** (通常在 -0.7 到 -0.1 之间)。这表明 POD 空间分布是左偏的，即大部分区域的 POD 表现较好。
        *   **峰度：** POD 的空间峰度值多数情况下为正或接近零（例如，2016年多数在0.0到0.8之间），表明其分布形态与正态分布相似或略尖峭。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 POD 在 2016 年 (0.6073), 2017 年 (0.5929), 2018 年 (0.6127), 2019 年 (0.5903), 2020 年 (0.6353)。年际间波动相对较小，2020年表现最佳。

*   **10.2.5.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** GSMAP 的平均空间 FAR 在所有产品中表现**最优异**，始终保持在极低的水平，通常在 0.0017 到 0.015 之间。这是一个非常显著的优势。
        *   **随阈值变化：** 平均空间 FAR 随着阈值的增加呈现非常缓慢的上升趋势，但即使在高阈值下也远低于其他产品。例如，2016 年从 0.0mm/d 的 0.0025 上升到 0.95mm/d 的 0.0145。
    *   **空间 FAR 中位数：**
        *   与前面分析的产品一致，在所有年份和所有阈值下，GSMAP 的 FAR 空间中位数几乎**始终为 0.0000**。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差也极低，通常在 0.007 到 0.05 之间波动，显著低于其他产品。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** GSMAP 的 FAR 空间偏度值同样表现出**极高的正偏** (通常在 4.0 到 5.4 之间)，指示误报具有显著的局部集中性，但其偏度值相较于其他产品可能略低或在相似范围内。
        *   **峰度：** GSMAP 的 FAR 空间峰度值同样表现出**极高的正峰度** (通常在 14 到 30+ 之间)，证实了误报分布的尖峰厚尾特征。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0053), 2017 年 (0.0049), 2018 年 (0.0042), 2019 年 (0.0050), 2020 年 (0.0046)。年际间波动非常小，且始终维持在极低水平。

*   **10.2.5.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** GSMAP 的平均空间 CSI 表现非常出色，通常在 0.53 到 0.65 之间波动，整体优于 CMORPH, CHIRPS，与 IMERG 相当或略优，尤其是在中高阈值下表现稳健。
        *   **随阈值变化：** CSI 随阈值的变化相对平稳，通常在低阈值处达到一个平台期后，在高阈值下也仅有轻微下降或甚至继续小幅上升（如2020年）。例如，2016年从0.0mm/d的0.5454，在0.3-0.5mm/d附近达到约0.596的峰值，之后略降至0.95mm/d的0.5816。这种在高阈值下CSI不明显下降的特性是GSMAP的一个优点。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.05 至 0.07 之间，与 IMERG 和 CHIRPS 类似，表明其综合评分在空间上也具有较好的一致性。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在所有年份和阈值下几乎都为**负偏** (通常在 -0.8 到 -0.1 之间)。这表明 CSI 空间分布是左偏的，大部分区域表现较好。
        *   **峰度：** CSI 的空间峰度值多数情况下接近0或为正值（如2016年多数在0.0到0.8之间），表明其分布形态与正态分布相似或略尖峭。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.5815), 2017 年 (0.5704), 2018 年 (0.5915), 2019 年 (0.5670), 2020 年 (0.6096)。年际间表现相对稳定，2020年表现最佳。

*   **10.2.5.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **回顾与其他产品的关系：** 与CMORPH的POD、FAR、CSI空间相关性均较高。与CHIRPS的FAR空间相关性极高，POD、CSI相关性较高。与SM2RAIN的FAR空间相关性高，但POD、CSI相关性较低。与IMERG的FAR空间相关性极高，POD、CSI相关性高。
    *   **与 PERSIANN：** POD 相关性较低至中等 (0.22-0.35)，FAR 相关性非常高 (0.96-0.97)，CSI 相关性中高 (0.50-0.67)。
    *   **总结：** GSMAP 在 FAR 的空间分布模式上与所有其他产品都表现出高度一致性。在 POD 和 CSI 方面，与 CMORPH, CHIRPS, IMERG 的空间分布模式都有较高程度的相似性，与 PERSIANN 的 POD 相似性较低但 CSI 相似性尚可，而与 SM2RAIN 的 POD 和 CSI 相似性均较低。

*   **10.2.5.5 GSMAP 年度表现小结**
    *   **整体性能：** GSMAP 最显著的优势在于其**极低的 FAR**，在所有分析产品中误报控制最佳。其 POD 随阈值增加有缓慢上升趋势，CSI 在中高阈值下表现稳健甚至略有提升，显示出良好的综合性能，尤其是在区分中高强度降雨事件上。
    *   **年际差异：** GSMAP 的各项性能指标在年际间表现出较好的稳定性，其中2020年在高阈值下的POD和CSI表现尤为突出。
    *   **空间特征：** POD 和 CSI 的空间分布通常左偏，且空间一致性非常好（标准差小）。FAR 的误报同样高度集中在少数区域，但整体误报水平极低。
    *   **与其他产品关系：** 其误报高发区的空间位置（尽管误报本身很少）与其他产品相似；在综合性能的空间好坏区域分布上，与IMERG、CMORPH等产品有较强的同步性。GSMAP的低FAR特性使其在数据融合中可能扮演关键角色，尤其是在对误报敏感的应用场景。

**10.2.6 PERSIANN 产品性能分析**

PERSIANN 系列产品也是一类常用的历史较久的卫星降雨估算产品。

*   **10.2.6.1 POD (命中率) 空间统计特征**
    *   **平均空间 POD 随阈值变化：**
        *   **普遍趋势：** PERSIANN 的平均空间 POD 整体水平属于中等，通常在 0.58 到 0.69 之间波动。其POD随阈值增加的变化趋势与其他产品（除GSMAP外）类似，即在极低阈值下可能略有上升，之后趋于平稳或缓慢下降。
        *   **2016年：** 从 0.0mm/d 的 0.6326 上升至 0.2-0.4mm/d 区间的约 0.68-0.69，之后缓慢下降至 0.95mm/d 的 0.6596。
        *   **2017年：** 从 0.0mm/d 的 0.5959 上升至 0.35-0.45mm/d 区间的约 0.658-0.659，之后略有下降至 0.95mm/d 的 0.6329。
        *   **2018年：** 从 0.0mm/d 的 0.6197 上升至 0.3-0.45mm/d 区间的约 0.667-0.668，之后在高位波动。
        *   **2019年：** 从 0.0mm/d 的 0.5817 上升至 0.3-0.45mm/d 区间的约 0.64-0.645，之后在高位波动或略降。
        *   **2020年：** 从 0.0mm/d 的 0.6020 上升至 0.4-0.55mm/d 区间的约 0.675-0.678，之后在高位波动或略降。
    *   **空间 POD 离散度 (标准差)：**
        *   **普遍趋势：** PERSIANN 的 POD 空间标准差相对较大，通常在 0.07 至 0.14 之间，表明其命中率在全国范围内的空间差异性较大，是所有分析产品中空间变异性最大的之一（尤其是2016和2019年）。
        *   **随阈值变化：** 标准差随阈值的变化没有非常一致的模式，但整体维持在一个相对较高的水平。
    *   **空间 POD 分布形态 (偏度、峰度)：**
        *   **偏度：** POD 的空间偏度值在不同年份和阈值下正负皆有，但绝对值通常不大（多数在-0.6到0.8之间），表明其空间分布相对接近对称或轻微偏斜。
        *   **峰度：** POD 的空间峰度值多数情况下为负值（通常在-0.9到-0.0之间），表明其分布形态通常比正态分布更平坦。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 POD 在 2016 年 (0.6711), 2017 年 (0.6313), 2018 年 (0.6519), 2019 年 (0.6184), 2020 年 (0.6452)。年际间存在较明显的波动，2016年表现最好，2019年相对较差。

*   **10.2.6.2 FAR (空报率) 空间统计特征**
    *   **平均空间 FAR：**
        *   **整体水平：** PERSIANN 的平均空间 FAR 水平与 CMORPH、CHIRPS、IMERG 相当，通常在 0.010 到 0.03 之间。
        *   **随阈值变化：** 与其他产品类似，平均空间 FAR 随着阈值的增加呈现缓慢上升的趋势。例如，2016 年从 0.0mm/d 的 0.0126 上升到 0.95mm/d 的 0.0301。
    *   **空间 FAR 中位数：**
        *   与前面分析的产品一致，在所有年份和所有阈值下，PERSIANN 的 FAR 空间中位数几乎**始终为 0.0000**。
    *   **空间 FAR 离散度 (标准差)：**
        *   FAR 的空间标准差通常在 0.04 到 0.11 之间波动，与CMORPH、CHIRPS、IMERG在相似的范围内。
    *   **空间 FAR 分布形态 (偏度、峰度)：**
        *   **偏度：** PERSIANN 的 FAR 空间偏度值同样表现出**极高的正偏** (通常在 3.3 到 5.2 之间)，指示误报具有显著的局部集中性。
        *   **峰度：** PERSIANN 的 FAR 空间峰度值同样表现出**极高的正峰度** (通常在 10 到 30+ 之间)，证实了误报分布的尖峰厚尾特征。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 FAR 在 2016 年 (0.0193), 2017 年 (0.0190), 2018 年 (0.0183), 2019 年 (0.0181), 2020 年 (0.0174)。年际间波动非常小，表现出良好的一致性，且有逐年略微下降的趋势。

*   **10.2.6.3 CSI (临界成功指数) 空间统计特征**
    *   **平均空间 CSI：**
        *   **整体水平：** PERSIANN 的平均空间 CSI 通常在 0.44 到 0.55 之间波动，整体水平与CMORPH相当或略低，低于IMERG和GSMAP。
        *   **随阈值变化：** CSI 随阈值的变化趋势通常是先略微上升或保持平稳，然后在较高阈值下逐渐下降。例如，2016年从0.0mm/d的0.5495，在0.05-0.25mm/d附近达到约0.53-0.54的平台期，之后缓慢下降至0.95mm/d的0.4663。
    *   **空间 CSI 离散度 (标准差)：**
        *   CSI 的空间标准差通常在 0.045 到 0.09 之间，其中2016年和2019年的标准差较大，与POD的空间变异性相对应。
    *   **空间 CSI 分布形态 (偏度、峰度)：**
        *   **偏度：** CSI 的空间偏度在不同年份和阈值下多为正值或接近零（例如，2016年多数在0.1到0.3之间），表明其空间分布可能略微右偏或接近对称。
        *   **峰度：** CSI 的空间峰度值多数情况下为负值（通常在-0.6到-0.0之间），表明其空间分布通常比正态分布更平坦。
    *   **年际变化：**
        *   以 0.1mm/d 阈值为例，平均空间 CSI 在 2016 年 (0.5389), 2017 年 (0.5156), 2018 年 (0.5346), 2019 年 (0.5067), 2020 年 (0.5343)。2016年和2018/2020年表现相对较好，2019年表现最差。

*   **10.2.6.4 产品间性能指标空间相关性 (基于每年 0.95mm/d 阈值)**
    *   **回顾与其他产品的关系：** 与CMORPH的POD、FAR、CSI空间相关性均较高。与CHIRPS的FAR空间相关性极高，POD、CSI相关性较高。与SM2RAIN的FAR空间相关性高，但POD相关性极低，CSI相关性中等。与IMERG的FAR空间相关性极高，POD、CSI相关性较高。与GSMAP的FAR空间相关性极高，POD相关性较低，CSI相关性中高。
    *   **总结：** PERSIANN 在 FAR 的空间分布模式上与所有其他产品都表现出高度一致性。在 POD 和 CSI 方面，与 CMORPH, CHIRPS, IMERG 的空间分布模式都有较高程度的相似性。与 GSMAP 的 POD 相似性较低，但 CSI 相似性尚可。与 SM2RAIN 的 POD 相似性极低，但 CSI 相似性中等。

*   **10.2.6.5 PERSIANN 年度表现小结**
    *   **整体性能：** PERSIANN 是一款性能表现中规中矩的产品，其平均 POD 和 CSI 水平与 CMORPH 接近。FAR 控制良好，误报也呈现局部化特征。其 POD 的空间变异性相对较大。
    *   **年际差异：** PERSIANN 的性能在年际间存在一定波动，2016年整体表现稍好，而2019年相对偏弱。FAR逐年有微弱改善。
    *   **空间特征：** POD 的空间一致性较差，是其一个弱点。CSI 空间分布相对平坦。FAR 的误报同样高度集中。
    *   **与其他产品关系：** 其误报高发区的空间位置与其他产品高度相似；在综合性能的空间好坏区域分布上，与多数传统卫星遥感产品有较强的同步性。

**10.3 跨产品横向比较与综合评估**

在对各产品进行逐年逐阈值分析的基础上，本节进行跨产品的横向比较，旨在更清晰地揭示各产品间的相对性能优劣、共性与特性。

*   **10.3.1 总体性能均值对比 (基于多年平均)**

    为了更直观地比较各产品在不同降雨强度下的平均表现，我们选取关键阈值（例如，代表小雨的 0.1 mm/d，代表中等强度关注点的 0.5 mm/d，以及接近常用评估上限的 0.95 mm/d），对各产品在 2016-2020 年间的平均空间 POD, FAR, CSI 进行比较。*(注：由于原始文本未直接给出多年平均值，此处基于对各年数据的观察进行定性总结，后续可补充精确计算值)*

    *   **POD (命中率)：**
        *   **SM2RAIN：** 在所有阈值下，其多年平均空间 POD 均显著高于其他所有产品，通常在 0.85 以上，展现出在“有无降雨”判断上的绝对优势。
        *   **IMERG：** 多年平均空间 POD 表现优异，通常在 0.60-0.75 范围，仅次于 SM2RAIN。
        *   **GSMAP：** 其 POD 随阈值增加而上升的特性使其在高阈值下（如0.95mm/d）的 POD 表现突出，多年平均可能接近或略高于 IMERG 在该阈值下的表现。在低阈值下表现中等偏上。
        *   **CMORPH & PERSIANN：** 两者 POD 水平相当，属于中等水平，通常在 0.50-0.68 范围波动，PERSIANN 某些年份在特定阈值下略高。
        *   **CHIRPS：** 多年平均空间 POD 相对最低，通常在 0.38-0.50 范围。
    *   **FAR (空报率)：**
        *   **GSMAP：** 多年平均空间 FAR 在所有阈值下均表现**最优异**，始终维持在极低水平 (约 0.002-0.015)。
        *   **CMORPH, CHIRPS, IMERG, PERSIANN：** 这四类产品的多年平均空间 FAR 水平相当，均控制在较低水平 (约 0.01-0.03)，略高于 GSMAP，但整体表现良好。它们之间的细微差异不显著。
        *   **SM2RAIN：** 多年平均空间 FAR 略高于其他产品 (约 0.015-0.035)，但考虑到其极高的 POD，这种略高的 FAR 在某些应用场景下可能是可接受的。
    *   **CSI (临界成功指数)：**
        *   **SM2RAIN：** 在**极低阈值 (如 0.0-0.1 mm/d)** 下，其多年平均空间 CSI 表现最佳，显著优于其他产品，通常可达 0.60-0.70+。
        *   **GSMAP & IMERG：** 在**中高阈值 (如 0.5 mm/d 及以上)** 下，这两款产品的多年平均空间 CSI 表现最为出色，通常在 0.50-0.65 范围，GSMAP 可能因其极低的 FAR 而在综合评分上略占优势或在高阈值下更稳定。
        *   **CMORPH & PERSIANN：** 两者的多年平均空间 CSI 水平相当，属于中等水平，通常在 0.45-0.55 范围。
        *   **CHIRPS：** 多年平均空间 CSI 相对最低，通常在 0.35-0.45 范围。

*   **10.3.2 空间稳定性对比 (基于多年平均的空间标准差)**

    评估产品性能在全国范围内的空间一致性。空间标准差越小，表明产品在不同地区的表现越均一。

    *   **POD 空间标准差：**
        *   **GSMAP & CHIRPS：** 这两款产品的 POD 空间标准差多年平均表现最小（约 0.05-0.07），表明其命中率在空间分布上最为均匀。
        *   **IMERG & SM2RAIN：** POD 空间标准差处于中等水平（约 0.07-0.09）。
        *   **CMORPH & PERSIANN：** POD 空间标准差相对较大（约 0.07-0.14，PERSIANN波动更大），表明其命中率在空间上的变异性较大。
    *   **FAR 空间标准差：**
        *   **GSMAP：** FAR 空间标准差最小（约 0.007-0.05），再次印证其误报控制的优异性和空间一致性。
        *   **CMORPH, CHIRPS, IMERG, PERSIANN, SM2RAIN：** 其他产品的 FAR 空间标准差处于相似的较高水平（约 0.03-0.12），表明它们的误报（尽管中位数都为0）在空间上的离散程度相似，即少数高误报区域的离群程度相似。
    *   **CSI 空间标准差：**
        *   **GSMAP, CHIRPS, IMERG：** 这三款产品的 CSI 空间标准差多年平均表现较小（约 0.05-0.07），表明其综合评分在空间上较为均一。
        *   **SM2RAIN：** CSI 空间标准差略高于上述三者（约 0.05-0.08）。
        *   **CMORPH & PERSIANN：** CSI 空间标准差相对较大（约 0.06-0.09，PERSIANN波动更大），与它们 POD 的空间变异性一致。

*   **10.3.3 误报 (FAR) 特征深度对比**

    误报是本项目关注的重点。

    *   **FAR 均值与中位数：**
        *   **GSMAP** 的 FAR 均值显著低于所有其他产品。
        *   所有产品的 FAR **中位数常年为 0**，这表明误报问题在空间上是高度集中的，而非普遍现象。
    *   **FAR 空间偏度和峰度：**
        *   **所有产品**的 FAR 空间分布均呈现**极高的正偏度（右偏）和极高的正峰度（尖峰厚尾）**。这意味着：
            1.  绝大多数地区的误报率非常低（接近于零）。
            2.  存在少数“热点”区域，其误报率远高于平均水平，这些区域是导致整体平均 FAR 上升的主要原因，也是未来误差分析和模型优化的关键靶区。
        *   **CMORPH** 和 **CHIRPS** 的 FAR 偏度和峰度值通常是最高的，表明其误报可能在空间上更为集中于极少数区域，或者这些区域的误报程度更为极端。
        *   **GSMAP** 虽然整体 FAR 最低，但其误报同样呈现高度集中的空间格局。
        *   理解这种空间分布特性对于设计针对性的误报修正策略至关重要。

*   **10.3.4 年际稳定性综合对比**

    评估各产品关键性能指标在 2016-2020 年间的波动幅度。

    *   **POD 均值：** SM2RAIN 的年际稳定性最好；其次是 IMERG, GSMAP, CMORPH；CHIRPS 和 PERSIANN 的年际波动相对较大。
    *   **FAR 均值：** 所有产品的 FAR 均值在年际间都表现出非常高的一致性和稳定性。
    *   **CSI 均值：** SM2RAIN, IMERG, GSMAP 的年际稳定性较好；CMORPH, CHIRPS, PERSIANN 的年际波动相对更明显一些。
    *   **结论：** 在误报控制方面，所有产品都表现出良好的年际稳定性。在命中能力和综合评分方面，SM2RAIN、IMERG 和 GSMAP 的表现更为稳健。

*   **10.3.5 产品间性能空间分布相关性格局**

    总结不同产品在性能指标空间分布模式上的相似性。

    *   **FAR 的空间分布模式高度一致：** 这是一个非常重要的发现。所有六种降雨产品，尽管其物理原理和算法各不相同，但在**误报事件高发的空间区域上表现出惊人的一致性**（FAR指标间的空间相关系数普遍在0.88以上，很多接近0.95）。这强烈暗示这些误报可能与某些共同的、难以准确观测或反演的地理环境因素、特定天气系统或遥感探测的固有局限性有关。
    *   **POD 和 CSI 的空间分布模式：**
        *   **传统卫星产品组 (CMORPH, CHIRPS, IMERG, GSMAP, PERSIANN)：** 这些产品在 POD 和 CSI 的空间分布模式上表现出中高到高度的相关性。这意味着它们往往在相似的区域表现良好，也在相似的区域表现较差。IMERG 通常与其他产品有较高的相关性，可以看作是这个组的一个性能较好的代表。
        *   **SM2RAIN 的独特性：** SM2RAIN 在 POD 和 CSI 的空间分布模式上与其他所有产品均表现出较低的相关性。这再次突显了其基于土壤湿度反演的独特性，其性能好的区域和差的区域可能与其他卫星产品有显著不同。


**10.4 分析总结与关键洞察**

综合以上对各降雨产品逐年逐阈值空间性能统计特征的深度分析以及跨产品的横向比较，可以总结出以下关键规律、洞察和对本项目研究的启示：

*   **10.4.1 各类产品性能指标随降雨阈值变化的普遍规律与独特模式**
    *   **POD (命中率)：**
        *   **普遍规律：** 大多数产品 (CMORPH, CHIRPS, IMERG, PERSIANN) 的 POD 在极低阈值下略微上升后，随阈值增加而平稳或下降。这可能是因为极小阈值包含了更多噪声或微量降水，略微提高阈值有助于识别更确切的降雨事件。
        *   **独特模式：**
            *   **SM2RAIN** 的 POD 始终维持在极高水平，并随阈值增加缓慢下降，显示其对“有无降雨”的判断非常敏感。
            *   **GSMAP** 的 POD 随阈值增加反而呈现缓慢上升或在高位持平的趋势，这表明其在探测中高强度降雨事件方面具有相对优势或不同的响应机制。
    *   **FAR (空报率)：**
        *   **普遍规律：** 所有产品的平均空间 FAR 都随阈值的增加而缓慢上升。这可能是因为随着阈值提高，真实降雨事件的样本量减少，使得少量误报事件在比例计算中更为突出。
        *   **关键特征：** 所有产品的 FAR 空间中位数几乎恒为0，且其空间分布均呈现极高的正偏度和高峰度。这揭示了一个核心现象：**误报并非均匀随机发生，而是高度集中在特定的少数“热点”区域。**
    *   **CSI (临界成功指数)：**
        *   **普遍规律：** 大多数产品的 CSI 在中低阈值达到峰值后，随阈值增加而下降，这反映了 POD 下降在高阈值区间的主导作用。
        *   **独特模式：**
            *   **SM2RAIN** 在极低阈值下的 CSI 表现最佳。
            *   **GSMAP** 的 CSI 在中高阈值下表现稳健，下降趋势平缓，甚至在某些年份（如2020）有所上升，这得益于其持续上升的 POD 和极低的 FAR。

*   **10.4.2 不同产品在命中率、误报控制和综合评分上的优势与劣势区间**
    *   **SM2RAIN：**
        *   **优势：** 极高的 POD（尤其擅长判断有无降雨），在小雨量级（低阈值）下拥有最高的 CSI。年际稳定性极好。
        *   **劣势：** FAR 相对略高于其他产品（尽管仍属低水平）；其 POD 和 CSI 的空间分布模式与其他产品差异大。
    *   **GSMAP：**
        *   **优势：** **FAR 控制最佳，误报率最低**；POD 随阈值增加而上升；CSI 在中高阈值下表现优异且稳定；POD 和 CSI 的空间一致性好。
        *   **劣势：** 在极低阈值下的 POD 和 CSI 不如 SM2RAIN。
    *   **IMERG：**
        *   **优势：** 综合性能强且稳定，POD 和 CSI 均处于较高水平；FAR 控制良好；各项指标的年际稳定性好，空间一致性也不错。
        *   **劣势：** 各项指标均非最优，但没有明显短板。
    *   **CMORPH & PERSIANN：**
        *   **优势：** FAR 控制良好；POD 和 CSI 属于中等水平。
        *   **劣势：** POD 和 CSI 的空间变异性相对较大；年际稳定性一般。PERSIANN 的 POD 空间变异性尤其突出。
    *   **CHIRPS：**
        *   **优势：** FAR 控制良好；POD 和 CSI 的空间一致性好。
        *   **劣势：** POD 和 CSI 的整体水平均相对较低。

*   **10.4.3 产品性能在年际间的共性波动与特性差异，可能的气候影响因素探讨（初步）**
    *   **共性波动：** 虽然各产品有其固有性能水平，但在某些年份，如 2020 年，多数产品（如 CHIRPS, GSMAP, IMERG）的 POD 和/或 CSI 表现均有不同程度的提升，尤其是在较高阈值下。这可能与当年的整体降雨特征（如大范围强降水事件增多）有关，使得产品的探测能力得到更好发挥。相反，在某些年份（如2017或2019年对某些产品而言），性能可能相对偏弱。
    *   **特性差异：** SM2RAIN 和 IMERG 展现出非常好的年际稳定性。而 CHIRPS 和 PERSIANN 的年际波动相对更明显。
    *   **气候影响探讨（初步）：** 年际间的气候背景（如 ENSO 事件、季风强度异常等）可能通过改变降雨类型、强度、空间分布等，从而影响各卫星产品的遥感物理机制和反演算法的适用性，导致性能的年际波动。例如，某年若小雨、层云降水偏多，可能对某些产品的探测构成挑战。

*   **10.4.4 空间异质性表现：哪些产品在全国范围内表现更稳定，哪些产品性能空间差异大**
    *   **空间表现更稳定 (低空间标准差)：**
        *   **GSMAP** 和 **CHIRPS** 在 POD 和 CSI 的空间一致性上表现突出。
        *   **IMERG** 在 CSI 的空间一致性上也表现良好。
        *   所有产品在 FAR 中位数上表现出高度的空间一致性（均为0）。
    *   **空间差异大 (高空间标准差)：**
        *   **CMORPH** 和 **PERSIANN** (尤其是 PERSIANN) 在 POD 和 CSI 上的空间标准差相对较大，表明其性能在不同区域差异显著。
        *   尽管所有产品的 FAR 中位数均为0，但其 FAR 的空间标准差、偏度和峰度均较高，表明少数高误报区域的“离群”程度是普遍存在的。

*   **10.4.5 对特征工程和模型融合策略的启示**
    *   **特征工程：**
        *   **误报热点区域特征挖掘：** 鉴于所有产品 FAR 空间分布的高度相关性和局部集中性，可以重点针对这些误报高发区域，挖掘其独特的地理、气象或产品自身特征（例如，特定地形、特定季节的特定天气系统影响、产品在这些区域的原始反演质量指标等），以期在模型中学习并修正这些系统性偏差。
        *   **产品独特性特征：** SM2RAIN 的性能模式与其他产品差异显著，其数据或基于其数据衍生的特征（如土壤湿度相关的特征）可能为融合模型提供独特的、其他产品难以捕捉的信息，尤其是在小雨和降雨启动阶段。
        *   **阈值依赖特征：** 不同产品在不同降雨强度（阈值）下的表现各异，可以考虑设计随降雨强度变化的特征权重，或构建针对不同强度区间的子模型。
    *   **模型融合：**
        *   **优势互补：**
            *   可考虑在小雨量级（低阈值）预测时赋予 SM2RAIN 更高权重。
            *   在需要严格控制误报的应用中，GSMAP 的权重应较高。
            *   IMERG 作为综合性能稳健的产品，可作为融合模型的坚实基础。
        *   **空间适应性融合：** 考虑到 CMORPH 和 PERSIANN 性能的空间变异性较大，如果计算资源允许，可以探索分区建模或地理加权融合策略，即在不同区域赋予不同产品组合或权重。
        *   **FAR 修正模型：** 鉴于 FAR 的高度局部化，可以考虑训练一个专门的“误报识别与修正”子模型（类似项目中已探索的 FP 专家模型），该模型可以利用所有产品在误报高发区的共性特征进行学习。

*   **10.4.6 未来研究方向建议**
    *   **误报/漏报成因的精细化地理诊断：** 结合高分辨率地形、土地利用、气候分区以及典型天气过程分析，深入探究导致各产品（尤其是共性）误报和漏报高发区的具体物理机制。
    *   **极端降雨事件的评估：** 本分析主要基于0-1mm/d的阈值，未来可扩展到更高阈值，专门评估各产品对暴雨等极端降雨事件的探测能力和空间特征。
    *   **不同季节性能差异：** 分析各产品在不同季节（如汛期 vs. 非汛期，夏季 vs. 冬季）的性能表现差异。
    *   **动态融合策略研究：** 基于实时气象条件、产品自身质量控制信息等，开发动态调整各产品融合权重的先进算法。
    *   **不确定性量化：** 不仅评估确定性预报性能，还需关注各产品及融合结果的不确定性量化。
