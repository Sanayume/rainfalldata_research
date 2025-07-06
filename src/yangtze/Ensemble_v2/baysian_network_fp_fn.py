#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯网络FP/FN专家模型训练脚本
===================================

基于气象学原理和V6特征集，构建两个专门的贝叶斯网络：
1. FP专家网络：专注于识别和预防误报事件
2. FN专家网络：专注于识别和预防漏报事件

Author: Claude & User
Date: 2025-07-05
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import optuna
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# 贝叶斯网络相关库
try:
    # 适配pgmpy版本差异
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
        print("✅ 使用DiscreteBayesianNetwork (pgmpy 1.0+)")
    except ImportError:
        from pgmpy.models import BayesianNetwork
        print("✅ 使用BayesianNetwork (pgmpy 0.x)")
    
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.factors.discrete import TabularCPD
    print("✅ pgmpy库导入成功")
except ImportError:
    print("警告：未安装pgmpy库，将尝试安装...")
    os.system("pip install pgmpy")
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.factors.discrete import TabularCPD

# 数据处理和评估
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           matthews_corrcoef, balanced_accuracy_score)
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

class BayesianNetworkFPFNExpert:
    """
    贝叶斯网络FP/FN专家模型类
    """
    
    def __init__(self, expert_type: str, random_state: int = 42):
        """
        初始化贝叶斯网络专家
        
        Args:
            expert_type: 'FP' 或 'FN'
            random_state: 随机种子
        """
        self.expert_type = expert_type.upper()
        self.random_state = random_state
        self.model = None
        self.inference = None
        self.feature_bins = {}
        self.discretizer = None
        self.feature_names = []
        
        # 预定义网络结构
        self.network_structure = self._design_network_structure()
        
        np.random.seed(random_state)
        
    def _design_network_structure(self) -> Dict[str, List[str]]:
        """
        基于气象学原理设计增强版网络结构 - 更复杂的层次化架构
        """
        if self.expert_type == 'FP':
            # FP专家网络：深层次结构关注产品分歧、不确定性传播、时空模式
            structure = {
                # ===== 第一层：基础观测特征 =====
                'multi_product_std': [],  # 产品间标准差（根节点）
                'multi_product_mean': [],  # 产品间均值（根节点）
                'coef_of_variation': [],  # 变异系数（根节点）
                'sin_day': [],  # 日周期（独立根节点）
                'season_spring': [],  # 季节信号（独立根节点）
                
                # ===== 第二层：不确定性量化层 =====
                'product_range': ['multi_product_std'],  # 产品极差依赖于标准差
                'rain_product_count': ['multi_product_std', 'multi_product_mean'],  # 产品数依赖于不确定性
                'low_intensity_std': ['multi_product_std', 'coef_of_variation'],  # 低强度不确定性
                'product_uncertainty': ['multi_product_std', 'product_range'],  # 复合不确定性指标
                
                # ===== 第三层：时序记忆层 =====
                'lag_1_std': ['multi_product_std'],  # 滞后标准差
                'lag_1_mean': ['multi_product_mean'],  # 滞后均值
                'diff_multi_product_mean': ['multi_product_mean', 'lag_1_mean'],  # 变化率
                'temporal_consistency': ['lag_1_std', 'multi_product_std'],  # 时序一致性
                
                # ===== 第四层：窗口统计层 =====
                'window_3d_mean': ['multi_product_mean', 'lag_1_mean'],  # 3天窗口依赖于当前和历史
                'window_7d_std': ['multi_product_std', 'temporal_consistency'],  # 7天窗口依赖于不确定性
                'window_stability': ['window_3d_mean', 'window_7d_std'],  # 窗口稳定性
                
                # ===== 第五层：交互与模式层 =====
                'std_diurnal_interaction': ['multi_product_std', 'sin_day'],  # 不确定性-日周期交互
                'uncertainty_seasonal': ['product_uncertainty', 'season_spring'],  # 不确定性-季节交互
                'low_std_cv_interaction': ['low_intensity_std', 'coef_of_variation'],  # 低强度交互
                'temporal_seasonal_pattern': ['temporal_consistency', 'season_spring'],  # 时序-季节模式
                
                # ===== 第六层：高级特征融合层 =====
                'signal_clarity': ['rain_product_count', 'window_stability'],  # 信号清晰度
                'prediction_confidence': ['signal_clarity', 'uncertainty_seasonal'],  # 预测置信度
                'false_signal_risk': ['std_diurnal_interaction', 'low_std_cv_interaction'],  # 假信号风险
                'meteorological_plausibility': ['prediction_confidence', 'temporal_seasonal_pattern'],  # 气象合理性
                
                # ===== 第七层：决策层 =====
                'high_uncertainty_flag': ['false_signal_risk', 'product_uncertainty'],  # 高不确定性标志
                'temporal_inconsistency_flag': ['diff_multi_product_mean', 'temporal_consistency'],  # 时序不一致标志
                'weak_signal_flag': ['signal_clarity', 'low_intensity_std'],  # 弱信号标志
                
                # ===== 第八层：最终输出层 =====
                'is_fp': ['high_uncertainty_flag', 'meteorological_plausibility', 
                         'temporal_inconsistency_flag', 'weak_signal_flag']  # FP概率
            }
            
        else:  # FN专家网络
            # FN专家网络：深层次结构关注强信号遗漏、系统性偏差、极值检测
            structure = {
                # ===== 第一层：基础强度特征 =====
                'multi_product_max': [],  # 产品最大值（根节点）
                'multi_product_mean': [],  # 产品均值（根节点）
                'multi_product_std': [],  # 产品标准差（根节点）
                'sin_day': [],  # 日周期（独立根节点）
                'season_spring': [],  # 季节信号（独立根节点）
                
                # ===== 第二层：强度量化层 =====
                'product_range': ['multi_product_max', 'multi_product_mean'],  # 极差依赖于最值和均值
                'rain_product_count': ['multi_product_mean', 'multi_product_max'],  # 产品数依赖于强度
                'intensity_signal_strength': ['multi_product_max', 'product_range'],  # 强度信号强度
                'agreement_strength': ['rain_product_count', 'multi_product_std'],  # 产品一致性强度
                
                # ===== 第三层：时序动力层 =====
                'lag_1_mean': ['multi_product_mean'],  # 滞后均值
                'lag_2_mean': ['lag_1_mean'],  # 更深层滞后
                'lag_1_max': ['multi_product_max'],  # 滞后最大值
                'diff_multi_product_mean': ['multi_product_mean', 'lag_1_mean'],  # 变化率
                'acceleration': ['diff_multi_product_mean', 'lag_1_mean'],  # 加速度（二阶变化）
                
                # ===== 第四层：累积与持续性层 =====
                'window_3d_mean': ['multi_product_mean', 'lag_1_mean'],  # 3天窗口
                'window_7d_std': ['multi_product_std', 'lag_1_mean'],  # 7天变异
                'persistence_indicator': ['window_3d_mean', 'lag_2_mean'],  # 持续性指标
                'intensity_trend': ['acceleration', 'window_3d_mean'],  # 强度趋势
                
                # ===== 第五层：环境与模式层 =====
                'max_diurnal_interaction': ['multi_product_max', 'sin_day'],  # 最大值-日周期交互
                'intensity_seasonal': ['intensity_signal_strength', 'season_spring'],  # 强度-季节交互
                'persistence_seasonal': ['persistence_indicator', 'season_spring'],  # 持续性-季节模式
                'trend_diurnal_coupling': ['intensity_trend', 'sin_day'],  # 趋势-日周期耦合
                
                # ===== 第六层：预测能力评估层 =====
                'strong_signal_clarity': ['agreement_strength', 'persistence_indicator'],  # 强信号清晰度
                'predictability_index': ['strong_signal_clarity', 'intensity_seasonal'],  # 可预测性指数
                'systematic_bias_risk': ['max_diurnal_interaction', 'trend_diurnal_coupling'],  # 系统性偏差风险
                'extreme_event_potential': ['intensity_signal_strength', 'acceleration'],  # 极端事件潜力
                
                # ===== 第七层：风险识别层 =====
                'underestimation_risk': ['predictability_index', 'systematic_bias_risk'],  # 低估风险
                'strong_signal_missed': ['extreme_event_potential', 'strong_signal_clarity'],  # 强信号遗漏
                'persistence_bias': ['persistence_seasonal', 'trend_diurnal_coupling'],  # 持续性偏差
                'intensity_threshold_bias': ['intensity_seasonal', 'agreement_strength'],  # 强度阈值偏差
                
                # ===== 第八层：最终输出层 =====
                'is_fn': ['underestimation_risk', 'strong_signal_missed', 
                         'persistence_bias', 'intensity_threshold_bias']  # FN概率
            }
            
        return structure
    
    def _select_and_map_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        选择并映射V6特征集到网络节点，并计算增强版网络所需的复合特征
        """
        # V6特征集到网络节点的基础映射
        feature_mapping = {
            # 基础多产品特征
            'multi_product_std': 'multi_product_std',
            'multi_product_mean': 'multi_product_mean', 
            'multi_product_max': 'multi_product_max',
            'multi_product_range': 'product_range',
            'rain_product_count': 'rain_product_count',
            'coef_of_variation': 'coef_of_variation',
            
            # 时序特征
            'sin_day': 'sin_day',
            'lag_1_multi_product_mean': 'lag_1_mean',
            'lag_1_multi_product_std': 'lag_1_std', 
            'lag_1_multi_product_max': 'lag_1_max',
            'lag_2_multi_product_mean': 'lag_2_mean',
            'diff_multi_product_mean': 'diff_multi_product_mean',
            
            # 弱信号特征
            'low_intensity_std': 'low_intensity_std',
            
            # 窗口特征
            'window_3d_mean_multi_product_mean': 'window_3d_mean',
            'window_7d_std_multi_product_mean': 'window_7d_std',
            
            # 季节特征
            'season_spring': 'season_spring',
        }
        
        # 选择可用特征
        selected_features = {}
        
        for network_node, v6_feature in feature_mapping.items():
            if v6_feature in X.columns:
                selected_features[network_node] = X[v6_feature]
            else:
                # 寻找近似特征
                similar_features = [col for col in X.columns if v6_feature.lower() in col.lower()]
                if similar_features:
                    selected_features[network_node] = X[similar_features[0]]
                    logger.info(f"使用近似特征 {similar_features[0]} 代替 {v6_feature}")
        
        # 计算增强版网络所需的复合特征
        logger.info("计算增强版网络的复合特征...")
        
        # 为FP专家计算特殊复合特征
        if self.expert_type == 'FP':
            # 第二层复合特征
            if 'multi_product_std' in selected_features and 'product_range' in selected_features:
                selected_features['product_uncertainty'] = selected_features['multi_product_std'] * selected_features['product_range']
            
            # 第三层复合特征
            if 'lag_1_std' in selected_features and 'multi_product_std' in selected_features:
                selected_features['temporal_consistency'] = 1.0 / (1.0 + np.abs(selected_features['lag_1_std'] - selected_features['multi_product_std']))
            
            # 第四层复合特征
            if 'window_3d_mean' in selected_features and 'window_7d_std' in selected_features:
                selected_features['window_stability'] = selected_features['window_3d_mean'] / (1.0 + selected_features['window_7d_std'])
            
            # 第五层交互特征
            if 'multi_product_std' in selected_features and 'sin_day' in selected_features:
                selected_features['std_diurnal_interaction'] = selected_features['multi_product_std'] * selected_features['sin_day']
            
            if 'product_uncertainty' in selected_features and 'season_spring' in selected_features:
                selected_features['uncertainty_seasonal'] = selected_features['product_uncertainty'] * selected_features['season_spring']
                
            if 'low_intensity_std' in selected_features and 'coef_of_variation' in selected_features:
                selected_features['low_std_cv_interaction'] = selected_features['low_intensity_std'] * selected_features['coef_of_variation']
                
            if 'temporal_consistency' in selected_features and 'season_spring' in selected_features:
                selected_features['temporal_seasonal_pattern'] = selected_features['temporal_consistency'] * selected_features['season_spring']
            
            # 第六层融合特征
            if 'rain_product_count' in selected_features and 'window_stability' in selected_features:
                selected_features['signal_clarity'] = selected_features['rain_product_count'] * selected_features['window_stability']
                
            if 'signal_clarity' in selected_features and 'uncertainty_seasonal' in selected_features:
                selected_features['prediction_confidence'] = selected_features['signal_clarity'] / (1.0 + selected_features['uncertainty_seasonal'])
                
            if 'std_diurnal_interaction' in selected_features and 'low_std_cv_interaction' in selected_features:
                selected_features['false_signal_risk'] = selected_features['std_diurnal_interaction'] + selected_features['low_std_cv_interaction']
                
            if 'prediction_confidence' in selected_features and 'temporal_seasonal_pattern' in selected_features:
                selected_features['meteorological_plausibility'] = selected_features['prediction_confidence'] * selected_features['temporal_seasonal_pattern']
            
            # 第七层决策特征
            if 'false_signal_risk' in selected_features and 'product_uncertainty' in selected_features:
                selected_features['high_uncertainty_flag'] = (selected_features['false_signal_risk'] > selected_features['false_signal_risk'].quantile(0.75)).astype(float)
                
            if 'diff_multi_product_mean' in selected_features and 'temporal_consistency' in selected_features:
                selected_features['temporal_inconsistency_flag'] = (selected_features['diff_multi_product_mean'] > selected_features['diff_multi_product_mean'].quantile(0.8)).astype(float)
                
            if 'signal_clarity' in selected_features and 'low_intensity_std' in selected_features:
                selected_features['weak_signal_flag'] = (selected_features['signal_clarity'] < selected_features['signal_clarity'].quantile(0.25)).astype(float)
                
        # 为FN专家计算特殊复合特征
        elif self.expert_type == 'FN':
            # 第二层复合特征
            if 'multi_product_max' in selected_features and 'product_range' in selected_features:
                selected_features['intensity_signal_strength'] = selected_features['multi_product_max'] * selected_features['product_range']
                
            if 'rain_product_count' in selected_features and 'multi_product_std' in selected_features:
                selected_features['agreement_strength'] = selected_features['rain_product_count'] / (1.0 + selected_features['multi_product_std'])
            
            # 第三层复合特征
            if 'diff_multi_product_mean' in selected_features and 'lag_1_mean' in selected_features:
                # 计算加速度（二阶差分）
                selected_features['acceleration'] = selected_features['diff_multi_product_mean'] - selected_features['lag_1_mean'].shift(1).fillna(0)
            
            # 第四层复合特征
            if 'window_3d_mean' in selected_features and 'lag_2_mean' in selected_features:
                selected_features['persistence_indicator'] = selected_features['window_3d_mean'] / (1.0 + np.abs(selected_features['lag_2_mean']))
                
            if 'acceleration' in selected_features and 'window_3d_mean' in selected_features:
                selected_features['intensity_trend'] = selected_features['acceleration'] * selected_features['window_3d_mean']
            
            # 第五层交互特征
            if 'multi_product_max' in selected_features and 'sin_day' in selected_features:
                selected_features['max_diurnal_interaction'] = selected_features['multi_product_max'] * selected_features['sin_day']
                
            if 'intensity_signal_strength' in selected_features and 'season_spring' in selected_features:
                selected_features['intensity_seasonal'] = selected_features['intensity_signal_strength'] * selected_features['season_spring']
                
            if 'persistence_indicator' in selected_features and 'season_spring' in selected_features:
                selected_features['persistence_seasonal'] = selected_features['persistence_indicator'] * selected_features['season_spring']
                
            if 'intensity_trend' in selected_features and 'sin_day' in selected_features:
                selected_features['trend_diurnal_coupling'] = selected_features['intensity_trend'] * selected_features['sin_day']
            
            # 第六层评估特征
            if 'agreement_strength' in selected_features and 'persistence_indicator' in selected_features:
                selected_features['strong_signal_clarity'] = selected_features['agreement_strength'] * selected_features['persistence_indicator']
                
            if 'strong_signal_clarity' in selected_features and 'intensity_seasonal' in selected_features:
                selected_features['predictability_index'] = selected_features['strong_signal_clarity'] + selected_features['intensity_seasonal']
                
            if 'max_diurnal_interaction' in selected_features and 'trend_diurnal_coupling' in selected_features:
                selected_features['systematic_bias_risk'] = selected_features['max_diurnal_interaction'] + selected_features['trend_diurnal_coupling']
                
            if 'intensity_signal_strength' in selected_features and 'acceleration' in selected_features:
                selected_features['extreme_event_potential'] = selected_features['intensity_signal_strength'] * selected_features['acceleration']
            
            # 第七层风险特征
            if 'predictability_index' in selected_features and 'systematic_bias_risk' in selected_features:
                selected_features['underestimation_risk'] = selected_features['predictability_index'] - selected_features['systematic_bias_risk']
                
            if 'extreme_event_potential' in selected_features and 'strong_signal_clarity' in selected_features:
                selected_features['strong_signal_missed'] = (selected_features['extreme_event_potential'] > selected_features['extreme_event_potential'].quantile(0.8)).astype(float)
                
            if 'persistence_seasonal' in selected_features and 'trend_diurnal_coupling' in selected_features:
                selected_features['persistence_bias'] = selected_features['persistence_seasonal'] - selected_features['trend_diurnal_coupling']
                
            if 'intensity_seasonal' in selected_features and 'agreement_strength' in selected_features:
                selected_features['intensity_threshold_bias'] = selected_features['intensity_seasonal'] / (1.0 + selected_features['agreement_strength'])
        
        # 创建网络特征DataFrame
        network_features = pd.DataFrame(selected_features)
        
        # 只保留网络结构中定义的特征
        structure_features = set()
        for node, parents in self.network_structure.items():
            structure_features.add(node)
            structure_features.update(parents)
        
        # 过滤可用特征
        available_features = [f for f in structure_features if f in network_features.columns]
        network_features = network_features[available_features]
        
        logger.info(f"{self.expert_type}专家选择了 {len(available_features)} 个特征: {available_features[:10]}{'...' if len(available_features) > 10 else ''}")
        
        return network_features
    
    def _discretize_features(self, X: pd.DataFrame, n_bins: int = 3) -> pd.DataFrame:
        """
        将连续特征离散化为分类变量
        """
        if self.discretizer is None:
            self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
            X_discretized = self.discretizer.fit_transform(X)
        else:
            X_discretized = self.discretizer.transform(X)
        
        # 转换为整数标签
        X_discretized = X_discretized.astype(int)
        
        # 创建DataFrame
        X_discrete = pd.DataFrame(X_discretized, columns=X.columns, index=X.index)
        
        # 记录分箱信息
        for i, feature in enumerate(X.columns):
            if feature not in self.feature_bins:
                bin_edges = self.discretizer.bin_edges_[i]
                self.feature_bins[feature] = {
                    'edges': bin_edges,
                    'labels': [f'{feature}_bin_{j}' for j in range(n_bins)]
                }
        
        return X_discrete
    
    def _balance_data(self, X: pd.DataFrame, y: np.ndarray, 
                     strategy: str = 'hybrid', positive_ratio: float = 0.25) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        平衡FP/FN样本 - 修复版
        """
        unique, counts = np.unique(y, return_counts=True)
        logger.info(f"原始样本分布: {dict(zip(unique, counts))}")
        
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        
        if strategy == 'hybrid':
            # 混合策略：适度处理，避免过度采样
            max_total_samples = min(1000000, len(y))  # 限制最大样本数
            target_n_positive = min(n_positive, int(max_total_samples * positive_ratio))
            target_n_negative = max_total_samples - target_n_positive
            
            # 确保不超过原有样本数
            target_n_negative = min(target_n_negative, n_negative)
            
            # 如果正样本不够，进行有限的过采样
            if target_n_positive > n_positive:
                # 计算需要重复的倍数，但限制在合理范围内
                repeat_factor = min(3, target_n_positive // n_positive)  # 最多重复3倍
                target_n_positive = n_positive * repeat_factor
                
                # 重新调整负样本数
                target_n_negative = min(target_n_negative, 
                                      int(target_n_positive * (1 - positive_ratio) / positive_ratio))
            
            # 下采样负样本
            neg_indices = np.where(y == 0)[0]
            pos_indices = np.where(y == 1)[0]
            
            np.random.seed(self.random_state)
            selected_neg_indices = np.random.choice(neg_indices, size=target_n_negative, replace=False)
            
            # 处理正样本
            if target_n_positive <= n_positive:
                selected_pos_indices = np.random.choice(pos_indices, size=target_n_positive, replace=False)
            else:
                # 重复采样正样本
                n_repeats = target_n_positive // n_positive
                n_remainder = target_n_positive % n_positive
                selected_pos_indices = np.tile(pos_indices, n_repeats)
                if n_remainder > 0:
                    additional = np.random.choice(pos_indices, size=n_remainder, replace=False)
                    selected_pos_indices = np.concatenate([selected_pos_indices, additional])
            
            # 合并索引
            balanced_indices = np.concatenate([selected_pos_indices, selected_neg_indices])
            
        elif strategy == 'undersample':
            # 纯下采样策略：只减少多数类
            target_n_negative = min(n_negative, int(n_positive * (1 - positive_ratio) / positive_ratio))
            
            neg_indices = np.where(y == 0)[0]
            pos_indices = np.where(y == 1)[0]
            
            np.random.seed(self.random_state)
            selected_neg_indices = np.random.choice(neg_indices, size=target_n_negative, replace=False)
            
            balanced_indices = np.concatenate([pos_indices, selected_neg_indices])
            
        elif strategy == 'oversample':
            # 纯过采样策略：只增加少数类，但有限制
            max_positive_multiplier = 5  # 最多增加5倍正样本
            target_n_positive = min(n_positive * max_positive_multiplier, 
                                   int(n_negative * positive_ratio / (1 - positive_ratio)))
            
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            if target_n_positive > n_positive:
                n_repeats = target_n_positive // n_positive
                n_remainder = target_n_positive % n_positive
                repeated_pos_indices = np.tile(pos_indices, n_repeats)
                if n_remainder > 0:
                    np.random.seed(self.random_state)
                    additional = np.random.choice(pos_indices, size=n_remainder, replace=False)
                    repeated_pos_indices = np.concatenate([repeated_pos_indices, additional])
            else:
                repeated_pos_indices = pos_indices
                
            balanced_indices = np.concatenate([repeated_pos_indices, neg_indices])
        
        # 打乱并转换
        np.random.seed(self.random_state + 2)
        np.random.shuffle(balanced_indices)
        
        # 转换回DataFrame和数组
        if isinstance(X, pd.DataFrame):
            X_balanced = X.iloc[balanced_indices].copy()
        else:
            X_balanced = pd.DataFrame(X[balanced_indices], columns=X.columns)
        
        y_balanced = y[balanced_indices]
        
        unique_bal, counts_bal = np.unique(y_balanced, return_counts=True)
        logger.info(f"平衡后样本分布: {dict(zip(unique_bal, counts_bal))}")
        
        return X_balanced, y_balanced
    
    def _build_network(self, X: pd.DataFrame) -> BayesianNetwork:
        """
        构建贝叶斯网络结构
        """
        # 过滤网络结构，只保留数据中存在的特征
        available_nodes = set(X.columns)
        if self.expert_type == 'FP':
            available_nodes.add('is_fp')
        else:
            available_nodes.add('is_fn')
        
        # 构建边列表
        edges = []
        filtered_structure = {}
        
        for child, parents in self.network_structure.items():
            if child in available_nodes:
                valid_parents = [p for p in parents if p in available_nodes]
                if valid_parents or child.startswith('is_'):  # 输出节点可以没有父节点
                    filtered_structure[child] = valid_parents
                    for parent in valid_parents:
                        edges.append((parent, child))
        
        logger.info(f"{self.expert_type}专家网络边: {edges}")
        
        # 创建贝叶斯网络
        model = BayesianNetwork(edges)
        
        return model
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            balance_strategy: str = 'hybrid',
            positive_ratio: float = 0.25,
            n_bins: int = 3,
            prior_strength: float = 1.0) -> 'BayesianNetworkFPFNExpert':
        """
        训练贝叶斯网络专家
        """
        logger.info(f"开始训练{self.expert_type}专家...")
        
        # 1. 特征选择和映射
        X_network = self._select_and_map_features(X)
        logger.info(f"选择了{len(X_network.columns)}个网络特征")
        
        # 2. 数据平衡
        X_balanced, y_balanced = self._balance_data(X_network, y, balance_strategy, positive_ratio)
        
        # 3. 特征离散化
        X_discrete = self._discretize_features(X_balanced, n_bins)
        
        # 4. 添加目标变量
        target_name = f'is_{self.expert_type.lower()}'
        X_discrete[target_name] = y_balanced
        
        # 5. 构建网络
        self.model = self._build_network(X_discrete)
        
        # 6. 参数学习 - 修复pgmpy 1.0.0兼容性
        try:
            # pgmpy 1.0.0+版本的新方法：直接使用fit方法
            self.model.fit(X_discrete)
            logger.info("使用默认参数估计成功")
        except Exception as e:
            logger.warning(f"默认参数估计失败: {e}")
            try:
                # 尝试手动参数估计
                estimator = BayesianEstimator(model=self.model, data=X_discrete)
                self.model.fit(data=X_discrete, estimator=estimator)
                logger.info("使用BayesianEstimator成功")
            except Exception as e2:
                logger.warning(f"BayesianEstimator失败: {e2}")
                try:
                    # 最后备选：使用MaximumLikelihoodEstimator
                    estimator = MaximumLikelihoodEstimator(model=self.model, data=X_discrete)
                    self.model.fit(data=X_discrete, estimator=estimator)
                    logger.info("使用MaximumLikelihoodEstimator成功")
                except Exception as e3:
                    logger.error(f"所有参数估计方法都失败: {e3}")
                    # 简化网络：只保留基础节点
                    logger.info("尝试简化网络结构...")
                    simple_edges = [(parent, child) for parent, child in self.model.edges() 
                                   if not (parent.startswith('temp') or child.startswith('temp'))][:3]  # 最多3条边
                    self.model = BayesianNetwork(simple_edges)
                    self.model.fit(X_discrete)
        
        # 7. 创建推理引擎
        self.inference = VariableElimination(self.model)
        
        # 保存特征名称
        self.feature_names = list(X_discrete.columns)
        
        logger.info(f"{self.expert_type}专家训练完成")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测概率
        """
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        # 特征处理
        X_network = self._select_and_map_features(X)
        X_discrete = self._discretize_features(X_network)
        
        target_name = f'is_{self.expert_type.lower()}'
        probabilities = []
        
        # 逐样本推理
        for idx in range(len(X_discrete)):
            evidence = {}
            for feature in X_discrete.columns:
                if feature != target_name:
                    evidence[feature] = X_discrete.iloc[idx][feature]
            
            try:
                # 查询目标变量的概率分布
                result = self.inference.query([target_name], evidence=evidence)
                prob_positive = result.values[1]  # 正类概率
                probabilities.append([1 - prob_positive, prob_positive])
            except Exception as e:
                logger.warning(f"推理失败，使用默认概率: {e}")
                probabilities.append([0.5, 0.5])  # 默认概率
        
        return np.array(probabilities)
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        预测类别
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] > threshold).astype(int)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型性能
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y, y_pred),
            'mcc': matthews_corrcoef(y, y_pred)
        }
        
        try:
            metrics['auc'] = roc_auc_score(y, y_proba)
        except ValueError:
            metrics['auc'] = 0.5
        
        return metrics


class BayesianNetworkOptimizer:
    """
    贝叶斯网络超参数优化器
    """
    
    def __init__(self, expert_type: str, X: pd.DataFrame, y: np.ndarray, 
                 cv_folds: int = 3, random_state: int = 42):
        self.expert_type = expert_type
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_params = None
        self.best_score = -np.inf
        
    def objective(self, trial):
        """
        Optuna目标函数
        """
        # 超参数空间
        params = {
            'balance_strategy': trial.suggest_categorical('balance_strategy', ['hybrid', 'undersample']),
            'positive_ratio': trial.suggest_float('positive_ratio', 0.15, 0.4, step=0.05),
            'n_bins': trial.suggest_int('n_bins', 3, 5),
            'prior_strength': trial.suggest_float('prior_strength', 0.1, 3.0, log=True)
        }
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(self.X, self.y):
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            try:
                # 训练模型
                model = BayesianNetworkFPFNExpert(self.expert_type, self.random_state)
                model.fit(X_train, y_train, **params)
                
                # 评估
                metrics = model.evaluate(X_val, y_val)
                cv_scores.append(metrics['f1'])  # 使用F1作为主要指标
                
            except Exception as e:
                logger.warning(f"训练失败: {e}")
                cv_scores.append(0.0)
        
        return np.mean(cv_scores)
    
    def optimize(self, n_trials: int = 50) -> Dict[str, Any]:
        """
        执行超参数优化
        """
        study = optuna.create_study(direction='maximize', 
                                  study_name=f'{self.expert_type}_bayesian_network_optimization')
        
        study.optimize(self.objective, n_trials=n_trials)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"{self.expert_type}专家最佳参数: {self.best_params}")
        logger.info(f"{self.expert_type}专家最佳F1分数: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study
        }


def load_data_and_predictions():
    """
    加载数据和基础模型预测结果
    """
    logger.info("加载数据...")
    
    # 自动检测项目根目录路径，适配Windows和Linux
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 向上查找rainfalldata目录
    while current_dir and not current_dir.endswith('rainfalldata'):
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:  # 已经到根目录了
            break
        current_dir = parent_dir
    
    if current_dir.endswith('rainfalldata'):
        project_root = current_dir
    else:
        # 备用方案：基于当前文件位置推断
        project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
        project_root = os.path.abspath(project_root)
    
    features_dir = os.path.join(project_root, "results", "yangtze", "features")
    logger.info(f"项目根目录: {project_root}")
    logger.info(f"数据目录: {features_dir}")
    
    # 加载V6特征集
    try:
        X_path = os.path.join(features_dir, "X_Yangtsu_flat_features_v6.npy")
        y_path = os.path.join(features_dir, "Y_Yangtsu_flat_target_v6.npy")
        feature_names_path = os.path.join(features_dir, "feature_names_yangtsu_v6.txt")
        
        logger.info(f"加载特征文件: {X_path}")
        logger.info(f"加载目标文件: {y_path}")
        
        X = np.load(X_path)
        y = np.load(y_path)
        
        # 读取特征名称
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r', encoding='utf-8') as f:
                feature_names = [line.strip() for line in f.readlines()]
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 创建DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        
        logger.info(f"加载特征矩阵: {X.shape}")
        logger.info(f"目标变量分布: {np.unique(y, return_counts=True)}")
        
        return X_df, y
        
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        raise


def generate_fp_fn_labels(X: pd.DataFrame, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成FP/FN标签（需要基础模型的预测结果）
    """
    logger.info("生成FP/FN标签...")
    
    # 这里需要基础XGBoost模型的预测结果
    # 为了演示，我们使用简单的阈值规则生成伪FP/FN标签
    
    # 简化版：基于多产品特征生成伪预测
    if 'multi_product_mean' in X.columns:
        y_pred_proba = X['multi_product_mean'].values
        y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        # 随机生成（仅用于测试）
        np.random.seed(42)
        y_pred = np.random.binomial(1, 0.6, len(y))
    
    # 生成FP/FN标签
    fp_labels = ((y == 0) & (y_pred == 1)).astype(int)  # 实际无雨，预测有雨
    fn_labels = ((y == 1) & (y_pred == 0)).astype(int)  # 实际有雨，预测无雨
    
    fp_count = np.sum(fp_labels)
    fn_count = np.sum(fn_labels)
    
    logger.info(f"FP样本数: {fp_count} ({fp_count/len(y)*100:.2f}%)")
    logger.info(f"FN样本数: {fn_count} ({fn_count/len(y)*100:.2f}%)")
    
    return fp_labels, fn_labels


def main():
    """
    主函数
    """
    logger.info("=" * 80)
    logger.info("贝叶斯网络FP/FN专家模型训练开始")
    logger.info("=" * 80)
    
    # 1. 加载数据
    X, y = load_data_and_predictions()
    
    # 2. 生成FP/FN标签
    fp_labels, fn_labels = generate_fp_fn_labels(X, y)
    
    # 3. 训练FP专家
    logger.info("\n" + "=" * 50)
    logger.info("训练FP专家")
    logger.info("=" * 50)
    
    if np.sum(fp_labels) > 10:  # 确保有足够的正样本
        fp_optimizer = BayesianNetworkOptimizer('FP', X, fp_labels, cv_folds=3)
        fp_results = fp_optimizer.optimize(n_trials=30)
        
        # 使用最佳参数训练最终FP专家
        fp_expert = BayesianNetworkFPFNExpert('FP')
        fp_expert.fit(X, fp_labels, **fp_results['best_params'])
        
        # 保存FP专家
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "Ensemble_v2", "models")
        os.makedirs(output_dir, exist_ok=True)
        
        joblib.dump(fp_expert, os.path.join(output_dir, 'bayesian_fp_expert.pkl'))
        logger.info("FP专家已保存")
    else:
        logger.warning("FP样本太少，跳过FP专家训练")
    
    # 4. 训练FN专家
    logger.info("\n" + "=" * 50)
    logger.info("训练FN专家")
    logger.info("=" * 50)
    
    if np.sum(fn_labels) > 10:  # 确保有足够的正样本
        fn_optimizer = BayesianNetworkOptimizer('FN', X, fn_labels, cv_folds=3)
        fn_results = fn_optimizer.optimize(n_trials=30)
        
        # 使用最佳参数训练最终FN专家
        fn_expert = BayesianNetworkFPFNExpert('FN')
        fn_expert.fit(X, fn_labels, **fn_results['best_params'])
        
        # 保存FN专家
        joblib.dump(fn_expert, os.path.join(output_dir, 'bayesian_fn_expert.pkl'))
        logger.info("FN专家已保存")
    else:
        logger.warning("FN样本太少，跳过FN专家训练")
    
    logger.info("\n" + "=" * 80)
    logger.info("贝叶斯网络FP/FN专家模型训练完成")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()