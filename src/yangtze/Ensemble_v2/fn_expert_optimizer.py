#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FN专家优化训练脚本
===================

针对FN专家的特殊困难，实现专门的优化策略：
1. 更激进的平衡策略
2. FN专门的特征工程
3. 多阶段训练方法
4. 专门的评估指标

Author: Claude & User
Date: 2025-07-05
"""

import os
import numpy as np
import pandas as pd
import joblib
import optuna
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                           roc_auc_score, f1_score, recall_score, precision_score)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN, SMOTETomek
import xgboost as xgb

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FNExpertOptimizer:
    """
    FN专家优化器 - 专门针对FN检测的困难
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.best_params = None
        self.best_score = -np.inf
        self.feature_importance = None
        
    def create_fn_specific_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        创建FN专门的特征工程
        """
        logger.info("创建FN专门的特征...")
        
        X_enhanced = X.copy()
        
        # 1. 强信号检测特征
        if 'multi_product_max' in X.columns and 'multi_product_mean' in X.columns:
            X_enhanced['intensity_signal_strength'] = X['multi_product_max'] / (X['multi_product_mean'] + 1e-6)
            X_enhanced['peak_to_mean_ratio'] = X['multi_product_max'] / (X['multi_product_mean'] + 1e-6)
            
        # 2. 一致性特征 - FN通常发生在产品分歧较大时
        if 'multi_product_std' in X.columns and 'multi_product_mean' in X.columns:
            X_enhanced['consistency_index'] = 1.0 / (1.0 + X['multi_product_std'] / (X['multi_product_mean'] + 1e-6))
            
        # 3. 时序异常特征
        if 'lag_1_multi_product_mean' in X.columns and 'multi_product_mean' in X.columns:
            X_enhanced['temporal_anomaly'] = np.abs(X['multi_product_mean'] - X['lag_1_multi_product_mean'])
            X_enhanced['temporal_acceleration'] = X['multi_product_mean'] - 2*X['lag_1_multi_product_mean']
            
        # 4. 产品计数相关特征
        if 'rain_product_count' in X.columns:
            X_enhanced['product_diversity'] = X['rain_product_count'] / 6.0  # 标准化到6个产品
            
        # 5. 季节性和日周期特征增强
        if 'season_spring' in X.columns and 'sin_day' in X.columns:
            X_enhanced['seasonal_diurnal_interaction'] = X['season_spring'] * X['sin_day']
            
        # 6. 极值检测特征
        if 'multi_product_max' in X.columns:
            X_enhanced['is_extreme_event'] = (X['multi_product_max'] > X['multi_product_max'].quantile(0.95)).astype(float)
            
        # 7. 低强度偏差特征
        if 'low_intensity_std' in X.columns and 'multi_product_mean' in X.columns:
            X_enhanced['low_intensity_bias'] = X['low_intensity_std'] / (X['multi_product_mean'] + 1e-6)
            
        logger.info(f"FN特征工程完成: {X.shape[1]} -> {X_enhanced.shape[1]}")
        return X_enhanced
    
    def advanced_fn_balancing(self, X: pd.DataFrame, y: np.ndarray, 
                            strategy: str = 'multi_stage') -> Tuple[pd.DataFrame, np.ndarray]:
        """
        FN专门的高级平衡策略
        """
        logger.info(f"FN专门平衡策略: {strategy}")
        
        n_positive = np.sum(y == 1)
        n_negative = np.sum(y == 0)
        original_ratio = n_positive / len(y)
        
        logger.info(f"原始分布: 正={n_positive}, 负={n_negative}, 比例={original_ratio:.4f}")
        
        if strategy == 'multi_stage':
            # 多阶段平衡：先欠采样，再过采样
            target_ratio = 0.3  # 目标30%正样本
            
            # 第一阶段：下采样负样本到合理范围
            stage1_neg_samples = min(n_negative, n_positive * 3)  # 最多3:1
            rus = RandomUnderSampler(sampling_strategy={0: stage1_neg_samples}, random_state=self.random_state)
            X_stage1, y_stage1 = rus.fit_resample(X, y)
            
            # 第二阶段：使用BorderlineSMOTE增强边界正样本
            target_pos_samples = int(stage1_neg_samples * target_ratio / (1 - target_ratio))
            if target_pos_samples > n_positive:
                smote = BorderlineSMOTE(sampling_strategy={1: target_pos_samples}, 
                                      random_state=self.random_state, k_neighbors=3)
                X_balanced, y_balanced = smote.fit_resample(X_stage1, y_stage1)
            else:
                X_balanced, y_balanced = X_stage1, y_stage1
                
        elif strategy == 'smoteenn':
            # SMOTEENN：结合过采样和欠采样
            smoteenn = SMOTEENN(sampling_strategy=0.25, random_state=self.random_state)
            X_balanced, y_balanced = smoteenn.fit_resample(X, y)
            
        elif strategy == 'adasyn':
            # ADASYN：自适应合成采样
            # 先下采样到合理范围
            max_neg_samples = min(n_negative, n_positive * 4)
            rus = RandomUnderSampler(sampling_strategy={0: max_neg_samples}, random_state=self.random_state)
            X_stage1, y_stage1 = rus.fit_resample(X, y)
            
            adasyn = ADASYN(sampling_strategy=0.3, random_state=self.random_state)
            X_balanced, y_balanced = adasyn.fit_resample(X_stage1, y_stage1)
            
        elif strategy == 'cost_sensitive':
            # 成本敏感：保持原始分布，但使用权重
            return X, y
            
        else:
            # 默认策略：简单过采样
            target_pos_samples = int(n_negative * 0.25)
            if target_pos_samples > n_positive:
                smote = SMOTE(sampling_strategy={1: target_pos_samples}, random_state=self.random_state)
                X_balanced, y_balanced = smote.fit_resample(X, y)
            else:
                X_balanced, y_balanced = X, y
        
        final_pos = np.sum(y_balanced == 1)
        final_neg = np.sum(y_balanced == 0)
        final_ratio = final_pos / len(y_balanced)
        
        logger.info(f"平衡后分布: 正={final_pos}, 负={final_neg}, 比例={final_ratio:.4f}")
        
        return X_balanced, y_balanced
    
    def fn_specific_objective(self, trial, X: pd.DataFrame, y: np.ndarray):
        """
        FN专门的Optuna目标函数
        """
        # 超参数空间
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
            
            # 平衡策略
            'balance_strategy': trial.suggest_categorical('balance_strategy', 
                                                       ['multi_stage', 'smoteenn', 'adasyn', 'cost_sensitive']),
            
            # 成本敏感参数
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0) if 
                               trial.suggest_categorical('use_scale_pos_weight', [True, False]) else 1.0,
        }
        
        # 交叉验证
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # 特征工程
                X_train_enhanced = self.create_fn_specific_features(X_train)
                X_val_enhanced = self.create_fn_specific_features(X_val)
                
                # 数据平衡
                balance_strategy = params['balance_strategy']
                if balance_strategy != 'cost_sensitive':
                    X_train_balanced, y_train_balanced = self.advanced_fn_balancing(
                        X_train_enhanced, y_train, balance_strategy
                    )
                else:
                    X_train_balanced, y_train_balanced = X_train_enhanced, y_train
                
                # 清理特征名称
                X_train_clean = X_train_balanced.copy()
                X_val_clean = X_val_enhanced.copy()
                
                # 清理列名中的特殊字符
                clean_columns = []
                for col in X_train_clean.columns:
                    clean_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
                    clean_columns.append(clean_col)
                
                X_train_clean.columns = clean_columns
                X_val_clean.columns = clean_columns
                
                # 计算样本权重
                if balance_strategy == 'cost_sensitive':
                    n_pos = np.sum(y_train_balanced == 1)
                    n_neg = np.sum(y_train_balanced == 0)
                    sample_weight = np.where(y_train_balanced == 1, 
                                           n_neg / n_pos, 1.0)
                else:
                    sample_weight = None
                
                # 训练模型
                model = xgb.XGBClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    min_child_weight=params['min_child_weight'],
                    gamma=params['gamma'],
                    scale_pos_weight=params.get('scale_pos_weight', 1.0),
                    random_state=self.random_state,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
                
                model.fit(X_train_clean, y_train_balanced, sample_weight=sample_weight)
                
                # 预测
                y_pred_proba = model.predict_proba(X_val_clean)[:, 1]
                y_pred = model.predict(X_val_clean)
                
                # FN专门的评估指标
                recall = recall_score(y_val, y_pred, zero_division=0)
                precision = precision_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                
                try:
                    auc = roc_auc_score(y_val, y_pred_proba)
                    ap = average_precision_score(y_val, y_pred_proba)
                except:
                    auc = 0.5
                    ap = 0.0
                
                # 组合评分：重视召回率和AP
                score = 0.4 * recall + 0.3 * f1 + 0.2 * auc + 0.1 * ap
                cv_scores.append(score)
                
            except Exception as e:
                logger.warning(f"训练失败: {e}")
                cv_scores.append(0.0)
        
        return np.mean(cv_scores)
    
    def optimize_fn_expert(self, X: pd.DataFrame, y: np.ndarray, 
                          n_trials: int = 100) -> Dict[str, Any]:
        """
        优化FN专家
        """
        logger.info("开始FN专家优化...")
        
        # 创建study
        study = optuna.create_study(
            direction='maximize',
            study_name='fn_expert_optimization',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # 优化
        study.optimize(
            lambda trial: self.fn_specific_objective(trial, X, y),
            n_trials=n_trials,
            timeout=7200  # 2小时超时
        )
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"FN专家优化完成")
        logger.info(f"最佳参数: {self.best_params}")
        logger.info(f"最佳分数: {self.best_score:.4f}")
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study
        }
    
    def train_final_fn_expert(self, X: pd.DataFrame, y: np.ndarray) -> Any:
        """
        训练最终的FN专家模型
        """
        logger.info("训练最终FN专家模型...")
        
        if self.best_params is None:
            raise ValueError("请先运行优化过程")
        
        # 特征工程
        X_enhanced = self.create_fn_specific_features(X)
        
        # 数据平衡
        balance_strategy = self.best_params['balance_strategy']
        if balance_strategy != 'cost_sensitive':
            X_balanced, y_balanced = self.advanced_fn_balancing(X_enhanced, y, balance_strategy)
        else:
            X_balanced, y_balanced = X_enhanced, y
        
        # 清理特征名称
        X_clean = X_balanced.copy()
        clean_columns = []
        for col in X_clean.columns:
            clean_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
            clean_columns.append(clean_col)
        X_clean.columns = clean_columns
        
        # 计算样本权重
        if balance_strategy == 'cost_sensitive':
            n_pos = np.sum(y_balanced == 1)
            n_neg = np.sum(y_balanced == 0)
            sample_weight = np.where(y_balanced == 1, n_neg / n_pos, 1.0)
        else:
            sample_weight = None
        
        # 训练最终模型
        model = xgb.XGBClassifier(
            n_estimators=self.best_params['n_estimators'],
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            reg_alpha=self.best_params['reg_alpha'],
            reg_lambda=self.best_params['reg_lambda'],
            min_child_weight=self.best_params['min_child_weight'],
            gamma=self.best_params['gamma'],
            scale_pos_weight=self.best_params.get('scale_pos_weight', 1.0),
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        model.fit(X_clean, y_balanced, sample_weight=sample_weight)
        
        # 保存特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_clean.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("最终FN专家模型训练完成")
        return model

def load_data():
    """
    加载数据
    """
    logger.info("加载数据...")
    
    # 找到项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    while project_root and not project_root.endswith('rainfalldata'):
        project_root = os.path.dirname(project_root)
    
    features_dir = os.path.join(project_root, "results", "yangtze", "features")
    
    # 加载特征和标签
    X = np.load(os.path.join(features_dir, "X_Yangtsu_flat_features_v6.npy"))
    y = np.load(os.path.join(features_dir, "Y_Yangtsu_flat_target_v6.npy"))
    
    with open(os.path.join(features_dir, "feature_names_yangtsu_v6.txt"), 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_binary = (y > 0.1).astype(int)
    
    logger.info(f"数据加载完成: {X_df.shape}")
    return X_df, y_binary

def generate_fn_labels(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """
    生成FN标签
    """
    logger.info("生成FN标签...")
    
    # 使用简单的基础预测器
    if 'multi_product_mean' in X.columns:
        base_pred = (X['multi_product_mean'] > X['multi_product_mean'].quantile(0.4)).astype(int)
    else:
        np.random.seed(42)
        base_pred = np.random.binomial(1, 0.6, len(y))
    
    # FN: 实际有雨，预测无雨
    fn_labels = ((y == 1) & (base_pred == 0)).astype(int)
    
    logger.info(f"FN样本: {np.sum(fn_labels)} ({np.sum(fn_labels)/len(y)*100:.2f}%)")
    return fn_labels

def main():
    """
    主函数
    """
    logger.info("=== FN专家优化训练 ===")
    
    # 加载数据
    X, y = load_data()
    
    # 采样数据以加速训练
    sample_size = min(500000, len(X))
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample = X.iloc[indices]
    y_sample = y[indices]
    
    # 生成FN标签
    fn_labels = generate_fn_labels(X_sample, y_sample)
    
    if np.sum(fn_labels) < 50:
        logger.error("FN样本太少，无法训练")
        return
    
    # 初始化优化器
    optimizer = FNExpertOptimizer()
    
    # 执行优化
    results = optimizer.optimize_fn_expert(X_sample, fn_labels, n_trials=10)  # 减少试验次数
    
    # 训练最终模型
    final_model = optimizer.train_final_fn_expert(X_sample, fn_labels)
    
    # 保存结果
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "models")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(output_dir, "optimized_fn_expert.pkl")
    joblib.dump(final_model, model_path)
    
    # 保存优化器
    optimizer_path = os.path.join(output_dir, "fn_optimizer.pkl")
    joblib.dump(optimizer, optimizer_path)
    
    logger.info(f"FN专家优化完成，模型已保存: {model_path}")
    
    # 打印特征重要性
    if optimizer.feature_importance is not None:
        logger.info("前10个重要特征:")
        for i, row in optimizer.feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

if __name__ == "__main__":
    main()