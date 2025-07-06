#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
4专家模型最优参数5折交叉验证训练 - 按原始策略
====================================================

按照每个专家原始寻优脚本的训练策略来训练最终模型：
- TP专家：标准训练策略 + CUDA
- TN专家：标准训练策略 + CUDA 
- FP专家：多策略平衡训练 + 组合评分 + CUDA
- FN专家：标准训练策略 + CUDA

Author: Claude & User
Date: 2025-07-06
"""

import os
import numpy as np
import pandas as pd
import joblib
import time
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, 
                           f1_score, accuracy_score, average_precision_score)
from sklearn.utils import resample
import xgboost as xgb
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
RANDOM_STATE = 42
N_SPLITS = 5
EARLY_STOPPING_ROUNDS = 50

# 输出目录 - 使用绝对路径
current_script_dir = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(current_script_dir, "Ensemble_v2")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
OOF_DIR = os.path.join(OUTPUT_DIR, "oof_predictions")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OOF_DIR, exist_ok=True)

# 4个专家的最佳参数
EXPERT_BEST_PARAMS = {
    'tp': {
        'n_estimators': 2766,
        'learning_rate': 0.08410630089914119,
        'max_depth': 15,
        'subsample': 0.9923265361208282,
        'colsample_bytree': 0.9467567526259696,
        'gamma': 3.496734754692614e-05,
        'lambda': 7.84507553202397e-06,
        'alpha': 0.09746807240360679,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': ['logloss', 'auc'],
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'verbosity': 1
    },
    'tn': {
        'n_estimators': 2554,
        'learning_rate': 0.08332667314055044,
        'max_depth': 16,
        'subsample': 0.9835753824630262,
        'colsample_bytree': 0.9588466316033053,
        'gamma': 0.018349301531321232,
        'lambda': 0.0008444353870671067,
        'alpha': 6.160264077313662e-07,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': ['logloss', 'auc'],
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'verbosity': 1
    },
    'fp': {
        'n_estimators': 1049,
        'learning_rate': 0.03681984256177111,
        'max_depth': 18,
        'subsample': 0.7788619794159749,
        'colsample_bytree': 0.799506885003064,
        'gamma': 0.032688002920481345,
        'lambda': 0.004466244092261626,
        'alpha': 0.013833047801898195,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': ['logloss', 'auc'],
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'verbosity': 0
    },
    'fn': {
        'n_estimators': 2770,
        'learning_rate': 0.026846510683785577,
        'max_depth': 19,
        'subsample': 0.9169738452674724,
        'colsample_bytree': 0.9840043109708252,
        'gamma': 0.25549498200638876,
        'lambda': 0.0006317042813657675,
        'alpha': 9.632034103362201,
        'random_state': RANDOM_STATE,
        'tree_method': 'hist',
        'device': 'cuda',
        'eval_metric': ['logloss', 'auc'],
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
        'verbosity': 1
    }
}

# FP专家的数据平衡策略配置（来自原始脚本）
FP_BALANCE_STRATEGIES = [
    {'ratio': 0.20, 'strategy': 'hybrid', 'max_samples': 800000},
    {'ratio': 0.25, 'strategy': 'hybrid', 'max_samples': 600000}, 
    {'ratio': 0.30, 'strategy': 'undersample', 'max_samples': 500000},
]

def load_data():
    """
    加载数据 - 确保与专家目标标签维度一致
    """
    logger.info("加载数据...")
    
    # 使用相对路径，从当前脚本目录向上导航
    current_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(current_dir, "..", "..", "..", "results", "yangtze", "features")
    features_dir = os.path.abspath(features_dir)
    
    # 加载特征和标签
    X = np.load(os.path.join(features_dir, "X_Yangtsu_flat_features_v6.npy"))
    y = np.load(os.path.join(features_dir, "Y_Yangtsu_flat_target_v6.npy"))
    
    with open(os.path.join(features_dir, "feature_names_yangtsu_v6.txt"), 'r', encoding='utf-8') as f:
        feature_names = [line.strip() for line in f.readlines()]
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_binary = (y > 0.1).astype(int)
    
    # 清理特征名称中的特殊字符
    clean_columns = []
    for col in feature_names:
        clean_col = str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_')
        clean_columns.append(clean_col)
    X_df.columns = clean_columns
    
    # 根据01_prepare_l1_targets.py的逻辑，重新划分数据以匹配专家目标
    # 使用相同的train_test_split参数以确保一致性
    from sklearn.model_selection import train_test_split
    
    X_train_pool, _, y_train_pool, _ = train_test_split(
        X_df, y_binary,
        test_size=0.2,  # 与01_prepare_l1_targets.py中的TEST_SIZE_RATIO_HOLDOUT一致
        random_state=42,
        stratify=y_binary
    )
    
    logger.info(f"全量数据: {X_df.shape}")
    logger.info(f"训练池数据: {X_train_pool.shape}, 正样本比例: {np.mean(y_train_pool):.4f}")
    
    return X_train_pool, y_train_pool

def load_expert_targets():
    """
    加载4个专家的目标标签
    """
    logger.info("加载专家目标标签...")
    
    # 获取当前脚本的绝对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    ensemble_dir = os.path.join(current_script_dir, "Ensemble_v2")
    
    expert_targets = {}
    for expert_type in ['tp', 'tn', 'fp', 'fn']:
        target_path = os.path.join(ensemble_dir, f"y_is_{expert_type}.npy")
        if os.path.exists(target_path):
            y_expert = np.load(target_path)
            expert_targets[expert_type] = y_expert
            pos_ratio = np.mean(y_expert)
            logger.info(f"{expert_type.upper()}专家目标: {len(y_expert)}样本, 正样本比例: {pos_ratio:.4f}")
        else:
            logger.error(f"找不到{expert_type.upper()}专家目标文件: {target_path}")
            return None
    
    return expert_targets

def create_fp_balanced_dataset(X, y, target_ratio=0.3, strategy='hybrid', max_samples=500000, random_state=42):
    """
    FP专家的数据平衡策略（来自train_fp_fn_experts_v2.py）
    """
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    logger.info(f"FP专家原始分布: 正={n_pos}, 负={n_neg} ({n_pos/(n_pos+n_neg)*100:.2f}%)")
    logger.info(f"平衡策略: {strategy}, 目标比例: {target_ratio}, 最大样本: {max_samples}")
    
    if strategy == 'hybrid':
        # 混合策略：适度保留更多数据
        target_total = min(max_samples, int((n_pos + n_neg) * 0.3))  # 保留30%原始数据
        target_n_pos = int(target_total * target_ratio)
        target_n_neg = target_total - target_n_pos
        
        # 处理正样本 - 如果不够就重复采样
        if target_n_pos <= n_pos:
            np.random.seed(random_state)
            selected_pos_indices = np.random.choice(pos_indices, size=target_n_pos, replace=False)
        else:
            # 重复采样
            n_repeats = target_n_pos // n_pos
            n_remainder = target_n_pos % n_pos
            selected_pos_indices = np.tile(pos_indices, n_repeats)
            if n_remainder > 0:
                np.random.seed(random_state)
                additional = np.random.choice(pos_indices, size=n_remainder, replace=False)
                selected_pos_indices = np.concatenate([selected_pos_indices, additional])
        
        # 处理负样本 - 随机下采样
        np.random.seed(random_state + 1)
        selected_neg_indices = np.random.choice(neg_indices, size=target_n_neg, replace=False)
        
        balanced_indices = np.concatenate([selected_pos_indices, selected_neg_indices])
        
    elif strategy == 'undersample':
        # 改进的下采样：保留更多负样本
        target_n_neg = min(int(n_pos * (1 - target_ratio) / target_ratio), 
                          min(max_samples - n_pos, n_neg))
        
        np.random.seed(random_state)
        selected_neg_indices = np.random.choice(neg_indices, size=target_n_neg, replace=False)
        balanced_indices = np.concatenate([pos_indices, selected_neg_indices])
        
    else:  # oversample
        # 改进的上采样：控制总样本数
        target_total = min(max_samples, n_neg + n_pos * 3)  # 控制增长
        target_n_pos = int(target_total * target_ratio)
        
        if target_n_pos > n_pos:
            n_repeats = target_n_pos // n_pos
            n_remainder = target_n_pos % n_pos
            repeated_pos_indices = np.tile(pos_indices, n_repeats)
            if n_remainder > 0:
                np.random.seed(random_state)
                additional = np.random.choice(pos_indices, size=n_remainder, replace=False)
                repeated_pos_indices = np.concatenate([repeated_pos_indices, additional])
        else:
            repeated_pos_indices = pos_indices
            
        balanced_indices = np.concatenate([repeated_pos_indices, neg_indices])
    
    # 打乱并限制总数
    np.random.seed(random_state + 2)
    np.random.shuffle(balanced_indices)
    
    if len(balanced_indices) > max_samples:
        balanced_indices = balanced_indices[:max_samples]
    
    X_balanced = X.iloc[balanced_indices] if isinstance(X, pd.DataFrame) else X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    # 统计结果
    n_pos_balanced = np.sum(y_balanced == 1)
    n_neg_balanced = np.sum(y_balanced == 0)
    total_balanced = len(y_balanced)
    
    logger.info(f"平衡后: 正={n_pos_balanced} ({n_pos_balanced/total_balanced*100:.1f}%), 负={n_neg_balanced}")
    logger.info(f"样本数: {len(y)} -> {total_balanced} ({total_balanced/len(y)*100:.1f}%)")
    
    return X_balanced, y_balanced

def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    计算评估指标
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
            metrics['ap'] = average_precision_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.5
            metrics['ap'] = 0.0
    
    return metrics

def train_standard_expert(expert_type, X, y_expert, params):
    """
    标准专家训练策略（用于TP、TN、FN专家）
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练{expert_type.upper()}专家 - 标准策略")
    logger.info(f"{'='*60}")
    
    # 检查数据平衡性
    pos_count = np.sum(y_expert == 1)
    neg_count = np.sum(y_expert == 0)
    pos_ratio = pos_count / len(y_expert)
    
    logger.info(f"数据分布: 正样本={pos_count}, 负样本={neg_count}, 比例={pos_ratio:.4f}")
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_results = []
    oof_predictions = np.zeros(len(y_expert))
    fold_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_expert)):
        logger.info(f"\n--- 第 {fold_idx + 1}/{N_SPLITS} 折 ---")
        
        # 分割数据
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_expert[train_idx], y_expert[val_idx]
        
        # 训练集平衡性检查
        train_pos_ratio = np.mean(y_train)
        val_pos_ratio = np.mean(y_val)
        logger.info(f"训练集: {len(y_train)}样本, 正样本比例: {train_pos_ratio:.4f}")
        logger.info(f"验证集: {len(y_val)}样本, 正样本比例: {val_pos_ratio:.4f}")
        
        # 添加scale_pos_weight参数
        model_params = params.copy()
        model_params['scale_pos_weight'] = neg_count / pos_count if pos_count > 0 else 1
        
        # 创建模型
        model = xgb.XGBClassifier(**model_params)
        
        # 训练模型
        start_time = time.time()
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)]
        )
        train_time = time.time() - start_time
        
        # 预测
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        
        # 保存折外预测
        oof_predictions[val_idx] = y_val_proba
        
        # 计算指标
        fold_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
        fold_metrics['train_time'] = train_time
        fold_metrics['best_iteration'] = model.best_iteration
        
        fold_results.append(fold_metrics)
        fold_models.append(model)
        
        # 记录当前折性能
        logger.info(f"第{fold_idx + 1}折性能:")
        logger.info(f"  AUC: {fold_metrics['auc']:.4f}")
        logger.info(f"  Precision: {fold_metrics['precision']:.4f}")
        logger.info(f"  Recall: {fold_metrics['recall']:.4f}")
        logger.info(f"  F1: {fold_metrics['f1']:.4f}")
        logger.info(f"  训练时间: {train_time:.1f}秒")
        logger.info(f"  最佳迭代: {model.best_iteration}")
    
    return fold_results, oof_predictions, fold_models

def train_fp_expert_with_balancing(expert_type, X, y_expert, params):
    """
    FP专家的多策略平衡训练（来自train_fp_fn_experts_v2.py）
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"开始训练{expert_type.upper()}专家 - 多策略平衡训练")
    logger.info(f"{'='*60}")
    
    # 直接使用最佳策略：25% hybrid (根据原始脚本结果)
    best_strategy_config = {'ratio': 0.25, 'strategy': 'hybrid', 'max_samples': 600000}
    
    logger.info(f"🏆 使用已知最佳策略: {best_strategy_config}")
    
    # 创建平衡数据集
    X_balanced, y_balanced = create_fp_balanced_dataset(
        X, y_expert,
        target_ratio=best_strategy_config['ratio'],
        strategy=best_strategy_config['strategy'],
        max_samples=best_strategy_config['max_samples'],
        random_state=RANDOM_STATE
    )
    
    # 5折交叉验证
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    oof_predictions = np.zeros(len(y_expert))  # 注意：这里应该是原始数据长度
    fold_models = []
    
    # 创建原始数据到平衡数据的映射
    balanced_to_original_idx = y_balanced.index if hasattr(y_balanced, 'index') else np.arange(len(y_balanced))
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_balanced, y_balanced)):
        logger.info(f"\n--- 第 {fold_idx + 1}/{N_SPLITS} 折 ---")
        
        # 分割平衡数据
        X_train = X_balanced.iloc[train_idx] if isinstance(X_balanced, pd.DataFrame) else X_balanced[train_idx]
        X_val = X_balanced.iloc[val_idx] if isinstance(X_balanced, pd.DataFrame) else X_balanced[val_idx]
        y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]
        
        logger.info(f"训练集: {len(y_train)}样本, 正样本比例: {np.mean(y_train):.4f}")
        logger.info(f"验证集: {len(y_val)}样本, 正样本比例: {np.mean(y_val):.4f}")
        
        # 添加权重参数 - 使用FP专家的custom_weight策略
        pos_count_balanced = np.sum(y_balanced == 1)
        neg_count_balanced = np.sum(y_balanced == 0)
        model_params = params.copy()
        
        # FP专家使用原始脚本中的custom_weight: 0.5017661265304088
        model_params['scale_pos_weight'] = 0.5017661265304088
        
        # 创建模型
        model = xgb.XGBClassifier(**model_params)
        
        # 训练模型
        start_time = time.time()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        train_time = time.time() - start_time
        
        # 在平衡验证集上预测
        y_val_proba = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_proba >= 0.5).astype(int)
        
        # 计算指标，包括FP专家的组合评分
        fold_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        # 计算FP专家特有的组合评分
        fp_combined_score = 0.4 * fold_metrics['auc'] + 0.4 * fold_metrics['precision'] + 0.2 * fold_metrics['f1']
        fold_metrics['fp_combined_score'] = fp_combined_score
        fold_metrics['val_proba'] = y_val_proba  # 保存验证集概率预测
        
        fold_metrics['train_time'] = train_time
        fold_metrics['best_iteration'] = model.best_iteration
        
        fold_results.append(fold_metrics)
        fold_models.append(model)
        
        logger.info(f"第{fold_idx + 1}折性能:")
        logger.info(f"  AUC: {fold_metrics['auc']:.4f}")
        logger.info(f"  Precision: {fold_metrics['precision']:.4f}")
        logger.info(f"  Recall: {fold_metrics['recall']:.4f}")
        logger.info(f"  F1: {fold_metrics['f1']:.4f}")
        logger.info(f"  FP组合评分: {fp_combined_score:.4f}")
        logger.info(f"  训练时间: {train_time:.1f}秒")
    
    # 计算平衡数据上的交叉验证性能（用于与原始脚本对比）
    balanced_oof_predictions = np.zeros(len(y_balanced))
    fold_idx = 0
    for train_idx, val_idx in skf.split(X_balanced, y_balanced):
        balanced_oof_predictions[val_idx] = fold_results[fold_idx]['val_proba']
        fold_idx += 1
    
    # 在平衡数据上的整体性能（这个应该与原始脚本一致）
    balanced_oof_pred_binary = (balanced_oof_predictions >= 0.5).astype(int)
    balanced_oof_metrics = calculate_metrics(y_balanced, balanced_oof_pred_binary, balanced_oof_predictions)
    
    logger.info(f"\n平衡数据CV性能 (与原始脚本对比):")
    logger.info(f"平衡数据AUC: {balanced_oof_metrics['auc']:.4f}")
    logger.info(f"平衡数据Precision: {balanced_oof_metrics['precision']:.4f}")
    logger.info(f"平衡数据Recall: {balanced_oof_metrics['recall']:.4f}")
    logger.info(f"平衡数据F1: {balanced_oof_metrics['f1']:.4f}")
    
    # 计算平衡数据上的组合评分
    balanced_combined_score = 0.4 * balanced_oof_metrics['auc'] + 0.4 * balanced_oof_metrics['precision'] + 0.2 * balanced_oof_metrics['f1']
    logger.info(f"平衡数据组合评分: {balanced_combined_score:.4f}")
    
    # 在原始完整数据上生成预测
    logger.info(f"\n--- 生成原始数据预测 ---")
    ensemble_predictions = np.zeros(len(y_expert))
    for model in fold_models:
        ensemble_predictions += model.predict_proba(X)[:, 1]
    ensemble_predictions /= len(fold_models)
    
    return fold_results, ensemble_predictions, fold_models

def train_expert_with_cv(expert_type, X, y_expert, params):
    """
    使用对应的训练策略训练单个专家
    """
    if expert_type == 'fp':
        # FP专家使用多策略平衡训练
        return train_fp_expert_with_balancing(expert_type, X, y_expert, params)
    else:
        # TP、TN、FN专家使用标准训练策略
        return train_standard_expert(expert_type, X, y_expert, params)

def save_results(expert_results):
    """
    保存所有结果
    """
    logger.info(f"\n{'='*60}")
    logger.info("保存训练结果")
    logger.info(f"{'='*60}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存每个专家的结果
    for expert_type, results in expert_results.items():
        # 保存模型
        models_path = os.path.join(MODELS_DIR, f"{expert_type}_expert_cv_models_{timestamp}.joblib")
        joblib.dump(results['models'], models_path)
        logger.info(f"{expert_type.upper()}专家模型已保存: {models_path}")
        
        # 保存OOF预测
        oof_path = os.path.join(OOF_DIR, f"{expert_type}_expert_oof_predictions_{timestamp}.npy")
        np.save(oof_path, results['oof_predictions'])
        logger.info(f"{expert_type.upper()}专家OOF预测已保存: {oof_path}")
        
        # 保存详细结果
        results_path = os.path.join(OUTPUT_DIR, f"{expert_type}_expert_cv_results_{timestamp}.joblib")
        joblib.dump(results, results_path)
        logger.info(f"{expert_type.upper()}专家详细结果已保存: {results_path}")
    
    # 保存汇总报告
    summary_report = {
        'timestamp': datetime.now().isoformat(),
        'training_config': {
            'n_splits': N_SPLITS,
            'random_state': RANDOM_STATE,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS
        },
        'expert_summary': {}
    }
    
    for expert_type, results in expert_results.items():
        summary_report['expert_summary'][expert_type] = {
            'avg_auc': results['avg_metrics']['auc_mean'],
            'avg_auc_std': results['avg_metrics']['auc_std'],
            'oof_auc': results['oof_metrics']['auc'],
            'avg_f1': results['avg_metrics']['f1_mean'],
            'oof_f1': results['oof_metrics']['f1'],
            'total_samples': results['training_info']['total_samples'],
            'positive_ratio': results['training_info']['positive_ratio']
        }
    
    summary_path = os.path.join(OUTPUT_DIR, f"experts_cv_summary_{timestamp}.joblib")
    joblib.dump(summary_report, summary_path)
    logger.info(f"汇总报告已保存: {summary_path}")
    
    return summary_report

def print_final_summary(expert_results):
    """
    打印最终汇总
    """
    logger.info(f"\n{'='*80}")
    logger.info("4专家模型 - 5折交叉验证最终汇总")
    logger.info(f"{'='*80}")
    
    print(f"\n{'专家类型':<8} {'5折平均AUC':<12} {'OOF AUC':<10} {'5折平均F1':<12} {'OOF F1':<10} {'正样本比例':<10}")
    print("-" * 70)
    
    for expert_type in ['tp', 'tn', 'fp', 'fn']:
        if expert_type in expert_results:
            results = expert_results[expert_type]
            avg_auc = results['avg_metrics']['auc_mean']
            oof_auc = results['oof_metrics']['auc']
            avg_f1 = results['avg_metrics']['f1_mean']
            oof_f1 = results['oof_metrics']['f1']
            pos_ratio = results['training_info']['positive_ratio']
            
            print(f"{expert_type.upper():<8} {avg_auc:<12.4f} {oof_auc:<10.4f} {avg_f1:<12.4f} {oof_f1:<10.4f} {pos_ratio:<10.4f}")
    
    # 与基线对比
    baseline_auc = 0.9887
    print(f"\n基线单个XGBoost AUC: {baseline_auc}")
    print(f"与基线对比 (OOF AUC):")
    for expert_type in ['tp', 'tn', 'fp', 'fn']:
        if expert_type in expert_results:
            oof_auc = expert_results[expert_type]['oof_metrics']['auc']
            diff = oof_auc - baseline_auc
            status = "✅ 超越" if diff > 0 else "❌ 低于"
            print(f"  {expert_type.upper()}专家: {oof_auc:.4f} ({diff:+.4f}) {status}")

def main():
    """
    主函数
    """
    logger.info("开始4专家模型最优参数5折交叉验证训练 - 按原始策略")
    
    start_time = time.time()
    
    # 1. 加载数据
    X, y_binary = load_data()
    
    # 2. 加载专家目标
    expert_targets = load_expert_targets()
    if expert_targets is None:
        logger.error("无法加载专家目标，程序退出")
        return
    
    # 3. 训练所有专家
    expert_results = {}
    
    for expert_type in ['fp']:
        logger.info(f"\n开始训练{expert_type.upper()}专家...")
        
        y_expert = expert_targets[expert_type]
        params = EXPERT_BEST_PARAMS[expert_type]
        
        fold_results, oof_predictions, fold_models = train_expert_with_cv(expert_type, X, y_expert, params)
        
        if fold_results is not None:
            # 计算平均性能
            avg_metrics = {}
            for metric in ['auc', 'precision', 'recall', 'f1', 'accuracy', 'ap', 'train_time']:
                values = [fold[metric] for fold in fold_results]
                avg_metrics[f'{metric}_mean'] = np.mean(values)
                avg_metrics[f'{metric}_std'] = np.std(values)
            
            # 整体OOF性能
            oof_pred_binary = (oof_predictions >= 0.5).astype(int)
            oof_metrics = calculate_metrics(y_expert, oof_pred_binary, oof_predictions)
            
            logger.info(f"\n🎯 {expert_type.upper()}专家 - 5折交叉验证结果:")
            logger.info(f"平均AUC: {avg_metrics['auc_mean']:.4f} ± {avg_metrics['auc_std']:.4f}")
            logger.info(f"平均Precision: {avg_metrics['precision_mean']:.4f} ± {avg_metrics['precision_std']:.4f}")
            logger.info(f"平均Recall: {avg_metrics['recall_mean']:.4f} ± {avg_metrics['recall_std']:.4f}")
            logger.info(f"平均F1: {avg_metrics['f1_mean']:.4f} ± {avg_metrics['f1_std']:.4f}")
            
            logger.info(f"\n整体OOF性能:")
            logger.info(f"OOF AUC: {oof_metrics['auc']:.4f}")
            logger.info(f"OOF Precision: {oof_metrics['precision']:.4f}")
            logger.info(f"OOF Recall: {oof_metrics['recall']:.4f}")
            logger.info(f"OOF F1: {oof_metrics['f1']:.4f}")
            
            # 保存结果
            pos_count = np.sum(y_expert == 1)
            neg_count = np.sum(y_expert == 0)
            pos_ratio = pos_count / len(y_expert)
            
            results = {
                'expert_type': expert_type,
                'fold_results': fold_results,
                'avg_metrics': avg_metrics,
                'oof_metrics': oof_metrics,
                'oof_predictions': oof_predictions,
                'models': fold_models,
                'training_info': {
                    'n_splits': N_SPLITS,
                    'random_state': RANDOM_STATE,
                    'total_samples': len(y_expert),
                    'positive_samples': pos_count,
                    'negative_samples': neg_count,
                    'positive_ratio': pos_ratio,
                    'params': params,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            expert_results[expert_type] = results
    
    # 4. 保存结果
    summary_report = save_results(expert_results)
    
    # 5. 打印最终汇总
    print_final_summary(expert_results)
    
    total_time = time.time() - start_time
    logger.info(f"\n✅ 所有专家训练完成！总耗时: {total_time/60:.1f}分钟")

if __name__ == "__main__":
    main()