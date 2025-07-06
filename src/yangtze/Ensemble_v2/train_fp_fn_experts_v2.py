#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的FP/FN专家模型训练脚本 v2
===================================

针对初版平衡训练效果不理想(AUC~87%)的问题，进行以下改进：
1. 尝试多种平衡比例 (20%, 25%, 30%, 35%)
2. 使用更大的数据样本 (混合策略)
3. 针对FP/FN任务优化超参数范围
4. 使用多种评估指标 (AUC, Precision, Recall, F1)
5. 增加更多集成子模型

Author: Claude & User
Date: 2025-07-05
"""

import os
import sys
import time
import signal
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数 v2
# =============================================================================

# 项目路径配置
PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "results", "yangtze", "features")
ENSEMBLE_V2_DIR = os.path.dirname(__file__)
TARGETS_DIR = os.path.join(ENSEMBLE_V2_DIR, "Ensemble_v2")
OUTPUT_DIR = os.path.join(ENSEMBLE_V2_DIR, "Ensemble_v2")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
OPTUNA_DB_DIR = os.path.join(OUTPUT_DIR, "optuna_db")

# 创建必要的目录
for dir_path in [OUTPUT_DIR, LOGS_DIR, MODELS_DIR, OPTUNA_DB_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# 数据文件路径
X_FLAT_PATH = os.path.join(PROJECT_DIR, "X_Yangtsu_flat_features_v6.npy")
Y_FLAT_PATH = os.path.join(PROJECT_DIR, "Y_Yangtsu_flat_target_v6.npy")
FEATURE_NAMES_PATH = os.path.join(PROJECT_DIR, "feature_names_yangtsu_v6.txt")

# 训练参数 v2 - 改进版
RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO_HOLDOUT = 0.2
N_SPLITS_KFOLD = 5
N_TRIALS_OPTUNA = 100  # 快速测试
OPTUNA_TIMEOUT = 3600 * 20  
EARLY_STOPPING_ROUNDS_OPTUNA = 20
EARLY_STOPPING_ROUNDS_FINAL = 30
OPTIMIZE_METRIC = 'auc'

# 改进的数据平衡策略参数
BALANCE_STRATEGIES = [
    {'ratio': 0.20, 'strategy': 'hybrid', 'max_samples': 800000},
    {'ratio': 0.25, 'strategy': 'hybrid', 'max_samples': 600000}, 
    {'ratio': 0.30, 'strategy': 'undersample', 'max_samples': 500000},
]
N_BALANCED_SUBSETS = 5  # 增加子集数量
RANDOM_STATE = 42

# 专家模型配置 - 改进版
EXPERT_MODELS = {
    "fp": {
        "target_file": "y_is_fp.npy", 
        "description": "False Positive Expert v2 - 多策略平衡训练",
        "study_name": "fp_expert_v2_optimization",
        "storage_file": "fp_expert_v2_optimization.db"
    },
    "fn": {
        "target_file": "y_is_fn.npy",
        "description": "False Negative Expert v2 - 多策略平衡训练", 
        "study_name": "fn_expert_v2_optimization",
        "storage_file": "fn_expert_v2_optimization.db"
    }
}
EXPERT_MODELS.pop("fp", None)

# 全局变量
current_expert = None
current_study = None
interrupted = False

def setup_logger():
    """设置日志记录器 v2"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"fp_fn_training_v2_{timestamp}.log")
    
    logger = logging.getLogger("fp_fn_v2")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 文件handler
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"=== FP/FN专家模型训练 v2 开始 ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"改进策略: 多比例平衡 + 混合采样 + 扩大数据集")
    
    return logger

def create_improved_balanced_dataset(X, y, target_ratio=0.3, strategy='hybrid', max_samples=500000, random_state=42):
    """
    改进的平衡数据集创建策略
    
    Args:
        X: 特征矩阵
        y: 目标变量
        target_ratio: 目标正样本比例
        strategy: 'undersample', 'oversample', 'hybrid'
        max_samples: 最大样本数限制
        random_state: 随机种子
    """
    
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    logger.info(f"原始分布: 正={n_pos}, 负={n_neg} ({n_pos/(n_pos+n_neg)*100:.2f}%)")
    logger.info(f"策略: {strategy}, 目标比例: {target_ratio}, 最大样本: {max_samples}")
    
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
    
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    # 统计结果
    n_pos_balanced = np.sum(y_balanced == 1)
    n_neg_balanced = np.sum(y_balanced == 0)
    total_balanced = len(y_balanced)
    
    logger.info(f"平衡后: 正={n_pos_balanced} ({n_pos_balanced/total_balanced*100:.1f}%), 负={n_neg_balanced}")
    logger.info(f"样本数: {len(y)} -> {total_balanced} ({total_balanced/len(y)*100:.1f}%)")
    
    return X_balanced, y_balanced

def create_objective_function_v2(X_train, y_train, X_val, y_val, expert_name):
    """改进的Optuna目标函数 v2"""
    
    def objective(trial):
        global interrupted
        
        if interrupted:
            raise optuna.TrialPruned()
        
        # 针对FP/FN任务优化的超参数空间
        param = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'tree_method': 'hist',
            'verbosity': 0,
            # 调整搜索空间 - 针对平衡数据优化
            'n_estimators': trial.suggest_int('n_estimators', 800, 2500),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 6, 18),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.3),
            'lambda': trial.suggest_float('lambda', 1e-6, 5.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-6, 5.0, log=True),
            'random_state': 42,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA,
            'device': 'cuda' if 'fp' in expert_name.lower() else 'cpu'
        }
        
        # 动态调整类别权重
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        
        # 对于平衡数据，可以尝试不同的权重策略
        weight_strategy = trial.suggest_categorical('weight_strategy', ['balanced', 'auto', 'custom'])
        
        if weight_strategy == 'balanced':
            param['scale_pos_weight'] = scale_pos_weight
        elif weight_strategy == 'auto':
            param['scale_pos_weight'] = 1.0
        else:  # custom
            param['scale_pos_weight'] = trial.suggest_float('custom_weight', 0.5, 3.0)
        
        try:
            model = xgb.XGBClassifier(**param)
            eval_set = [(X_val, y_val)]
            
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            
            # 获取预测结果
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 计算多个指标
            auc_score = roc_auc_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            # 组合评分 - 针对FP/FN任务的特殊权重
            if 'fp' in expert_name.lower():
                # FP专家更关注precision（减少误报）
                combined_score = 0.4 * auc_score + 0.4 * precision + 0.2 * f1
            else:  # FN专家
                # FN专家更关注recall（减少漏报）
                combined_score = 0.4 * auc_score + 0.4 * recall + 0.2 * f1
            
            return combined_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return 0.0
    
    return objective

def train_improved_expert_model(expert_name, expert_config, X_train_pool, y_train_pool, feature_names):
    """改进的专家模型训练函数 v2"""
    global current_expert, current_study, interrupted
    
    if interrupted:
        return None
    
    current_expert = expert_name
    expert_desc = expert_config["description"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"开始训练改进版 {expert_name.upper()} 专家模型")
    logger.info(f"描述: {expert_desc}")
    logger.info(f"{'='*80}")
    
    # 1. 加载目标标签
    target_file_path = os.path.join(TARGETS_DIR, expert_config["target_file"])
    if not os.path.exists(target_file_path):
        logger.error(f"目标文件不存在: {target_file_path}")
        return None
    
    y_expert_target = np.load(target_file_path)
    pos_count = np.sum(y_expert_target == 1)
    neg_count = np.sum(y_expert_target == 0)
    pos_ratio = pos_count / len(y_expert_target) * 100
    
    logger.info(f"原始类别分布 - 正样本: {pos_count} ({pos_ratio:.2f}%), 负样本: {neg_count}")
    
    # 2. 尝试多种平衡策略
    best_strategy_result = None
    best_auc = 0
    
    for strategy_config in BALANCE_STRATEGIES:
        logger.info(f"\n--- 测试平衡策略: {strategy_config} ---")
        
        # 创建平衡数据集
        X_balanced, y_balanced = create_improved_balanced_dataset(
            X_train_pool, y_expert_target,
            target_ratio=strategy_config['ratio'],
            strategy=strategy_config['strategy'],
            max_samples=strategy_config['max_samples'],
            random_state=RANDOM_STATE
        )
        
        # 快速评估这个策略
        X_test, X_val, y_test, y_val = train_test_split(
            X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
        )
        
        # 简单模型快速测试
        quick_model = xgb.XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=42, device='cuda' if 'fp' in expert_name.lower() else 'cpu'
        )
        quick_model.fit(X_test, y_test)
        
        quick_auc = roc_auc_score(y_val, quick_model.predict_proba(X_val)[:, 1])
        logger.info(f"策略快速评估AUC: {quick_auc:.4f}")
        
        if quick_auc > best_auc:
            best_auc = quick_auc
            best_strategy_result = {
                'config': strategy_config,
                'X_balanced': X_balanced,
                'y_balanced': y_balanced,
                'quick_auc': quick_auc
            }
    
    if best_strategy_result is None:
        logger.error("所有平衡策略都失败了")
        return None
    
    logger.info(f"\n🏆 选择最佳策略: {best_strategy_result['config']}")
    logger.info(f"预期AUC: {best_strategy_result['quick_auc']:.4f}")
    
    # 3. 使用最佳策略进行完整训练
    X_balanced = best_strategy_result['X_balanced']
    y_balanced = best_strategy_result['y_balanced']
    
    # 数据划分用于Optuna优化
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    logger.info(f"优化数据划分: 训练={X_opt_train.shape[0]}, 验证={X_opt_val.shape[0]}")
    
    # 4. Optuna超参数优化
    storage_url = f"sqlite:///{os.path.join(OPTUNA_DB_DIR, expert_config['storage_file'])}"
    study_name = expert_config["study_name"]
    
    logger.info(f"\n--- 开始Optuna超参数优化 v2 ---")
    
    try:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',
            sampler=sampler,
            load_if_exists=True
        )
        current_study = study
        
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"已完成试验: {completed_trials}")
        
        if completed_trials < N_TRIALS_OPTUNA:
            remaining_trials = N_TRIALS_OPTUNA - completed_trials
            logger.info(f"将进行 {remaining_trials} 次试验")
            
            objective_func = create_objective_function_v2(X_opt_train, y_opt_train, X_opt_val, y_opt_val, expert_name)
            
            start_time = time.time()
            study.optimize(objective_func, n_trials=remaining_trials, timeout=OPTUNA_TIMEOUT, n_jobs=1)
            end_time = time.time()
            
            logger.info(f"优化耗时: {end_time - start_time:.2f} 秒")
        
        if study.trials:
            best_trial = study.best_trial
            logger.info(f"\n🏆 最佳试验结果:")
            logger.info(f"最佳组合评分: {best_trial.value:.6f}")
            logger.info(f"最佳参数:")
            for key, value in best_trial.params.items():
                logger.info(f"  {key}: {value}")
            
            best_params = best_trial.params
        else:
            logger.error("没有完成的试验")
            return None
            
    except Exception as e:
        logger.error(f"Optuna优化失败: {e}")
        return None
    
    # 5. 最终训练和评估
    logger.info(f"\n--- 最终模型训练 ---")
    
    # 构建最终参数
    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        'random_state': 42,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL,
        'device': 'cuda' if 'fp' in expert_name.lower() else 'cpu'
    }
    
    # 添加最佳参数，但排除自定义的参数
    for key, value in best_params.items():
        if key not in ['weight_strategy', 'custom_weight']:
            final_params[key] = value
    
    # 处理权重策略
    pos_count_balanced = np.sum(y_balanced == 1)
    neg_count_balanced = np.sum(y_balanced == 0)
    
    if best_params.get('weight_strategy') == 'custom':
        final_params['scale_pos_weight'] = best_params.get('custom_weight', 1.0)
    elif best_params.get('weight_strategy') == 'balanced':
        final_params['scale_pos_weight'] = neg_count_balanced / pos_count_balanced
    else:
        final_params['scale_pos_weight'] = 1.0
    
    # 交叉验证训练
    kf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42)
    fold_models = []
    fold_metrics = []
    oof_predictions = np.zeros(len(y_balanced))
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_balanced, y_balanced)):
        if interrupted:
            break
            
        logger.info(f"\n--- 第 {fold_num + 1}/{N_SPLITS_KFOLD} 折 ---")
        
        X_fold_train, X_fold_val = X_balanced[train_idx], X_balanced[val_idx]
        y_fold_train, y_fold_val = y_balanced[train_idx], y_balanced[val_idx]
        
        model = xgb.XGBClassifier(**final_params)
        eval_set = [(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)]
        
        model.fit(X_fold_train, y_fold_train, eval_set=eval_set, verbose=False)
        
        # 预测和评估
        y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        fold_auc = roc_auc_score(y_fold_val, y_pred_proba)
        fold_precision = precision_score(y_fold_val, y_pred, zero_division=0)
        fold_recall = recall_score(y_fold_val, y_pred, zero_division=0)
        fold_f1 = f1_score(y_fold_val, y_pred, zero_division=0)
        
        logger.info(f"Fold {fold_num + 1} - AUC: {fold_auc:.4f}, Precision: {fold_precision:.4f}, Recall: {fold_recall:.4f}, F1: {fold_f1:.4f}")
        
        oof_predictions[val_idx] = y_pred_proba
        fold_models.append(model)
        fold_metrics.append({
            'auc': fold_auc, 'precision': fold_precision, 
            'recall': fold_recall, 'f1': fold_f1
        })
    
    if interrupted:
        return None
    
    # 6. 生成原始数据预测
    logger.info(f"\n--- 生成原始数据预测 ---")
    
    ensemble_predictions = np.zeros(len(y_expert_target))
    for model in fold_models:
        ensemble_predictions += model.predict_proba(X_train_pool)[:, 1]
    ensemble_predictions /= len(fold_models)
    
    # 7. 最终评估
    overall_auc = roc_auc_score(y_expert_target, ensemble_predictions)
    y_pred_overall = (ensemble_predictions >= 0.5).astype(int)
    overall_precision = precision_score(y_expert_target, y_pred_overall, zero_division=0)
    overall_recall = recall_score(y_expert_target, y_pred_overall, zero_division=0)
    overall_f1 = f1_score(y_expert_target, y_pred_overall, zero_division=0)
    
    logger.info(f"\n🎯 {expert_name.upper()} v2 最终性能:")
    logger.info(f"AUC: {overall_auc:.6f}")
    logger.info(f"Precision: {overall_precision:.6f}")
    logger.info(f"Recall: {overall_recall:.6f}")
    logger.info(f"F1-Score: {overall_f1:.6f}")
    
    # 平均折验证性能
    avg_metrics = {
        'auc': np.mean([m['auc'] for m in fold_metrics]),
        'precision': np.mean([m['precision'] for m in fold_metrics]),
        'recall': np.mean([m['recall'] for m in fold_metrics]),
        'f1': np.mean([m['f1'] for m in fold_metrics])
    }
    
    logger.info(f"平均CV性能: AUC={avg_metrics['auc']:.4f}, P={avg_metrics['precision']:.4f}, R={avg_metrics['recall']:.4f}, F1={avg_metrics['f1']:.4f}")
    
    # 8. 保存结果
    oof_save_path = os.path.join(OUTPUT_DIR, f"l1_meta_feature_{expert_name}_v2.npy")
    np.save(oof_save_path, ensemble_predictions)
    
    models_save_path = os.path.join(MODELS_DIR, f"{expert_name}_expert_v2_models.joblib")
    joblib.dump(fold_models, models_save_path)
    
    logger.info(f"预测结果保存至: {oof_save_path}")
    logger.info(f"模型保存至: {models_save_path}")
    
    return {
        'expert_name': expert_name,
        'ensemble_predictions': ensemble_predictions,
        'models': fold_models,
        'best_strategy': best_strategy_result['config'],
        'final_metrics': {
            'overall_auc': overall_auc,
            'overall_precision': overall_precision,
            'overall_recall': overall_recall,
            'overall_f1': overall_f1,
            'cv_metrics': avg_metrics
        }
    }

def signal_handler(signum, frame):
    """信号处理"""
    global interrupted, logger
    interrupted = True
    if logger:
        logger.warning(f"接收到中断信号 {signum}，正在安全退出...")
    time.sleep(2)
    sys.exit(0)

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    """主函数 v2"""
    global logger, interrupted
    
    setup_signal_handlers()
    logger = setup_logger()
    
    try:
        # 检查文件
        required_files = [X_FLAT_PATH, Y_FLAT_PATH, FEATURE_NAMES_PATH]
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"文件不存在: {file_path}")
                return
        
        for expert_name, config in EXPERT_MODELS.items():
            target_path = os.path.join(TARGETS_DIR, config["target_file"])
            if not os.path.exists(target_path):
                logger.error(f"目标文件不存在: {target_path}")
                return
        
        logger.info("所有文件检查通过")
        
        # 加载数据
        logger.info(f"\n--- 加载数据 ---")
        start_time = time.time()
        
        X_flat_full = np.load(X_FLAT_PATH, mmap_mode='r')
        Y_flat_full_raw = np.load(Y_FLAT_PATH)
        
        with open(FEATURE_NAMES_PATH, "r", encoding='utf-8') as f:
            feature_names = [line.strip() for line in f]
        
        end_time = time.time()
        logger.info(f"数据加载完成，耗时: {end_time - start_time:.2f} 秒")
        logger.info(f"特征矩阵: {X_flat_full.shape}, 目标变量: {Y_flat_full_raw.shape}")
        
        # 准备训练数据
        Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD).astype(int)
        
        indices = np.arange(len(Y_flat_full_binary))
        train_indices, test_indices, y_train_pool, y_test = train_test_split(
            indices, Y_flat_full_binary,
            test_size=TEST_SIZE_RATIO_HOLDOUT,
            random_state=42,
            stratify=Y_flat_full_binary
        )
        
        X_train_pool = X_flat_full[train_indices]
        
        logger.info(f"训练数据: {X_train_pool.shape}, 正样本比例: {np.mean(y_train_pool):.4f}")
        
        # 训练专家模型
        expert_results = {}
        
        for expert_name, expert_config in EXPERT_MODELS.items():
            if interrupted:
                break
            
            try:
                result = train_improved_expert_model(expert_name, expert_config, X_train_pool, y_train_pool, feature_names)
                if result is not None:
                    expert_results[expert_name] = result
                    logger.info(f"✅ {expert_name.upper()} v2 训练成功")
                    
                    # 显示关键指标
                    metrics = result['final_metrics']
                    logger.info(f"关键指标 - AUC: {metrics['overall_auc']:.4f}, "
                              f"Precision: {metrics['overall_precision']:.4f}, "
                              f"Recall: {metrics['overall_recall']:.4f}")
                else:
                    logger.warning(f"❌ {expert_name.upper()} v2 训练失败")
                    
            except Exception as e:
                logger.error(f"❌ {expert_name.upper()} v2 训练出错: {e}", exc_info=True)
        
        # 生成总结
        if expert_results and not interrupted:
            logger.info(f"\n{'='*80}")
            logger.info(f"FP/FN专家模型 v2 训练总结")
            logger.info(f"{'='*80}")
            
            for expert_name, result in expert_results.items():
                metrics = result['final_metrics']
                strategy = result['best_strategy']
                
                logger.info(f"\n{expert_name.upper()} 专家:")
                logger.info(f"  最佳策略: {strategy}")
                logger.info(f"  最终AUC: {metrics['overall_auc']:.6f}")
                logger.info(f"  Precision: {metrics['overall_precision']:.6f}")
                logger.info(f"  Recall: {metrics['overall_recall']:.6f}")
                logger.info(f"  F1-Score: {metrics['overall_f1']:.6f}")
            
            logger.info(f"\n🎉 训练完成！输出目录: {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"主程序出错: {e}", exc_info=True)
    finally:
        if logger:
            logger.info("=== FP/FN专家模型 v2 训练结束 ===")

if __name__ == "__main__":
    main()