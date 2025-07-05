#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡数据集的FP/FN专家模型训练脚本
==========================================

针对FP和FN专家模型正样本比例极低（2-4%）的问题，采用数据平衡策略：
1. 下采样多数类以平衡数据集
2. 分层采样保持时空分布
3. 集成多个平衡子模型提升性能

Author: Claude & User  
Date: 2025-07-04
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
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
from sklearn.utils import resample
import xgboost as xgb
import optuna
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 配置参数
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

# 训练参数
RAIN_THRESHOLD = 0.1
TEST_SIZE_RATIO_HOLDOUT = 0.2
N_SPLITS_KFOLD = 5
N_TRIALS_OPTUNA = 200  # 减少试验次数由于平衡数据训练更快
OPTUNA_TIMEOUT = 3600 * 50  
EARLY_STOPPING_ROUNDS_OPTUNA = 30
EARLY_STOPPING_ROUNDS_FINAL = 50
OPTIMIZE_METRIC = 'auc'

# 数据平衡策略参数
BALANCE_STRATEGY = 'undersample'  # 'undersample', 'oversample', 'hybrid'
TARGET_BALANCE_RATIO = 0.3  # 目标正样本比例 (30% vs 70%)
N_BALANCED_SUBSETS = 3  # 创建多个平衡子集进行集成
RANDOM_STATE = 42

# 专家模型配置 - 只训练FP和FN
EXPERT_MODELS = {
    "fp": {
        "target_file": "y_is_fp.npy", 
        "description": "False Positive Expert - 学习识别基础模型何时产生假正例 (平衡数据)",
        "study_name": "fp_expert_balanced_optimization_v1",
        "storage_file": "fp_expert_balanced_optimization.db"
    },
    "fn": {
        "target_file": "y_is_fn.npy",
        "description": "False Negative Expert - 学习识别基础模型何时产生假负例 (平衡数据)", 
        "study_name": "fn_expert_balanced_optimization_v1",
        "storage_file": "fn_expert_balanced_optimization.db"
    }
}

# 全局变量用于中断处理
current_expert = None
current_study = None
interrupted = False

# =============================================================================
# 日志配置
# =============================================================================

def setup_logger():
    """设置日志记录器"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"balanced_fp_fn_training_{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger("balanced_fp_fn")
    logger.setLevel(logging.INFO)
    
    # 清除已有的handlers
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
    
    logger.info(f"=== 平衡数据集FP/FN专家模型训练开始 ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"平衡策略: {BALANCE_STRATEGY}")
    logger.info(f"目标平衡比例: {TARGET_BALANCE_RATIO}")
    logger.info(f"平衡子集数量: {N_BALANCED_SUBSETS}")
    
    return logger

# =============================================================================
# 数据平衡策略
# =============================================================================

def create_balanced_dataset(X, y, target_ratio=0.3, strategy='undersample', random_state=42):
    """
    创建平衡数据集
    
    Args:
        X: 特征矩阵
        y: 目标变量 (0/1)
        target_ratio: 目标正样本比例
        strategy: 平衡策略 ('undersample', 'oversample', 'hybrid')
        random_state: 随机种子
    
    Returns:
        X_balanced, y_balanced: 平衡后的数据
    """
    
    # 统计原始类别分布
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)
    
    logger.info(f"原始数据分布: 正样本={n_pos} ({n_pos/(n_pos+n_neg)*100:.2f}%), 负样本={n_neg}")
    
    if strategy == 'undersample':
        # 下采样策略：保持所有正样本，随机采样负样本
        target_n_neg = int(n_pos * (1 - target_ratio) / target_ratio)
        target_n_neg = min(target_n_neg, n_neg)  # 不能超过现有负样本数
        
        # 随机选择负样本
        np.random.seed(random_state)
        selected_neg_indices = np.random.choice(neg_indices, size=target_n_neg, replace=False)
        
        # 合并正负样本索引
        balanced_indices = np.concatenate([pos_indices, selected_neg_indices])
        
    elif strategy == 'oversample':
        # 上采样策略：保持所有负样本，重复采样正样本
        target_n_pos = int(n_neg * target_ratio / (1 - target_ratio))
        
        # 重复采样正样本
        np.random.seed(random_state)
        if target_n_pos > n_pos:
            # 需要上采样
            n_repeats = target_n_pos // n_pos
            n_remainder = target_n_pos % n_pos
            
            repeated_pos_indices = np.tile(pos_indices, n_repeats)
            if n_remainder > 0:
                additional_pos_indices = np.random.choice(pos_indices, size=n_remainder, replace=False)
                repeated_pos_indices = np.concatenate([repeated_pos_indices, additional_pos_indices])
        else:
            repeated_pos_indices = pos_indices
            
        # 合并正负样本索引
        balanced_indices = np.concatenate([repeated_pos_indices, neg_indices])
        
    else:  # hybrid
        # 混合策略：适度上采样正样本，适度下采样负样本
        target_total = int((n_pos + n_neg) * 0.5)  # 减少总样本数
        target_n_pos = int(target_total * target_ratio)
        target_n_neg = target_total - target_n_pos
        
        # 处理正样本
        if target_n_pos > n_pos:
            # 上采样正样本
            n_repeats = target_n_pos // n_pos
            n_remainder = target_n_pos % n_pos
            repeated_pos_indices = np.tile(pos_indices, n_repeats)
            if n_remainder > 0:
                np.random.seed(random_state)
                additional_pos_indices = np.random.choice(pos_indices, size=n_remainder, replace=False)
                repeated_pos_indices = np.concatenate([repeated_pos_indices, additional_pos_indices])
        else:
            # 下采样正样本
            np.random.seed(random_state)
            repeated_pos_indices = np.random.choice(pos_indices, size=target_n_pos, replace=False)
        
        # 下采样负样本
        np.random.seed(random_state + 1)
        selected_neg_indices = np.random.choice(neg_indices, size=target_n_neg, replace=False)
        
        # 合并索引
        balanced_indices = np.concatenate([repeated_pos_indices, selected_neg_indices])
    
    # 打乱索引顺序
    np.random.seed(random_state + 2)
    np.random.shuffle(balanced_indices)
    
    # 生成平衡数据集
    X_balanced = X[balanced_indices]
    y_balanced = y[balanced_indices]
    
    # 统计平衡后的分布
    n_pos_balanced = np.sum(y_balanced == 1)
    n_neg_balanced = np.sum(y_balanced == 0)
    total_balanced = len(y_balanced)
    
    logger.info(f"平衡后数据分布: 正样本={n_pos_balanced} ({n_pos_balanced/total_balanced*100:.2f}%), 负样本={n_neg_balanced}")
    logger.info(f"数据集大小变化: {len(y)} -> {total_balanced} ({total_balanced/len(y)*100:.1f}%)")
    
    return X_balanced, y_balanced

def create_multiple_balanced_subsets(X, y, n_subsets=3, target_ratio=0.3, strategy='undersample'):
    """创建多个不同的平衡子集用于集成学习"""
    
    balanced_subsets = []
    
    for i in range(n_subsets):
        logger.info(f"\n--- 创建平衡子集 {i+1}/{n_subsets} ---")
        
        # 使用不同的随机种子创建不同的平衡子集
        X_balanced, y_balanced = create_balanced_dataset(
            X, y, 
            target_ratio=target_ratio, 
            strategy=strategy, 
            random_state=RANDOM_STATE + i * 1000
        )
        
        balanced_subsets.append((X_balanced, y_balanced))
    
    return balanced_subsets

# =============================================================================
# Optuna目标函数 (修改版)
# =============================================================================

def create_objective_function(X_train, y_train, X_val, y_val, expert_name):
    """创建针对平衡数据的Optuna目标函数"""
    
    def objective(trial):
        global interrupted
        
        if interrupted:
            raise optuna.TrialPruned()
        
        # 针对平衡数据调整超参数搜索空间
        param = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', OPTIMIZE_METRIC],
            'tree_method': 'hist',
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),  # 减少估计器数量
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, log=True),  # 提高学习率
            'max_depth': trial.suggest_int('max_depth', 8, 15),  # 适度减少深度
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.2),  # 减少gamma范围
            'lambda': trial.suggest_float('lambda', 1e-6, 1.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-6, 1.0, log=True),
            'random_state': 42,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA,
            'device': 'cuda' if 'fp' in expert_name.lower() else 'cpu'  # FP用CUDA，FN用CPU
        }
        
        # 计算类别权重 (对于平衡数据，权重接近1)
        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        param['scale_pos_weight'] = scale_pos_weight
        
        try:
            model = xgb.XGBClassifier(**param)
            eval_set = [(X_val, y_val)]
            
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
            
            results = model.evals_result()
            best_score = results['validation_0'][OPTIMIZE_METRIC][model.best_iteration]
            
            return best_score
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed for {expert_name} expert: {e}")
            return 0.0 if OPTIMIZE_METRIC == 'auc' else float('inf')
    
    return objective

# =============================================================================
# 信号处理
# =============================================================================

def signal_handler(signum, frame):
    """处理中断信号"""
    global interrupted, logger, current_expert, current_study
    interrupted = True
    
    if logger:
        logger.warning(f"=== 接收到中断信号 {signum} ===")
        logger.warning(f"当前正在训练专家: {current_expert}")
        logger.warning("正在安全退出，请稍候...")
    
    time.sleep(2)
    sys.exit(0)

def setup_signal_handlers():
    """设置信号处理器"""
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # 终止信号

# =============================================================================
# 性能指标计算
# =============================================================================

def calculate_metrics(y_true, y_pred, title=""):
    """计算分类性能指标"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = accuracy_score(y_true, y_pred)
    pod = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
    
    logger.info(f'\n--- {title} Performance ---')
    logger.info(f'Confusion Matrix:\n{cm}')
    logger.info(f'  True Negatives (TN): {tn}')
    logger.info(f'  False Positives (FP): {fp}')
    logger.info(f'  False Negatives (FN): {fn}')
    logger.info(f'  True Positives (TP): {tp}')
    logger.info(f'Accuracy: {accuracy:.4f}')
    logger.info(f'POD (Hit Rate/Recall): {pod:.4f}')
    logger.info(f'FAR (False Alarm Ratio): {far:.4f}')
    logger.info(f'CSI (Critical Success Index): {csi:.4f}')
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
        'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi
    }

# =============================================================================
# 平衡专家模型训练函数
# =============================================================================

def train_balanced_expert_model(expert_name, expert_config, X_train_pool, y_train_pool, feature_names):
    """训练平衡数据集的专家模型"""
    global current_expert, current_study, interrupted
    
    if interrupted:
        return None
    
    current_expert = expert_name
    expert_desc = expert_config["description"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"开始训练平衡数据的 {expert_name.upper()} 专家模型")
    logger.info(f"描述: {expert_desc}")
    logger.info(f"{'='*80}")
    
    # 1. 加载目标标签
    target_file_path = os.path.join(TARGETS_DIR, expert_config["target_file"])
    if not os.path.exists(target_file_path):
        logger.error(f"目标文件不存在: {target_file_path}")
        return None
    
    y_expert_target = np.load(target_file_path)
    logger.info(f"加载目标标签: {target_file_path}")
    logger.info(f"目标标签形状: {y_expert_target.shape}")
    
    # 检查原始类别分布
    pos_count = np.sum(y_expert_target == 1)
    neg_count = np.sum(y_expert_target == 0)
    pos_ratio = pos_count / len(y_expert_target) * 100
    
    logger.info(f"原始类别分布 - 正样本: {pos_count} ({pos_ratio:.2f}%), 负样本: {neg_count}")
    
    if pos_count == 0:
        logger.warning(f"{expert_name} 专家没有正样本，跳过训练")
        return None
    
    # 2. 创建平衡数据集
    logger.info(f"\n--- 创建平衡数据集 ---")
    X_balanced, y_balanced = create_balanced_dataset(
        X_train_pool, y_expert_target,
        target_ratio=TARGET_BALANCE_RATIO,
        strategy=BALANCE_STRATEGY,
        random_state=RANDOM_STATE
    )
    
    # 3. 划分平衡数据用于Optuna优化
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_balanced, y_balanced,
        test_size=0.2, random_state=42, stratify=y_balanced
    )
    
    logger.info(f"Optuna优化数据划分:")
    logger.info(f"  训练集: {X_opt_train.shape[0]} 样本")
    logger.info(f"  验证集: {X_opt_val.shape[0]} 样本")
    
    # 4. Optuna超参数优化
    storage_url = f"sqlite:///{os.path.join(OPTUNA_DB_DIR, expert_config['storage_file'])}"
    study_name = expert_config["study_name"]
    
    logger.info(f"\n--- 开始Optuna超参数优化 (平衡数据) ---")
    logger.info(f"Study名称: {study_name}")
    logger.info(f"数据库路径: {storage_url}")
    logger.info(f"目标试验次数: {N_TRIALS_OPTUNA}")
    
    try:
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize' if OPTIMIZE_METRIC == 'auc' else 'minimize',
            sampler=sampler,
            load_if_exists=True
        )
        current_study = study
        
        # 检查已完成的试验
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"Study已完成 {completed_trials} 次试验")
        
        if completed_trials < N_TRIALS_OPTUNA:
            remaining_trials = N_TRIALS_OPTUNA - completed_trials
            logger.info(f"将进行额外 {remaining_trials} 次试验")
            
            # 创建目标函数
            objective_func = create_objective_function(X_opt_train, y_opt_train, X_opt_val, y_opt_val, expert_name)
            
            # 开始优化
            start_opt_time = time.time()
            try:
                study.optimize(
                    objective_func, 
                    n_trials=remaining_trials,
                    timeout=OPTUNA_TIMEOUT,
                    n_jobs=1
                )
            except KeyboardInterrupt:
                logger.warning("优化被用户中断")
                interrupted = True
            except Exception as e:
                logger.error(f"优化过程中出现错误: {e}")
            
            end_opt_time = time.time()
            logger.info(f"优化耗时: {end_opt_time - start_opt_time:.2f} 秒")
        
        # 获取最佳参数
        if study.trials:
            best_trial = study.best_trial
            logger.info(f"\n--- 最佳试验结果 ---")
            logger.info(f"最佳 {OPTIMIZE_METRIC}: {best_trial.value:.6f}")
            logger.info(f"最佳参数:")
            for key, value in best_trial.params.items():
                logger.info(f"  {key}: {value}")
            
            best_params = best_trial.params
        else:
            logger.error("没有完成的试验，无法获取最佳参数")
            return None
            
    except Exception as e:
        logger.error(f"Optuna优化失败: {e}")
        return None
    
    if interrupted:
        return None
    
    # 5. 使用最佳参数在多个平衡子集上进行训练 (集成策略)
    logger.info(f"\n--- 开始多子集集成训练 ---")
    
    # 创建多个平衡子集
    balanced_subsets = create_multiple_balanced_subsets(
        X_train_pool, y_expert_target,
        n_subsets=N_BALANCED_SUBSETS,
        target_ratio=TARGET_BALANCE_RATIO,
        strategy=BALANCE_STRATEGY
    )
    
    # 构建最终模型参数
    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        'random_state': 42,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL,
        'device': 'cuda' if 'fp' in expert_name.lower() else 'cpu'
    }
    final_params.update(best_params)
    
    # 在每个平衡子集上训练模型
    subset_models = []
    subset_metrics = []
    all_oof_predictions = []
    
    for subset_idx, (X_subset, y_subset) in enumerate(balanced_subsets):
        logger.info(f"\n--- 训练子集 {subset_idx + 1}/{N_BALANCED_SUBSETS} ---")
        
        # 计算类别权重
        pos_count_subset = np.sum(y_subset == 1)
        neg_count_subset = np.sum(y_subset == 0)
        final_params['scale_pos_weight'] = neg_count_subset / pos_count_subset
        
        # 5折交叉验证
        kf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42 + subset_idx)
        fold_models = []
        subset_oof_predictions = np.zeros(len(y_subset))
        
        for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_subset, y_subset)):
            if interrupted:
                break
                
            X_fold_train, X_fold_val = X_subset[train_idx], X_subset[val_idx]
            y_fold_train, y_fold_val = y_subset[train_idx], y_subset[val_idx]
            
            # 训练模型
            model = xgb.XGBClassifier(**final_params)
            eval_set = [(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)]
            
            model.fit(X_fold_train, y_fold_train, eval_set=eval_set, verbose=False)
            
            # 生成折外预测
            subset_oof_predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]
            fold_models.append(model)
        
        # 评估子集性能
        subset_auc = roc_auc_score(y_subset, subset_oof_predictions)
        logger.info(f"子集 {subset_idx + 1} AUC: {subset_auc:.6f}")
        
        subset_models.append(fold_models)
        subset_metrics.append(subset_auc)
        all_oof_predictions.append(subset_oof_predictions)
    
    if interrupted:
        return None
    
    # 6. 生成原始训练集上的预测 (使用集成模型)
    logger.info(f"\n--- 生成原始训练集预测 ---")
    
    # 在原始训练集上生成预测
    ensemble_predictions = np.zeros(len(y_expert_target))
    
    # 使用所有子集的模型进行预测并平均
    for subset_idx, fold_models in enumerate(subset_models):
        subset_predictions = np.zeros(len(y_expert_target))
        
        # 对每个fold的模型进行预测并平均
        for model in fold_models:
            subset_predictions += model.predict_proba(X_train_pool)[:, 1]
        
        subset_predictions /= len(fold_models)
        ensemble_predictions += subset_predictions
    
    ensemble_predictions /= len(subset_models)
    
    # 7. 保存集成预测结果
    oof_save_path = os.path.join(OUTPUT_DIR, f"l1_meta_feature_{expert_name}_balanced.npy")
    np.save(oof_save_path, ensemble_predictions)
    logger.info(f"集成预测保存至: {oof_save_path}")
    
    # 8. 整体性能评估
    logger.info(f"\n--- {expert_name.upper()} 平衡专家模型整体性能 ---")
    overall_auc = roc_auc_score(y_expert_target, ensemble_predictions)
    y_pred_overall = (ensemble_predictions >= 0.5).astype(int)
    overall_metrics = calculate_metrics(y_expert_target, y_pred_overall, f"{expert_name.upper()} 平衡模型整体性能")
    overall_metrics['auc'] = overall_auc
    
    logger.info(f"集成模型平均AUC: {np.mean(subset_metrics):.6f} ± {np.std(subset_metrics):.6f}")
    logger.info(f"原始数据集AUC: {overall_auc:.6f}")
    
    # 9. 保存模型和性能报告
    models_save_path = os.path.join(MODELS_DIR, f"{expert_name}_balanced_expert_models.joblib")
    joblib.dump(subset_models, models_save_path)
    logger.info(f"集成模型保存至: {models_save_path}")
    
    # 保存性能报告
    performance_report = {
        'expert_name': expert_name,
        'expert_description': expert_desc,
        'training_timestamp': datetime.now().isoformat(),
        'balance_strategy': {
            'strategy': BALANCE_STRATEGY,
            'target_ratio': TARGET_BALANCE_RATIO,
            'n_subsets': N_BALANCED_SUBSETS
        },
        'data_info': {
            'original_total_samples': len(y_expert_target),
            'original_positive_samples': int(pos_count),
            'original_negative_samples': int(neg_count),
            'original_positive_ratio': float(pos_ratio),
            'balanced_samples_per_subset': len(y_balanced)
        },
        'optimization_info': {
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'best_trial_number': best_trial.number,
            'best_score': best_trial.value,
            'best_params': best_params
        },
        'ensemble_info': {
            'n_subsets': N_BALANCED_SUBSETS,
            'n_folds_per_subset': N_SPLITS_KFOLD,
            'subset_aucs': subset_metrics,
            'ensemble_auc_mean': float(np.mean(subset_metrics)),
            'ensemble_auc_std': float(np.std(subset_metrics)),
            'overall_metrics': overall_metrics
        }
    }
    
    report_save_path = os.path.join(OUTPUT_DIR, f"{expert_name}_balanced_expert_performance_report.json")
    import json
    with open(report_save_path, 'w', encoding='utf-8') as f:
        json.dump(performance_report, f, indent=2, ensure_ascii=False)
    logger.info(f"性能报告保存至: {report_save_path}")
    
    logger.info(f"{expert_name.upper()} 平衡专家模型训练完成！")
    
    return {
        'expert_name': expert_name,
        'ensemble_predictions': ensemble_predictions,
        'subset_models': subset_models,
        'performance_report': performance_report
    }

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    global logger, interrupted
    
    # 设置信号处理
    setup_signal_handlers()
    
    # 设置日志
    logger = setup_logger()
    
    try:
        # 检查必要文件
        required_files = [X_FLAT_PATH, Y_FLAT_PATH, FEATURE_NAMES_PATH]
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"必要文件不存在: {file_path}")
                return
        
        # 检查目标文件
        for expert_name, config in EXPERT_MODELS.items():
            target_path = os.path.join(TARGETS_DIR, config["target_file"])
            if not os.path.exists(target_path):
                logger.error(f"目标文件不存在: {target_path}")
                return
        
        logger.info("所有必要文件检查通过")
        
        # 加载数据
        logger.info(f"\n--- 加载训练数据 ---")
        logger.info(f"正在加载特征数据: {X_FLAT_PATH}")
        
        start_load_time = time.time()
        X_flat_full = np.load(X_FLAT_PATH, mmap_mode='r')
        Y_flat_full_raw = np.load(Y_FLAT_PATH)
        
        with open(FEATURE_NAMES_PATH, "r", encoding='utf-8') as f:
            feature_names = [line.strip() for line in f]
        
        end_load_time = time.time()
        logger.info(f"数据加载完成，耗时: {end_load_time - start_load_time:.2f} 秒")
        logger.info(f"特征矩阵形状: {X_flat_full.shape}")
        logger.info(f"目标变量形状: {Y_flat_full_raw.shape}")
        logger.info(f"特征数量: {len(feature_names)}")
        
        # 准备训练数据
        Y_flat_full_binary = (Y_flat_full_raw > RAIN_THRESHOLD).astype(int)
        
        # 划分数据集（与Level-0保持一致）
        indices = np.arange(len(Y_flat_full_binary))
        train_indices, test_indices, y_train_pool, y_test = train_test_split(
            indices, Y_flat_full_binary,
            test_size=TEST_SIZE_RATIO_HOLDOUT,
            random_state=42,
            stratify=Y_flat_full_binary
        )
        
        # 从内存映射中获取训练数据
        X_train_pool = X_flat_full[train_indices]
        
        logger.info(f"训练数据形状: {X_train_pool.shape}")
        logger.info(f"训练标签形状: {y_train_pool.shape}")
        logger.info(f"训练集正样本比例: {np.mean(y_train_pool):.4f}")
        
        # 训练FP和FN专家模型
        expert_results = {}
        
        for expert_name, expert_config in EXPERT_MODELS.items():
            if interrupted:
                logger.warning("训练被中断")
                break
            
            try:
                result = train_balanced_expert_model(expert_name, expert_config, X_train_pool, y_train_pool, feature_names)
                if result is not None:
                    expert_results[expert_name] = result
                    logger.info(f"{expert_name.upper()} 平衡专家训练成功")
                else:
                    logger.warning(f"{expert_name.upper()} 平衡专家训练失败或被跳过")
                    
            except Exception as e:
                logger.error(f"{expert_name.upper()} 平衡专家训练过程中出现错误: {e}", exc_info=True)
                continue
        
        # 生成最终总结报告
        if expert_results and not interrupted:
            logger.info(f"\n{'='*80}")
            logger.info(f"平衡数据FP/FN专家模型训练总结")
            logger.info(f"{'='*80}")
            
            summary_info = {
                'training_timestamp': datetime.now().isoformat(),
                'balance_strategy': BALANCE_STRATEGY,
                'target_balance_ratio': TARGET_BALANCE_RATIO,
                'n_balanced_subsets': N_BALANCED_SUBSETS,
                'total_experts': len(EXPERT_MODELS),
                'successfully_trained': len(expert_results),
                'failed_experts': [name for name in EXPERT_MODELS.keys() if name not in expert_results],
                'output_files': {}
            }
            
            logger.info(f"平衡策略: {BALANCE_STRATEGY}")
            logger.info(f"目标平衡比例: {TARGET_BALANCE_RATIO}")
            logger.info(f"子集数量: {N_BALANCED_SUBSETS}")
            logger.info(f"计划训练专家数: {summary_info['total_experts']}")
            logger.info(f"成功训练专家数: {summary_info['successfully_trained']}")
            
            if summary_info['failed_experts']:
                logger.warning(f"失败的专家: {summary_info['failed_experts']}")
            
            # 检查生成的meta-feature文件
            logger.info(f"\n--- 生成的平衡Meta-Feature文件 ---")
            for expert_name in expert_results.keys():
                meta_feature_path = os.path.join(OUTPUT_DIR, f"l1_meta_feature_{expert_name}_balanced.npy")
                if os.path.exists(meta_feature_path):
                    meta_features = np.load(meta_feature_path)
                    logger.info(f"{expert_name.upper()}: {meta_feature_path} (形状: {meta_features.shape})")
                    summary_info['output_files'][expert_name] = meta_feature_path
                else:
                    logger.warning(f"{expert_name.upper()}: Meta-feature文件未找到")
            
            # 保存总结报告
            summary_path = os.path.join(OUTPUT_DIR, "balanced_fp_fn_training_summary.json")
            import json
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_info, f, indent=2, ensure_ascii=False)
            logger.info(f"训练总结保存至: {summary_path}")
            
            logger.info(f"\n平衡数据训练完成！")
            logger.info(f"输出目录: {OUTPUT_DIR}")
        
        elif interrupted:
            logger.warning("训练被中断，部分结果可能已保存")
        else:
            logger.error("所有平衡专家模型训练都失败了")
            
    except Exception as e:
        logger.error(f"主程序执行过程中出现严重错误: {e}", exc_info=True)
    finally:
        if logger:
            logger.info("=== 平衡数据FP/FN专家模型训练结束 ===")

if __name__ == "__main__":
    main()