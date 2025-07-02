#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ensemble_v2 Level-1 Expert Models Training Script
=================================================

本脚本训练4个专家模型（TP、FP、FN、TN），每个专家模型都经过完整的Optuna超参数优化过程：
1. TP专家：学习识别基础模型何时产生真正例(True Positive)
2. FP专家：学习识别基础模型何时产生假正例(False Positive) 
3. FN专家：学习识别基础模型何时产生假负例(False Negative)
4. TN专家：学习识别基础模型何时产生真负例(True Negative)

每个专家模型的训练流程：
- Optuna超参数寻优（50次试验无提升早停）
- SQLite数据库记录优化历史
- 使用最优参数进行5折交叉验证
- 生成折外预测作为Level-2的meta-features

Author: Claude & User
Date: 2025-07-02
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
import xgboost as xgb
import optuna

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
N_TRIALS_OPTUNA = 300  # 每个专家模型的寻优试验次数
OPTUNA_TIMEOUT = 3600 * 100  # 1小时超时
EARLY_STOPPING_ROUNDS_OPTUNA = 30
EARLY_STOPPING_ROUNDS_FINAL = 50
OPTIMIZE_METRIC = 'auc'

# 专家模型配置
EXPERT_MODELS = {
    "tp": {
        "target_file": "y_is_tp.npy",
        "description": "True Positive Expert - 学习识别基础模型何时产生真正例",
        "study_name": "tp_expert_optimization_v1",
        "storage_file": "tp_expert_optimization.db"
    },
    "fp": {
        "target_file": "y_is_fp.npy", 
        "description": "False Positive Expert - 学习识别基础模型何时产生假正例",
        "study_name": "fp_expert_optimization_v1",
        "storage_file": "fp_expert_optimization.db"
    }
}
'''
    "fn": {
        "target_file": "y_is_fn.npy",
        "description": "False Negative Expert - 学习识别基础模型何时产生假负例", 
        "study_name": "fn_expert_optimization_v1",
        "storage_file": "fn_expert_optimization.db"
    },
    "tn": {
        "target_file": "y_is_tn.npy",
        "description": "True Negative Expert - 学习识别基础模型何时产生真负例",
        "study_name": "tn_expert_optimization_v1", 
        "storage_file": "tn_expert_optimization.db"
'''


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
    log_file = os.path.join(LOGS_DIR, f"ensemble_v2_l1_training_{timestamp}.log")
    
    # 创建logger
    logger = logging.getLogger("ensemble_v2_l1")
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
    
    logger.info(f"=== Ensemble_v2 Level-1 Expert Models Training Started ===")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Optuna DB directory: {OPTUNA_DB_DIR}")
    
    return logger

# =============================================================================
# 中断处理
# =============================================================================

def signal_handler(signum, frame):
    """处理中断信号"""
    global interrupted, logger, current_expert, current_study
    interrupted = True
    
    if logger:
        logger.warning(f"=== 接收到中断信号 {signum} ===")
        logger.warning(f"当前正在训练专家: {current_expert}")
        
        if current_study:
            try:
                completed_trials = len([t for t in current_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
                logger.info(f"当前Study已完成 {completed_trials} 次试验，优化历史已保存到数据库")
            except:
                logger.warning("无法获取Study状态信息")
        
        logger.warning("正在安全退出，请稍候...")
    
    # 给程序一些时间来清理
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
    logger.info('\nClassification Report:')
    logger.info(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))
    
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 
        'accuracy': accuracy, 'pod': pod, 'far': far, 'csi': csi
    }

# =============================================================================
# Optuna目标函数
# =============================================================================

def create_objective_function(X_train, y_train, X_val, y_val, expert_name):
    """创建Optuna目标函数"""
    
    def objective(trial):
        """Optuna目标函数"""
        global interrupted
        
        if interrupted:
            raise optuna.TrialPruned()
        
        # 超参数搜索空间
        param = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', OPTIMIZE_METRIC],
            'tree_method': 'hist',
            'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 10, 20),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
            'random_state': 42,
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS_OPTUNA,
            'device': 'cuda'
        }
        
        # 计算类别权重
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
# 专家模型训练函数
# =============================================================================

def train_expert_model(expert_name, expert_config, X_train_pool, y_train_pool, feature_names):
    """训练单个专家模型"""
    global current_expert, current_study, interrupted
    
    if interrupted:
        return None
    
    current_expert = expert_name
    expert_desc = expert_config["description"]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"开始训练 {expert_name.upper()} 专家模型")
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
    
    # 检查类别分布
    pos_count = np.sum(y_expert_target == 1)
    neg_count = np.sum(y_expert_target == 0)
    pos_ratio = pos_count / len(y_expert_target) * 100
    
    logger.info(f"类别分布 - 正样本: {pos_count} ({pos_ratio:.2f}%), 负样本: {neg_count}")
    
    if pos_count == 0:
        logger.warning(f"{expert_name} 专家没有正样本，跳过训练")
        return None
    
    # 2. 数据划分用于Optuna优化
    X_opt_train, X_opt_val, y_opt_train, y_opt_val = train_test_split(
        X_train_pool, y_expert_target,
        test_size=0.2, random_state=42, stratify=y_expert_target
    )
    
    logger.info(f"Optuna优化数据划分:")
    logger.info(f"  训练集: {X_opt_train.shape[0]} 样本")
    logger.info(f"  验证集: {X_opt_val.shape[0]} 样本")
    
    # 3. Optuna超参数优化
    storage_url = f"sqlite:///{os.path.join(OPTUNA_DB_DIR, expert_config['storage_file'])}"
    study_name = expert_config["study_name"]
    
    logger.info(f"\n--- 开始Optuna超参数优化 ---")
    logger.info(f"Study名称: {study_name}")
    logger.info(f"数据库路径: {storage_url}")
    logger.info(f"目标试验次数: {N_TRIALS_OPTUNA}")
    logger.info(f"早停轮次: 50次无提升")
    
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
                    n_jobs=1  # 避免并行导致的问题
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
    
    # 4. 使用最佳参数进行5折交叉验证
    logger.info(f"\n--- 开始5折交叉验证训练 ---")
    
    # 构建最终模型参数
    final_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['logloss', 'auc'],
        'tree_method': 'hist',
        'scale_pos_weight': neg_count / pos_count,
        'random_state': 42,
        'early_stopping_rounds': EARLY_STOPPING_ROUNDS_FINAL,
        'device': 'cuda'
    }
    final_params.update(best_params)
    
    logger.info(f"最终模型参数:")
    for key, value in final_params.items():
        logger.info(f"  {key}: {value}")
    
    # 5折交叉验证
    kf = StratifiedKFold(n_splits=N_SPLITS_KFOLD, shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(y_expert_target))
    fold_models = []
    fold_metrics = []
    
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X_train_pool, y_expert_target)):
        if interrupted:
            break
            
        logger.info(f"\n--- 第 {fold_num + 1}/{N_SPLITS_KFOLD} 折 ---")
        
        X_fold_train, X_fold_val = X_train_pool[train_idx], X_train_pool[val_idx]
        y_fold_train, y_fold_val = y_expert_target[train_idx], y_expert_target[val_idx]
        
        logger.info(f"训练集大小: {X_fold_train.shape[0]}")
        logger.info(f"验证集大小: {X_fold_val.shape[0]}")
        logger.info(f"验证集正样本比例: {np.mean(y_fold_val):.4f}")
        
        # 训练模型
        model = xgb.XGBClassifier(**final_params)
        eval_set = [(X_fold_train, y_fold_train), (X_fold_val, y_fold_val)]
        
        start_fold_time = time.time()
        model.fit(X_fold_train, y_fold_train, eval_set=eval_set, verbose=50)
        end_fold_time = time.time()
        
        logger.info(f"第 {fold_num + 1} 折训练耗时: {end_fold_time - start_fold_time:.2f} 秒")
        logger.info(f"最佳迭代: {model.best_iteration}")
        
        # 生成折外预测
        oof_predictions[val_idx] = model.predict_proba(X_fold_val)[:, 1]
        
        # 评估性能
        y_pred_binary = (model.predict_proba(X_fold_val)[:, 1] >= 0.5).astype(int)
        fold_auc = roc_auc_score(y_fold_val, model.predict_proba(X_fold_val)[:, 1])
        metrics = calculate_metrics(y_fold_val, y_pred_binary, f"{expert_name.upper()} 第{fold_num + 1}折")
        metrics['auc'] = fold_auc
        fold_metrics.append(metrics)
        
        # 保存模型
        model_save_path = os.path.join(MODELS_DIR, f"{expert_name}_expert_fold_{fold_num + 1}.joblib")
        joblib.dump(model, model_save_path)
        logger.info(f"模型保存至: {model_save_path}")
        
        fold_models.append(model)
    
    if interrupted:
        return None
    
    # 6. 保存折外预测
    oof_save_path = os.path.join(OUTPUT_DIR, f"l1_meta_feature_{expert_name}.npy")
    np.save(oof_save_path, oof_predictions)
    logger.info(f"折外预测保存至: {oof_save_path}")
    
    # 7. 整体性能评估
    logger.info(f"\n--- {expert_name.upper()} 专家模型整体性能 ---")
    overall_auc = roc_auc_score(y_expert_target, oof_predictions)
    y_pred_overall = (oof_predictions >= 0.5).astype(int)
    overall_metrics = calculate_metrics(y_expert_target, y_pred_overall, f"{expert_name.upper()} 整体性能")
    overall_metrics['auc'] = overall_auc
    
    # 8. 保存性能报告
    performance_report = {
        'expert_name': expert_name,
        'expert_description': expert_desc,
        'training_timestamp': datetime.now().isoformat(),
        'data_info': {
            'total_samples': len(y_expert_target),
            'positive_samples': int(pos_count),
            'negative_samples': int(neg_count),
            'positive_ratio': float(pos_ratio)
        },
        'optimization_info': {
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'best_trial_number': best_trial.number,
            'best_score': best_trial.value,
            'best_params': best_params
        },
        'cross_validation_info': {
            'n_folds': N_SPLITS_KFOLD,
            'fold_metrics': fold_metrics,
            'overall_metrics': overall_metrics
        }
    }
    
    report_save_path = os.path.join(OUTPUT_DIR, f"{expert_name}_expert_performance_report.json")
    import json
    with open(report_save_path, 'w', encoding='utf-8') as f:
        json.dump(performance_report, f, indent=2, ensure_ascii=False)
    logger.info(f"性能报告保存至: {report_save_path}")
    
    logger.info(f"{expert_name.upper()} 专家模型训练完成！")
    return {
        'expert_name': expert_name,
        'oof_predictions': oof_predictions,
        'models': fold_models,
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
        X_flat_full = np.load(X_FLAT_PATH, mmap_mode='r')  # 使用内存映射
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
        
        # 训练各个专家模型
        expert_results = {}
        
        for expert_name, expert_config in EXPERT_MODELS.items():
            if interrupted:
                logger.warning("训练被中断")
                break
            
            try:
                result = train_expert_model(expert_name, expert_config, X_train_pool, y_train_pool, feature_names)
                if result is not None:
                    expert_results[expert_name] = result
                    logger.info(f"{expert_name.upper()} 专家训练成功")
                else:
                    logger.warning(f"{expert_name.upper()} 专家训练失败或被跳过")
                    
            except Exception as e:
                logger.error(f"{expert_name.upper()} 专家训练过程中出现错误: {e}", exc_info=True)
                continue
        
        # 生成最终总结报告
        if expert_results and not interrupted:
            logger.info(f"\n{'='*80}")
            logger.info(f"Ensemble_v2 Level-1 专家模型训练总结")
            logger.info(f"{'='*80}")
            
            summary_info = {
                'training_timestamp': datetime.now().isoformat(),
                'total_experts': len(EXPERT_MODELS),
                'successfully_trained': len(expert_results),
                'failed_experts': [name for name in EXPERT_MODELS.keys() if name not in expert_results],
                'output_files': {}
            }
            
            logger.info(f"计划训练专家数: {summary_info['total_experts']}")
            logger.info(f"成功训练专家数: {summary_info['successfully_trained']}")
            
            if summary_info['failed_experts']:
                logger.warning(f"失败的专家: {summary_info['failed_experts']}")
            
            # 检查生成的meta-feature文件
            logger.info(f"\n--- 生成的Meta-Feature文件 ---")
            for expert_name in expert_results.keys():
                meta_feature_path = os.path.join(OUTPUT_DIR, f"l1_meta_feature_{expert_name}.npy")
                if os.path.exists(meta_feature_path):
                    meta_features = np.load(meta_feature_path)
                    logger.info(f"{expert_name.upper()}: {meta_feature_path} (形状: {meta_features.shape})")
                    summary_info['output_files'][expert_name] = meta_feature_path
                else:
                    logger.warning(f"{expert_name.upper()}: Meta-feature文件未找到")
            
            # 保存总结报告
            summary_path = os.path.join(OUTPUT_DIR, "ensemble_v2_l1_training_summary.json")
            import json
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary_info, f, indent=2, ensure_ascii=False)
            logger.info(f"训练总结保存至: {summary_path}")
            
            logger.info(f"\n训练完成！准备进行Level-2融合模型训练。")
            logger.info(f"输出目录: {OUTPUT_DIR}")
        
        elif interrupted:
            logger.warning("训练被中断，部分结果可能已保存")
        else:
            logger.error("所有专家模型训练都失败了")
            
    except Exception as e:
        logger.error(f"主程序执行过程中出现严重错误: {e}", exc_info=True)
    finally:
        if logger:
            logger.info("=== Ensemble_v2 Level-1 Expert Models Training Finished ===")

if __name__ == "__main__":
    main()