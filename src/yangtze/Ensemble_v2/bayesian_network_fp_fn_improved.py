#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
贝叶斯网络FP/FN专家模型改进版
================================

基于原版本的改进：
1. 修复数据路径问题
2. 集成真实的TP/TN专家预测结果
3. 添加更详细的评估和可视化
4. 优化网络结构学习

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
    print("正在安装pgmpy库...")
    os.system("pip install pgmpy")
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
    except ImportError:
        from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
    from pgmpy.estimators import HillClimbSearch, ExhaustiveSearch
    from pgmpy.inference import VariableElimination
    from pgmpy.factors.discrete import TabularCPD

# 导入原有的BayesianNetworkFPFNExpert类
from baysian_network_fp_fn import BayesianNetworkFPFNExpert, BayesianNetworkOptimizer

# 数据处理和评估
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix,
                           matthews_corrcoef, balanced_accuracy_score)
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def load_yangtze_v6_data():
    """
    加载长江流域V6特征集数据
    """
    logger.info("加载长江流域V6数据...")
    
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
    logger.info(f"特征目录: {features_dir}")
    
    try:
        # 加载V6特征集
        X = np.load(os.path.join(features_dir, "X_Yangtsu_flat_features_v6.npy"))
        y = np.load(os.path.join(features_dir, "Y_Yangtsu_flat_target_v6.npy"))
        
        # 读取特征名称
        feature_names_file = os.path.join(features_dir, "feature_names_yangtsu_v6.txt")
        if os.path.exists(feature_names_file):
            with open(feature_names_file, 'r', encoding='utf-8') as f:
                feature_names = [line.strip() for line in f.readlines()]
        else:
            logger.warning("特征名称文件不存在，使用默认名称")
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 创建DataFrame
        X_df = pd.DataFrame(X, columns=feature_names)
        
        # 二值化目标变量
        y_binary = (y > 0.1).astype(int)  # 0.1mm阈值
        
        logger.info(f"加载特征矩阵: {X.shape}")
        logger.info(f"特征数量: {len(feature_names)}")
        logger.info(f"目标变量分布: {np.unique(y_binary, return_counts=True)}")
        
        return X_df, y_binary
        
    except FileNotFoundError as e:
        logger.error(f"数据文件未找到: {e}")
        raise

def load_tp_tn_expert_predictions():
    """
    加载TP/TN专家的预测结果
    """
    logger.info("加载TP/TN专家预测结果...")
    
    # 使用相对路径，自动适配Windows和Linux
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ensemble_dir = os.path.join(current_dir, "Ensemble_v2")
    
    try:
        # 尝试加载已存在的meta-features
        tp_pred_path = os.path.join(ensemble_dir, "l1_meta_feature_tp.npy")
        tn_pred_path = os.path.join(ensemble_dir, "l1_meta_feature_tn.npy")
        
        logger.info(f"查找TP预测文件: {tp_pred_path}")
        logger.info(f"查找TN预测文件: {tn_pred_path}")
        
        if os.path.exists(tp_pred_path) and os.path.exists(tn_pred_path):
            tp_predictions = np.load(tp_pred_path)
            tn_predictions = np.load(tn_pred_path)
            
            logger.info(f"加载TP专家预测: {tp_predictions.shape}")
            logger.info(f"加载TN专家预测: {tn_predictions.shape}")
            
            return tp_predictions, tn_predictions
        else:
            logger.warning("TP/TN专家预测文件不存在，将使用模拟预测")
            return None, None
            
    except Exception as e:
        logger.error(f"加载TP/TN预测失败: {e}")
        return None, None

def generate_enhanced_fp_fn_labels(X: pd.DataFrame, y: np.ndarray, 
                                 tp_predictions=None, tn_predictions=None,
                                 threshold=0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用TP/TN专家预测结果生成高质量的FP/FN标签
    """
    logger.info("生成增强的FP/FN标签...")
    
    if tp_predictions is not None and tn_predictions is not None:
        # 使用真实的TP/TN专家预测
        logger.info("使用TP/TN专家预测结果")
        
        # 组合TP/TN预测形成基础预测
        # 简单策略：如果TP概率高或TN概率低，则预测为正类
        tp_prob = tp_predictions
        tn_prob = tn_predictions
        
        # 基础预测逻辑：TP高概率或TN低概率 -> 预测为雨
        base_pred_prob = 0.6 * tp_prob + 0.4 * (1 - tn_prob)
        y_pred = (base_pred_prob > threshold).astype(int)
        
        logger.info(f"基础模型预测分布: {np.unique(y_pred, return_counts=True)}")
        
    else:
        # 使用特征生成模拟预测
        logger.warning("使用特征模拟基础预测")
        
        if 'multi_product_mean' in X.columns:
            # 基于多产品均值的简单预测模型
            base_feature = X['multi_product_mean'].values
            # 标准化
            base_feature_norm = (base_feature - base_feature.min()) / (base_feature.max() - base_feature.min())
            y_pred = (base_feature_norm > threshold).astype(int)
        else:
            # 随机生成（仅用于测试）
            np.random.seed(42)
            y_pred = np.random.binomial(1, 0.6, len(y))
    
    # 生成FP/FN标签
    fp_labels = ((y == 0) & (y_pred == 1)).astype(int)  # False Positive: 实际无雨，预测有雨
    fn_labels = ((y == 1) & (y_pred == 0)).astype(int)  # False Negative: 实际有雨，预测无雨
    
    # 统计信息
    total_samples = len(y)
    fp_count = np.sum(fp_labels)
    fn_count = np.sum(fn_labels)
    tp_count = np.sum((y == 1) & (y_pred == 1))
    tn_count = np.sum((y == 0) & (y_pred == 0))
    
    logger.info(f"混淆矩阵统计:")
    logger.info(f"  TP: {tp_count} ({tp_count/total_samples*100:.2f}%)")
    logger.info(f"  TN: {tn_count} ({tn_count/total_samples*100:.2f}%)")
    logger.info(f"  FP: {fp_count} ({fp_count/total_samples*100:.2f}%) <- 目标")
    logger.info(f"  FN: {fn_count} ({fn_count/total_samples*100:.2f}%) <- 目标")
    
    return fp_labels, fn_labels

def evaluate_bayesian_experts(fp_expert, fn_expert, X, fp_labels, fn_labels):
    """
    评估贝叶斯专家模型性能
    """
    logger.info("评估贝叶斯专家模型...")
    
    results = {}
    
    # 评估FP专家
    if fp_expert is not None and np.sum(fp_labels) > 0:
        fp_metrics = fp_expert.evaluate(X, fp_labels)
        results['FP'] = fp_metrics
        
        logger.info(f"FP专家性能:")
        for metric, value in fp_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # 评估FN专家
    if fn_expert is not None and np.sum(fn_labels) > 0:
        fn_metrics = fn_expert.evaluate(X, fn_labels)
        results['FN'] = fn_metrics
        
        logger.info(f"FN专家性能:")
        for metric, value in fn_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    return results

def visualize_bayesian_network_results(fp_expert, fn_expert, evaluation_results):
    """
    可视化贝叶斯网络结果
    """
    logger.info("生成可视化结果...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('贝叶斯网络FP/FN专家模型性能分析', fontsize=16, fontweight='bold')
    
    # 1. 性能对比
    ax1 = axes[0, 0]
    if 'FP' in evaluation_results and 'FN' in evaluation_results:
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        fp_values = [evaluation_results['FP'].get(m, 0) for m in metrics]
        fn_values = [evaluation_results['FN'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, fp_values, width, label='FP Expert', alpha=0.8, color='red')
        ax1.bar(x + width/2, fn_values, width, label='FN Expert', alpha=0.8, color='green')
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('FP vs FN Expert Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # 2. 网络复杂度对比
    ax2 = axes[0, 1]
    complexities = []
    expert_names = []
    
    if fp_expert and hasattr(fp_expert, 'model') and fp_expert.model:
        fp_nodes = len(fp_expert.model.nodes())
        fp_edges = len(fp_expert.model.edges())
        complexities.append([fp_nodes, fp_edges])
        expert_names.append('FP Expert')
    
    if fn_expert and hasattr(fn_expert, 'model') and fn_expert.model:
        fn_nodes = len(fn_expert.model.nodes())
        fn_edges = len(fn_expert.model.edges())
        complexities.append([fn_nodes, fn_edges])
        expert_names.append('FN Expert')
    
    if complexities:
        complexities = np.array(complexities)
        x = np.arange(len(expert_names))
        width = 0.35
        
        ax2.bar(x - width/2, complexities[:, 0], width, label='Nodes', alpha=0.8)
        ax2.bar(x + width/2, complexities[:, 1], width, label='Edges', alpha=0.8)
        
        ax2.set_xlabel('Expert Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Network Complexity')
        ax2.set_xticks(x)
        ax2.set_xticklabels(expert_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # 3. 特征重要性（如果可获取）
    ax3 = axes[1, 0]
    ax3.text(0.5, 0.5, 'Feature Importance Analysis\n(待实现)', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Feature Importance')
    
    # 4. 不确定性分析
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.5, 'Uncertainty Analysis\n(待实现)', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Prediction Uncertainty')
    
    plt.tight_layout()
    
    # 保存图片
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, "Ensemble_v2", "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_path = os.path.join(output_dir, "bayesian_network_fp_fn_analysis.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化结果已保存: {plot_path}")
    
    plt.show()

def main():
    """
    改进版主函数
    """
    logger.info("=" * 80)
    logger.info("贝叶斯网络FP/FN专家模型训练 - 改进版")
    logger.info("=" * 80)
    
    try:
        # 1. 加载数据
        X, y = load_yangtze_v6_data()
        
        # 2. 加载TP/TN专家预测
        tp_predictions, tn_predictions = load_tp_tn_expert_predictions()
        
        # 3. 生成FP/FN标签
        fp_labels, fn_labels = generate_enhanced_fp_fn_labels(
            X, y, tp_predictions, tn_predictions
        )
        
        # 4. 检查数据质量
        min_samples_required = 50
        
        fp_expert = None
        fn_expert = None
        
        # 5. 训练FP专家
        if np.sum(fp_labels) >= min_samples_required:
            logger.info(f"\n{'='*50}")
            logger.info("训练FP专家 (贝叶斯网络)")
            logger.info(f"{'='*50}")
            
            fp_optimizer = BayesianNetworkOptimizer('FP', X, fp_labels, cv_folds=3)
            fp_results = fp_optimizer.optimize(n_trials=20)  # 减少试验次数以节省时间
            
            # 训练最终FP专家
            fp_expert = BayesianNetworkFPFNExpert('FP')
            fp_expert.fit(X, fp_labels, **fp_results['best_params'])
            
            # 保存FP专家
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(current_dir, "Ensemble_v2", "models")
            os.makedirs(output_dir, exist_ok=True)
            
            fp_save_path = os.path.join(output_dir, 'bayesian_fp_expert_v2.pkl')
            joblib.dump(fp_expert, fp_save_path)
            logger.info(f"FP专家已保存: {fp_save_path}")
            
        else:
            logger.warning(f"FP样本太少 ({np.sum(fp_labels)}<{min_samples_required})，跳过训练")
        
        # 6. 训练FN专家
        if np.sum(fn_labels) >= min_samples_required:
            logger.info(f"\n{'='*50}")
            logger.info("训练FN专家 (贝叶斯网络)")
            logger.info(f"{'='*50}")
            
            fn_optimizer = BayesianNetworkOptimizer('FN', X, fn_labels, cv_folds=3)
            fn_results = fn_optimizer.optimize(n_trials=20)
            
            # 训练最终FN专家
            fn_expert = BayesianNetworkFPFNExpert('FN')
            fn_expert.fit(X, fn_labels, **fn_results['best_params'])
            
            # 保存FN专家
            fn_save_path = os.path.join(output_dir, 'bayesian_fn_expert_v2.pkl')
            joblib.dump(fn_expert, fn_save_path)
            logger.info(f"FN专家已保存: {fn_save_path}")
            
        else:
            logger.warning(f"FN样本太少 ({np.sum(fn_labels)}<{min_samples_required})，跳过训练")
        
        # 7. 评估模型
        if fp_expert or fn_expert:
            evaluation_results = evaluate_bayesian_experts(fp_expert, fn_expert, X, fp_labels, fn_labels)
            
            # 8. 生成可视化
            visualize_bayesian_network_results(fp_expert, fn_expert, evaluation_results)
            
            # 9. 保存预测结果
            if fp_expert:
                fp_predictions = fp_expert.predict_proba(X)[:, 1]
                fp_pred_path = os.path.join(output_dir, 'bayesian_fp_predictions.npy')
                np.save(fp_pred_path, fp_predictions)
                logger.info(f"FP预测结果已保存: {fp_pred_path}")
            
            if fn_expert:
                fn_predictions = fn_expert.predict_proba(X)[:, 1]
                fn_pred_path = os.path.join(output_dir, 'bayesian_fn_predictions.npy')
                np.save(fn_pred_path, fn_predictions)
                logger.info(f"FN预测结果已保存: {fn_pred_path}")
        
        logger.info(f"\n{'='*80}")
        logger.info("贝叶斯网络FP/FN专家模型训练完成")
        logger.info(f"{'='*80}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}", exc_info=True)

if __name__ == "__main__":
    main()