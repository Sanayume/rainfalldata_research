#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FN专家快速优化方案
==================

针对你的FN专家训练困难，提供具体的优化建议和策略

Author: Claude & User
Date: 2025-07-05
"""

import numpy as np
import pandas as pd

def analyze_fn_training_difficulties():
    """
    分析FN专家训练困难的原因
    """
    print("=" * 80)
    print("FN专家训练困难分析")
    print("=" * 80)
    
    print("\n🎯 核心问题分析：")
    print("1. 更严重的类别不平衡")
    print("   - FN事件仅占3.72%，比FP的2.23%略高但更难学习")
    print("   - FN代表'漏报'，通常发生在边界情况或异常天气下")
    
    print("\n2. FN事件的本质特征")
    print("   - 信号微弱：真实降雨但强度接近检测阈值")
    print("   - 产品分歧：不同卫星产品对同一事件判断不一致")
    print("   - 时空复杂性：局地性强降雨或极端天气事件")
    
    print("\n3. 特征表征困难")
    print("   - FN事件往往是多因素复合的结果")
    print("   - 需要更复杂的时序依赖和空间关联特征")

def provide_fn_optimization_strategies():
    """
    提供FN专家优化策略
    """
    print("\n" + "=" * 80)
    print("FN专家优化策略")
    print("=" * 80)
    
    print("\n🚀 立即可用的优化方案：")
    
    print("\n1. 调整评估指标权重")
    print("   修改train_balanced_fp_fn_experts.py中的eval_metric：")
    print("   ```python")
    print("   def weighted_score(y_true, y_pred):")
    print("       recall = recall_score(y_true, y_pred)")
    print("       precision = precision_score(y_true, y_pred)")
    print("       f1 = f1_score(y_true, y_pred)")
    print("       auc = roc_auc_score(y_true, y_pred)")
    print("       # FN专家重视召回率")
    print("       return 0.5 * recall + 0.2 * f1 + 0.2 * auc + 0.1 * precision")
    print("   ```")
    
    print("\n2. 更激进的数据平衡")
    print("   在train_balanced_fp_fn_experts.py中调整FN专家的平衡策略：")
    print("   ```python")
    print("   # 针对FN专家的特殊平衡")
    print("   if expert_type == 'FN':")
    print("       target_ratio = 0.4  # 提高到40%正样本")
    print("       max_samples = 300000  # 减少总样本数加速训练")
    print("   ```")
    
    print("\n3. 特征工程增强")
    print("   添加FN专门的特征：")
    print("   ```python")
    print("   # 强信号检测特征")
    print("   df['intensity_signal'] = df['multi_product_max'] / df['multi_product_mean']")
    print("   # 一致性特征")
    print("   df['consistency_index'] = 1.0 / (1.0 + df['multi_product_std'] / df['multi_product_mean'])")
    print("   # 极值事件特征")
    print("   df['is_extreme'] = (df['multi_product_max'] > df['multi_product_max'].quantile(0.95)).astype(float)")
    print("   ```")
    
    print("\n4. 成本敏感学习")
    print("   在XGBoost参数中增加：")
    print("   ```python")
    print("   xgb_params = {")
    print("       'scale_pos_weight': 20,  # FN专家使用更高的权重")
    print("       'max_delta_step': 1,     # 控制更新步长")
    print("       'eval_metric': 'aucpr',  # 使用PR-AUC作为评估指标")
    print("   }")
    print("   ```")

def suggest_immediate_actions():
    """
    建议立即可执行的操作
    """
    print("\n" + "=" * 80)
    print("立即可执行的操作建议")
    print("=" * 80)
    
    print("\n✅ 快速改进方案（5分钟内完成）：")
    
    print("\n1. 修改现有FN专家训练参数")
    print("   在当前的训练脚本中添加这些参数：")
    
    fn_optuna_params = """
    # FN专家专门的Optuna参数空间
    'scale_pos_weight': trial.suggest_float('scale_pos_weight', 15.0, 50.0),
    'max_delta_step': trial.suggest_float('max_delta_step', 0.5, 2.0),
    'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # 提高子采样比例
    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
    """
    
    print(f"   {fn_optuna_params}")
    
    print("\n2. 调整数据预处理")
    print("   在数据平衡阶段：")
    print("   - 将FN专家的positive_ratio提高到0.35-0.4")
    print("   - 减少max_samples到200000-300000以加速训练")
    print("   - 使用'hybrid'策略结合过采样和欠采样")
    
    print("\n3. 修改评估函数")
    print("   重点关注召回率和PR-AUC：")
    print("   ```python")
    print("   def fn_evaluation_score(y_true, y_pred_proba, y_pred):")
    print("       recall = recall_score(y_true, y_pred)")
    print("       pr_auc = average_precision_score(y_true, y_pred_proba)")
    print("       f1 = f1_score(y_true, y_pred)")
    print("       return 0.6 * recall + 0.3 * pr_auc + 0.1 * f1")
    print("   ```")

def create_fn_config_template():
    """
    创建FN专家配置模板
    """
    print("\n" + "=" * 80)
    print("FN专家配置模板")
    print("=" * 80)
    
    config = {
        'data_balance': {
            'positive_ratio': 0.4,
            'strategy': 'hybrid',
            'max_samples': 250000,
            'enable_smote': True
        },
        'xgb_params': {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'scale_pos_weight': 25.0,
            'max_delta_step': 1.0,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'eval_metric': 'aucpr'
        },
        'evaluation': {
            'primary_metric': 'recall',
            'weights': {
                'recall': 0.5,
                'pr_auc': 0.3,
                'f1': 0.2
            }
        },
        'feature_engineering': {
            'add_intensity_features': True,
            'add_consistency_features': True,
            'add_extreme_features': True,
            'add_temporal_features': True
        }
    }
    
    print("推荐的FN专家配置：")
    for section, params in config.items():
        print(f"\n[{section}]")
        for key, value in params.items():
            print(f"  {key}: {value}")

def main():
    """
    主函数：提供完整的FN专家优化分析
    """
    analyze_fn_training_difficulties()
    provide_fn_optimization_strategies()
    suggest_immediate_actions()
    create_fn_config_template()
    
    print("\n" + "=" * 80)
    print("总结和建议")
    print("=" * 80)
    
    print("\n🎯 关键洞察：")
    print("1. FP专家成功（AUC: 0.9998）说明方法论正确")
    print("2. FN专家困难在于事件稀少性和复杂性")
    print("3. 需要专门针对FN的优化策略")
    
    print("\n🚀 优先执行顺序：")
    print("1. 调整现有脚本的FN专家参数（scale_pos_weight, positive_ratio）")
    print("2. 修改评估指标重视召回率")
    print("3. 添加FN专门的特征工程")
    print("4. 考虑多模型集成（如果单模型效果仍不佳）")
    
    print("\n💡 预期改进：")
    print("- 通过参数调整，FN专家AUC可能提升到0.85-0.90")
    print("- 重视召回率的评估可能发现隐藏的优势")
    print("- 特征工程可能显著提升FN检测能力")

if __name__ == "__main__":
    main()