#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FN专家集成策略改进
==================

基于FN专家的特殊困难，设计多层次集成策略

Author: Claude & User  
Date: 2025-07-05
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FNEnsembleStrategy:
    """
    FN专家集成策略类
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.base_models = {}
        self.ensemble_model = None
        self.meta_model = None
        
    def create_diverse_fn_experts(self, X: pd.DataFrame, y: np.ndarray):
        """
        创建多样化的FN专家模型
        """
        logger.info("创建多样化的FN专家模型...")
        
        # 1. 基于不同数据平衡策略的专家
        strategies = [
            {'name': 'balanced_expert', 'balance_ratio': 0.3, 'scale_pos_weight': 15},
            {'name': 'recall_focused', 'balance_ratio': 0.4, 'scale_pos_weight': 25},
            {'name': 'precision_focused', 'balance_ratio': 0.2, 'scale_pos_weight': 10}
        ]
        
        for strategy in strategies:
            model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.9,
                scale_pos_weight=strategy['scale_pos_weight'],
                random_state=self.random_state,
                eval_metric='aucpr'
            )
            
            # 简单数据平衡
            pos_indices = np.where(y == 1)[0]
            neg_indices = np.where(y == 0)[0]
            
            n_pos = len(pos_indices)
            n_neg = int(n_pos * (1 - strategy['balance_ratio']) / strategy['balance_ratio'])
            n_neg = min(n_neg, len(neg_indices))
            
            selected_neg = np.random.choice(neg_indices, size=n_neg, replace=False)
            balanced_indices = np.concatenate([pos_indices, selected_neg])
            
            X_balanced = X.iloc[balanced_indices]
            y_balanced = y[balanced_indices]
            
            model.fit(X_balanced, y_balanced)
            self.base_models[strategy['name']] = model
            
            logger.info(f"训练完成：{strategy['name']}")
        
        # 2. 基于不同特征子集的专家
        feature_groups = {
            'intensity_expert': ['multi_product_max', 'multi_product_mean', 'product_range'],
            'consistency_expert': ['multi_product_std', 'coef_of_variation', 'rain_product_count'],
            'temporal_expert': ['lag_1_multi_product_mean', 'diff_multi_product_mean', 'window_3d_mean']
        }
        
        for expert_name, features in feature_groups.items():
            available_features = [f for f in features if f in X.columns]
            if len(available_features) >= 2:
                X_subset = X[available_features]
                
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=5,
                    learning_rate=0.1,
                    scale_pos_weight=20,
                    random_state=self.random_state
                )
                
                # 使用适中的平衡策略
                pos_indices = np.where(y == 1)[0]
                neg_indices = np.where(y == 0)[0]
                n_neg = min(len(pos_indices) * 3, len(neg_indices))
                selected_neg = np.random.choice(neg_indices, size=n_neg, replace=False)
                balanced_indices = np.concatenate([pos_indices, selected_neg])
                
                X_balanced = X_subset.iloc[balanced_indices]
                y_balanced = y[balanced_indices]
                
                model.fit(X_balanced, y_balanced)
                self.base_models[expert_name] = {'model': model, 'features': available_features}
                
                logger.info(f"训练完成：{expert_name} (特征：{available_features})")
    
    def create_stacking_ensemble(self, X: pd.DataFrame, y: np.ndarray):
        """
        创建堆叠集成模型
        """
        logger.info("创建堆叠集成模型...")
        
        # 准备基础模型列表
        base_estimators = []
        
        # 添加完整特征的模型
        for name, model in self.base_models.items():
            if isinstance(model, dict):
                continue  # 跳过特征子集模型
            base_estimators.append((name, model))
        
        # 元学习器：专门为FN优化的逻辑回归
        meta_learner = LogisticRegression(
            class_weight='balanced',
            C=0.1,  # 较强的正则化
            random_state=self.random_state
        )
        
        # 创建堆叠分类器
        self.ensemble_model = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,  # 3折交叉验证
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # 数据平衡
        pos_indices = np.where(y == 1)[0]
        neg_indices = np.where(y == 0)[0]
        n_neg = min(len(pos_indices) * 2, len(neg_indices))  # 1:2比例
        selected_neg = np.random.choice(neg_indices, size=n_neg, replace=False)
        balanced_indices = np.concatenate([pos_indices, selected_neg])
        
        X_balanced = X.iloc[balanced_indices]
        y_balanced = y[balanced_indices]
        
        # 训练堆叠模型
        self.ensemble_model.fit(X_balanced, y_balanced)
        logger.info("堆叠集成模型训练完成")
    
    def create_weighted_voting_ensemble(self, X: pd.DataFrame, y: np.ndarray):
        """
        创建加权投票集成
        """
        logger.info("创建加权投票集成...")
        
        # 计算各模型在验证集上的召回率作为权重
        from sklearn.model_selection import train_test_split
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        model_weights = {}
        
        for name, model in self.base_models.items():
            if isinstance(model, dict):
                continue  # 跳过特征子集模型
                
            try:
                y_pred = model.predict(X_val)
                recall = recall_score(y_val, y_pred, zero_division=0)
                model_weights[name] = max(0.1, recall)  # 最小权重0.1
                logger.info(f"{name} 召回率: {recall:.4f}, 权重: {model_weights[name]:.4f}")
            except Exception as e:
                logger.warning(f"评估模型 {name} 失败: {e}")
                model_weights[name] = 0.1
        
        # 归一化权重
        total_weight = sum(model_weights.values())
        for name in model_weights:
            model_weights[name] /= total_weight
        
        # 存储权重用于预测
        self.model_weights = model_weights
        logger.info(f"权重分配: {model_weights}")
    
    def ensemble_predict_proba(self, X: pd.DataFrame):
        """
        集成预测概率
        """
        if self.ensemble_model is not None:
            # 使用堆叠模型
            return self.ensemble_model.predict_proba(X)[:, 1]
        
        # 使用加权投票
        weighted_proba = np.zeros(len(X))
        
        for name, model in self.base_models.items():
            if isinstance(model, dict):
                # 特征子集模型
                X_subset = X[model['features']]
                proba = model['model'].predict_proba(X_subset)[:, 1]
                weight = self.model_weights.get(name, 0.1)
            else:
                # 完整特征模型
                proba = model.predict_proba(X)[:, 1]
                weight = self.model_weights.get(name, 0.1)
            
            weighted_proba += weight * proba
        
        return weighted_proba
    
    def evaluate_ensemble(self, X: pd.DataFrame, y: np.ndarray):
        """
        评估集成模型
        """
        logger.info("评估集成FN专家...")
        
        y_proba = self.ensemble_predict_proba(X)
        y_pred = (y_proba > 0.5).astype(int)
        
        metrics = {
            'recall': recall_score(y, y_pred, zero_division=0),
            'precision': precision_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.5
        }
        
        # FN专门的综合评分
        fn_score = 0.5 * metrics['recall'] + 0.3 * metrics['f1'] + 0.2 * metrics['auc']
        metrics['fn_score'] = fn_score
        
        logger.info("集成FN专家性能:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics

def create_fn_ensemble_recommendations():
    """
    创建FN专家集成建议
    """
    print("=" * 80)
    print("FN专家集成策略建议")
    print("=" * 80)
    
    print("\n🎯 集成策略核心思想：")
    print("1. 多样化：创建不同特性的FN专家")
    print("   - 不同数据平衡比例的专家")
    print("   - 不同特征子集的专家")
    print("   - 不同超参数配置的专家")
    
    print("\n2. 组合方式：")
    print("   - 堆叠集成：使用元学习器学习最优组合")
    print("   - 加权投票：基于召回率动态分配权重")
    print("   - 动态选择：根据输入特征选择最佳专家")
    
    print("\n🚀 立即可实施的集成方案：")
    
    print("\n方案1：多专家投票")
    print("```python")
    print("# 训练3个不同配置的FN专家")
    print("fn_expert_conservative = XGBClassifier(scale_pos_weight=15, max_depth=4)")
    print("fn_expert_aggressive = XGBClassifier(scale_pos_weight=30, max_depth=8)")
    print("fn_expert_balanced = XGBClassifier(scale_pos_weight=20, max_depth=6)")
    print("")
    print("# 加权投票")
    print("final_proba = 0.2*conservative + 0.5*balanced + 0.3*aggressive")
    print("```")
    
    print("\n方案2：特征专家组合")
    print("```python")
    print("# 强度专家：关注降雨强度特征")
    print("intensity_features = ['multi_product_max', 'multi_product_mean']")
    print("intensity_expert = train_expert(X[intensity_features], y)")
    print("")
    print("# 一致性专家：关注产品一致性")
    print("consistency_features = ['multi_product_std', 'coef_of_variation']") 
    print("consistency_expert = train_expert(X[consistency_features], y)")
    print("")
    print("# 时序专家：关注时间模式")
    print("temporal_features = ['lag_1_mean', 'diff_mean', 'window_3d_mean']")
    print("temporal_expert = train_expert(X[temporal_features], y)")
    print("```")
    
    print("\n方案3：动态权重调整")
    print("```python")
    print("def dynamic_ensemble_predict(X):")
    print("    predictions = []")
    print("    weights = []")
    print("    ")
    print("    for expert in fn_experts:")
    print("        pred = expert.predict_proba(X)[:, 1]")
    print("        # 基于预测置信度调整权重")
    print("        confidence = np.abs(pred - 0.5)")
    print("        weight = confidence * expert.recall_score")
    print("        ")
    print("        predictions.append(pred)")
    print("        weights.append(weight)")
    print("    ")
    print("    # 归一化权重并加权平均")
    print("    weights = np.array(weights)")
    print("    weights = weights / weights.sum(axis=0)")
    print("    ")
    print("    final_pred = np.average(predictions, weights=weights, axis=0)")
    print("    return final_pred")
    print("```")

def main():
    """
    展示完整的FN专家集成改进方案
    """
    create_fn_ensemble_recommendations()
    
    print("\n" + "=" * 80)
    print("实施建议")
    print("=" * 80)
    
    print("\n✅ 阶段性实施计划：")
    
    print("\n阶段1：基础多专家（1-2天）")
    print("- 使用不同scale_pos_weight训练3个FN专家")
    print("- 实现简单加权投票")
    print("- 预期提升：AUC +0.02-0.05")
    
    print("\n阶段2：特征专家（2-3天）")
    print("- 设计特征子集专家")
    print("- 实现动态权重分配")
    print("- 预期提升：召回率 +5-10%")
    
    print("\n阶段3：高级集成（3-5天）")
    print("- 实现堆叠集成")
    print("- 优化元学习器")
    print("- 预期提升：F1-score +0.05-0.10")
    
    print("\n🎯 成功指标：")
    print("- FN专家召回率 > 0.80")
    print("- FN专家F1-score > 0.70")
    print("- FN专家AUC > 0.85")
    print("- 整体集成性能稳定")
    
    print("\n💡 关键成功因素：")
    print("1. 确保基础专家的多样性")
    print("2. 重视召回率而非精确率")
    print("3. 动态调整权重机制")
    print("4. 持续监控和优化")

if __name__ == "__main__":
    main()