#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
平衡数据FP/FN专家模型Optuna优化结果读取脚本
===============================================

读取并分析使用平衡数据训练的FP/FN专家模型的Optuna超参数优化结果

Author: Claude & User
Date: 2025-07-04
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import optuna
import warnings
warnings.filterwarnings('ignore')

def read_balanced_experts_optuna_results():
    """读取平衡数据FP/FN专家模型的Optuna优化结果"""
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    optuna_db_dir = os.path.join(current_dir, "Ensemble_v2", "optuna_db")
    
    experts = {
        "FN": {
            "db_file": "fn_expert_balanced_optimization.db", 
            "study_name": "fn_expert_balanced_optimization_v1",
            "color": "green"
        }
    }
    
    print("=" * 80)
    print("平衡数据FP/FN专家模型 - Optuna优化结果分析")
    print("=" * 80)
    
    all_results = {}
    
    for expert_name, expert_info in experts.items():
        print(f"\n{'-'*60}")
        print(f"分析 {expert_name} 专家模型 (平衡数据)")
        print(f"{'-'*60}")
        
        db_path = os.path.join(optuna_db_dir, expert_info["db_file"])
        
        if not os.path.exists(db_path):
            print(f"⚠️  数据库文件不存在: {db_path}")
            print(f"   该专家模型可能尚未开始训练或使用不同的数据库文件名")
            continue
        
        try:
            # 连接Optuna study
            storage_url = f"sqlite:///{db_path}"
            study = optuna.load_study(
                study_name=expert_info["study_name"],
                storage=storage_url
            )
            
            print(f"📊 Study名称: {study.study_name}")
            print(f"📈 优化方向: {study.direction}")
            print(f"🔢 总试验次数: {len(study.trials)}")
            
            # 获取最佳试验
            if study.trials:
                best_trial = study.best_trial
                print(f"\n🏆 最佳试验信息:")
                print(f"   试验编号: {best_trial.number}")
                print(f"   最佳AUC值: {best_trial.value:.6f}")
                print(f"   最佳参数:")
                for key, value in best_trial.params.items():
                    if isinstance(value, float):
                        print(f"     {key}: {value:.6f}")
                    else:
                        print(f"     {key}: {value}")
                
                # 创建DataFrame分析所有试验
                trials_data = []
                for trial in study.trials:
                    trial_info = {
                        'number': trial.number,
                        'value': trial.value,
                        'state': trial.state.name,
                        'datetime_start': trial.datetime_start,
                        'datetime_complete': trial.datetime_complete,
                        'duration': (trial.datetime_complete - trial.datetime_start).total_seconds() if trial.datetime_complete else None
                    }
                    # 添加参数
                    trial_info.update(trial.params)
                    trials_data.append(trial_info)
                
                df_trials = pd.DataFrame(trials_data)
                
                # 过滤完成的试验
                completed_trials = df_trials[df_trials['state'] == 'COMPLETE'].copy()
                print(f"\n📈 完成的试验数: {len(completed_trials)}")
                
                if len(completed_trials) > 0:
                    print(f"📊 AUC统计:")
                    print(f"   最大值: {completed_trials['value'].max():.6f}")
                    print(f"   最小值: {completed_trials['value'].min():.6f}")
                    print(f"   平均值: {completed_trials['value'].mean():.6f}")
                    print(f"   标准差: {completed_trials['value'].std():.6f}")
                    print(f"   中位数: {completed_trials['value'].median():.6f}")
                    
                    # 找出前5个最佳试验
                    top_5 = completed_trials.nlargest(5, 'value')
                    print(f"\n🔝 前5个最佳试验:")
                    for idx, row in top_5.iterrows():
                        print(f"   试验{row['number']:3d}: AUC={row['value']:.6f}")
                    
                    # 分析参数分布
                    print(f"\n📊 关键参数统计:")
                    key_params = ['n_estimators', 'learning_rate', 'max_depth', 'subsample', 'colsample_bytree']
                    for param in key_params:
                        if param in completed_trials.columns:
                            values = completed_trials[param].dropna()
                            if len(values) > 0:
                                print(f"   {param}: 均值={values.mean():.4f}, 标准差={values.std():.4f}")
                
                all_results[expert_name] = {
                    'study': study,
                    'trials_df': df_trials,
                    'completed_trials': completed_trials,
                    'best_trial': best_trial,
                    'color': expert_info['color']
                }
                
            else:
                print("⚠️  没有找到任何试验数据")
                
        except Exception as e:
            print(f"❌ 读取 {expert_name} 专家结果时出错: {str(e)}")
    
    # 保存详细结果和生成对比图
    if all_results:
        output_dir = os.path.join(current_dir, "Ensemble_v2", "balanced_optuna_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存各专家详细结果
        for expert_name, results in all_results.items():
            # 保存试验结果CSV
            csv_path = os.path.join(output_dir, f"{expert_name.lower()}_expert_balanced_optuna_trials.csv")
            results['trials_df'].to_csv(csv_path, index=False)
            print(f"\n💾 {expert_name}专家试验结果已保存到: {csv_path}")
            
            # 保存最佳参数
            best_params_path = os.path.join(output_dir, f"{expert_name.lower()}_expert_balanced_best_params.txt")
            with open(best_params_path, 'w', encoding='utf-8') as f:
                f.write(f"{expert_name}专家模型最佳参数 (平衡数据)\n")
                f.write("=" * 60 + "\n")
                f.write(f"最佳AUC值: {results['best_trial'].value:.6f}\n")
                f.write(f"试验编号: {results['best_trial'].number}\n\n")
                f.write("最佳参数:\n")
                for key, value in results['best_trial'].params.items():
                    f.write(f"  {key}: {value}\n")
            print(f"💾 {expert_name}专家最佳参数已保存到: {best_params_path}")
        
        # 生成对比分析图
        if len(all_results) > 1:
            print(f"\n📊 生成FP vs FN专家对比分析图...")
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('平衡数据FP vs FN专家模型优化对比分析', fontsize=16, fontweight='bold')
            
            # 1. 优化历史对比
            ax1 = axes[0, 0]
            for expert_name, results in all_results.items():
                completed = results['completed_trials']
                if len(completed) > 1:
                    ax1.plot(completed['number'], completed['value'], 
                            label=f'{expert_name} Expert', color=results['color'], alpha=0.7)
                    ax1.scatter(completed['number'], completed['value'], 
                               c=results['color'], s=20, alpha=0.6)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('AUC Score')
            ax1.set_title('优化历史对比')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. AUC分布对比
            ax2 = axes[0, 1]
            auc_data = []
            labels = []
            colors = []
            for expert_name, results in all_results.items():
                completed = results['completed_trials']
                if len(completed) > 0:
                    auc_data.append(completed['value'].values)
                    labels.append(f'{expert_name} Expert')
                    colors.append(results['color'])
            
            if auc_data:
                bp = ax2.boxplot(auc_data, labels=labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
            ax2.set_ylabel('AUC Score')
            ax2.set_title('AUC分布对比')
            ax2.grid(True, alpha=0.3)
            
            # 3. 累积最佳值对比
            ax3 = axes[0, 2]
            for expert_name, results in all_results.items():
                completed = results['completed_trials']
                if len(completed) > 1:
                    cumulative_best = completed['value'].cummax()
                    ax3.plot(completed['number'], cumulative_best, 
                            label=f'{expert_name} Expert', color=results['color'], linewidth=2)
            ax3.set_xlabel('Trial Number')
            ax3.set_ylabel('Best AUC So Far')
            ax3.set_title('累积最佳AUC对比')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # 4. 学习率vs性能
            ax4 = axes[1, 0]
            for expert_name, results in all_results.items():
                completed = results['completed_trials']
                if len(completed) > 0 and 'learning_rate' in completed.columns:
                    valid_data = completed.dropna(subset=['learning_rate', 'value'])
                    if len(valid_data) > 0:
                        ax4.scatter(valid_data['learning_rate'], valid_data['value'], 
                                   label=f'{expert_name} Expert', color=results['color'], alpha=0.6)
            ax4.set_xlabel('Learning Rate')
            ax4.set_ylabel('AUC Score')
            ax4.set_title('学习率 vs AUC性能')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            # 5. 深度vs性能
            ax5 = axes[1, 1]
            for expert_name, results in all_results.items():
                completed = results['completed_trials']
                if len(completed) > 0 and 'max_depth' in completed.columns:
                    valid_data = completed.dropna(subset=['max_depth', 'value'])
                    if len(valid_data) > 0:
                        ax5.scatter(valid_data['max_depth'], valid_data['value'], 
                                   label=f'{expert_name} Expert', color=results['color'], alpha=0.6)
            ax5.set_xlabel('Max Depth')
            ax5.set_ylabel('AUC Score')
            ax5.set_title('树深度 vs AUC性能')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # 6. 估计器数量vs性能
            ax6 = axes[1, 2]
            for expert_name, results in all_results.items():
                completed = results['completed_trials']
                if len(completed) > 0 and 'n_estimators' in completed.columns:
                    valid_data = completed.dropna(subset=['n_estimators', 'value'])
                    if len(valid_data) > 0:
                        ax6.scatter(valid_data['n_estimators'], valid_data['value'], 
                                   label=f'{expert_name} Expert', color=results['color'], alpha=0.6)
            ax6.set_xlabel('N Estimators')
            ax6.set_ylabel('AUC Score')
            ax6.set_title('估计器数量 vs AUC性能')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存对比图
            comparison_plot_path = os.path.join(output_dir, "fp_vs_fn_balanced_experts_comparison.png")
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 FP vs FN专家对比分析图已保存到: {comparison_plot_path}")
            plt.show()
        
        # 生成总结报告
        summary_path = os.path.join(output_dir, "balanced_experts_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("平衡数据FP/FN专家模型训练总结\n")
            f.write("=" * 60 + "\n")
            f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for expert_name, results in all_results.items():
                f.write(f"{expert_name}专家模型:\n")
                f.write(f"  总试验次数: {len(results['study'].trials)}\n")
                completed = results['completed_trials']
                f.write(f"  完成试验数: {len(completed)}\n")
                if len(completed) > 0:
                    f.write(f"  最佳AUC: {results['best_trial'].value:.6f}\n")
                    f.write(f"  平均AUC: {completed['value'].mean():.6f}\n")
                    f.write(f"  AUC标准差: {completed['value'].std():.6f}\n")
                f.write("\n")
            
            if len(all_results) > 1:
                # 对比分析
                expert_names = list(all_results.keys())
                if len(expert_names) == 2:
                    expert1, expert2 = expert_names
                    best1 = all_results[expert1]['best_trial'].value
                    best2 = all_results[expert2]['best_trial'].value
                    f.write("对比分析:\n")
                    f.write(f"  {expert1} vs {expert2} 最佳AUC差异: {abs(best1 - best2):.6f}\n")
                    if best1 > best2:
                        f.write(f"  {expert1}专家表现更优\n")
                    elif best2 > best1:
                        f.write(f"  {expert2}专家表现更优\n")
                    else:
                        f.write(f"  两个专家表现相当\n")
        
        print(f"📄 总结报告已保存到: {summary_path}")
    
    print("\n" + "=" * 80)
    print("平衡数据FP/FN专家模型Optuna结果分析完成!")
    print("=" * 80)

if __name__ == "__main__":
    read_balanced_experts_optuna_results()