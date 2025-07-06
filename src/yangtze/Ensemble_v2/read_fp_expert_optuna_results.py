#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FP专家模型Optuna优化结果读取脚本
==================================

读取并分析FP专家模型的Optuna超参数优化结果

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

def read_fp_expert_optuna_results():
    """读取FP专家模型的Optuna优化结果"""
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    optuna_db_path = os.path.join(current_dir, "Ensemble_v2", "optuna_db", "fp_expert_v2_optimization.db")
    
    print("=" * 80)
    print("FP专家模型 - Optuna优化结果分析")
    print("=" * 80)
    print(f"数据库路径: {optuna_db_path}")
    
    if not os.path.exists(optuna_db_path):
        print(f"错误: 数据库文件不存在: {optuna_db_path}")
        return
    
    try:
        # 连接Optuna study
        storage_url = f"sqlite:///{optuna_db_path}"
        study = optuna.load_study(
            study_name="fp_expert_v2_optimization",
            storage=storage_url
        )
        
        print(f"Study名称: {study.study_name}")
        print(f"优化方向: {study.direction}")
        print(f"总试验次数: {len(study.trials)}")
        
        # 获取最佳试验
        best_trial = study.best_trial
        print(f"\n最佳试验信息:")
        print(f"  试验编号: {best_trial.number}")
        print(f"  最佳AUC值: {best_trial.value:.6f}")
        print(f"  最佳参数:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        
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
        print(f"\n完成的试验数: {len(completed_trials)}")
        
        if len(completed_trials) > 0:
            print(f"AUC统计:")
            print(f"  最大值: {completed_trials['value'].max():.6f}")
            print(f"  最小值: {completed_trials['value'].min():.6f}")
            print(f"  平均值: {completed_trials['value'].mean():.6f}")
            print(f"  标准差: {completed_trials['value'].std():.6f}")
            
            # 找出前10个最佳试验
            top_10 = completed_trials.nlargest(10, 'value')
            print(f"\n前10个最佳试验:")
            for idx, row in top_10.iterrows():
                print(f"  试验{row['number']:3d}: AUC={row['value']:.6f}")
        
        # 保存详细结果
        output_dir = os.path.join(current_dir, "Ensemble_v2", "optuna_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存试验结果CSV
        csv_path = os.path.join(output_dir, "fp_expert_optuna_trials.csv")
        df_trials.to_csv(csv_path, index=False)
        print(f"\n试验结果已保存到: {csv_path}")
        
        # 保存最佳参数
        best_params_path = os.path.join(output_dir, "fp_expert_best_params.txt")
        with open(best_params_path, 'w', encoding='utf-8') as f:
            f.write("FP专家模型最佳参数\n")
            f.write("=" * 50 + "\n")
            f.write(f"最佳AUC值: {best_trial.value:.6f}\n")
            f.write(f"试验编号: {best_trial.number}\n\n")
            f.write("最佳参数:\n")
            for key, value in best_trial.params.items():
                f.write(f"  {key}: {value}\n")
        print(f"最佳参数已保存到: {best_params_path}")
        
        # 绘制优化历史图
        if len(completed_trials) > 1:
            plt.figure(figsize=(12, 8))
            
            # 子图1: 优化历史
            plt.subplot(2, 2, 1)
            plt.plot(completed_trials['number'], completed_trials['value'], 'r-', alpha=0.7)
            plt.scatter(completed_trials['number'], completed_trials['value'], c=completed_trials['value'], cmap='plasma', s=30)
            plt.xlabel('Trial Number')
            plt.ylabel('AUC Score')
            plt.title('FP Expert Optimization History')
            plt.grid(True, alpha=0.3)
            
            # 子图2: AUC分布直方图
            plt.subplot(2, 2, 2)
            plt.hist(completed_trials['value'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.axvline(best_trial.value, color='red', linestyle='--', label=f'Best: {best_trial.value:.4f}')
            plt.xlabel('AUC Score')
            plt.ylabel('Frequency')
            plt.title('AUC Score Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # 子图3: 运行时间分析
            if completed_trials['duration'].notna().any():
                plt.subplot(2, 2, 3)
                valid_durations = completed_trials.dropna(subset=['duration'])
                plt.scatter(valid_durations['number'], valid_durations['duration'], alpha=0.6, color='orange')
                plt.xlabel('Trial Number')
                plt.ylabel('Duration (seconds)')
                plt.title('Trial Duration')
                plt.grid(True, alpha=0.3)
            
            # 子图4: 累积最佳值
            plt.subplot(2, 2, 4)
            cumulative_best = completed_trials['value'].cummax()
            plt.plot(completed_trials['number'], cumulative_best, 'r-', linewidth=2)
            plt.xlabel('Trial Number')
            plt.ylabel('Best AUC So Far')
            plt.title('Cumulative Best AUC')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            plot_path = os.path.join(output_dir, "fp_expert_optimization_analysis.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"优化分析图已保存到: {plot_path}")
            plt.show()
        
        print("\n" + "=" * 80)
        print("FP专家模型Optuna结果分析完成!")
        print("=" * 80)
        
    except Exception as e:
        print(f"读取Optuna结果时出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    read_fp_expert_optuna_results()