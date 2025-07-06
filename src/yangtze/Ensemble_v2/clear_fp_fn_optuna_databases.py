#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清空FP/FN专家Optuna数据库脚本
===============================

由于数据平衡策略的改变，需要清空原有的不平衡数据训练历史，
为平衡数据训练开启全新的超参数优化过程。

Author: Claude & User
Date: 2025-07-04
"""

import os
import shutil
from datetime import datetime

def backup_and_clear_fp_fn_databases():
    """备份并清空FP/FN专家的Optuna数据库"""
    
    # 设置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    optuna_db_dir = os.path.join(current_dir, "Ensemble_v2", "optuna_db")
    backup_dir = os.path.join(current_dir, "Ensemble_v2", "optuna_db_backup")
    
    # 创建备份目录
    os.makedirs(backup_dir, exist_ok=True)
    
    # 需要处理的数据库文件
    fp_fn_databases = [
        "fp_expert_optimization.db",
        "fn_expert_optimization.db",
        "fp_expert_balanced_optimization.db"
    ]
    
    print("=" * 80)
    print("FP/FN专家Optuna数据库清理工具")
    print("=" * 80)
    print("原因: 数据平衡策略变更，原有不平衡数据的优化历史不再适用")
    print(f"备份目录: {backup_dir}")
    print(f"时间戳: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    for db_file in fp_fn_databases:
        db_path = os.path.join(optuna_db_dir, db_file)
        
        if os.path.exists(db_path):
            # 创建带时间戳的备份文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{db_file.replace('.db', '')}_backup_{timestamp}.db"
            backup_path = os.path.join(backup_dir, backup_filename)
            
            # 备份原文件
            shutil.copy2(db_path, backup_path)
            print(f"✅ 已备份: {db_file}")
            print(f"   备份位置: {backup_path}")
            
            # 删除原文件
            os.remove(db_path)
            print(f"🗑️  已删除: {db_file}")
            print(f"   原位置: {db_path}")
            
            # 检查文件大小
            backup_size = os.path.getsize(backup_path) / 1024  # KB
            print(f"   文件大小: {backup_size:.1f} KB")
            
        else:
            print(f"⚠️  文件不存在: {db_file}")
        
        print()
    
    # 保留TP/TN专家的数据库（它们训练效果很好）
    preserved_databases = [
        "tp_expert_optimization.db", 
        "tn_expert_optimization.db"
    ]
    
    print("保留的数据库文件 (TP/TN专家训练效果优秀):")
    for db_file in preserved_databases:
        db_path = os.path.join(optuna_db_dir, db_file)
        if os.path.exists(db_path):
            db_size = os.path.getsize(db_path) / 1024  # KB
            print(f"✅ {db_file} (大小: {db_size:.1f} KB)")
        else:
            print(f"⚠️  {db_file} (不存在)")
    
    print()
    print("=" * 80)
    print("数据库清理完成!")
    print("=" * 80)
    print("说明:")
    print("- FP/FN专家的原始优化历史已备份并清空")
    print("- 现在可以使用平衡数据进行全新的超参数优化")
    print("- TP/TN专家的优化历史保持不变 (性能优秀)")
    print("- 备份文件可在需要时恢复")
    print()
    print("下一步:")
    print("运行: python train_balanced_fp_fn_experts.py")
    print()

if __name__ == "__main__":
    backup_and_clear_fp_fn_databases()