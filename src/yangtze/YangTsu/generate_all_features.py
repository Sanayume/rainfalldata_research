#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主控特征生成器
管理所有类别的特征生成，支持选择性运行
"""

import os
import sys
import time
import subprocess
from typing import List, Dict

# 配置
OUTPUT_DIR = "/mnt/f/rainfalldata/results/yangtze/features/features"

# 特征生成器映射
FEATURE_GENERATORS = {
    'basic': {
        'script': 'generate_basic_features.py',
        'description': '基础原始特征 (原始数据和目标变量)',
        'estimated_time': '2-3分钟',
        'dependencies': []
    },
    'multi_product': {
        'script': 'generate_multi_product_features.py', 
        'description': '多产品协同特征 (产品间统计关系)',
        'estimated_time': '3-5分钟',
        'dependencies': []
    },
    'temporal': {
        'script': 'generate_temporal_features.py',
        'description': '时序动态特征 (周期性、季节性、趋势)',
        'estimated_time': '2-3分钟', 
        'dependencies': []
    },
    'lag': {
        'script': 'generate_lag_features.py',
        'description': '滞后特征 (时间依赖性)',
        'estimated_time': '3-4分钟',
        'dependencies': []
    },
    'spatial': {
        'script': 'generate_spatial_features.py',
        'description': '真实空间特征 (梯度、邻域、聚集性)',
        'estimated_time': '5-8分钟',
        'dependencies': []
    },
    'advanced': {
        'script': 'generate_advanced_features.py',
        'description': '高级统计特征 (分位数、极值、异常检测)',
        'estimated_time': '4-6分钟',
        'dependencies': []
    },
    'interaction': {
        'script': 'generate_interaction_features.py',
        'description': '交互特征 (特征间交互和组合)',
        'estimated_time': '3-5分钟',
        'dependencies': ['basic', 'temporal']  # 需要基础和时序特征的一些输入
    }
}

def run_generator(generator_name: str, verbose: bool = True) -> bool:
    """
    运行指定的特征生成器
    
    Args:
        generator_name: 生成器名称
        verbose: 是否显示详细输出
    
    Returns:
        是否成功运行
    """
    if generator_name not in FEATURE_GENERATORS:
        print(f"错误: 未知的生成器 '{generator_name}'")
        return False
    
    generator = FEATURE_GENERATORS[generator_name]
    script_path = generator['script']
    
    if not os.path.exists(script_path):
        print(f"错误: 脚本文件不存在 '{script_path}'")
        return False
    
    print(f"\n{'='*60}")
    print(f"运行: {generator_name}")
    print(f"描述: {generator['description']}")
    print(f"预估时间: {generator['estimated_time']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # 运行生成器脚本
        if verbose:
            result = subprocess.run([sys.executable, script_path], 
                                  check=True, text=True)
        else:
            result = subprocess.run([sys.executable, script_path], 
                                  check=True, text=True, 
                                  capture_output=True)
        
        elapsed_time = time.time() - start_time
        print(f"\n✓ {generator_name} 生成器完成，耗时: {elapsed_time:.1f}秒")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {generator_name} 生成器失败，耗时: {elapsed_time:.1f}秒")
        print(f"错误码: {e.returncode}")
        if not verbose and e.stdout:
            print(f"标准输出: {e.stdout}")
        if not verbose and e.stderr:
            print(f"错误输出: {e.stderr}")
        return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n✗ {generator_name} 生成器异常，耗时: {elapsed_time:.1f}秒")
        print(f"异常信息: {e}")
        return False

def check_dependencies(generator_name: str, completed: List[str]) -> bool:
    """检查依赖是否满足"""
    dependencies = FEATURE_GENERATORS[generator_name]['dependencies']
    missing_deps = [dep for dep in dependencies if dep not in completed]
    
    if missing_deps:
        print(f"警告: {generator_name} 需要先完成依赖: {missing_deps}")
        return False
    return True

def get_feature_count() -> int:
    """获取当前特征文件数量"""
    if os.path.exists(OUTPUT_DIR):
        return len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')])
    return 0

def print_usage():
    """打印使用说明"""
    print("="*80)
    print("长江流域单独特征生成系统")
    print("="*80)
    print("\n可用的特征类别:")
    for name, info in FEATURE_GENERATORS.items():
        deps_str = f" (依赖: {', '.join(info['dependencies'])})" if info['dependencies'] else ""
        print(f"  {name:12} - {info['description']}{deps_str}")
        print(f"  {'':12}   预估时间: {info['estimated_time']}")
    
    print(f"\n当前特征目录: {OUTPUT_DIR}")
    print(f"已有特征文件: {get_feature_count()}个")
    
    print("\n使用方法:")
    print("  python generate_all_features.py [选项] [特征类别...]")
    print("\n选项:")
    print("  --all          生成所有特征")
    print("  --list         列出所有可用的特征类别")
    print("  --check        检查现有特征文件")
    print("  --quiet        静默模式，减少输出")
    print("  --help         显示此帮助信息")
    print("\n示例:")
    print("  python generate_all_features.py --all")
    print("  python generate_all_features.py basic temporal")
    print("  python generate_all_features.py spatial advanced --quiet")

def check_existing_features():
    """检查现有特征文件"""
    print("\n检查现有特征文件...")
    
    if not os.path.exists(OUTPUT_DIR):
        print(f"特征目录不存在: {OUTPUT_DIR}")
        return
    
    all_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')]
    print(f"总特征文件数: {len(all_files)}")
    
    # 按类别统计
    categories = {
        'basic': ['raw_', 'target_'],
        'multi_product': ['multi_product_', 'rain_product_', 'correlation_', 'product_', 'weighted_', 'max_product_', 'min_product_'],
        'temporal': ['sin_', 'cos_', 'season_', 'month_', 'time_', 'normalized_', 'quadratic_', 'diff_', 'relative_', 'day_of_', 'cumulative_', 'days_in_'],
        'lag': ['lag_'],
        'spatial': ['spatial_'],
        'advanced': ['quantile_', 'extreme_', 'daily_', 'coefficient_', 'anomaly_', 'min_distance_', 'near_threshold_', 'low_intensity_', 'intensity_bin_', 'rain_count_bin_', 'product_consistency_', 'product_disagreement', 'product_entropy', 'target_rolling_', 'target_anomaly_'],
        'interaction': ['interaction_']
    }
    
    print("\n按类别分布:")
    for category, prefixes in categories.items():
        count = sum(1 for f in all_files if any(f.startswith(p) for p in prefixes))
        print(f"  {category:12}: {count:3d}个")
    
    # 检查是否有未分类的文件
    categorized_files = set()
    for prefixes in categories.values():
        for prefix in prefixes:
            categorized_files.update(f for f in all_files if f.startswith(prefix))
    
    uncategorized = set(all_files) - categorized_files
    if uncategorized:
        print(f"\n未分类文件 ({len(uncategorized)}个):")
        for f in sorted(uncategorized):
            print(f"  {f}")

def main():
    """主函数"""
    args = sys.argv[1:]
    
    # 解析参数
    if not args or '--help' in args:
        print_usage()
        return
    
    if '--list' in args:
        print("\n可用的特征类别:")
        for name, info in FEATURE_GENERATORS.items():
            print(f"  {name}: {info['description']}")
        return
    
    if '--check' in args:
        check_existing_features()
        return
    
    verbose = '--quiet' not in args
    run_all = '--all' in args
    
    # 确定要运行的生成器
    if run_all:
        generators_to_run = list(FEATURE_GENERATORS.keys())
    else:
        generators_to_run = [arg for arg in args if arg in FEATURE_GENERATORS and not arg.startswith('--')]
    
    if not generators_to_run:
        print("错误: 未指定有效的特征类别")
        print("使用 --help 查看使用说明")
        return
    
    # 检查依赖关系并排序
    ordered_generators = []
    remaining = generators_to_run.copy()
    
    while remaining:
        made_progress = False
        for gen in remaining.copy():
            if check_dependencies(gen, ordered_generators):
                ordered_generators.append(gen)
                remaining.remove(gen)
                made_progress = True
        
        if not made_progress:
            print(f"错误: 存在循环依赖或无法满足的依赖: {remaining}")
            return
    
    # 开始生成特征
    print(f"\n开始生成特征...")
    print(f"目标目录: {OUTPUT_DIR}")
    print(f"计划运行: {len(ordered_generators)}个生成器")
    print(f"运行顺序: {' -> '.join(ordered_generators)}")
    
    initial_count = get_feature_count()
    print(f"开始时特征文件数: {initial_count}")
    
    start_time = time.time()
    completed = []
    failed = []
    
    for generator_name in ordered_generators:
        success = run_generator(generator_name, verbose)
        if success:
            completed.append(generator_name)
        else:
            failed.append(generator_name)
            # 可以选择继续或停止
            if input(f"\n{generator_name} 失败，是否继续运行其他生成器? (y/n): ").lower() != 'y':
                break
    
    # 总结
    total_time = time.time() - start_time
    final_count = get_feature_count()
    new_features = final_count - initial_count
    
    print(f"\n{'='*80}")
    print("特征生成完成")
    print(f"{'='*80}")
    print(f"总耗时: {total_time:.1f}秒")
    print(f"成功完成: {len(completed)}个生成器")
    if failed:
        print(f"失败: {len(failed)}个生成器 - {', '.join(failed)}")
    print(f"新增特征文件: {new_features}个")
    print(f"最终特征文件总数: {final_count}个")
    
    if completed:
        print(f"\n成功完成的生成器:")
        for gen in completed:
            print(f"  ✓ {gen}")
    
    if failed:
        print(f"\n失败的生成器:")
        for gen in failed:
            print(f"  ✗ {gen}")
        print("\n可以单独重新运行失败的生成器")
    
    print(f"\n特征文件目录: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()