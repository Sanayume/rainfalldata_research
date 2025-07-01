#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征加载系统使用示例
演示如何使用FeatureLoader类进行特征加载和管理
"""

import numpy as np
from loadfeatures import FeatureLoader

def demo_basic_usage():
    """基础使用示例"""
    print("=== 基础使用示例 ===")
    
    # 初始化特征加载器
    loader = FeatureLoader()
    
    # 查看特征概览
    loader.print_summary()
    
    print("\n--- 可用类别 ---")
    categories = loader.get_categories()
    for cat in categories:
        count = len(loader.get_category_features(cat))
        print(f"{cat}: {count}个特征")

def demo_single_feature_loading():
    """单个特征加载示例"""
    print("\n=== 单个特征加载示例 ===")
    
    loader = FeatureLoader()
    
    # 加载单个特征
    feature = loader.load_feature("multi_product_mean")
    print(f"multi_product_mean 形状: {feature.shape}")
    print(f"数据范围: {np.nanmin(feature):.3f} - {np.nanmax(feature):.3f}")
    
    # 获取特征详细信息
    info = loader.get_feature_info("multi_product_mean")
    print(f"特征信息: {info}")

def demo_category_loading():
    """按类别加载示例"""
    print("\n=== 按类别加载示例 ===")
    
    loader = FeatureLoader()
    
    # 加载时序特征
    temporal_features = loader.load_category("temporal")
    print(f"加载了 {len(temporal_features)} 个时序特征:")
    for name, data in list(temporal_features.items())[:5]:
        print(f"  {name}: {data.shape}")

def demo_custom_feature_matrix():
    """自定义特征矩阵构建示例"""
    print("\n=== 自定义特征矩阵构建示例 ===")
    
    loader = FeatureLoader()
    
    # 选择特定特征构建矩阵
    selected_features = [
        "multi_product_mean",
        "multi_product_std", 
        "lag_1_multi_product_mean",
        "sin_day_of_year",
        "cos_day_of_year"
    ]
    
    feature_matrix, feature_names = loader.build_feature_matrix(selected_features)
    print(f"特征矩阵形状: {feature_matrix.shape}")
    print(f"特征名称: {feature_names}")

def demo_feature_search():
    """特征搜索示例"""
    print("\n=== 特征搜索示例 ===")
    
    loader = FeatureLoader()
    
    # 搜索GSMAP相关特征
    gsmap_features = loader.search_features("GSMAP")
    print(f"找到 {len(gsmap_features)} 个GSMAP相关特征:")
    for feature in gsmap_features[:10]:
        print(f"  {feature}")
    
    # 搜索滞后特征
    lag_features = loader.search_features("lag", category="lag")
    print(f"\n找到 {len(lag_features)} 个滞后特征 (显示前10个):")
    for feature in lag_features[:10]:
        print(f"  {feature}")

def demo_recommended_features():
    """推荐特征组合示例"""
    print("\n=== 推荐特征组合示例 ===")
    
    loader = FeatureLoader()
    
    # 不同模型类型的推荐特征
    model_types = ["basic", "temporal", "spatial", "advanced"]
    
    for model_type in model_types:
        recommended = loader.get_recommended_features(model_type)
        print(f"\n{model_type}模型推荐特征 ({len(recommended)}个):")
        for feature in recommended:
            print(f"  {feature}")

def demo_feature_subset():
    """特征子集创建示例"""
    print("\n=== 特征子集创建示例 ===")
    
    loader = FeatureLoader()
    
    # 创建包含特定产品的特征子集
    subset = loader.create_feature_subset(
        include_categories=["basic", "multi_product", "temporal"],
        include_products=["GSMAP", "IMERG"],
        max_features=20,
        priority_keywords=["mean", "std"]
    )
    
    print(f"创建的特征子集 ({len(subset)}个):")
    for feature in subset:
        print(f"  {feature}")

def demo_batch_loading():
    """批量特征加载示例"""
    print("\n=== 批量特征加载示例 ===")
    
    loader = FeatureLoader()
    
    # 批量加载多个特征
    feature_list = [
        "multi_product_mean",
        "multi_product_std",
        "lag_1_points_GSMAP",
        "sin_day_of_year",
        "spatial_variance_GSMAP"
    ]
    
    features = loader.load_multiple_features(feature_list)
    print(f"批量加载了 {len(features)} 个特征:")
    for name, data in features.items():
        print(f"  {name}: {data.shape}, 大小: {data.nbytes/(1024*1024):.1f}MB")

def demo_memory_efficient_loading():
    """内存高效加载示例"""
    print("\n=== 内存高效加载示例 ===")
    
    loader = FeatureLoader()
    
    # 逐步加载和处理，避免内存溢出
    print("逐类别处理特征:")
    
    for category in ["basic", "temporal", "multi_product"]:
        print(f"\n处理 {category} 类别:")
        category_features = loader.load_category(category)
        
        # 计算类别统计
        total_size = sum(data.nbytes for data in category_features.values())
        print(f"  特征数量: {len(category_features)}")
        print(f"  总内存: {total_size/(1024*1024):.1f}MB")
        
        # 处理完后可以删除以释放内存
        del category_features

def main():
    """主演示函数"""
    print("长江流域降雨预测特征加载系统演示")
    print("=" * 60)
    
    try:
        # 基础使用
        demo_basic_usage()
        
        # 单个特征加载
        demo_single_feature_loading()
        
        # 按类别加载
        demo_category_loading()
        
        # 自定义特征矩阵
        demo_custom_feature_matrix()
        
        # 特征搜索
        demo_feature_search()
        
        # 推荐特征
        demo_recommended_features()
        
        # 特征子集
        demo_feature_subset()
        
        # 批量加载
        demo_batch_loading()
        
        # 内存高效加载
        demo_memory_efficient_loading()
        
        print("\n=== 演示完成 ===")
        print("更多用法请参考 loadfeatures.py 中的文档和方法")
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()