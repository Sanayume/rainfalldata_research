#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单独特征加载器
方便灵活地加载和组合不同的特征进行实验
"""

import numpy as np
import os
from typing import List, Dict, Tuple, Optional

class IndividualFeatureLoader:
    """单独特征加载器类"""
    
    def __init__(self, features_dir: str = None):
        """
        初始化特征加载器
        
        Args:
            features_dir: 特征文件目录，默认为标准路径
        """
        if features_dir is None:
            self.features_dir = "/mnt/f/rainfalldata/results/yangtze/features/features"
        else:
            self.features_dir = features_dir
        
        if not os.path.exists(self.features_dir):
            raise FileNotFoundError(f"特征目录不存在: {self.features_dir}")
        
        # 扫描可用特征
        self.available_features = self._scan_features()
        self.feature_categories = self._categorize_features()
        
    def _scan_features(self) -> List[str]:
        """扫描可用的特征文件"""
        feature_files = [f for f in os.listdir(self.features_dir) if f.endswith('.npy')]
        return sorted(feature_files)
    
    def _categorize_features(self) -> Dict[str, List[str]]:
        """将特征按类别分组"""
        categories = {
            'raw_data': [],
            'multi_product': [],
            'temporal': [],
            'lag': [],
            'spatial': [],
            'interaction': [],
            'advanced': []
        }
        
        for feature in self.available_features:
            if feature.startswith('raw_'):
                categories['raw_data'].append(feature)
            elif feature.startswith('target_'):
                categories['raw_data'].append(feature)
            elif feature.startswith('multi_product_') or feature.startswith('rain_product_'):
                categories['multi_product'].append(feature)
            elif feature.startswith('sin_') or feature.startswith('cos_') or feature.startswith('season_'):
                categories['temporal'].append(feature)
            elif feature.startswith('lag_') or feature.startswith('diff_'):
                categories['lag'].append(feature)
            elif 'spatial' in feature or 'neighbor' in feature or 'gradient' in feature:
                categories['spatial'].append(feature)
            elif feature.startswith('interaction_'):
                categories['interaction'].append(feature)
            else:
                categories['advanced'].append(feature)
        
        return categories
    
    def list_features(self, category: str = None) -> List[str]:
        """
        列出可用特征
        
        Args:
            category: 特征类别，可选值: 'raw_data', 'multi_product', 'temporal', 'lag', 'spatial', 'interaction', 'advanced'
        
        Returns:
            特征文件名列表
        """
        if category is None:
            return self.available_features
        elif category in self.feature_categories:
            return self.feature_categories[category]
        else:
            raise ValueError(f"未知类别: {category}. 可用类别: {list(self.feature_categories.keys())}")
    
    def load_feature(self, feature_name: str) -> np.ndarray:
        """
        加载单个特征
        
        Args:
            feature_name: 特征文件名（可以带.npy后缀，也可以不带）
        
        Returns:
            特征数据数组
        """
        if not feature_name.endswith('.npy'):
            feature_name += '.npy'
        
        if feature_name not in self.available_features:
            raise FileNotFoundError(f"特征文件不存在: {feature_name}")
        
        filepath = os.path.join(self.features_dir, feature_name)
        return np.load(filepath)
    
    def load_multiple_features(self, feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        加载多个特征
        
        Args:
            feature_names: 特征文件名列表
        
        Returns:
            特征名到数据的字典
        """
        features_data = {}
        for name in feature_names:
            clean_name = name.replace('.npy', '')
            features_data[clean_name] = self.load_feature(name)
        return features_data
    
    def load_feature_category(self, category: str) -> Dict[str, np.ndarray]:
        """
        加载某个类别的所有特征
        
        Args:
            category: 特征类别
        
        Returns:
            特征名到数据的字典
        """
        feature_names = self.list_features(category)
        return self.load_multiple_features(feature_names)
    
    def get_feature_info(self, feature_name: str) -> Dict:
        """
        获取特征信息
        
        Args:
            feature_name: 特征文件名
        
        Returns:
            特征信息字典
        """
        data = self.load_feature(feature_name)
        return {
            'name': feature_name,
            'shape': data.shape,
            'dtype': data.dtype,
            'size_mb': data.nbytes / (1024 * 1024),
            'min': np.nanmin(data),
            'max': np.nanmax(data),
            'mean': np.nanmean(data),
            'std': np.nanstd(data),
            'nan_count': np.sum(np.isnan(data)),
            'nan_ratio': np.sum(np.isnan(data)) / data.size
        }
    
    def build_feature_matrix(self, feature_names: List[str], 
                           target_shape: str = 'flat') -> Tuple[np.ndarray, List[str]]:
        """
        构建特征矩阵
        
        Args:
            feature_names: 要组合的特征名列表
            target_shape: 目标形状，'flat' 或 'temporal'
        
        Returns:
            (特征矩阵, 特征名列表)
        """
        features_data = self.load_multiple_features(feature_names)
        
        # 确定基准维度
        first_feature = list(features_data.values())[0]
        if len(first_feature.shape) == 1:
            # 1D特征（如周期性特征）
            n_samples = first_feature.shape[0]
            base_shape = (n_samples,)
        elif len(first_feature.shape) == 2:
            # 2D特征（如点数据）
            n_samples, n_points = first_feature.shape
            base_shape = (n_samples, n_points)
        else:
            raise ValueError(f"不支持的特征维度: {first_feature.shape}")
        
        combined_features = []
        feature_columns = []
        
        for name, data in features_data.items():
            clean_name = name.replace('.npy', '')
            
            if len(data.shape) == 1:
                # 1D特征需要扩展到所有点
                if len(base_shape) == 2:
                    data_expanded = np.broadcast_to(data[:, np.newaxis], base_shape)
                else:
                    data_expanded = data
                combined_features.append(data_expanded.flatten())
                feature_columns.append(clean_name)
            
            elif len(data.shape) == 2:
                if target_shape == 'flat':
                    combined_features.append(data.flatten())
                    feature_columns.append(clean_name)
                else:
                    # 保持时序结构
                    combined_features.append(data)
                    feature_columns.append(clean_name)
            
            elif len(data.shape) == 3:
                # 空间特征，需要展平空间维度
                if target_shape == 'flat':
                    reshaped = data.reshape(data.shape[0], -1)
                    for point_idx in range(reshaped.shape[1]):
                        combined_features.append(reshaped[:, point_idx])
                        feature_columns.append(f"{clean_name}_point_{point_idx}")
                else:
                    combined_features.append(data)
                    feature_columns.append(clean_name)
        
        if target_shape == 'flat':
            feature_matrix = np.column_stack(combined_features)
        else:
            # 处理不同维度的特征组合
            feature_matrix = np.stack(combined_features, axis=-1)
        
        return feature_matrix, feature_columns
    
    def create_feature_subset(self, 
                            base_features: List[str] = None,
                            include_lag: bool = True,
                            include_temporal: bool = True,
                            include_interactions: bool = False,
                            max_features: int = None) -> List[str]:
        """
        创建特征子集
        
        Args:
            base_features: 基础特征列表
            include_lag: 是否包含滞后特征
            include_temporal: 是否包含时序特征
            include_interactions: 是否包含交互特征
            max_features: 最大特征数量
        
        Returns:
            特征名列表
        """
        if base_features is None:
            # 默认核心特征
            base_features = [
                'multi_product_mean.npy',
                'multi_product_std.npy',
                'rain_product_count.npy',
                'raw_points_GSMAP_valid.npy',
                'raw_points_IMERG_valid.npy'
            ]
        
        selected_features = base_features.copy()
        
        if include_temporal:
            temporal_features = self.list_features('temporal')
            selected_features.extend(temporal_features)
        
        if include_lag:
            # 选择关键滞后特征
            key_lag_features = [
                'lag_1_multi_product_mean.npy',
                'lag_1_points_GSMAP.npy',
                'lag_2_multi_product_mean.npy',
                'lag_3_multi_product_mean.npy'
            ]
            for feature in key_lag_features:
                if feature in self.available_features:
                    selected_features.append(feature)
        
        if include_interactions:
            interaction_features = self.list_features('interaction')
            selected_features.extend(interaction_features)
        
        # 去重
        selected_features = list(set(selected_features))
        
        # 限制特征数量
        if max_features and len(selected_features) > max_features:
            selected_features = selected_features[:max_features]
        
        return selected_features
    
    def print_summary(self):
        """打印特征库概要信息"""
        print("=== 单独特征库概要 ===")
        print(f"特征目录: {self.features_dir}")
        print(f"总特征数量: {len(self.available_features)}")
        print("\n按类别分布:")
        for category, features in self.feature_categories.items():
            print(f"  {category}: {len(features)}个")
        
        print("\n主要特征类型:")
        print("- 原始数据: 各产品的原始值和目标变量")
        print("- 多产品协同: 产品间的统计关系")
        print("- 时序特征: 周期性和季节性特征")
        print("- 滞后特征: 历史信息的时间依赖")
        print("- 空间特征: 真实空间结构特征")
        print("- 高级特征: 交互和统计特征")

# 使用示例
def example_usage():
    """使用示例"""
    # 初始化加载器
    loader = IndividualFeatureLoader()
    
    # 查看概要
    loader.print_summary()
    
    # 列出多产品特征
    multi_product_features = loader.list_features('multi_product')
    print(f"\n多产品特征: {multi_product_features}")
    
    # 加载单个特征
    product_mean = loader.load_feature('multi_product_mean')
    print(f"\n产品均值特征形状: {product_mean.shape}")
    
    # 创建特征子集
    feature_subset = loader.create_feature_subset(
        include_lag=True,
        include_temporal=True,
        max_features=20
    )
    print(f"\n特征子集 ({len(feature_subset)}个): {feature_subset}")
    
    # 构建特征矩阵
    X, feature_names = loader.build_feature_matrix(feature_subset[:5])
    print(f"\n特征矩阵形状: {X.shape}")
    print(f"特征名称: {feature_names}")

if __name__ == "__main__":
    example_usage()