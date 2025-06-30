#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
长江流域降雨预测特征加载器
与loaddata.py类似的设计模式，提供便捷的特征加载和管理功能
"""

import numpy as np
import os
import glob
from typing import List, Dict, Tuple, Optional, Union
import warnings
import json
from datetime import datetime

class FeatureLoader:
    """
    长江流域降雨预测特征加载器
    
    提供灵活、高效的特征加载和管理功能，支持:
    - 按类别加载特征
    - 自定义特征组合
    - 特征矩阵构建
    - 内存优化加载
    - 特征统计和分析
    """
    
    def __init__(self, features_dir="/mnt/f/rainfalldata/results/yangtze/features/features"):
        """
        初始化特征加载器
        
        Args:
            features_dir: 特征文件目录路径
        """
        self.features_dir = features_dir
        self.feature_files = []
        self.feature_categories = {}
        self.feature_metadata = {}
        
        # 检查目录是否存在
        if not os.path.exists(features_dir):
            raise FileNotFoundError(f"特征目录不存在: {features_dir}")
        
        # 扫描特征文件
        self._scan_features()
        
        # 分类特征
        self._categorize_features()
        
        print(f"特征加载器初始化完成: 发现 {len(self.feature_files)} 个特征文件")
    
    def _scan_features(self):
        """扫描特征文件"""
        pattern = os.path.join(self.features_dir, "*.npy")
        self.feature_files = glob.glob(pattern)
        self.feature_files = [os.path.basename(f) for f in self.feature_files]
        self.feature_files.sort()
    
    def _categorize_features(self):
        """按类别分类特征"""
        categories = {
            'basic': ['raw_', 'target_'],
            'multi_product': ['multi_product_', 'rain_product_', 'correlation_', 'product_', 'weighted_', 'max_product_', 'min_product_'],
            'temporal': ['sin_', 'cos_', 'season_', 'month_', 'time_', 'normalized_', 'quadratic_', 'diff_', 'relative_', 'day_of_', 'cumulative_', 'days_in_'],
            'lag': ['lag_'],
            'spatial': ['spatial_'],
            'advanced': ['quantile_', 'extreme_', 'daily_', 'coefficient_', 'anomaly_', 'min_distance_', 'near_threshold_', 'low_intensity_', 'intensity_bin_', 'rain_count_bin_', 'product_consistency_', 'product_disagreement', 'product_entropy', 'target_rolling_', 'target_anomaly_'],
            'interaction': ['interaction_'],
            'rolling_quantile': ['rolling_quantile_']  # 新增未分类的滚动分位数
        }
        
        # 初始化分类字典
        for category in categories:
            self.feature_categories[category] = []
        
        # 分类特征
        for feature_file in self.feature_files:
            categorized = False
            for category, prefixes in categories.items():
                if any(feature_file.startswith(prefix) for prefix in prefixes):
                    self.feature_categories[category].append(feature_file)
                    categorized = True
                    break
            
            if not categorized:
                if 'uncategorized' not in self.feature_categories:
                    self.feature_categories['uncategorized'] = []
                self.feature_categories['uncategorized'].append(feature_file)
    
    def get_feature_path(self, feature_name: str) -> str:
        """获取特征文件的完整路径"""
        if not feature_name.endswith('.npy'):
            feature_name += '.npy'
        return os.path.join(self.features_dir, feature_name)
    
    def load_feature(self, feature_name: str, safe_load: bool = True) -> np.ndarray:
        """
        加载单个特征
        
        Args:
            feature_name: 特征名称 (可以包含或不包含.npy后缀)
            safe_load: 是否进行安全加载 (处理异常)
        
        Returns:
            特征数据数组
        """
        feature_path = self.get_feature_path(feature_name)
        
        if not os.path.exists(feature_path):
            if safe_load:
                warnings.warn(f"特征文件不存在: {feature_name}")
                return None
            else:
                raise FileNotFoundError(f"特征文件不存在: {feature_path}")
        
        try:
            data = np.load(feature_path)
            return data
        except Exception as e:
            if safe_load:
                warnings.warn(f"加载特征失败 {feature_name}: {e}")
                return None
            else:
                raise e
    
    def load_multiple_features(self, feature_names: List[str], safe_load: bool = True) -> Dict[str, np.ndarray]:
        """
        批量加载多个特征
        
        Args:
            feature_names: 特征名称列表
            safe_load: 是否进行安全加载
        
        Returns:
            特征名称到数据的字典
        """
        features = {}
        for name in feature_names:
            data = self.load_feature(name, safe_load)
            if data is not None:
                clean_name = name.replace('.npy', '')
                features[clean_name] = data
        
        return features
    
    def load_category(self, category: str, safe_load: bool = True) -> Dict[str, np.ndarray]:
        """
        加载指定类别的所有特征
        
        Args:
            category: 特征类别名称
            safe_load: 是否进行安全加载
        
        Returns:
            特征字典
        """
        if category not in self.feature_categories:
            available_categories = list(self.feature_categories.keys())
            raise ValueError(f"未知类别 '{category}'. 可用类别: {available_categories}")
        
        feature_files = self.feature_categories[category]
        return self.load_multiple_features(feature_files, safe_load)
    
    def get_categories(self) -> List[str]:
        """获取所有可用的特征类别"""
        return list(self.feature_categories.keys())
    
    def get_category_features(self, category: str) -> List[str]:
        """获取指定类别的特征列表"""
        if category not in self.feature_categories:
            return []
        return [f.replace('.npy', '') for f in self.feature_categories[category]]
    
    def build_feature_matrix(self, 
                           feature_selection: Union[List[str], Dict[str, List[str]]], 
                           align_time: bool = True,
                           flatten_spatial: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        构建特征矩阵
        
        Args:
            feature_selection: 特征选择，可以是特征名称列表或类别字典
            align_time: 是否对齐时间维度
            flatten_spatial: 是否展平空间维度
        
        Returns:
            (特征矩阵, 特征名称列表)
        """
        # 处理特征选择
        if isinstance(feature_selection, dict):
            # 按类别选择
            selected_features = []
            for category, features in feature_selection.items():
                if features is None:
                    # 选择整个类别
                    selected_features.extend(self.get_category_features(category))
                else:
                    # 选择类别中的特定特征
                    selected_features.extend(features)
        else:
            # 直接特征列表
            selected_features = feature_selection
        
        # 加载特征
        feature_data = self.load_multiple_features(selected_features)
        
        if not feature_data:
            raise ValueError("没有成功加载任何特征")
        
        # 处理特征维度和对齐
        processed_features = []
        feature_names = []
        
        for name, data in feature_data.items():
            processed_data = self._process_feature_for_matrix(data, flatten_spatial)
            if processed_data is not None:
                processed_features.append(processed_data)
                feature_names.append(name)
        
        if not processed_features:
            raise ValueError("没有有效的特征数据用于构建矩阵")
        
        # 构建矩阵
        if align_time:
            # 找到最小时间长度
            min_time_len = min(f.shape[0] for f in processed_features)
            processed_features = [f[:min_time_len] for f in processed_features]
        
        # 水平堆叠特征
        feature_matrix = np.hstack(processed_features)
        
        return feature_matrix, feature_names
    
    def _process_feature_for_matrix(self, data: np.ndarray, flatten_spatial: bool) -> Optional[np.ndarray]:
        """处理单个特征用于矩阵构建"""
        if data.ndim == 1:
            # 时间序列特征，需要扩展为 (time, 1)
            return data.reshape(-1, 1)
        elif data.ndim == 2:
            # (time, points) 或 (lat, lon) 格式
            if flatten_spatial and data.shape[0] > 2000:  # 假设time维度大于2000
                # (time, points) 格式，直接返回
                return data
            elif flatten_spatial:
                # (lat, lon) 格式，展平并复制到所有时间步
                flattened = data.flatten()
                # 需要确定时间长度，这里假设1797
                return np.tile(flattened, (1797, 1))
            else:
                return data
        elif data.ndim == 3:
            # (time, lat, lon) 格式
            if flatten_spatial:
                # 展平空间维度
                time_len, lat, lon = data.shape
                return data.reshape(time_len, -1)
            else:
                return data
        else:
            warnings.warn(f"不支持的特征维度: {data.shape}")
            return None
    
    def get_feature_info(self, feature_name: str) -> Dict:
        """获取特征信息"""
        feature_path = self.get_feature_path(feature_name)
        
        if not os.path.exists(feature_path):
            return {"error": "文件不存在"}
        
        try:
            data = np.load(feature_path)
            return {
                "name": feature_name.replace('.npy', ''),
                "shape": data.shape,
                "dtype": str(data.dtype),
                "size_mb": data.nbytes / (1024 * 1024),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "mean": float(np.nanmean(data)),
                "std": float(np.nanstd(data)),
                "nan_count": int(np.isnan(data).sum()),
                "file_path": feature_path
            }
        except Exception as e:
            return {"error": str(e)}
    
    def print_summary(self):
        """打印特征库概览"""
        print("=" * 60)
        print("长江流域降雨预测特征库概览")
        print("=" * 60)
        
        print(f"特征目录: {self.features_dir}")
        print(f"总特征数: {len(self.feature_files)}")
        
        print("\n按类别分布:")
        total_size_mb = 0
        for category, features in self.feature_categories.items():
            print(f"  {category:15}: {len(features):3d} 个")
            
            # 计算类别大小
            category_size = 0
            for feature in features[:5]:  # 只检查前几个文件以估算
                try:
                    data = np.load(self.get_feature_path(feature))
                    category_size += data.nbytes
                except:
                    continue
            
            if len(features) > 0:
                avg_size = category_size / min(5, len(features))
                estimated_category_size = avg_size * len(features)
                total_size_mb += estimated_category_size / (1024 * 1024)
        
        print(f"\n估算总存储大小: {total_size_mb:.1f} MB")
        
        # 显示一些示例特征
        print("\n示例特征:")
        for category, features in list(self.feature_categories.items())[:3]:
            if features:
                example_feature = features[0]
                info = self.get_feature_info(example_feature)
                if "error" not in info:
                    print(f"  {category}: {info['name']} {info['shape']} ({info['size_mb']:.1f} MB)")
    
    def search_features(self, keyword: str, category: str = None) -> List[str]:
        """
        搜索特征
        
        Args:
            keyword: 搜索关键词
            category: 限制搜索的类别
        
        Returns:
            匹配的特征名称列表
        """
        search_pool = self.feature_files
        
        if category:
            if category in self.feature_categories:
                search_pool = self.feature_categories[category]
            else:
                return []
        
        matches = [f for f in search_pool if keyword.lower() in f.lower()]
        return [f.replace('.npy', '') for f in matches]
    
    def create_feature_subset(self, 
                            include_categories: List[str] = None,
                            exclude_categories: List[str] = None,
                            include_products: List[str] = None,
                            max_features: int = None,
                            priority_keywords: List[str] = None) -> List[str]:
        """
        创建特征子集
        
        Args:
            include_categories: 包含的类别
            exclude_categories: 排除的类别
            include_products: 包含的产品
            max_features: 最大特征数量
            priority_keywords: 优先级关键词
        
        Returns:
            特征名称列表
        """
        selected_features = []
        
        # 按类别选择
        categories_to_use = include_categories or list(self.feature_categories.keys())
        if exclude_categories:
            categories_to_use = [c for c in categories_to_use if c not in exclude_categories]
        
        for category in categories_to_use:
            if category in self.feature_categories:
                category_features = self.get_category_features(category)
                
                # 按产品过滤
                if include_products:
                    filtered_features = []
                    for feature in category_features:
                        if any(product in feature for product in include_products):
                            filtered_features.append(feature)
                    category_features = filtered_features
                
                selected_features.extend(category_features)
        
        # 优先级排序
        if priority_keywords:
            priority_features = []
            other_features = []
            
            for feature in selected_features:
                is_priority = any(keyword.lower() in feature.lower() for keyword in priority_keywords)
                if is_priority:
                    priority_features.append(feature)
                else:
                    other_features.append(feature)
            
            selected_features = priority_features + other_features
        
        # 限制数量
        if max_features and len(selected_features) > max_features:
            selected_features = selected_features[:max_features]
        
        return selected_features
    
    def export_feature_list(self, filepath: str, include_metadata: bool = True):
        """导出特征列表到文件"""
        export_data = {
            "export_time": datetime.now().isoformat(),
            "total_features": len(self.feature_files),
            "categories": {}
        }
        
        for category, features in self.feature_categories.items():
            category_data = {
                "count": len(features),
                "features": []
            }
            
            for feature in features:
                feature_info = {"name": feature.replace('.npy', '')}
                if include_metadata:
                    info = self.get_feature_info(feature)
                    if "error" not in info:
                        feature_info.update(info)
                
                category_data["features"].append(feature_info)
            
            export_data["categories"][category] = category_data
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"特征列表已导出到: {filepath}")
    
    def get_recommended_features(self, model_type: str = "basic") -> List[str]:
        """
        获取推荐的特征组合
        
        Args:
            model_type: 模型类型 ("basic", "temporal", "spatial", "advanced", "full")
        
        Returns:
            推荐特征列表
        """
        recommendations = {
            "basic": [
                "multi_product_mean", "multi_product_std", "multi_product_max",
                "rain_product_count", "product_consistency_ratio"
            ],
            "temporal": [
                "multi_product_mean", "multi_product_std", "sin_day_of_year", "cos_day_of_year",
                "lag_1_multi_product_mean", "lag_3_multi_product_mean", "lag_7_multi_product_mean",
                "season_onehot_0", "season_onehot_1", "season_onehot_2", "season_onehot_3"
            ],
            "spatial": [
                "multi_product_mean", "spatial_variance_GSMAP", "spatial_variance_IMERG",
                "spatial_gradient_magnitude_GSMAP", "spatial_correlation_GSMAP_IMERG"
            ],
            "advanced": [
                "multi_product_mean", "multi_product_quantile_25", "multi_product_quantile_75",
                "extreme_ratio_above_1.0", "anomaly_zscore_GSMAP", "product_entropy"
            ]
        }
        
        if model_type == "full":
            # 返回所有推荐特征的组合
            all_recommended = set()
            for features in recommendations.values():
                all_recommended.update(features)
            return list(all_recommended)
        
        return recommendations.get(model_type, recommendations["basic"])


# 便捷函数
def quick_load_features(feature_names: List[str], features_dir: str = None) -> Dict[str, np.ndarray]:
    """快速加载特征的便捷函数"""
    if features_dir is None:
        features_dir = "/mnt/f/rainfalldata/results/yangtze/features/features"
    
    loader = FeatureLoader(features_dir)
    return loader.load_multiple_features(feature_names)


def get_basic_feature_matrix(features_dir: str = None) -> Tuple[np.ndarray, List[str]]:
    """获取基础特征矩阵的便捷函数"""
    if features_dir is None:
        features_dir = "/mnt/f/rainfalldata/results/yangtze/features/features"
    
    loader = FeatureLoader(features_dir)
    recommended_features = loader.get_recommended_features("basic")
    return loader.build_feature_matrix(recommended_features)


if __name__ == "__main__":
    # 测试代码
    print("测试特征加载器...")
    
    try:
        # 初始化加载器
        loader = FeatureLoader()
        
        # 打印概览
        loader.print_summary()
        
        # 测试加载单个特征
        print("\n测试加载单个特征...")
        feature = loader.load_feature("multi_product_mean")
        if feature is not None:
            print(f"multi_product_mean shape: {feature.shape}")
        
        # 测试搜索功能
        print("\n搜索'GSMAP'相关特征:")
        gsmap_features = loader.search_features("GSMAP")
        print(f"找到 {len(gsmap_features)} 个相关特征")
        
        # 测试推荐特征
        print("\n基础推荐特征:")
        recommended = loader.get_recommended_features("basic")
        print(recommended[:5])
        
    except Exception as e:
        print(f"测试失败: {e}")