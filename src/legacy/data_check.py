import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import font_manager

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 首选黑体
    plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
except:
    try:
        # 如果没有SimHei，尝试使用Microsoft YaHei
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    except:
        print("警告：未能找到中文字体，可能会导致中文显示异常")

def check_missing_data(data, mask, dataset_name, special_mask=None):
    """检查数据集中的缺失值情况
    
    Args:
        data: 输入数据，形状为(lat, lon, time)
        mask: 掩膜数据
        dataset_name: 数据集名称（用于显示）
        special_mask: 特殊掩膜（如CHIRPS的特殊处理）
    
    Returns:
        processed_data: 处理后的数据
        stats: 包含统计信息的字典
    """
    # 创建数据副本
    processed_data = data.copy()
    
    # 使用特殊掩膜或普通掩膜
    working_mask = special_mask if special_mask is not None else mask
    
    # 初始化统计变量
    nan_counts = []
    error_counts = []
    clustered_counts = []
    boundary_counts = []
    total_mask_points = np.sum(working_mask == 1)
    
    print(f"\n{'='*20} 检查{dataset_name}数据 {'='*20}")
    print(f"Mask内有效点总数: {total_mask_points}")
    
    # 第一步：检查错误值和NaN值
    for day in tqdm(range(data.shape[2]), desc="检查数据"):
        daily_data = data[:,:,day].copy()
        
        # 检查错误值(-99.9)
        error_mask = np.abs(daily_data + 99.9) < 0.01
        errors = np.sum(error_mask & (working_mask == 1))
        error_counts.append(errors)
        
        # 检查NaN值
        nan_mask = np.isnan(daily_data) & (working_mask == 1)
        nans = np.sum(nan_mask)
        nan_counts.append(nans)
        
        # 处理mask外的值
        daily_data[working_mask == 0] = -99.9
        processed_data[:,:,day] = daily_data
        
        # 检查NaN的聚集性和边界性
        nan_indices = np.where(nan_mask)
        clustered = 0
        boundary = 0
        
        for i, j in zip(nan_indices[0], nan_indices[1]):
            # 获取3x3邻域
            i_min, i_max = max(0, i-1), min(data.shape[0]-1, i+1)
            j_min, j_max = max(0, j-1), min(data.shape[1]-1, j+1)
            neighborhood = daily_data[i_min:i_max+1, j_min:j_max+1].copy()
            
            # 检查边界性
            if np.any(np.abs(neighborhood + 99.9) < 0.01):
                boundary += 1
            
            # 检查聚集性
            center_i, center_j = i - i_min, j - j_min
            orig_val = neighborhood[center_i, center_j]
            neighborhood[center_i, center_j] = -888
            if np.sum(np.isnan(neighborhood)) > 0:
                clustered += 1
            neighborhood[center_i, center_j] = orig_val
            
        clustered_counts.append(clustered)
        boundary_counts.append(boundary)
    
    # 计算统计信息
    stats = {
        'total_days': data.shape[2],
        'total_mask_points': total_mask_points,
        'nan_stats': {
            'min': min(nan_counts),
            'max': max(nan_counts),
            'mean': np.mean(nan_counts)
        },
        'error_stats': {
            'min': min(error_counts),
            'max': max(error_counts),
            'mean': np.mean(error_counts)
        },
        'clustered_stats': {
            'min': min(clustered_counts),
            'max': max(clustered_counts),
            'mean': np.mean(clustered_counts)
        },
        'boundary_stats': {
            'min': min(boundary_counts),
            'max': max(boundary_counts),
            'mean': np.mean(boundary_counts)
        },
        'daily_stats': {
            'nan_counts': nan_counts,
            'error_counts': error_counts,
            'clustered_counts': clustered_counts,
            'boundary_counts': boundary_counts
        }
    }
    
    # 在返回前添加最大缺失值天的索引到stats中
    max_nan_day = np.argmax(nan_counts)
    stats['max_nan_day'] = {
        'index': max_nan_day,
        'count': nan_counts[max_nan_day]
    }
    
    return processed_data, stats

def plot_missing_data_stats(stats, dataset_name):
    """绘制缺失值统计图"""
    daily_stats = stats['daily_stats']
    
    plt.figure(figsize=(12, 6))
    plt.plot(daily_stats['nan_counts'], 'r-', linewidth=2, label='NaN值总数')
    plt.plot(daily_stats['clustered_counts'], 'g--', linewidth=2, label='聚集NaN值')
    plt.plot(daily_stats['boundary_counts'], 'b:', linewidth=2, label='边界NaN值')
    plt.plot(daily_stats['error_counts'], 'm-.', linewidth=2, label='错误值(-99.9)')
    
    plt.xlabel('天数', fontsize=12)
    plt.ylabel('缺失点数量', fontsize=12)
    plt.title(f'{dataset_name}数据中mask内每天的缺失点情况', pad=20, fontsize=14)
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def print_stats_summary(stats, dataset_name):
    """打印统计信息摘要"""
    print(f"\n=== {dataset_name}数据统计结果 ===")
    print(f"总天数: {stats['total_days']}")
    print(f"Mask内有效点总数: {stats['total_mask_points']}")
    print(f"NaN值分布: 最少 {stats['nan_stats']['min']}, "
          f"最多 {stats['nan_stats']['max']}, "
          f"平均 {stats['nan_stats']['mean']:.2f}")
    print(f"错误值(-99.9)分布: 最少 {stats['error_stats']['min']}, "
          f"最多 {stats['error_stats']['max']}, "
          f"平均 {stats['error_stats']['mean']:.2f}")

def plot_max_missing_day(data, mask, stats, dataset_name):
    """绘制缺失值最多的那天的分布图"""
    # 找出缺失值最多的那天
    nan_counts = stats['daily_stats']['nan_counts']
    max_nan_day = np.argmax(nan_counts)
    
    # 获取那天的数据
    daily_data = data[:,:,max_nan_day]
    nan_mask = np.isnan(daily_data) & (mask == 1)
    
    # 创建彩色显示，使用更鲜明的颜色
    masked_display = np.zeros((nan_mask.shape[0], nan_mask.shape[1], 3))
    masked_display[mask == 1] = [0.9, 0.9, 0.9]      # 有效区域显示为亮灰色
    masked_display[nan_mask] = [0.8, 0.2, 0.2]       # NaN位置显示为深红色
    masked_display[mask == 0] = [0.3, 0.3, 0.3]      # 掩膜外区域显示为深灰色
    
    plt.figure(figsize=(12, 8))
    plt.imshow(masked_display)
    
    # 设置标题和说明，增加字体大小
    title = (f'{dataset_name}数据第{max_nan_day+1}天NaN分布图\n'
             f'缺失值数量: {nan_counts[max_nan_day]}')
    plt.title(title, pad=20, fontsize=14)
    
    # 添加颜色图例说明
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[0.8, 0.2, 0.2], label='NaN位置'),
        Patch(facecolor=[0.9, 0.9, 0.9], label='有效区域'),
        Patch(facecolor=[0.3, 0.3, 0.3], label='掩膜外区域')
    ]
    plt.legend(handles=legend_elements, loc='upper right', 
              fontsize=10, title='区域说明')
    
    # 调整布局
    plt.tight_layout()
    plt.show()

def hybrid_interpolation(data, mask, reference_data=None, dataset_name=""):
    """混合插值和替换方法处理缺失值
    
    Args:
        data: 需要处理的数据，形状为(lat, lon, time)
        mask: 掩膜数据
        reference_data: 可选的参考数据，用于替换无法插值的点
        dataset_name: 数据集名称，用于显示
    
    Returns:
        processed_data: 处理后的数据
        stats: 处理统计信息
    """
    processed_data = data.copy()
    stats = {
        'interpolated_points': 0,
        'replaced_points': 0,
        'remaining_nan_points': 0,
        'daily_stats': []
    }
    
    def check_neighborhood(neighborhood):
        """分析3x3邻域的特征"""
        total_nan = np.sum(np.isnan(neighborhood))
        is_boundary = np.any(np.abs(neighborhood + 99.9) < 0.01)
        return {
            'nan_count': total_nan,
            'is_boundary': is_boundary,
            'interpolatable': total_nan <= 3 and not is_boundary
        }
    
    def find_continuous_nans(daily_data, mask):
        """识别连续NaN区域"""
        from scipy.ndimage import label
        nan_mask = np.isnan(daily_data) & (mask == 1)
        labeled_array, num_features = label(nan_mask)
        return labeled_array, num_features
    
    # 遍历每一天的数据
    for day in tqdm(range(data.shape[2]), desc=f"处理{dataset_name}数据"):
        daily_data = data[:, :, day].copy()
        interpolation_values = np.zeros_like(daily_data)
        interpolation_mask = np.zeros_like(daily_data, dtype=bool)
        
        # 找出需要处理的NaN点
        nan_mask = np.isnan(daily_data) & (mask == 1)
        labeled_nans, num_regions = find_continuous_nans(daily_data, mask)
        
        daily_stats = {
            'total_nan': np.sum(nan_mask),
            'interpolated': 0,
            'replaced': 0,
            'remaining_nan': 0
        }
        
        # 第一步：进行插值
        for i, j in zip(*np.where(nan_mask)):
            # 获取8邻域
            i_min, i_max = max(0, i-1), min(data.shape[0]-1, i+1)
            j_min, j_max = max(0, j-1), min(data.shape[1]-1, j+1)
            neighborhood = daily_data[i_min:i_max+1, j_min:j_max+1].copy()
            
            # 分析邻域特征
            analysis = check_neighborhood(neighborhood)
            region_size = np.sum(labeled_nans == labeled_nans[i, j])
            
            # 判断是否适合插值
            if (analysis['interpolatable'] and region_size <= 4):
                valid_values = neighborhood[~np.isnan(neighborhood)]
                if len(valid_values) > 0:
                    interpolation_values[i, j] = np.mean(valid_values)
                    interpolation_mask[i, j] = True
                    daily_stats['interpolated'] += 1
        
        # 应用插值结果
        daily_data[interpolation_mask] = interpolation_values[interpolation_mask]
        
        # 第二步：对剩余的NaN点进行替换
        if reference_data is not None:
            remaining_nan_mask = np.isnan(daily_data) & (mask == 1)
            if np.any(remaining_nan_mask):
                reference_day = reference_data[:, :, day]
                valid_reference = ~np.isnan(reference_day)
                replace_mask = remaining_nan_mask & valid_reference
                daily_data[replace_mask] = reference_day[replace_mask]
                daily_stats['replaced'] = np.sum(replace_mask)
        
        # 统计剩余的NaN点
        daily_stats['remaining_nan'] = np.sum(np.isnan(daily_data) & (mask == 1))
        
        # 更新数据和统计信息
        processed_data[:, :, day] = daily_data
        stats['daily_stats'].append(daily_stats)
        stats['interpolated_points'] += daily_stats['interpolated']
        stats['replaced_points'] += daily_stats['replaced']
        stats['remaining_nan_points'] += daily_stats['remaining_nan']
        
        # 输出每天的处理信息
        if daily_stats['total_nan'] > 0:
            print(f"\n第{day+1}天处理结果:")
            print(f"  总缺失值: {daily_stats['total_nan']}")
            print(f"  成功插值: {daily_stats['interpolated']}")
            print(f"  数据替换: {daily_stats['replaced']}")
            print(f"  剩余缺失: {daily_stats['remaining_nan']}")
    
    # 打印总体统计信息
    print(f"\n{dataset_name}数据处理总结:")
    print(f"总插值点数: {stats['interpolated_points']}")
    print(f"总替换点数: {stats['replaced_points']}")
    print(f"剩余缺失点: {stats['remaining_nan_points']}")
    
    return processed_data, stats

def plot_interpolation_results(original, processed, mask, day_index, dataset_name):
    """可视化插值和替换结果"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 原始数据
    masked_orig = np.ma.masked_where(np.isnan(original[:,:,day_index]) | (mask == 0), 
                                   original[:,:,day_index])
    im1 = ax1.imshow(masked_orig, cmap='viridis')
    ax1.set_title('处理前')
    plt.colorbar(im1, ax=ax1)
    
    # 处理后的数据
    masked_proc = np.ma.masked_where(np.isnan(processed[:,:,day_index]) | (mask == 0), 
                                   processed[:,:,day_index])
    im2 = ax2.imshow(masked_proc, cmap='viridis')
    ax2.set_title('处理后')
    plt.colorbar(im2, ax=ax2)
    
    plt.suptitle(f'{dataset_name}数据第{day_index+1}天处理效果对比', fontsize=14)
    plt.tight_layout()
    plt.show()

# 使用示例：
'''
# 对数据进行混合处理
processed_data, stats = hybrid_interpolation(data, mask, reference_data, "CHIRPS")

# 显示处理效果（选择缺失值最多的那天）
max_nan_day = np.argmax([np.sum(np.isnan(data[:,:,i]) & (mask == 1)) 
                         for i in range(data.shape[2])])
plot_interpolation_results(data, processed_data, mask, max_nan_day, "CHIRPS")
'''
