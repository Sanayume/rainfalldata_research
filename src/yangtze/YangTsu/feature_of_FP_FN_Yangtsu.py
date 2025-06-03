from loaddata import mydata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
#import torch.nn as nn
#import torch
#import torch.optim as optim
#import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
from scipy.interpolate import griddata
import platform
import os
import subprocess
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from matplotlib.font_manager import fontManager
from scipy import stats
from pysal.explore import esda
from pysal.lib import weights
from pysal.explore import esda
from pysal.lib import weights

# 定义函数计算各类错误率
def calculate_error_rates(x_data, y_data, threshold):
    """
    注意输入数据的shape的要求
    对数据形状的要求：
    只要x_data和y_data的shape一致即可
    """
    # 根据阈值筛选降雨事件
    selected_mask = np.where(x_data <= threshold, 1, 0)
    selected_x = np.where(x_data <= threshold, x_data, -999)
    total_selected = np.sum(selected_mask)  # 该阈值下被选中的总样本数
    if total_selected == 0:
        return None
    # 计算误报数
    FAR_points = (selected_x > 0) & (y_data == 0)
    MS_points = (selected_x == 0) & (y_data > 0)
    TRUE_points = ((selected_x > 0) & (y_data > 0)) | ((selected_x == 0) & (y_data == 0))
    FAR_sum = np.sum(FAR_points)
    MS_sum = np.sum(MS_points)
    TRUE_sum = np.sum(TRUE_points)
    FAR_rate = FAR_sum / total_selected
    MS_rate = MS_sum / total_selected
    TRUE_rate = TRUE_sum / total_selected
    return {
        'false_alarm_rate': FAR_rate,
        'miss_rate': MS_rate,
        'error_rate': (FAR_rate + MS_rate),
        'correct_rate': TRUE_rate
    }

def optimize_threshold(x, y, min_threshold=0, max_threshold=20, coarse_steps=100, fine_steps=50):
    """
    输入数据x和y的shape要求:
        对x和y只要shape一致即可
    函数用途：
        优化阈值，找到最佳的阈值，使得误报率，漏报率，总错误率最小
    参数解释：
        min_threshold: 最小阈值
        max_threshold: 最大阈值
        coarse_steps: 粗略扫描的步长
        fine_steps: 精细扫描的步长
    返回值：
        result: 存储所有有关分析的数据包括阈值，误报率，漏报率，总错误率，正确率
        result字典中包含以下内容：
        results_df: 存储所有有关分析的数据包括阈值，误报率，漏报率，总错误率，正确率
        fine_df_fa: 存储精细扫描的误报率数据
        fine_df_miss: 存储精细扫描的漏报率数据
        fine_df_error: 存储精细扫描的总错误率数据
        final_best_fa_threshold: 最佳误报率阈值
        final_best_miss_threshold: 最佳漏报率阈值
        final_best_error_threshold: 最佳总错误率阈值
        thresholds: 粗略扫描的阈值
        fine_thresholds_fa: 精细扫描的误报率阈值
        fine_thresholds_miss: 精细扫描的漏报率阈值
        fine_thresholds_error: 精细扫描的总错误率阈值
    """
    # 初始化阈值范围用于粗略扫描，扩大到20
    thresholds = np.linspace(min_threshold, max_threshold, coarse_steps)

    # 存储所有阈值的错误率
    results = []
    for threshold in thresholds:
        rates = calculate_error_rates(x, y, threshold)
        results.append({
            'threshold': threshold,
            **rates
        })
    results_df = pd.DataFrame(results)

    # 找出各类错误率最大的阈值
    max_false_alarm_idx = results_df['false_alarm_rate'].idxmax()
    max_miss_idx = results_df['miss_rate'].idxmax()
    max_error_idx = results_df['error_rate'].idxmax()

    # 在最佳误报率阈值附近进行精细扫描
    fine_thresholds_fa = np.linspace(max(0, results_df.iloc[max_false_alarm_idx]['threshold'] - 0.2), 
                                     min(20, results_df.iloc[max_false_alarm_idx]['threshold'] + 0.2), 50)
    fine_results_fa = []
    for threshold in fine_thresholds_fa:
        rates = calculate_error_rates(x, y, threshold)
        fine_results_fa.append({
            'threshold': threshold,
            **rates
        })
    fine_df_fa = pd.DataFrame(fine_results_fa)
    final_max_fa_idx = fine_df_fa['false_alarm_rate'].idxmax()
    final_best_fa_threshold = fine_df_fa.iloc[final_max_fa_idx]['threshold']

    # 在最佳漏报率阈值附近进行精细扫描
    fine_thresholds_miss = np.linspace(max(0, results_df.iloc[max_miss_idx]['threshold'] - 0.2), 
                                     min(20, results_df.iloc[max_miss_idx]['threshold'] + 0.2), 50)
    fine_results_miss = []
    for threshold in fine_thresholds_miss:
        rates = calculate_error_rates(x, y, threshold)
        fine_results_miss.append({
            'threshold': threshold,
            **rates
        })
    fine_df_miss = pd.DataFrame(fine_results_miss)
    final_max_miss_idx = fine_df_miss['miss_rate'].idxmax()
    final_best_miss_threshold = fine_df_miss.iloc[final_max_miss_idx]['threshold']

    # 在最佳总错误率阈值附近进行精细扫描
    fine_thresholds_error = np.linspace(max(0, results_df.iloc[max_error_idx]['threshold'] - 0.2), 
                                     min(20, results_df.iloc[max_error_idx]['threshold'] + 0.2), 50)
    fine_results_error = []
    for threshold in fine_thresholds_error:
        rates = calculate_error_rates(x, y, threshold)
        fine_results_error.append({
            'threshold': threshold,
            **rates
        })
    fine_df_error = pd.DataFrame(fine_results_error)
    final_max_error_idx = fine_df_error['error_rate'].idxmax()
    final_best_error_threshold = fine_df_error.iloc[final_max_error_idx]['threshold']

    # 打印最终结果
    #print(f"最佳误报率阈值: {final_best_fa_threshold:.4f}, 误报率: {fine_df_fa.iloc[final_max_fa_idx]['false_alarm_rate']:.4f}")
    #print(f"最佳漏报率阈值: {final_best_miss_threshold:.4f}, 漏报率: {fine_df_miss.iloc[final_max_miss_idx]['miss_rate']:.4f}")
    #print(f"最佳总错误率阈值: {final_best_error_threshold:.4f}, 错误率: {fine_df_error.iloc[final_max_error_idx]['error_rate']:.4f}")

    result = {
        'results_df': results_df,
        'fine_df_fa': fine_df_fa, 
        'fine_df_miss': fine_df_miss,
        'fine_df_error': fine_df_error,
        'final_best_fa_threshold': final_best_fa_threshold,
        'final_best_miss_threshold': final_best_miss_threshold,
        'final_best_error_threshold': final_best_error_threshold,
        'thresholds': thresholds,
        'fine_thresholds_fa': fine_thresholds_fa,
        'fine_thresholds_miss': fine_thresholds_miss,
        'fine_thresholds_error': fine_thresholds_error,
        'final_max_fa_idx': final_max_fa_idx,
        'final_max_miss_idx': final_max_miss_idx,
        'final_max_error_idx': final_max_error_idx
    }
    return result

def plot_threshold_results(result, save_path):
    """
    一次生成三个图像，分别对应误报率，漏报率，总错误率
    输入参数：
        result: 优化阈值的结果
        save_path: 保存路径,保护想要创建的文件名字,不带扩展名,默认png格式
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    # 解包结果字典
    results_df = result['results_df']
    fine_df_fa = result['fine_df_fa']
    fine_df_miss = result['fine_df_miss']
    fine_df_error = result['fine_df_error']
    final_best_fa_threshold = result['final_best_fa_threshold']
    final_best_miss_threshold = result['final_best_miss_threshold']
    final_best_error_threshold = result['final_best_error_threshold']
    thresholds = result['thresholds']
    fine_thresholds_fa = result['fine_thresholds_fa']
    fine_thresholds_miss = result['fine_thresholds_miss']
    fine_thresholds_error = result['fine_thresholds_error']
    final_max_fa_idx = result['final_max_fa_idx']
    final_max_miss_idx = result['final_max_miss_idx']
    final_max_error_idx = result['final_max_error_idx']

    fig, axes = plt.subplots(3, 1, figsize=(10, 15))

    # 绘制误报率曲线
    axes[0].plot(thresholds, results_df['false_alarm_rate'], 'b-', label='粗扫描', linewidth=2)
    axes[0].plot(fine_thresholds_fa, fine_df_fa['false_alarm_rate'], 'r--', label='精细扫描', linewidth=2)
    axes[0].axvline(x=final_best_fa_threshold, color='g', linestyle='-', label=f'最佳阈值: {final_best_fa_threshold:.4f}')
    axes[0].set_title('误报率与阈值的关系', fontsize=14)
    axes[0].set_xlabel('阈值', fontsize=12)
    axes[0].set_ylabel('误报率', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(loc='best')

    # 绘制漏报率曲线
    axes[1].plot(thresholds, results_df['miss_rate'], 'b-', label='粗扫描', linewidth=2)
    axes[1].plot(fine_thresholds_miss, fine_df_miss['miss_rate'], 'r--', label='精细扫描', linewidth=2)
    axes[1].axvline(x=final_best_miss_threshold, color='g', linestyle='-', label=f'最佳阈值: {final_best_miss_threshold:.4f}')
    axes[1].set_title('漏报率与阈值的关系', fontsize=14)
    axes[1].set_xlabel('阈值', fontsize=12)
    axes[1].set_ylabel('漏报率', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(loc='best')

    # 绘制总错误率曲线
    axes[2].plot(thresholds, results_df['error_rate'], 'b-', label='错误率(粗扫描)', linewidth=2)
    axes[2].plot(thresholds, results_df['correct_rate'], 'g-', label='正确率', linewidth=2)
    axes[2].plot(fine_thresholds_error, fine_df_error['error_rate'], 'r--', label='错误率(精细扫描)', linewidth=2)
    axes[2].axvline(x=final_best_error_threshold, color='m', linestyle='-', label=f'最佳阈值: {final_best_error_threshold:.4f}')
    axes[2].set_title('总错误率与阈值的关系', fontsize=14)
    axes[2].set_xlabel('阈值', fontsize=12)
    axes[2].set_ylabel('比率', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend(loc='best')

    plt.tight_layout()
    plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 将结果保存到CSV
    results_summary = pd.DataFrame([
        {'错误类型': '误报', '最佳阈值': final_best_fa_threshold, '错误率': fine_df_fa.iloc[final_max_fa_idx]['false_alarm_rate']},
        {'错误类型': '漏报', '最佳阈值': final_best_miss_threshold, '错误率': fine_df_miss.iloc[final_max_miss_idx]['miss_rate']},
        {'错误类型': '总错误', '最佳阈值': final_best_error_threshold, '错误率': fine_df_error.iloc[final_max_error_idx]['error_rate']}
    ])
    results_summary.to_csv(f'{save_path}.csv', index=False, encoding='utf-8-sig')
    print(f"分析结果已保存到 '{save_path}' 和 '{save_path.replace('.png', '_summary.csv')}'")

def analysis_fn_fp_with_time_feature(X, Y):
    """
    分析误报率和漏报率随时间变化的情况
    输入参数:
        X,Y的shape一致即可, 期望shape[0]为时间
    """
    #为X, Y添加时间编码
    time_steps = np.arange(X.shape[0]) #时间步,动态生成
    time_features = np.zeros((X.shape[0], 2)) #初始化时间特征数组
    time_features[:, 0] = np.sin(2 * np.pi * time_steps / 365) #正弦编码
    time_features[:, 1] = np.cos(2 * np.pi * time_steps / 365) #余弦编码
    #直接在X的最后一个维度添加时间特征
    X = np.concatenate([X, time_features[:, :, np.newaxis, np.newaxis]], axis=3)
    #Y保持不变 
my_data = mydata()
X,Y = my_data.get_basin_spatial_data(2)

X = np.transpose(X, (1,0,2,3))  # 转置X的维度，使其符合预期
"""
X :(1827, 6, 144, 256)
Y :(1827, 144, 256)
"""
is_rain_true_binary = np.where(Y > 0, 1, 0)
days = np.arange(1827)

product_list = my_data.get_products()

for product in range(X.shape[1]):
    product_name = product_list[product]
    X_product = X[:, product, :, :]
    print(X_product.shape)
    is_rain_pred_binary = np.where(X_product > 0, 1, 0)
    print(is_rain_pred_binary.shape)
    fp_rate_list = []
    fn_rate_list = []
    true_rate_list = []
    false_rate_list = []
    time_list = []
    for d in range(X_product.shape[0]):
        time_x = np.sin(2 * np.pi * d / 365)
        time_y = np.cos(2 * np.pi * d / 365)
        time_list.append([time_x, time_y])

        is_rain_true_binary_d = is_rain_true_binary[d, :, :]
        is_rain_pred_binary_d = is_rain_pred_binary[d, :, :]
        total_point = np.sum(np.where(is_rain_true_binary_d != np.nan, 1, 0))
        fp_point = np.where((is_rain_pred_binary_d == 1) & (is_rain_true_binary_d == 0), 1, 0)
        fn_point = np.where((is_rain_pred_binary_d == 0) & (is_rain_true_binary_d == 1), 1, 0)
        tn_point = np.where((is_rain_pred_binary_d == 0) & (is_rain_true_binary_d == 0), 1, 0)
        tp_point = np.where(((is_rain_pred_binary_d == 1) & (is_rain_true_binary_d == 1)), 1, 0)
        true_point = np.where((is_rain_pred_binary_d == 1) & (is_rain_true_binary_d == 1) | ((is_rain_pred_binary_d == 0) & (is_rain_true_binary_d == 0)), 1, 0)
        false_point = np.where(((is_rain_pred_binary_d == 1) & (is_rain_true_binary_d == 0)) | ((is_rain_pred_binary_d == 0) & (is_rain_true_binary_d == 1)), 1, 0)
        if  (np.sum(fp_point) + np.sum(tp_point) == 0):
            fp_rate = 0
        else:
            fp_rate = np.sum(fp_point) / (np.sum(fp_point) + np.sum(tp_point))
        if (np.sum(fn_point) + np.sum(tp_point) == 0):
            fn_rate = 0
        else:
            fn_rate = np.sum(fn_point) / (np.sum(fn_point) + np.sum(tp_point))
        if (np.sum(true_point) + np.sum(false_point) == 0):
            true_rate = 0
        else:
            true_rate = np.sum(true_point) / (np.sum(true_point) + np.sum(false_point))
        if (np.sum(false_point) + np.sum(true_point) == 0):
            false_rate = 0
        else:
            false_rate = np.sum(false_point) / (np.sum(false_point) + np.sum(true_point))

        #print(f"第{d}天, 误报率: {fp_rate}\n漏报率: {fn_rate}\n正确率: {true_rate}\n错误率: {false_rate}")
        #print(f"总点数: {total_point}, 正确点数: {true_point}, 错误点数: {false_point}, 误报点数: {fp_point}, 漏报点数: {fn_point}")

        fp_rate_list.append(fp_rate)
        fn_rate_list.append(fn_rate)
        true_rate_list.append(true_rate)
        false_rate_list.append(false_rate)
    fp_rate_list = np.array(fp_rate_list)
    fn_rate_list = np.array(fn_rate_list)
    true_rate_list = np.array(true_rate_list)
    false_rate_list = np.array(false_rate_list)
    time_list = np.array(time_list)
    far_list = fp_rate_list 
    pod_list = true_rate_list 
    fnr_list = fn_rate_list 

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    # 创建实际的日期信息（2016-2020）
    start_date = pd.Timestamp('2016-01-01')
    dates = [start_date + pd.Timedelta(days=int(d)) for d in range(len(days))]
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])
    days_of_year = np.array([d.dayofyear for d in dates])
    
    # 定义季节 - 根据月份
    season_map = {1: '冬季', 2: '冬季', 3: '春季', 4: '春季', 5: '春季', 
                  6: '夏季', 7: '夏季', 8: '夏季', 9: '秋季', 10: '秋季', 11: '秋季', 12: '冬季'}
    seasons = np.array([season_map[m] for m in months])
    
    # 为每个产品保存指标，以便最后汇总
    if product == 0:
        all_metrics = {}
    
    all_metrics[f'产品_{product+1}'] = {
        'far': far_list,
        'pod': pod_list, 
        'fnr': fnr_list,
        'days': days,
        'days_of_year': days_of_year,
        'months': months,
        'seasons': seasons,
        'years': years,
        'dates': dates
    }
    
    # 设置更美观的风格
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = ['#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#8c564b', '#e377c2']
    markers = ['o', 's', '^', 'D', 'p', '*']  # 增加不同形状的标记
    
    # 1. 增强的时间序列图 - 添加平滑趋势线
    fig = plt.figure(figsize=(20, 12))
    plt.subplot(2, 1, 1)
    
    # 使用7天移动平均使趋势更清晰
    window = 7
    far_smooth = np.convolve(far_list, np.ones(window)/window, mode='valid')
    pod_smooth = np.convolve(pod_list, np.ones(window)/window, mode='valid')
    fnr_smooth = np.convolve(fnr_list, np.ones(window)/window, mode='valid')
    days_smooth = days[window-1:]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    plt.plot(days, far_list, 'r-', alpha=0.3, label='误报率 (原始)')
    plt.plot(days, pod_list, 'g-', alpha=0.3, label='命中率 (原始)')
    plt.plot(days, fnr_list, 'b-', alpha=0.3, label='漏报率 (原始)')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    plt.plot(days_smooth, far_smooth, 'ro-', linewidth=2.5, markevery=30, markersize=8, label='误报率 (7天移动平均)')
    plt.plot(days_smooth, pod_smooth, 'gs-', linewidth=2.5, markevery=30, markersize=8, label='命中率 (7天移动平均)')
    plt.plot(days_smooth, fnr_smooth, 'b^-', linewidth=2.5, markevery=30, markersize=8, label='漏报率 (7天移动平均)')
    
    # 添加年份标记
    for year in range(2016, 2021):
        year_start = (pd.Timestamp(f'{year}-01-01') - start_date).days
        plt.axvline(x=year_start, color='gray', linestyle='--', alpha=0.5)
        plt.text(year_start+10, 0.95, f'{year}年', rotation=90, alpha=0.7, fontsize=12)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    plt.xlabel('天数', fontsize=14)
    plt.ylabel('比率', fontsize=14)
    plt.title(f'{product_name} - 降雨检测指标随时间变化趋势', fontsize=16)
    plt.legend(loc='best', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 2. 极坐标季节图 - 新增的可视化方式
    ax = plt.subplot(2, 2, 3, polar=True)
    
    # 将日期转换为角度（弧度）
    theta = days_of_year * 2 * np.pi / 366  # 使用366以考虑闰年
    
    # 绘制散点
    cmap = plt.cm.plasma
    norm = plt.Normalize(min(far_list), max(far_list))
    
    # 按年份使用不同标记
    for year in np.unique(years):
        year_mask = years == year
        year_theta = theta[year_mask]
        year_far = far_list[year_mask]
        marker_idx = (year - 2016) % len(markers)
        sc = ax.scatter(year_theta, year_far, c=year_far, cmap=cmap, norm=norm,
                   marker=markers[marker_idx], alpha=0.7, s=40, edgecolor='none', 
                   label=f'{year}年')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号

    plt.colorbar(sc, label='误报率')
    
    # 设置极坐标刻度
    ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
    ax.set_xticklabels(['一月', '二月', '三月', '四月', '五月', '六月', 
                        '七月', '八月', '九月', '十月', '十一月', '十二月'])
    ax.set_title('按月份的误报率分布（极坐标视图）', fontsize=14)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    # 3. 热力图 - 按年份和月份
    ax = plt.subplot(2, 2, 4)
    
    # 准备按月和年份分组的数据
    # 只考虑2016-2020年
    unique_years = np.arange(2016, 2021)
    num_years = len(unique_years)
    
    # 创建月度平均FAR热图
    heatmap_data = np.zeros((num_years, 12))
    for i, y in enumerate(unique_years):
        for m in range(12):
            mask = (years == y) & (months == m+1)
            if np.any(mask):
                heatmap_data[i, m] = np.mean(far_list[mask])
    
    im = ax.imshow(heatmap_data, cmap='plasma', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(im, label='平均误报率')
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_yticks(range(num_years))
    ax.set_yticklabels([f'{y}年' for y in unique_years])
    ax.set_xticks(range(12))
    ax.set_xticklabels(['一月', '二月', '三月', '四月', '五月', '六月', 
                       '七月', '八月', '九月', '十月', '十一月', '十二月'])
    
    # 添加数值标签
    for i in range(num_years):
        for j in range(12):
            if not np.isnan(heatmap_data[i, j]):
                text = ax.text(j, i, f"{heatmap_data[i, j]:.2f}",
                           ha="center", va="center", color="white" if heatmap_data[i, j] > 0.5 else "black",
                           fontsize=8)
    
    ax.set_title('按年份和月份的平均误报率', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f'增强指标_{product_name}.png', dpi=300, bbox_inches='tight')
    
    # 4. 季节性变化箱线图
    plt.figure(figsize=(18, 15))
    
    # 按季节分组的数据
    seasons_unique = np.array(['冬季', '春季', '夏季', '秋季'])
    season_data = {metric: {season: [] for season in seasons_unique} for metric in ['误报率', '命中率', '漏报率']}
    
    for i in range(len(seasons)):
        season_data['误报率'][seasons[i]].append(far_list[i])
        season_data['命中率'][seasons[i]].append(pod_list[i])
        season_data['漏报率'][seasons[i]].append(fnr_list[i])
    
    metrics = ['误报率', '命中率', '漏报率']
    colors = ['#ff9999', '#66b3ff', '#99ff99']
    
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i+1)
        
        boxdata = [season_data[metric][season] for season in seasons_unique]
        box = plt.boxplot(boxdata, patch_artist=True)
        
        # 自定义箱线图颜色
        for patch, color in zip(box['boxes'], [colors[i]]*4):
            patch.set_facecolor(color)
        
        # 添加散点以显示实际分布
        for j, season in enumerate(seasons_unique):
            y = season_data[metric][season]
            x = np.random.normal(j+1, 0.08, size=len(y))
            
            # 按年份使用不同标记
            for year in np.unique(years):
                year_mask = np.array([years[idx] == year for idx in range(len(years)) if seasons[idx] == season])
                if len(year_mask) > 0:
                    year_y = np.array(y)[year_mask]
                    year_x = np.array(x)[year_mask]
                    marker_idx = (year - 2016) % len(markers)
                    plt.scatter(year_x, year_y, alpha=0.4, s=30, 
                            marker=markers[marker_idx], label=f'{year}年' if j == 0 else "", 
                            edgecolor='black', linewidth=0.5)
        
        if i == 0:
            plt.legend(loc='upper right', fontsize=10)
        
        plt.xlabel('季节', fontsize=14)
        plt.ylabel(f'{metric}', fontsize=14)
        plt.title(f'{product_name} - 按季节的{metric}分布', fontsize=16)
        plt.xticks([1, 2, 3, 4], seasons_unique)
        plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'季节性箱线图_{product_name}.png', dpi=300, bbox_inches='tight')
    
    # 5. 月度趋势图 - 显示每个月的平均值和标准差
    plt.figure(figsize=(16, 12))
    
    metric_names = ['误报率', '命中率', '漏报率']
    for i, (metric, title, color) in enumerate(zip(
            [far_list, pod_list, fnr_list], 
            metric_names,
            ['#ff7f0e', '#2ca02c', '#1f77b4'])):
        
        plt.subplot(3, 1, i+1)
        
        means = []
        stds = []
        
        for m in range(1, 13):
            mask = months == m
            month_data = metric[mask]
            means.append(np.mean(month_data))
            stds.append(np.std(month_data))
        
        x = np.arange(1, 13)
        bars = plt.bar(x, means, yerr=stds, color=color, alpha=0.7,
                capsize=7, edgecolor='black', linewidth=1.5)
        
        # 在每个柱子上添加数值标签
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{mean:.2f}', ha='center', va='bottom', fontsize=10)
        
        plt.xlabel('月份', fontsize=14)
        plt.ylabel(title, fontsize=14)
        plt.title(f'{product_name} - 月度{title}（均值 ± 标准差）', fontsize=16)
        plt.xticks(x, ['一月', '二月', '三月', '四月', '五月', '六月', 
                      '七月', '八月', '九月', '十月', '十一月', '十二月'])
        plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'月度趋势_{product_name}.png', dpi=300, bbox_inches='tight')
    plt.show()

# 最后，创建汇总展示所有产品的大图
if product == X.shape[1] - 1:  # 当处理完所有产品后
    plt.figure(figsize=(24, 16))
    
    # 1. 所有产品随时间变化的FAR比较
    plt.subplot(3, 1, 1)
    for p_idx, (prod, metrics) in enumerate(all_metrics.items()):
        # 使用滑动平均平滑曲线
        window = 30  # 30天窗口使曲线更平滑
        far_smooth = np.convolve(metrics['far'], np.ones(window)/window, mode='valid')
        days_smooth = metrics['days'][window-1:]
        plt.plot(days_smooth, far_smooth, '-', 
                linewidth=2, marker=markers[p_idx % len(markers)], markevery=60,
                label=f'{product_list[p_idx]} 误报率')
    
    # 添加年份标记
    for year in range(2016, 2021):
        year_start = (pd.Timestamp(f'{year}-01-01') - start_date).days
        plt.axvline(x=year_start, color='gray', linestyle='--', alpha=0.5)
        plt.text(year_start+10, 0.95, f'{year}年', rotation=90, alpha=0.7, fontsize=12)
    
    plt.xlabel('天数', fontsize=14)
    plt.ylabel('误报率', fontsize=14)
    plt.title('各产品误报率对比', fontsize=16)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # 2. 季节性热图 - 每个产品每个月的平均指标
    metrics_to_plot = ['far', 'pod']
    titles = ['误报率', '命中率']
    cmaps = ['plasma', 'viridis']
    
    for i, (metric, title, cmap) in enumerate(zip(metrics_to_plot, titles, cmaps)):
        plt.subplot(3, 2, 3+i)
        
        # 创建每个产品每个月的平均值矩阵
        matrix = np.zeros((len(all_metrics), 12))
        
        for p_idx, (prod, data) in enumerate(all_metrics.items()):
            for m in range(1, 13):
                mask = data['months'] == m
                if np.any(mask):
                    matrix[p_idx, m-1] = np.mean(data[metric][mask])
        
        im = plt.imshow(matrix, aspect='auto', cmap=cmap)
        cbar = plt.colorbar(im, label=title)
        cbar.ax.tick_params(labelsize=10)
        
        # 添加数值标签
        for i in range(len(all_metrics)):
            for j in range(12):
                text = plt.text(j, i, f"{matrix[i, j]:.2f}",
                            ha="center", va="center", color="white" if matrix[i, j] > 0.5 else "black",
                            fontsize=8)
        
        plt.yticks(range(len(all_metrics)), product_list)
        plt.xticks(range(12), ['一月', '二月', '三月', '四月', '五月', '六月', 
                              '七月', '八月', '九月', '十月', '十一月', '十二月'])
        plt.title(f'各产品月度平均{title}', fontsize=14)
        plt.xlabel('月份', fontsize=12)
        plt.ylabel('产品', fontsize=12)
    
    # 3. 季节饼图 - 显示各产品在不同季节的性能
    seasons_unique = ['冬季', '春季', '夏季', '秋季']
    metrics = ['far', 'pod', 'fnr']
    metric_labels = ['误报率', '命中率', '漏报率']
    
    for s_idx, season in enumerate(seasons_unique):
        plt.subplot(3, 4, 9+s_idx)
        
        # 收集每个产品在当前季节的平均性能
        prod_means = {prod_name: [] for prod_name in product_list}
        
        for p_idx, (prod, data) in enumerate(all_metrics.items()):
            season_mask = data['seasons'] == season
            if np.any(season_mask):
                prod_means[product_list[p_idx]] = [
                    np.mean(data['far'][season_mask]),
                    np.mean(data['pod'][season_mask]),
                    np.mean(data['fnr'][season_mask])
                ]
        
        # 创建堆叠条形图
        prods = list(prod_means.keys())
        ind = np.arange(len(prods))
        width = 0.25
        
        bottom = np.zeros(len(prods))
        for m_idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [prod_means[prod][m_idx] if prod_means[prod] else 0 for prod in prods]
            plt.bar(ind, values, width, bottom=bottom, label=label,
                   alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # 添加数值标签
            for i, v in enumerate(values):
                plt.text(i, bottom[i] + v/2, f"{v:.2f}", 
                      ha='center', va='center', fontsize=8,
                      color='white' if v > 0.3 else 'black')
            
            bottom += values
        
        plt.title(f'{season}表现', fontsize=14)
        if s_idx == 0:
            plt.ylabel('平均比率', fontsize=12)
        plt.xticks(ind, product_list, rotation=45)
        if s_idx == 3:
            plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('所有产品总结概览.png', dpi=300, bbox_inches='tight')
    
    # 4. 创建径向图 - 显示所有产品的季节性变化
    plt.figure(figsize=(20, 20))
    
    # 将一年分成72个点(每5天一点)
    theta = np.linspace(0, 2*np.pi, 73)[:-1]
    r_grid = np.linspace(0, 1, 5)
    
    metrics = ['far', 'pod', 'fnr']
    metric_names = ['误报率', '命中率', '漏报率']
    cmaps = ['plasma_r', 'viridis', 'coolwarm']
    
    for m_idx, (metric, name, cmap) in enumerate(zip(metrics, metric_names, cmaps)):
        ax = plt.subplot(2, 2, m_idx+1, polar=True)
        
        for p_idx, (prod, data) in enumerate(all_metrics.items()):
            # 将数据重新采样为72点
            bins = np.linspace(0, 365, 73)
            digitized = np.digitize(data['days_of_year'], bins) - 1
            digitized[digitized >= 72] = 71  # 处理边界情况
            
            # 计算每个bin的平均值
            binned_data = np.zeros(72)
            for i in range(72):
                mask = digitized == i
                if np.any(mask):
                    binned_data[i] = np.mean(data[metric][mask])
            
            # 确保数据形成闭环
            values = np.append(binned_data, binned_data[0])
            theta_plot = np.append(theta, theta[0])
            
            plt.plot(theta_plot, values, '-', linewidth=2, 
                    label=f"{product_list[p_idx]}", color=colors[p_idx % len(colors)],
                    marker=markers[p_idx % len(markers)], markevery=8)
        
        # 添加月份标签
        ax.set_xticks(np.linspace(0, 2*np.pi, 12, endpoint=False))
        ax.set_xticklabels(['一月', '二月', '三月', '四月', '五月', '六月', 
                          '七月', '八月', '九月', '十月', '十一月', '十二月'])
        
        # 添加同心圆网格
        for r in r_grid:
            circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--', alpha=0.4)
            ax.add_artist(circle)
            if r > 0:
                plt.text(0, r, f'{r:.1f}', ha='center', va='bottom', color='gray')
        
        plt.title(f'各产品季节性{name}变化', fontsize=16)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    
    # 5. 热力图矩阵 - 产品间相关性
    ax = plt.subplot(2, 2, 4)
    corr_matrix = np.zeros((len(all_metrics), len(all_metrics)))
    
    products = product_list
    
    for i, prod1 in enumerate(products):
        for j, prod2 in enumerate(products):
            # 计算FAR的相关性
            corr = np.corrcoef(all_metrics[f'产品_{i+1}']['far'], all_metrics[f'产品_{j+1}']['far'])[0, 1]
            corr_matrix[i, j] = corr
    
    im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, label='相关性')
    plt.xticks(range(len(products)), product_list, rotation=45)
    plt.yticks(range(len(products)), product_list)
    plt.title('产品相关性矩阵（误报率）', fontsize=16)
    
    for i in range(len(products)):
        for j in range(len(products)):
            plt.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                    ha='center', va='center', 
                    color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white')
    
    plt.tight_layout()
    plt.savefig('产品相关性与径向分析.png', dpi=300, bbox_inches='tight')
    plt.show()
