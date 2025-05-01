"""
可视化辅助模块，用于创建专业的可视化图表和处理可视化问题
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, mean_squared_error, mean_absolute_error, r2_score, auc
from matplotlib.colors import LinearSegmentedColormap
import warnings

# 导入字体辅助模块
try:
    from font_helper import setup_chinese_matplotlib
    # 自动设置中文字体
    setup_chinese_matplotlib(test=False)
except ImportError:
    # 如果字体辅助模块不可用，使用默认设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

# 创建一个专业的颜色方案
COLORS = {
    'blue': '#1f77b4',
    'orange': '#ff7f0e',
    'green': '#2ca02c',
    'red': '#d62728',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'gray': '#7f7f7f',
    'olive': '#bcbd22',
    'cyan': '#17becf'
}

# 创建一个专业的颜色映射
COLORMAP = LinearSegmentedColormap.from_list('custom_cmap', 
                                            [COLORS['blue'], COLORS['orange'], COLORS['red']])

def set_style(style='whitegrid', context='paper', font_scale=1.2):
    """设置可视化样式"""
    sns.set_style(style)
    sns.set_context(context, font_scale=font_scale)
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['figure.dpi'] = 100

def ensure_dir(filepath):
    """确保目录存在"""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return filepath

def plot_confusion_matrix(y_true, y_pred, labels=None, title='混淆矩阵', 
                         filepath=None, normalize=False, figsize=(8, 6)):
    """
    绘制混淆矩阵
    
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        labels: 标签名称列表
        title: 图表标题
        filepath: 保存路径，如果为None则不保存
        normalize: 是否归一化
        figsize: 图表大小
    """
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 设置默认标签
    if labels is None:
        if np.max(y_true) == 1 and np.min(y_true) == 0:
            labels = ['无降雨', '有降雨']
        else:
            labels = [f'类别{i}' for i in range(cm.shape[0])]
    
    # 归一化
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # 绘制混淆矩阵
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # 使用更好的颜色方案
    cmap = sns.color_palette("Blues", as_cmap=True)
    
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, 
               xticklabels=labels, yticklabels=labels, ax=ax)
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('预测标签', fontsize=13)
    plt.ylabel('真实标签', fontsize=13)
    
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {filepath}")
    
    return cm

def plot_roc_curve(y_true, y_scores, title='ROC曲线', 
                  filepath=None, figsize=(8, 6)):
    """
    绘制ROC曲线
    
    参数:
        y_true: 真实标签
        y_scores: 预测概率
        title: 图表标题
        filepath: 保存路径，如果为None则不保存
        figsize: 图表大小
    """
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 绘制ROC曲线
    plt.figure(figsize=figsize)
    
    # 使用更好的颜色
    plt.plot(fpr, tpr, color=COLORS['blue'], lw=2, 
             label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color=COLORS['gray'], lw=1, linestyle='--',
             label='随机猜测')
    
    # 设置标题和标签
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)', fontsize=13)
    plt.ylabel('真正例率 (TPR)', fontsize=13)
    plt.title(title, fontsize=15)
    plt.legend(loc="lower right", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存至: {filepath}")
    
    return fpr, tpr, roc_auc

def plot_pr_curve(y_true, y_scores, title='PR曲线',
                 filepath=None, figsize=(8, 6)):
    """
    绘制精确率-召回率曲线
    
    参数:
        y_true: 真实标签
        y_scores: 预测概率
        title: 图表标题
        filepath: 保存路径，如果为None则不保存
        figsize: 图表大小
    """
    # 计算精确率-召回率曲线
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    
    # 计算无技能线（随机猜测）
    no_skill = np.sum(y_true) / len(y_true)
    
    # 绘制PR曲线
    plt.figure(figsize=figsize)
    
    # 使用更好的颜色
    plt.plot(recall, precision, color=COLORS['green'], lw=2,
             label=f'PR曲线 (AUC = {pr_auc:.4f})')
    plt.plot([0, 1], [no_skill, no_skill], color=COLORS['gray'], 
             linestyle='--', lw=1, label='随机猜测')
    
    # 设置标题和标签
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率', fontsize=13)
    plt.ylabel('精确率', fontsize=13)
    plt.title(title, fontsize=15)
    plt.legend(loc="best", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"PR曲线已保存至: {filepath}")
    
    return precision, recall, pr_auc

def plot_feature_importance(importance_dict, feature_names=None, title='特征重要性', 
                          filepath=None, figsize=(10, 8), top_n=None,
                          importance_type='gain', sort=True, error_bars=None):
    """
    绘制特征重要性条形图
    
    参数:
        importance_dict: 特征重要性字典或数组
        feature_names: 特征名称列表
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
        top_n: 只显示前N个特征
        importance_type: 重要性类型
        sort: 是否按重要性排序
        error_bars: 误差棒数据
    """
    # 准备数据
    if isinstance(importance_dict, dict):
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
    else:
        importances = importance_dict
        features = feature_names if feature_names is not None else [f'特征 {i+1}' for i in range(len(importances))]
    
    # 创建DataFrame
    data = pd.DataFrame({
        '特征': features,
        '重要性': importances
    })
    
    # 排序
    if sort:
        data = data.sort_values('重要性', ascending=False)
    
    # 选择前N个特征
    if top_n is not None:
        data = data.head(top_n)
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    ax = plt.gca()
    
    # 使用更好的颜色方案
    palette = sns.color_palette("viridis", len(data))
    
    if error_bars is not None:
        # 绘制带误差棒的条形图
        plt.barh(data['特征'], data['重要性'], xerr=error_bars, 
                 color=palette, alpha=0.7, error_kw={'ecolor': 'black'})
    else:
        # 绘制普通条形图
        sns.barplot(data=data, y='特征', x='重要性', palette=palette, ax=ax)
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel(f'重要性 ({importance_type})', fontsize=13)
    plt.ylabel('特征', fontsize=13)
    
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存至: {filepath}")
    
    return data

def plot_learning_curve(train_scores, val_scores=None, metric_name='错误率', 
                      title=None, filepath=None, figsize=(10, 6)):
    """
    绘制学习曲线
    
    参数:
        train_scores: 训练集上的评估分数
        val_scores: 验证集上的评估分数
        metric_name: 指标名称
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 绘制训练曲线
    plt.plot(train_scores, color=COLORS['blue'], label=f'训练集 {metric_name}')
    
    # 绘制验证曲线
    if val_scores is not None:
        plt.plot(val_scores, color=COLORS['orange'], label=f'验证集 {metric_name}')
    
    # 设置标题和标签
    title = title or f'学习曲线 ({metric_name})'
    plt.title(title, fontsize=15)
    plt.xlabel('迭代次数', fontsize=13)
    plt.ylabel(metric_name, fontsize=13)
    plt.legend(loc="best", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存至: {filepath}")
    
    return True

def plot_multiple_learning_curves(result_dict, title='学习曲线', 
                                filepath=None, figsize=(12, 6), 
                                cols=2, share_y=False):
    """
    绘制多个学习曲线
    
    参数:
        result_dict: 评估结果字典，格式为 {metric_name: {'train': [...], 'val': [...], ...}, ...}
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
        cols: 列数
        share_y: 是否共享y轴
    """
    metrics = list(result_dict.keys())
    rows = (len(metrics) + cols - 1) // cols  # 向上取整
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize, sharey=share_y)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            ax = axes[i]
            
            # 获取当前指标的训练和验证数据
            data = result_dict[metric]
            for key, values in data.items():
                color = COLORS['blue'] if key == 'train' else COLORS['orange']
                ax.plot(values, color=color, label=f'{key}')
            
            # 设置标题和标签
            ax.set_title(metric, fontsize=13)
            ax.set_xlabel('迭代次数', fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.legend(loc="best", fontsize=10)
            ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    
    # 设置总标题
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为suptitle留出空间
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"学习曲线已保存至: {filepath}")
    
    return fig

def plot_spatial_distribution(data_array, mask=None, title='空间分布',
                            filepath=None, figsize=(10, 8), 
                            cmap='viridis', vmin=None, vmax=None,
                            colorbar_label='值'):
    """
    绘制空间分布图
    
    参数:
        data_array: 二维数据数组
        mask: 掩膜数组，用于屏蔽部分区域
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
        cmap: 颜色映射
        vmin: 颜色范围最小值
        vmax: 颜色范围最大值
        colorbar_label: 颜色条标签
    """
    plt.figure(figsize=figsize)
    
    # 应用掩膜
    if mask is not None:
        masked_data = np.ma.masked_where(mask == 0, data_array)
    else:
        masked_data = data_array
    
    # 绘制热图
    im = plt.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label(colorbar_label, fontsize=12)
    
    # 设置标题
    plt.title(title, fontsize=15)
    plt.axis('off')  # 不显示坐标轴
    
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"空间分布图已保存至: {filepath}")
    
    return True

def plot_distributions(data, hue=None, title='数据分布', 
                      filepath=None, figsize=(12, 6)):
    """
    绘制数据分布(直方图和密度图)
    
    参数:
        data: DataFrame或类似数据结构
        hue: 用于分组的列
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    plt.figure(figsize=figsize)
    
    # 如果是DataFrame，遍历每一列
    if isinstance(data, pd.DataFrame):
        # 计算需要的行列数
        n_cols = min(3, len(data.columns))
        n_rows = (len(data.columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
        
        for i, col in enumerate(data.columns):
            if i < len(axes):
                ax = axes[i]
                sns.histplot(data[col], kde=True, ax=ax, hue=data[hue] if hue else None)
                ax.set_title(col)
                ax.set_xlabel('')
        
        # 隐藏多余的子图
        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])
            
    else:
        # 如果是简单的数组，直接绘制
        sns.histplot(data, kde=True, hue=hue)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 为suptitle留出空间
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"数据分布图已保存至: {filepath}")
    
    return True

def plot_prediction_scatter(y_true, y_pred, title='预测值与实际值对比',
                          filepath=None, figsize=(8, 8), alpha=0.5):
    """
    绘制预测值与实际值的散点图
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
        alpha: 点的透明度
    """
    plt.figure(figsize=figsize)
    
    # 计算误差指标
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # 绘制散点图
    plt.scatter(y_true, y_pred, alpha=alpha, color=COLORS['blue'])
    
    # 绘制理想线(y=x)
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    # 添加误差指标文本
    plt.annotate(f'MSE: {mse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('真实值', fontsize=13)
    plt.ylabel('预测值', fontsize=13)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"预测散点图已保存至: {filepath}")
    
    return mse, mae, r2

def plot_correlation_matrix(data, method='pearson', title='相关性矩阵',
                          filepath=None, figsize=(10, 8), 
                          cmap='coolwarm', annot=True):
    """
    绘制相关性矩阵
    
    参数:
        data: DataFrame格式的数据
        method: 相关系数方法，'pearson', 'kendall', 'spearman'
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
        cmap: 颜色映射
        annot: 是否在方格中显示数值
    """
    # 计算相关系数矩阵
    corr = data.corr(method=method)
    
    # 绘制热力图
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr, dtype=bool))  # 创建上三角掩码
    
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, 
               fmt='.2f', square=True, linewidths=0.5,
               vmax=1.0, vmin=-1.0, center=0,
               cbar_kws={"shrink": 0.8})
    
    # 设置标题
    plt.title(title, fontsize=15)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"相关性矩阵已保存至: {filepath}")
    
    return corr

def plot_radar_chart(values, categories, title='模型评估雷达图',
                    filepath=None, figsize=(8, 8), 
                    color=COLORS['blue'], alpha=0.25):
    """
    绘制雷达图，用于比较多个指标
    
    参数:
        values: 评估指标值数组
        categories: 指标名称数组
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
        color: 填充颜色
        alpha: 透明度
    """
    # 确保数据类型正确
    values = np.array(values)
    categories = np.array(categories)
    
    # 数据点数量
    N = len(categories)
    
    # 计算每个角度
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    
    # 闭合雷达图
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))
    categories = np.concatenate((categories, [categories[0]]))
    
    # 绘制雷达图
    plt.figure(figsize=figsize)
    ax = plt.subplot(111, polar=True)
    
    # 绘制线条和填充区域
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label='指标值')
    ax.fill(angles, values, color=color, alpha=alpha)
    
    # 设置刻度标签
    ax.set_thetagrids(np.degrees(angles), categories)
    
    # 添加标题
    plt.title(title, fontsize=15, y=1.1)
    
    # 设置y轴范围
    ax.set_ylim(0, np.max(values) * 1.1)
    
    # 添加图例
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"雷达图已保存至: {filepath}")
    
    return ax

def plot_residuals(y_true, y_pred, title='残差分析',
                  filepath=None, figsize=(12, 5)):
    """
    绘制残差分析图，包括残差散点图和残差分布图
    
    参数:
        y_true: 真实值
        y_pred: 预测值
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    # 计算残差
    residuals = y_true - y_pred
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 绘制残差散点图
    ax1.scatter(y_pred, residuals, alpha=0.6, color=COLORS['blue'])
    ax1.axhline(y=0, color=COLORS['red'], linestyle='--')
    ax1.set_title('残差散点图')
    ax1.set_xlabel('预测值')
    ax1.set_ylabel('残差')
    ax1.grid(True, alpha=0.3)
    
    # 绘制残差分布图
    sns.histplot(residuals, kde=True, ax=ax2, color=COLORS['green'])
    ax2.axvline(x=0, color=COLORS['red'], linestyle='--')
    ax2.set_title('残差分布')
    ax2.set_xlabel('残差')
    ax2.set_ylabel('频数')
    
    # 设置总标题
    plt.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"残差分析图已保存至: {filepath}")
    
    return fig

def plot_feature_interactions(X, y, feature1_idx, feature2_idx, feature_names=None,
                             title='特征交互图', filepath=None, figsize=(10, 8)):
    """
    绘制两个特征间的交互效应
    
    参数:
        X: 特征矩阵
        y: 目标变量
        feature1_idx: 第一个特征的索引
        feature2_idx: 第二个特征的索引
        feature_names: 特征名称列表
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    if feature_names is None:
        feature_names = [f'特征 {i+1}' for i in range(X.shape[1])]
    
    # 提取两个特征的值
    x1 = X[:, feature1_idx]
    x2 = X[:, feature2_idx]
    
    # 创建散点图
    plt.figure(figsize=figsize)
    
    # 根据y值的不同，使用不同颜色
    unique_y = np.unique(y)
    
    if len(unique_y) <= 10:  # 分类问题或有限类别
        # 使用离散颜色映射
        scatter = plt.scatter(x1, x2, c=y, cmap='viridis', 
                           alpha=0.7, edgecolor='w', linewidth=0.5)
        plt.colorbar(label='类别')
    else:  # 回归问题
        # 使用连续颜色映射
        scatter = plt.scatter(x1, x2, c=y, cmap='coolwarm', 
                           alpha=0.7, edgecolor='w', linewidth=0.5)
        plt.colorbar(label='目标值')
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel(feature_names[feature1_idx], fontsize=13)
    plt.ylabel(feature_names[feature2_idx], fontsize=13)
    
    # 添加网格线
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"特征交互图已保存至: {filepath}")
    
    return scatter

def plot_model_comparison(models_data, title='模型性能比较',
                        filepath=None, figsize=(12, 6)):
    """
    比较多个模型的性能指标
    
    参数:
        models_data: 字典，格式为 {model_name: {metric_name: value, ...}, ...}
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    # 获取所有度量指标
    all_metrics = set()
    for model_data in models_data.values():
        all_metrics.update(model_data.keys())
    
    # 将数据转换为DataFrame
    df = pd.DataFrame({model: {metric: model_data.get(metric, 0) for metric in all_metrics}
                     for model, model_data in models_data.items()}).T
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    df.plot(kind='bar', figsize=figsize, width=0.8, ax=plt.gca())
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('模型', fontsize=13)
    plt.ylabel('指标值', fontsize=13)
    plt.legend(title='指标', fontsize=12)
    
    # 添加网格线
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"模型比较图已保存至: {filepath}")
    
    return df

def plot_class_distribution(y, title='类别分布',
                          filepath=None, figsize=(10, 6)):
    """
    绘制类别分布图
    
    参数:
        y: 目标变量数组
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    # 计算类别分布
    unique_classes, counts = np.unique(y, return_counts=True)
    
    # 创建DataFrame
    df = pd.DataFrame({'类别': unique_classes, '数量': counts})
    
    # 绘制条形图
    plt.figure(figsize=figsize)
    ax = sns.barplot(x='类别', y='数量', data=df, palette='viridis')
    
    # 添加数值标签
    for i, count in enumerate(counts):
        ax.text(i, count + (max(counts) * 0.01), f'{count}', 
               ha='center', va='bottom', fontsize=12)
    
    # 设置标题和标签
    plt.title(title, fontsize=15)
    plt.xlabel('类别', fontsize=13)
    plt.ylabel('样本数量', fontsize=13)
    
    # 添加网格线
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"类别分布图已保存至: {filepath}")
    
    return df

def plot_missing_values(data, title='缺失值分析',
                      filepath=None, figsize=(12, 6)):
    """
    绘制数据集中的缺失值分析图
    
    参数:
        data: DataFrame格式的数据
        title: 图表标题
        filepath: 保存路径
        figsize: 图表大小
    """
    # 计算每列的缺失值数量和百分比
    missing_count = data.isnull().sum()
    missing_percent = 100 * missing_count / len(data)
    
    # 创建包含缺失值信息的DataFrame
    missing_data = pd.DataFrame({
        '缺失值数量': missing_count,
        '缺失值百分比': missing_percent
    })
    
    # 仅保留有缺失值的列
    missing_data = missing_data[missing_data['缺失值数量'] > 0]
    
    if missing_data.empty:
        print("数据中没有缺失值")
        return None
    
    # 按缺失值百分比排序
    missing_data = missing_data.sort_values('缺失值百分比', ascending=False)
    
    # 设置图表大小
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 绘制缺失值数量图
    missing_data['缺失值数量'].plot(kind='bar', ax=ax1, color=COLORS['blue'])
    ax1.set_title('缺失值数量')
    ax1.set_ylabel('缺失值数量')
    ax1.set_xlabel('特征')
    
    # 绘制缺失值百分比图
    missing_data['缺失值百分比'].plot(kind='bar', ax=ax2, color=COLORS['orange'])
    ax2.set_title('缺失值百分比')
    ax2.set_ylabel('缺失值百分比 (%)')
    ax2.set_xlabel('特征')
    
    # 设置总标题
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图片
    if filepath:
        ensure_dir(filepath)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"缺失值分析图已保存至: {filepath}")
    
    return missing_data

# 导出常用变量和函数
__all__ = [
    'set_style', 'ensure_dir', 'COLORS', 'COLORMAP',
    'plot_confusion_matrix', 'plot_roc_curve', 'plot_pr_curve',
    'plot_feature_importance', 'plot_learning_curve',
    'plot_multiple_learning_curves', 'plot_spatial_distribution',
    'plot_distributions', 'plot_prediction_scatter',
    'plot_correlation_matrix', 'plot_radar_chart', 'plot_residuals',
    'plot_feature_interactions', 'plot_model_comparison',
    'plot_class_distribution', 'plot_missing_values'
]