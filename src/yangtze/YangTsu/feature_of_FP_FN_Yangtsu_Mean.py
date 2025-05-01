from loaddata import mydata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# 定义函数计算指定子集内的错误率
def calculate_subset_error_rates(x_data_flat, y_data_flat, threshold):
    """
    计算预测值 <= threshold 的子集内的指标。
    - subset_miss_rate: 在预测值 <= threshold 的情况下，实际发生降雨的概率 (FN / (TN + FN)) - 即错误地预测为无雨的概率。
    - subset_correct_neg_rate: 在预测值 <= threshold 的情况下，实际未发生降雨的概率 (TN / (TN + FN)) - 即正确地预测为无雨的概率。
    """
    # 1. 筛选出预测值 <= threshold 的子集
    subset_indices = np.where(x_data_flat <= threshold)[0]
    if len(subset_indices) == 0:
        # 如果该阈值下没有样本被选中，返回 NaN
        return {'subset_size': 0, 'subset_miss_rate': np.nan, 'subset_correct_neg_rate': np.nan}

    x_subset = x_data_flat[subset_indices]
    y_subset = y_data_flat[subset_indices]
    subset_size = len(x_subset)

    # 2. 在子集内判断实际情况
    # 实际有雨 (Y > 0)
    actual_rain_in_subset = (y_subset > 0)
    # 实际无雨 (Y <= 0)
    actual_no_rain_in_subset = (y_subset <= 0)

    # 3. 计算子集内的计数
    # 在预测值 <= threshold 时，实际有雨的数量 (False Negatives relative to threshold decision)
    subset_FN_count = np.sum(actual_rain_in_subset)
    # 在预测值 <= threshold 时，实际无雨的数量 (True Negatives relative to threshold decision)
    subset_TN_count = np.sum(actual_no_rain_in_subset)

    # 4. 计算子集内的比率
    # 漏报率 (在预测值 <= threshold 的条件下，实际有雨的概率)
    subset_miss_rate = subset_FN_count / subset_size if subset_size > 0 else 0
    # 正确无雨率 (在预测值 <= threshold 的条件下，实际无雨的概率)
    subset_correct_neg_rate = subset_TN_count / subset_size if subset_size > 0 else 0

    return {
        'subset_size': subset_size,
        'subset_miss_rate': subset_miss_rate, # 预测值低但实际有雨的概率
        'subset_correct_neg_rate': subset_correct_neg_rate # 预测值低且实际无雨的概率
    }

# 修改阈值优化函数以使用新的指标
def optimize_threshold(x_flat, y_flat, min_threshold=0, max_threshold=20, coarse_steps=100, fine_steps=50):
    """
    扫描阈值，计算每个阈值下，预测值 <= threshold 子集内的漏报率和正确无雨率。
    找出子集漏报率最高的点。
    """
    print(f"  Optimizing threshold between {min_threshold} and {max_threshold} for subset analysis...")
    thresholds = np.linspace(min_threshold, max_threshold, coarse_steps)
    results = []
    for threshold in thresholds:
        # 使用新的计算函数
        rates = calculate_subset_error_rates(x_flat, y_flat, threshold)
        if rates: results.append({'threshold': threshold, **rates})

    if not results:
        print("  Warning: No valid rates calculated during coarse scan.")
        return None
    results_df = pd.DataFrame(results).dropna()
    if results_df.empty:
        print("  Warning: All rates were NaN during coarse scan.")
        return None

    # 找出子集漏报率最高的点
    max_subset_miss_idx = results_df['subset_miss_rate'].idxmax()

    # --- 精细扫描 ---
    # 在最高子集漏报率阈值附近进行精细扫描
    fine_thresholds_miss = np.linspace(max(min_threshold, results_df.loc[max_subset_miss_idx, 'threshold'] - 0.5),
                                       min(max_threshold, results_df.loc[max_subset_miss_idx, 'threshold'] + 0.5), fine_steps)
    fine_results_miss = []
    for threshold in fine_thresholds_miss:
        rates = calculate_subset_error_rates(x_flat, y_flat, threshold)
        if rates: fine_results_miss.append({'threshold': threshold, **rates})
    fine_df_miss = pd.DataFrame(fine_results_miss).dropna()
    final_max_miss_idx = fine_df_miss['subset_miss_rate'].idxmax() if not fine_df_miss.empty else None
    final_max_miss_threshold = fine_df_miss.loc[final_max_miss_idx, 'threshold'] if final_max_miss_idx is not None else np.nan

    # 打印最终结果
    if final_max_miss_idx is not None:
        print(f"  子集漏报率最高点阈值: {final_max_miss_threshold:.4f}, 此时子集漏报率: {fine_df_miss.loc[final_max_miss_idx, 'subset_miss_rate']:.4f}")
        print(f"  (在该阈值下，当预测值 <= {final_max_miss_threshold:.4f} 时，实际有雨的概率为 {fine_df_miss.loc[final_max_miss_idx, 'subset_miss_rate']:.4f})")

    result = {
        'results_df': results_df, # 粗扫描结果
        'fine_df_miss': fine_df_miss, # 精细扫描结果 (围绕最高漏报点)
        'final_max_miss_threshold': final_max_miss_threshold,
        'thresholds': thresholds,
        'fine_thresholds_miss': fine_thresholds_miss,
        'final_max_miss_idx': final_max_miss_idx,
    }
    return result

# 修改绘图函数以显示新的指标
def plot_threshold_results(result, feature_name, save_dir):
    """
    绘制阈值优化结果图 (子集分析)，并保存。
    """
    if result is None:
        print(f"  Skipping plotting for {feature_name} due to no valid results.")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 解包结果字典
    results_df = result['results_df']
    fine_df_miss = result['fine_df_miss']
    final_max_miss_threshold = result['final_max_miss_threshold']
    thresholds = result['thresholds']
    fine_thresholds_miss = result['fine_thresholds_miss']
    final_max_miss_idx = result['final_max_miss_idx']

    fig, axes = plt.subplots(2, 1, figsize=(12, 12)) # 调整为2个子图
    fig.suptitle(f'长江流域 - {feature_name} - 低预测值子集分析 (预测 <= 阈值)', fontsize=16)

    # 绘制子集漏报率曲线
    if not fine_df_miss.empty and final_max_miss_idx is not None:
        axes[0].plot(thresholds, results_df['subset_miss_rate'], 'b-', label='子集漏报率 (粗扫描)', linewidth=1.5, alpha=0.7)
        axes[0].plot(fine_thresholds_miss, fine_df_miss['subset_miss_rate'], 'r--', label='子集漏报率 (精细扫描)', linewidth=1.5)
        axes[0].axvline(x=final_max_miss_threshold, color='g', linestyle=':', label=f'最高点阈值: {final_max_miss_threshold:.4f}')
        axes[0].scatter(final_max_miss_threshold, fine_df_miss.loc[final_max_miss_idx, 'subset_miss_rate'], color='red', s=50, zorder=5, label='最高子集漏报率点')
    else:
        axes[0].plot(thresholds, results_df['subset_miss_rate'], 'b-', label='子集漏报率 (粗扫描)', linewidth=1.5, alpha=0.7)
    axes[0].set_title('子集漏报率 vs. 阈值上限', fontsize=14)
    axes[0].set_xlabel('阈值上限 (Threshold)', fontsize=12)
    axes[0].set_ylabel('子集漏报率\nP(实际有雨 | 预测 <= 阈值)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best')
    axes[0].set_ylim(bottom=0)

    # 绘制子集正确无雨率曲线
    # 注意：精细扫描是围绕漏报率最高点进行的，可能不是正确无雨率的最佳区域，但仍可绘制
    axes[1].plot(thresholds, results_df['subset_correct_neg_rate'], 'b-', label='子集正确无雨率 (粗扫描)', linewidth=1.5, alpha=0.7)
    if not fine_df_miss.empty: # 使用与漏报率相同的精细扫描范围
        axes[1].plot(fine_thresholds_miss, fine_df_miss['subset_correct_neg_rate'], 'r--', label='子集正确无雨率 (精细扫描)', linewidth=1.5)
    axes[1].set_title('子集正确无雨率 vs. 阈值上限', fontsize=14)
    axes[1].set_xlabel('阈值上限 (Threshold)', fontsize=12)
    axes[1].set_ylabel('子集正确无雨率\nP(实际无雨 | 预测 <= 阈值)', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='best')
    axes[1].set_ylim(bottom=0)


    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局
    safe_feature_name = feature_name.replace(' ', '_').replace('/', '_')
    # 更新文件名以反映新的分析类型
    save_filename = os.path.join(save_dir, f'Yangtsu_{safe_feature_name}_subset_analysis.png')
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"  Subset analysis plot saved to {save_filename}")
    plt.close(fig)

    # 将结果保存到CSV
    summary_data = []
    if final_max_miss_idx is not None:
        summary_data.append({
            '分析点': '子集漏报率最高点',
            '阈值上限': final_max_miss_threshold,
            '子集漏报率': fine_df_miss.loc[final_max_miss_idx, 'subset_miss_rate'],
            '子集正确无雨率': fine_df_miss.loc[final_max_miss_idx, 'subset_correct_neg_rate'],
            '子集大小': fine_df_miss.loc[final_max_miss_idx, 'subset_size']
        })

    if summary_data:
        results_summary = pd.DataFrame(summary_data)
        # 更新CSV文件名
        csv_filename = os.path.join(save_dir, f'Yangtsu_{safe_feature_name}_subset_summary.csv')
        results_summary.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        print(f"  Subset analysis summary saved to {csv_filename}")

# --- 主程序 ---
if __name__ == "__main__":
    start_analysis_time = time.time()
    # 更新标题
    print("--- 长江流域 - 产品平均值 - 低预测值子集分析 ---")

    # --- 1. 加载数据 ---
    print("Loading Yangtze data...")
    try:
        data_loader = mydata()
        # X_raw shape: (n_products, time, n_points), Y_raw shape: (time, n_points)
        X_raw, Y_raw = data_loader.yangtsu()
        print(f"Data loaded. X shape: {X_raw.shape}, Y shape: {Y_raw.shape}")
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- 2. 计算产品平均值 ---
    print("Calculating mean across products...")
    # X_raw shape: (n_products, time, n_points) -> axis=0
    X_mean = np.nanmean(X_raw, axis=0) # Shape: (time, n_points)
    print(f"Product mean calculated. Shape: {X_mean.shape}")
    del X_raw # 释放原始 X 数据内存

    # --- 3. 设置分析参数 ---
    FEATURE_NAME_TO_ANALYZE = "Product Mean" # 定义要分析的特征名称
    MIN_THRESHOLD = 0
    MAX_THRESHOLD = 10 # 假设分析降雨强度阈值，最大到 10 mm/day
    COARSE_STEPS = 100
    FINE_STEPS = 50
    SAVE_DIR = "F:/rainfalldata/YangTsu/threshold_analysis_results"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- 4. 准备数据进行分析 ---
    print(f"\nAnalyzing feature: {FEATURE_NAME_TO_ANALYZE}...")

    # 展平数据
    X_flat = X_mean.reshape(-1)
    Y_flat = Y_raw.reshape(-1)
    del X_mean, Y_raw # 释放内存

    # 处理 NaN 值
    valid_mask = ~np.isnan(X_flat) & ~np.isnan(Y_flat)
    X_flat_valid = X_flat[valid_mask]
    Y_flat_valid = Y_flat[valid_mask]
    del X_flat, Y_flat, valid_mask # 释放内存

    if len(X_flat_valid) == 0:
        print(f"  Skipping {FEATURE_NAME_TO_ANALYZE}: No valid data points after removing NaNs.")
        exit()

    print(f"  Number of valid data points for analysis: {len(X_flat_valid)}")

    # --- 计算并打印 Y > 0 的基础概率 ---
    base_rain_rate_y_gt_0 = np.sum(Y_flat_valid > 0) / len(Y_flat_valid)
    print(f"  Base rate of actual rain (Y > 0) in valid data: {base_rain_rate_y_gt_0:.4f}")
    # --- (可选) 计算 Y > 0.1 的基础概率 ---
    # base_rain_rate_y_gt_01 = np.sum(Y_flat_valid > 0.1) / len(Y_flat_valid)
    # print(f"  Base rate of actual rain (Y > 0.1) in valid data: {base_rain_rate_y_gt_01:.4f}")
    # --- (可选) 计算 Y > 1.0 的基础概率 ---
    # base_rain_rate_y_gt_1 = np.sum(Y_flat_valid > 1.0) / len(Y_flat_valid)
    # print(f"  Base rate of actual rain (Y > 1.0) in valid data: {base_rain_rate_y_gt_1:.4f}")
    # ---------------------------------------

    # --- 5. 执行阈值优化 ---
    optimization_result = optimize_threshold(
        X_flat_valid, Y_flat_valid,
        min_threshold=MIN_THRESHOLD,
        max_threshold=MAX_THRESHOLD,
        coarse_steps=COARSE_STEPS,
        fine_steps=FINE_STEPS
    )

    # --- 6. 绘制并保存结果 ---
    plot_threshold_results(optimization_result, FEATURE_NAME_TO_ANALYZE, SAVE_DIR)

    end_analysis_time = time.time()
    print(f"\n--- Analysis complete in {end_analysis_time - start_analysis_time:.2f} seconds ---")
    print(f"Results saved in: {SAVE_DIR}")
