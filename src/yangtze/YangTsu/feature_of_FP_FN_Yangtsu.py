from loaddata import mydata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# 定义降雨判断阈值 (用于计算混淆矩阵)
RAIN_THRESHOLD_OBS = 0.1 # 观测值大于此值为实际降雨
RAIN_THRESHOLD_PRED = 0.1 # 子集内预测值大于此值为预测降雨

# 新函数：计算指定子集内的混淆矩阵和指标
def calculate_subset_metrics(x_data_flat, y_data_flat, threshold):
    """
    计算预测值 <= threshold 的子集内的混淆矩阵和相关指标 (POD, FAR)。
    """
    # 1. 筛选出预测值 <= threshold 的子集
    subset_indices = np.where(x_data_flat <= threshold)[0] # 修改此处 < 为 <=
    if len(subset_indices) == 0:
        # 如果该阈值下没有样本被选中，返回 NaN
        return {
            'subset_size': 0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0,
            'POD': np.nan, 'FAR': np.nan
        }

    x_subset = x_data_flat[subset_indices]
    y_subset = y_data_flat[subset_indices]
    subset_size = len(x_subset)

    # 2. 在子集内判断预测和实际情况 (使用定义的降雨阈值)
    predicted_rain = (x_subset > 0)
    actual_rain = (y_subset > 0)
    predicted_no_rain = ~predicted_rain
    actual_no_rain = ~actual_rain

    # 3. 计算子集内的混淆矩阵计数
    TP = np.sum(predicted_rain & actual_rain)
    TN = np.sum(predicted_no_rain & actual_no_rain)
    FP = np.sum(predicted_rain & actual_no_rain)
    FN = np.sum(predicted_no_rain & actual_rain)

    # 4. 计算子集内的指标
    pod = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    # 如果 TP+FP 为 0 (即子集内没有预测降雨)，则 FAR 定义为 0 或 NaN 均可，这里用 NaN
    far = FP / (TP + FP) if (TP + FP) > 0 else np.nan

    return {
        'subset_size': subset_size,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'POD': pod,
        'FAR': far
    }

# 修改阈值分析函数以包含粗略和精细扫描 (寻找最大 FAR)
def analyze_thresholds(x_flat, y_flat, min_threshold=0, max_threshold=20, coarse_steps=100, fine_steps=50, fine_range_factor=0.1):
    """
    通过粗略和精细扫描阈值，计算预测值 < threshold 子集内的指标，
    并找出使子集 FAR 最大化的阈值。
    """
    print(f"  Analyzing thresholds between {min_threshold} and {max_threshold} for max FAR...")

    # --- 粗略扫描 ---
    print(f"  Performing coarse scan ({coarse_steps} steps)...")
    coarse_thresholds = np.linspace(min_threshold, max_threshold, coarse_steps)
    coarse_results = []
    for threshold in coarse_thresholds:
        metrics = calculate_subset_metrics(x_flat, y_flat, threshold)
        coarse_results.append({'threshold': threshold, **metrics})

    coarse_df = pd.DataFrame(coarse_results).dropna(subset=['FAR']) # 移除 FAR 为 NaN 的行进行查找
    if coarse_df.empty:
        print("  Warning: No valid FAR calculated during coarse scan.")
        return {'coarse_df': pd.DataFrame(coarse_results), 'fine_df': None, 'final_max_far_threshold': np.nan}

    # 找出粗略扫描中 FAR 最高的点
    max_far_coarse_idx = coarse_df['FAR'].idxmax()
    max_far_coarse_threshold = coarse_df.loc[max_far_coarse_idx, 'threshold']
    max_far_coarse_value = coarse_df.loc[max_far_coarse_idx, 'FAR']
    print(f"  Coarse max FAR found: {max_far_coarse_value:.4f} at threshold {max_far_coarse_threshold:.4f}")

    # --- 精细扫描 ---
    # 在粗略最大 FAR 阈值附近进行精细扫描
    scan_range = (max_threshold - min_threshold) * fine_range_factor / 2
    fine_min_threshold = max(min_threshold, max_far_coarse_threshold - scan_range)
    fine_max_threshold = min(max_threshold, max_far_coarse_threshold + scan_range)
    print(f"  Performing fine scan ({fine_steps} steps) around {max_far_coarse_threshold:.4f} (range: [{fine_min_threshold:.4f}, {fine_max_threshold:.4f}])...")

    fine_thresholds = np.linspace(fine_min_threshold, fine_max_threshold, fine_steps)
    fine_results = []
    for threshold in fine_thresholds:
        metrics = calculate_subset_metrics(x_flat, y_flat, threshold)
        fine_results.append({'threshold': threshold, **metrics})

    fine_df = pd.DataFrame(fine_results).dropna(subset=['FAR']) # 同样移除 NaN
    final_max_far_threshold = np.nan
    final_max_far_value = np.nan

    if not fine_df.empty:
        final_max_far_idx = fine_df['FAR'].idxmax()
        final_max_far_threshold = fine_df.loc[final_max_far_idx, 'threshold']
        final_max_far_value = fine_df.loc[final_max_far_idx, 'FAR']
        print(f"  Fine scan max FAR found: {final_max_far_value:.4f} at threshold {final_max_far_threshold:.4f}")
    else:
        print("  Warning: No valid FAR calculated during fine scan. Using coarse result.")
        # 如果精细扫描没有有效值，则回退到粗略扫描结果
        final_max_far_threshold = max_far_coarse_threshold
        final_max_far_value = max_far_coarse_value
        fine_df = None # 表示没有有效的精细扫描结果

    result = {
        'coarse_df': pd.DataFrame(coarse_results), # 返回包含 NaN 的完整粗略扫描结果用于绘图
        'fine_df': pd.DataFrame(fine_results) if fine_df is not None else None, # 返回包含 NaN 的完整精细扫描结果
        'final_max_far_threshold': final_max_far_threshold,
        'final_max_far_value': final_max_far_value,
        'coarse_thresholds': coarse_thresholds,
        'fine_thresholds': fine_thresholds if fine_df is not None else None,
    }
    return result

# 修改绘图函数以显示粗略/精细扫描和最大 FAR 点
def plot_subset_metrics(analysis_result, product_name, save_dir):
    """
    绘制子集分析结果图 (混淆矩阵计数, POD, FAR)，突出显示最大 FAR 点，并保存。
    """
    if analysis_result is None:
        print(f"  Skipping plotting for {product_name} due to no analysis results.")
        return

    coarse_df = analysis_result['coarse_df']
    fine_df = analysis_result['fine_df']
    final_max_far_threshold = analysis_result['final_max_far_threshold']
    final_max_far_value = analysis_result['final_max_far_value']
    coarse_thresholds = analysis_result['coarse_thresholds']
    fine_thresholds = analysis_result['fine_thresholds']

    if coarse_df.empty:
        print(f"  Skipping plotting for {product_name} due to empty coarse results.")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True) # 共享X轴
    fig.suptitle(f'长江流域 - {product_name} - 低预测值子集性能分析 (预测 < 阈值)', fontsize=16)

    # --- 子图 1: 混淆矩阵计数 (使用粗略扫描结果) ---
    axes[0].plot(coarse_thresholds, coarse_df['TP'], label='TP (命中)', color='green', alpha=0.8)
    axes[0].plot(coarse_thresholds, coarse_df['FN'], label='FN (漏报)', color='red', alpha=0.8)
    axes[0].plot(coarse_thresholds, coarse_df['FP'], label='FP (误报)', color='orange', alpha=0.8)
    axes[0].plot(coarse_thresholds, coarse_df['TN'], label='TN (正确无雨)', color='blue', alpha=0.8)
    axes[0].set_title('混淆矩阵计数 vs. 阈值上限 (粗略扫描)', fontsize=14)
    axes[0].set_ylabel('计数 (对数刻度)', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend(loc='best')
    # 尝试使用对数刻度，如果数据包含0或负数，则回退到线性刻度
    try:
        axes[0].set_yscale('log')
        # 为对数刻度设置一个小的正数下限，避免0值问题
        min_positive = coarse_df[['TP', 'FN', 'FP', 'TN']][coarse_df[['TP', 'FN', 'FP', 'TN']] > 0].min().min()
        if pd.notna(min_positive) and min_positive > 0:
             axes[0].set_ylim(bottom=min_positive * 0.5)
        else:
             axes[0].set_ylim(bottom=1e-1) # 默认一个小的下限
    except ValueError:
        axes[0].set_yscale('linear')
        axes[0].set_ylabel('计数 (线性刻度)', fontsize=12)
        print(f"  Note: Could not use log scale for counts in {product_name}, using linear scale.")


    # --- 子图 2: POD 和 FAR ---
    # 绘制粗略扫描曲线
    axes[1].plot(coarse_thresholds, coarse_df['POD'], label='POD (粗)', color='green', alpha=0.6)
    axes[1].plot(coarse_thresholds, coarse_df['FAR'], label='FAR (粗)', color='orange', alpha=0.6)

    # 绘制精细扫描曲线 (如果存在)
    if fine_df is not None and fine_thresholds is not None:
        axes[1].plot(fine_thresholds, fine_df['POD'], 'g--', label='POD (精)', linewidth=1.5)
        axes[1].plot(fine_thresholds, fine_df['FAR'], 'm--', label='FAR (精)', linewidth=1.5) # 用不同颜色突出

    # 标记最终最大 FAR 点
    if pd.notna(final_max_far_threshold) and pd.notna(final_max_far_value):
        axes[1].axvline(x=final_max_far_threshold, color='purple', linestyle=':', label=f'最大 FAR 阈值: {final_max_far_threshold:.4f}')
        # 在图上标记最大 FAR 点
        axes[1].scatter(final_max_far_threshold, final_max_far_value, color='purple', s=60, zorder=5, label=f'最大 FAR: {final_max_far_value:.4f}')

    axes[1].set_title('POD 和 FAR vs. 阈值上限', fontsize=14)
    axes[1].set_xlabel('阈值上限 (Threshold)', fontsize=12)
    axes[1].set_ylabel('比率', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend(loc='best')
    axes[1].set_ylim(0, 1.05) # POD 和 FAR 范围在 0 到 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # 调整布局
    safe_product_name = product_name.replace(' ', '_').replace('/', '_')
    # 更新文件名
    save_filename = os.path.join(save_dir, f'Yangtsu_{safe_product_name}_subset_metrics_maxFAR.png')
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    print(f"  Subset metrics plot (max FAR focus) saved to {save_filename}")
    plt.close(fig) # 关闭图形，释放内存

    # --- 保存结果到 CSV ---
    # 1. 保存详细的粗略扫描结果
    coarse_csv_filename = os.path.join(save_dir, f'Yangtsu_{safe_product_name}_subset_metrics_coarse_details.csv')
    coarse_df.to_csv(coarse_csv_filename, index=False, encoding='utf-8-sig')
    print(f"  Coarse scan details saved to {coarse_csv_filename}")

    # 2. 保存详细的精细扫描结果 (如果存在)
    if fine_df is not None:
        fine_csv_filename = os.path.join(save_dir, f'Yangtsu_{safe_product_name}_subset_metrics_fine_details.csv')
        fine_df.to_csv(fine_csv_filename, index=False, encoding='utf-8-sig')
        print(f"  Fine scan details saved to {fine_csv_filename}")

    # 3. 保存包含最优点的摘要信息
    summary_data = []
    if pd.notna(final_max_far_threshold):
        # 从对应的 DataFrame (优先精细，否则粗略) 中查找该阈值行的所有指标
        optimal_row = None
        if fine_df is not None:
             # 在精细扫描结果中查找最接近的阈值
             optimal_idx = (fine_df['threshold'] - final_max_far_threshold).abs().idxmin()
             optimal_row = fine_df.loc[optimal_idx]
        elif not coarse_df.dropna(subset=['FAR']).empty:
             # 在粗略扫描结果中查找最接近的阈值
             optimal_idx = (coarse_df['threshold'] - final_max_far_threshold).abs().idxmin()
             optimal_row = coarse_df.loc[optimal_idx]

        if optimal_row is not None:
            summary_data.append({
                '分析目标': '最大子集FAR',
                '最优阈值上限': final_max_far_threshold,
                '最大FAR': optimal_row['FAR'],
                '对应POD': optimal_row['POD'],
                '对应TP': optimal_row['TP'],
                '对应FN': optimal_row['FN'],
                '对应FP': optimal_row['FP'],
                '对应TN': optimal_row['TN'],
                '子集大小': optimal_row['subset_size']
            })
        else: # Fallback if row finding fails
             summary_data.append({
                '分析目标': '最大子集FAR',
                '最优阈值上限': final_max_far_threshold,
                '最大FAR': final_max_far_value,
                '对应POD': np.nan, # 无法获取其他指标
                '对应TP': np.nan, '对应FN': np.nan, '对应FP': np.nan, '对应TN': np.nan, '子集大小': np.nan
            })


    if summary_data:
        results_summary = pd.DataFrame(summary_data)
        summary_csv_filename = os.path.join(save_dir, f'Yangtsu_{safe_product_name}_subset_metrics_summary.csv')
        results_summary.to_csv(summary_csv_filename, index=False, encoding='utf-8-sig')
        print(f"  Subset metrics summary saved to {summary_csv_filename}")


# --- 主程序 ---
if __name__ == "__main__":
    start_analysis_time = time.time()
    # 更新标题
    print("--- 长江流域 - 各产品 - 低预测值子集性能分析 (预测 < 阈值, 寻找最大 FAR) ---")
    print(f"Using Observation Rain Threshold: > {RAIN_THRESHOLD_OBS} mm")
    print(f"Using Prediction Rain Threshold (within subset): > {RAIN_THRESHOLD_PRED} mm")

    # --- 1. 加载数据 ---
    print("Loading Yangtze data...")
    try:
        data_loader = mydata()
        # X_raw shape: (n_products, time, n_points), Y_raw shape: (time, n_points)
        X_raw, Y_raw = data_loader.yangtsu()
        product_names = data_loader.features
        n_products, n_time, n_points = X_raw.shape
        print(f"Data loaded. X shape: {X_raw.shape}, Y shape: {Y_raw.shape}")
        print(f"Number of products to analyze: {n_products}") # 诊断信息
        if n_products == 0:
            print("Error: No products found in the loaded data.")
            exit()
    except Exception as e:
        print(f"Error loading data: {e}")
        exit()

    # --- 2. 设置分析参数 ---
    MIN_THRESHOLD = 0.1 # 阈值下限
    MAX_THRESHOLD = 10 # 阈值上限
    COARSE_STEPS = 100   # 粗略扫描步数
    FINE_STEPS = 50     # 精细扫描步数
    FINE_RANGE_FACTOR = 0.2 # 精细扫描范围占总范围的比例 (例如 0.2 表示在最大点左右各 10% 范围扫描)
    SAVE_DIR = "F:/rainfalldata/YangTsu/threshold_subset_metrics_maxFAR_results" # 新建文件夹保存结果
    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"Analysis parameters set. Save directory: {SAVE_DIR}") # 诊断信息

    # --- 3. 遍历每个产品进行分析 ---
    print("Starting product analysis loop...") # 诊断信息
    for i in range(n_products):
        product_name = product_names[i]
        print(f"\n>>> Processing product {i+1}/{n_products}: {product_name}...") # 诊断信息

        try:
            # 提取当前产品数据
            print("  Extracting product data...") # 诊断信息
            X_product = X_raw[i, :, :]

            # 展平数据以便分析
            print("  Flattening data...") # 诊断信息
            X_flat = X_product.reshape(-1)
            Y_flat = Y_raw.reshape(-1) # Y 每次都需要重新展平

            # 处理 NaN 值
            print("  Handling NaN values...") # 诊断信息
            valid_mask = ~np.isnan(X_flat) & ~np.isnan(Y_flat)
            X_flat_valid = X_flat[valid_mask]
            Y_flat_valid = Y_flat[valid_mask]
            num_valid_points = len(X_flat_valid) # 诊断信息
            print(f"  Number of valid (non-NaN) data points for analysis: {num_valid_points}") # 诊断信息

            if num_valid_points == 0:
                print(f"  Skipping {product_name}: No valid data points after removing NaNs.")
                continue

            # 执行阈值分析 (调用修改后的函数)
            print("  Calling analyze_thresholds...") # 诊断信息
            analysis_result = analyze_thresholds(
                X_flat_valid, Y_flat_valid,
                min_threshold=MIN_THRESHOLD,
                max_threshold=MAX_THRESHOLD,
                coarse_steps=COARSE_STEPS,
                fine_steps=FINE_STEPS,
                fine_range_factor=FINE_RANGE_FACTOR
            )
            print("  analyze_thresholds finished.") # 诊断信息

            # 绘制并保存结果 (调用修改后的函数)
            if analysis_result:
                 print("  Calling plot_subset_metrics...") # 诊断信息
                 plot_subset_metrics(analysis_result, product_name, SAVE_DIR)
                 print("  plot_subset_metrics finished.") # 诊断信息
            else:
                 print("  Skipping plotting because analysis_result is None or empty.") # 诊断信息

        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!") # 突出显示错误
            print(f"  ERROR processing product {product_name}: {e}")
            import traceback
            traceback.print_exc() # 打印详细的错误堆栈
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            # 选择继续处理下一个产品还是停止
            # continue # 继续下一个
            # break    # 停止循环

    end_analysis_time = time.time()
    print(f"\n--- Analysis loop finished ---") # 诊断信息
    print(f"\n--- Analysis complete in {end_analysis_time - start_analysis_time:.2f} seconds ---")
    print(f"Results saved in: {SAVE_DIR}")
