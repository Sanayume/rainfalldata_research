import re
import pandas as pd
import ast # For safely evaluating the parameters dictionary string
from history import log_data

parsed_data = []
# Regex to capture trial number, value, parameters, and best trial info
# Corrected regex to handle missing "Best is trial..." line if it's the first trial
regex = re.compile(
    r"Trial\s+(?P<trial_num>\d+)\s+finished\s+with\s+value:\s+(?P<value>[\d.]+)\s+and\s+parameters:\s+(?P<params>\{.*?\})\.\s*(?:Best\s+is\s+trial\s+\d+\s+with\s+value:\s+(?P<best_value_so_far>[\d.]+)\.)?"
)

for line in log_data.strip().split('\n'):
    if "Trial" in line and "finished" in line:
        match = regex.search(line)
        if match:
            data_dict = match.groupdict()
            try:
                params_dict = ast.literal_eval(data_dict['params'])
                parsed_data.append({
                    'trial': int(data_dict['trial_num']),
                    'value': float(data_dict['value']),
                    'best_value_so_far': float(data_dict['best_value_so_far']) if data_dict.get('best_value_so_far') else float(data_dict['value']), # Handle first trial
                    **params_dict # Unpack parameters into the main dictionary
                })
            except Exception as e:
                print(f"Skipping line due to parsing error: {line}\nError: {e}")
        else:
            print(f"Regex did not match line: {line}")


df_trials = pd.DataFrame(parsed_data)
df_trials = df_trials.sort_values(by='trial').reset_index(drop=True)

# For the first trial, 'best_value_so_far' might be missing in the log or equal to its own value
# We can fill forward the 'best_value_so_far' or recalculate it
current_best = float('-inf')
best_values_progressive = []
for val in df_trials['value']:
    if val > current_best:
        current_best = val
    best_values_progressive.append(current_best)
df_trials['best_value_progressive'] = best_values_progressive


print(f"Parsed {len(df_trials)} trials into DataFrame.")
if df_trials.empty:
    print("No trials parsed. Check the log format and regex.")
    hyperparameters = [] # Define as empty list to avoid NameError later
else:
    print(df_trials.head())
    # Identify hyperparameters (assuming they are all columns except 'trial', 'value', 'best_value_so_far', 'best_value_progressive')
    hyperparameters = [col for col in df_trials.columns if col not in ['trial', 'value', 'best_value_so_far', 'best_value_progressive']]
    print("\nIdentified Hyperparameters:", hyperparameters)

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import plotly.graph_objects as go # For interactive 3D plots if desired
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.templates.default = "plotly_white" # Clean plotly theme
    # --- Google风格的全局Matplotlib设置 ---
    sns.set_theme(style="whitegrid") # 基础样式
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans'] # 专业字体
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'  # 浅灰色背景，提高可读性
    plt.rcParams['axes.edgecolor'] = '#424242'  # 深灰色坐标轴线
    plt.rcParams['axes.labelcolor'] = '#424242' # 深灰色标签
    plt.rcParams['xtick.color'] = '#424242'     # 深灰色刻度颜色
    plt.rcParams['ytick.color'] = '#424242'
    plt.rcParams['text.color'] = '#424242'
    plt.rcParams['axes.titlesize'] = 18         # 更大的标题字体大小
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['figure.titlesize'] = 20
    plt.rcParams['figure.dpi'] = 100 # 显示的DPI，保存时使用更高的DPI

    # 定义Google风格的配色方案
    colors_google = {
        'blue': '#4285F4',    # Google蓝
        'red': '#EA4335',     # Google红
        'yellow': '#FBBC05',  # Google黄
        'green': '#34A853',   # Google绿
        'grey': '#616161',    # Google深灰
        'light_grey': '#E0E0E0'  # 网格线的浅灰色
    }

    if not df_trials.empty:
        # --- 1. 优化历史图（最重要的可视化） ---
        # 展示目标值如何随着试验而改进
        plt.figure(figsize=(12, 7))
        
        # 设置浅灰色背景以提高可读性
        plt.gca().set_facecolor('#f8f9fa')
        
        # 绘制每个试验的目标值（灰色）
        plt.plot(df_trials['trial'], df_trials['value'], marker='o', linestyle='-', 
                 color=colors_google['grey'], alpha=0.6, label='Trial Values', zorder=1)
        
        # 绘制到目前为止的最佳值（Google红色）
        plt.plot(df_trials['trial'], df_trials['best_value_progressive'], marker='.', linestyle='-', 
                 color=colors_google['red'], linewidth=2, label='Best Value So Far', zorder=2)

        # 突出显示总体最佳试验点
        best_trial_overall_idx = df_trials['value'].idxmax()
        best_trial_overall = df_trials.loc[best_trial_overall_idx]
        plt.scatter(best_trial_overall['trial'], best_trial_overall['value'],
                    color=colors_google['yellow'], s=200, edgecolor='black', zorder=3, 
                    label=f"Best Overall (Trial {best_trial_overall['trial']:.0f})")
        
        # 为最佳点添加值注释
        plt.annotate(f"{best_trial_overall['value']:.4f}",
                    xy=(best_trial_overall['trial'], best_trial_overall['value']),
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=30, fontweight='bold', color=colors_google['red'])

        # 坐标轴标签和标题
        plt.xlabel("Trial Number", fontsize=30, fontweight='bold', labelpad=10, color='#424242')
        plt.ylabel("AUC (Accuracy)", fontsize=30, fontweight='bold', labelpad=10, color='#424242')
        
        # 添加Google风格的标题框（蓝色圆角矩形）
        plt.title("Hyperparameter Optimization History", fontweight='bold', color=colors_google['blue'], fontsize=30, pad=20, horizontalalignment='center') #解释各个参数 : 

        # 自定义图例样式
        plt.legend(frameon=True, loc='lower right', facecolor='white', framealpha=0.95, 
                   edgecolor='#424242', borderpad=1.0, handletextpad=1.0, fontsize=20)
        
        # 网格样式
        plt.grid(True, linestyle=':', alpha=0.6, color=colors_google['light_grey'])
        
        # 移除顶部和右侧边框
        sns.despine()
        
        # 使剩余的底部和左侧边框加粗
        plt.gca().spines['bottom'].set_linewidth(2)
        plt.gca().spines['left'].set_linewidth(2)
        
        # 获取坐标轴范围以添加箭头
        xlims = plt.gca().get_xlim()
        ylims = plt.gca().get_ylim()
        
        # 添加坐标轴箭头
        arrow_props = dict(facecolor=colors_google['blue'], edgecolor=colors_google['blue'],
                           width=1.5, headwidth=10, headlength=12,
                           shrinkA=0, shrinkB=0)
        
        # X轴箭头
        plt.annotate('',
                    xy=(xlims[1], ylims[0]),  # 箭头尖端
                    xytext=(-arrow_props['headlength'], 0),  # 相对于尖端的箭头尾部偏移
                    textcoords='offset points',
                    arrowprops=arrow_props,
                    xycoords='data',
                    clip_on=False)
        
        # Y轴箭头
        plt.annotate('',
                    xy=(xlims[0], ylims[1]),  # 箭头尖端
                    xytext=(0, -arrow_props['headlength']),  # 相对于尖端的箭头尾部偏移
                    textcoords='offset points',
                    arrowprops=arrow_props,
                    xycoords='data',
                    clip_on=False)

        plt.tight_layout()
        plt.savefig("optuna_optimization_history.png", dpi=300)
        plt.show()
        # plt.savefig("optuna_optimization_history.png", dpi=300) # 保存第一个图
        # plt.show() # 显示第一个图

        plt.show()
        # plt.savefig("optuna_optimization_history.png", dpi=300) # 保存第一个图
        # plt.show() # 显示第一个图

        plt.savefig("optuna_optimization_history.png", dpi=300)
        plt.show()
        # plt.savefig("optuna_optimization_history.png", dpi=300) # 保存第一个图
        # plt.show() # 显示第一个图

        # --- 2. 超参数平行坐标图 (Matplotlib) ---
        if not hyperparameters:
            print("没有识别出超参数，无法创建平行坐标图。")
        else:
            try:
                import matplotlib.colors
                import matplotlib.cm
                from matplotlib.lines import Line2D # 用于自定义图例
            except ImportError:
                print("请确保已安装 matplotlib: pip install matplotlib")


            # --- 字体大小定义 ---
            title_fontsize = 40.0
            axis_label_fontsize = 30.0  # Y轴标题 "Normalized Value"
            xtick_label_fontsize = 26.0 # X轴参数名
            ytick_label_fontsize = 20.0 # Y轴刻度数字 (0.0, 0.2, ...)
            param_axis_tick_fontsize = 20.0 # 新增：在每条参数轴上标注原始值的字体大小
            cbar_label_fontsize = 25.0
            cbar_tick_fontsize = 20.0
            legend_fontsize = 24

            # --- 线条样式 ---
            default_linewidth = 0.8
            best_linewidth = 2.6
            default_alpha = 0.45
            best_alpha = 0.95
            best_line_color = colors_google['red'] if 'colors_google' in locals() and isinstance(colors_google, dict) and 'red' in colors_google else '#E53935' # 更鲜明的红色

            # --- 辅助函数：格式化数值用于坐标轴标注 ---
            def format_axis_value_display(value, num_decimals=2):
                if pd.isna(value):
                    return "N/A"
                if abs(value) < 1e-9: # Very small numbers, treat as 0 for display
                    return "0"
                # 尝试整数格式化
                if abs(value - round(value)) < 1e-5 and abs(value) < 1e4 : # "整数型" 数值
                     return f"{value:.0f}"
                # 对于非常大或非常小的数，考虑科学计数法或调整小数位数
                if abs(value) >= 1e4 or (abs(value) < 1e-2 and value !=0) :
                     return f"{value:.{num_decimals}e}" # 使用科学计数法
                return f"{value:.{num_decimals}f}"


            if df_trials.empty or 'value' not in df_trials.columns or df_trials['value'].isnull().all():
                print("警告: 'value' 列为空或全为NaN，无法确定最佳试验。")
                best_trial_idx = -1
            else:
                if pd.api.types.is_numeric_dtype(df_trials['value']):
                    best_trial_idx = df_trials['value'].idxmax() # 假设最大化目标
                else:
                    print("警告: 'value' 列不是数值类型，无法确定最佳试验。")
                    best_trial_idx = -1

            data_to_plot_mpl = pd.DataFrame()
            dimension_info_mpl = [] # 将首先填充超参数，然后是AUC
            original_objective_values_mpl = df_trials['value'].copy()

            # --- 1. 准备AUC（目标值）维度信息，但先不添加到 dimension_info_mpl ---
            auc_dimension_data_for_df = {}
            auc_dimension_info_entry = None
            
            label_obj_plot_auc = 'AUC' # 确保名称为AUC
            values_obj_mpl_auc = original_objective_values_mpl.copy()
            obj_min_raw_auc, obj_max_raw_auc = (values_obj_mpl_auc.min(), values_obj_mpl_auc.max()) if pd.api.types.is_numeric_dtype(values_obj_mpl_auc) and not values_obj_mpl_auc.empty else (np.nan, np.nan)

            if not values_obj_mpl_auc.empty and pd.api.types.is_numeric_dtype(values_obj_mpl_auc):
                scaled_min_for_norm_auc, scaled_max_for_norm_auc = np.nan, np.nan
                if pd.isna(obj_min_raw_auc) or pd.isna(obj_max_raw_auc):
                    auc_dimension_data_for_df[label_obj_plot_auc] = np.full_like(values_obj_mpl_auc, 0.5, dtype=float)
                    scaled_min_for_norm_auc, scaled_max_for_norm_auc = (0.0, 0.0) if pd.isna(obj_min_raw_auc) else (obj_min_raw_auc, obj_min_raw_auc)
                elif obj_max_raw_auc > obj_min_raw_auc:
                    auc_dimension_data_for_df[label_obj_plot_auc] = (values_obj_mpl_auc - obj_min_raw_auc) / (obj_max_raw_auc - obj_min_raw_auc)
                    scaled_min_for_norm_auc, scaled_max_for_norm_auc = obj_min_raw_auc, obj_max_raw_auc
                else:
                    auc_dimension_data_for_df[label_obj_plot_auc] = np.full_like(values_obj_mpl_auc, 0.5, dtype=float)
                    scaled_min_for_norm_auc, scaled_max_for_norm_auc = obj_min_raw_auc, obj_max_raw_auc
                
                auc_dimension_info_entry = {
                    'label_for_plot': label_obj_plot_auc, 'original_min': obj_min_raw_auc, 'original_max': obj_max_raw_auc,
                    'scaled_min_for_norm': scaled_min_for_norm_auc, 'scaled_max_for_norm': scaled_max_for_norm_auc,
                    'is_log_transformed': False, 'raw_param_name': 'AUC'
                }
            elif not df_trials.empty: 
                auc_dimension_data_for_df[label_obj_plot_auc] = pd.Series(np.full(len(df_trials), 0.5), dtype=float)
                auc_dimension_info_entry = {
                    'label_for_plot': label_obj_plot_auc, 'original_min': np.nan, 'original_max': np.nan,
                    'scaled_min_for_norm': np.nan, 'scaled_max_for_norm': np.nan,
                    'is_log_transformed': False, 'raw_param_name': 'AUC'
                }
            
            # 将AUC的归一化数据添加到主DataFrame中，但其dimension_info_entry稍后添加
            if label_obj_plot_auc in auc_dimension_data_for_df:
                 data_to_plot_mpl[label_obj_plot_auc] = auc_dimension_data_for_df[label_obj_plot_auc]


            # --- 2. 超参数维度处理 (这些会先于AUC添加到dimension_info_mpl) ---
            log_params = ['learning_rate', 'gamma', 'lambda', 'alpha']
            for param in hyperparameters:
                if param == 'AUC': continue # 如果AUC意外出现在hyperparameters列表中，跳过

                if param not in df_trials.columns:
                    print(f"警告: 超参数 '{param}' 在 trials 数据中未找到。")
                    continue

                current_values_for_scaling = df_trials[param].copy()
                current_values_for_scaling = pd.to_numeric(current_values_for_scaling, errors='coerce')
                original_param_values_for_range = current_values_for_scaling.copy() # Keep original scale for min/max display

                param_orig_min = original_param_values_for_range.min()
                param_orig_max = original_param_values_for_range.max()
                
                is_log = param in log_params
                label_for_plot_col = f"log10({param})" if is_log else param

                if current_values_for_scaling.isnull().all():
                    print(f"警告: 超参数 '{param}' 全部为NaN。")
                    data_to_plot_mpl[label_for_plot_col] = np.full(len(df_trials), 0.5, dtype=float)
                    dim_info_entry = {
                        'label_for_plot': label_for_plot_col, 'original_min': np.nan, 'original_max': np.nan,
                        'scaled_min_for_norm': np.nan, 'scaled_max_for_norm': np.nan,
                        'is_log_transformed': is_log, 'raw_param_name': param }
                else:
                    if is_log:
                        nan_mask_before_log = current_values_for_scaling.isnull()
                        positive_values = np.maximum(current_values_for_scaling.fillna(0), 1e-9) # Avoid log(0) or log(-)
                        current_values_for_scaling = np.log10(positive_values)
                        current_values_for_scaling[nan_mask_before_log] = np.nan # Restore NaNs

                    param_min_for_norm = np.nanmin(current_values_for_scaling)
                    param_max_for_norm = np.nanmax(current_values_for_scaling)

                    scaled_col_data = np.full_like(current_values_for_scaling, 0.5, dtype=float)
                    if pd.isna(param_min_for_norm) or pd.isna(param_max_for_norm): # All NaNs after potential log
                        pass # Already 0.5
                    elif param_max_for_norm > param_min_for_norm:
                        scaled_col_data = (current_values_for_scaling - param_min_for_norm) / (param_max_for_norm - param_min_for_norm)
                        scaled_col_data[np.isnan(scaled_col_data)] = 0.5 # Handle original NaNs within data
                    else: # All values are the same (or single non-NaN value)
                        pass # Already 0.5

                    data_to_plot_mpl[label_for_plot_col] = scaled_col_data
                    dim_info_entry = {
                        'label_for_plot': label_for_plot_col, 'original_min': param_orig_min, 'original_max': param_orig_max,
                        'scaled_min_for_norm': param_min_for_norm, 'scaled_max_for_norm': param_max_for_norm,
                        'is_log_transformed': is_log, 'raw_param_name': param }
                dimension_info_mpl.append(dim_info_entry)
            
            # --- 3. 将之前准备的AUC维度信息添加到 dimension_info_mpl 的末尾 ---
            if auc_dimension_info_entry:
                dimension_info_mpl.append(auc_dimension_info_entry)

            plot_column_names = [info['label_for_plot'] for info in dimension_info_mpl]
            if not all(col_name in data_to_plot_mpl.columns for col_name in plot_column_names):
                 valid_plot_cols = [cn for cn in plot_column_names if cn in data_to_plot_mpl.columns]
                 dimension_info_mpl = [info for info in dimension_info_mpl if info['label_for_plot'] in valid_plot_cols]
                 plot_column_names = valid_plot_cols


            if len(dimension_info_mpl) > 1 and not data_to_plot_mpl.empty:
                hyperparam_cols_data = data_to_plot_mpl[[info['label_for_plot'] for info in dimension_info_mpl if info['raw_param_name'] != 'AUC' and info['label_for_plot'] in data_to_plot_mpl.columns]]
                if hyperparam_cols_data.isnull().all().all() and len(hyperparam_cols_data.columns) > 0 :
                     print("所有超参数维度的数据都为空，无法创建有意义的平行坐标图。")
                else:
                    fig_par_coords_mpl, ax_mpl = plt.subplots(figsize=(max(12, len(dimension_info_mpl) * 2.0 + 2), 10.5)) # Adjusted width factor and height
                    
                    cmap_mpl = plt.get_cmap('cividis') # Changed colormap
                    
                    valid_objectives = original_objective_values_mpl.dropna() if pd.api.types.is_numeric_dtype(original_objective_values_mpl) else pd.Series(dtype=float)
                    norm_mpl, sm_mpl = None, None
                    if not valid_objectives.empty and valid_objectives.min() < valid_objectives.max(): # Ensure range for norm
                        norm_mpl = matplotlib.colors.Normalize(vmin=valid_objectives.min(), vmax=valid_objectives.max())
                        sm_mpl = matplotlib.cm.ScalarMappable(cmap=cmap_mpl, norm=norm_mpl)
                        sm_mpl.set_array([])
                    elif not valid_objectives.empty: # All objectives are the same
                         print("警告: 所有有效目标值相同，将使用单一颜色（最优试验除外）。")


                    for i in range(len(df_trials)):
                        if i < len(data_to_plot_mpl):
                            y_values_mpl_series = data_to_plot_mpl.iloc[i][plot_column_names]
                            if y_values_mpl_series.isnull().all(): continue

                            obj_val_for_color = original_objective_values_mpl.iloc[i] if i < len(original_objective_values_mpl) else np.nan
                            current_color, current_lw, current_alpha, current_zorder = 'lightgrey', default_linewidth, default_alpha, 1

                            if norm_mpl and not pd.isna(obj_val_for_color):
                                current_color = cmap_mpl(norm_mpl(obj_val_for_color))
                            elif not valid_objectives.empty and valid_objectives.min() == valid_objectives.max() and not pd.isna(obj_val_for_color): # single AUC case
                                current_color = cmap_mpl(0.5) # middle of colormap
                            
                            if i == best_trial_idx and best_trial_idx != -1:
                                current_color, current_lw, current_alpha, current_zorder = best_line_color, best_linewidth, best_alpha, 10
                            
                            ax_mpl.plot(range(len(dimension_info_mpl)), y_values_mpl_series.values, 
                                        color=current_color, alpha=current_alpha, linewidth=current_lw, zorder=current_zorder)
                    
                    if valid_objectives.empty and not original_objective_values_mpl.empty:
                        print("警告: 所有目标值均为NaN或无法用于着色。")

                    xtick_labels_param_names = [info['label_for_plot'] for info in dimension_info_mpl]
                    ax_mpl.set_xticks(range(len(dimension_info_mpl)))
                    
                    # 设置X轴刻度标签，并对AUC标签进行加粗
                    xtick_objects = ax_mpl.set_xticklabels(xtick_labels_param_names, rotation=40, ha='right', fontsize=xtick_label_fontsize)
                    for tick_label_obj, original_label_text in zip(xtick_objects, xtick_labels_param_names):
                        if original_label_text == 'AUC':
                            tick_label_obj.set_fontweight('bold')
                            tick_label_obj.set_color(colors_google['red']) # AUC X轴标签为红色
                        else:
                            tick_label_obj.set_color(colors_google['blue']) # 其他X轴标签为蓝色

                    # --- 新增：在每条参数轴上标注原始值 ---
                    for idx, info in enumerate(dimension_info_mpl):
                        y_positions_norm = [0.0, 0.5, 1.0]
                        
                        # Values to display are from original scale
                        val_bottom_orig = info['original_min']
                        val_top_orig = info['original_max']
                        
                        # Calculate middle original value based on scaled range
                        val_mid_orig = np.nan
                        s_min, s_max = info['scaled_min_for_norm'], info['scaled_max_for_norm']

                        if not pd.isna(s_min) and not pd.isna(s_max):
                            if s_max > s_min:
                                val_at_0_5_scaled = s_min + 0.5 * (s_max - s_min)
                                if info['is_log_transformed']:
                                    val_mid_orig = 10**val_at_0_5_scaled
                                else:
                                    val_mid_orig = val_at_0_5_scaled
                            else: # min == max for scaled values
                                val_mid_orig = info['original_min'] # or original_max, they should be the same if scaled are same
                        elif not pd.isna(info['original_min']): # Fallback if scaled min/max are NaN but original is not
                            val_mid_orig = info['original_min']


                        original_values_to_label = [val_bottom_orig, val_mid_orig, val_top_orig]

                        for norm_y, orig_val in zip(y_positions_norm, original_values_to_label):
                            if not pd.isna(orig_val):
                                label_text = format_axis_value_display(orig_val)
                                # 根据轴的标签确定颜色
                                current_param_axis_value_color = colors_google['blue'] # 默认为蓝色
                                if info['label_for_plot'] == 'AUC':
                                    current_param_axis_value_color = colors_google['red'] # 如果是AUC轴，则为红色
                                
                                ax_mpl.text(idx + 0.08, norm_y, label_text, 
                                            ha='left', va='center', fontsize=param_axis_tick_fontsize, 
                                            color=current_param_axis_value_color, bbox=dict(facecolor='white', alpha=0.5, pad=0.1, edgecolor='none'))
                    # --- 结束新增标注 ---

                    ax_mpl.set_ylim(-0.05, 1.05)
                    ax_mpl.set_yticks(np.linspace(0, 1, 6))
                    ax_mpl.set_yticklabels([f"{y:.1f}" for y in np.linspace(0, 1, 6)], fontsize=ytick_label_fontsize)
                    ax_mpl.set_ylabel("Normalized Value", fontsize=axis_label_fontsize, color=colors_google['green'])
                    for lables in ax_mpl.get_yticklabels():
                        lables.set_color(colors_google['blue'])

                    # 绘制垂直参数轴线，并加粗AUC轴线
                    default_dim_line_lw = 2
                    auc_dim_line_lw = 8.0  # AUC轴线的线宽
                    default_dim_line_color = 'grey'
                    auc_dim_line_color = 'dimgray' # AUC轴线的颜色，可以更深一些

                    for j_ax_idx in range(len(dimension_info_mpl)):
                        current_alpha = 1
                        current_lw = default_dim_line_lw
                        current_color = colors_google['blue']
                        # 检查是否为AUC轴 (现在它在dimension_info_mpl的最后一个)
                        if dimension_info_mpl[j_ax_idx]['label_for_plot'] == 'AUC':
                            current_alpha = 1
                            current_lw = auc_dim_line_lw
                            current_color = colors_google['red']
                        
                        ax_mpl.axvline(j_ax_idx, color=current_color, linestyle='-', linewidth=current_lw, alpha=0.6)
                    
                    #ax_mpl.grid(True, axis='y', linestyle=':', alpha=0.5)

                    title_color_mpl = colors_google['blue'] if 'colors_google' in locals() and isinstance(colors_google, dict) and 'blue' in colors_google else 'darkblue'
                    ax_mpl.set_title('Hyperparameters vs AUC', fontsize=title_fontsize, color=title_color_mpl, pad=30) # pad adjusted

                    if sm_mpl and not valid_objectives.empty and valid_objectives.min() < valid_objectives.max():
                        cbar_mpl = fig_par_coords_mpl.colorbar(sm_mpl, ax=ax_mpl, pad=0.06, aspect=30, shrink=0.75) 
                        cbar_mpl.set_label('AUC', fontsize=cbar_label_fontsize, color=colors_google['red'])
                        cbar_mpl.ax.tick_params(labelsize=cbar_tick_fontsize, color=colors_google['red'])
                    
                    if best_trial_idx != -1 and best_trial_idx < len(df_trials) and 'value' in df_trials and pd.api.types.is_numeric_dtype(df_trials['value']):
                        best_val_obj = df_trials['value'].loc[best_trial_idx]
                        best_val_str = format_axis_value_display(best_val_obj, 4) if not pd.isna(best_val_obj) else "N/A"
                        legend_elements = [Line2D([0], [0], color=colors_google['red'], lw=best_linewidth, label=f'Best Trial (Value: {best_val_str})')]
                        # Adjust legend position dynamically based on figure height and font sizes
                        # Approximate vertical space needed by rotated x-labels: xtick_label_fontsize * sin(40deg)
                        # This is complex to get perfect, so using a slightly larger negative offset
                        ax_mpl.legend(handles=legend_elements, loc='upper center', 
                                      bbox_to_anchor=(0.9, -0.24), # Adjusted y-offset for legend
                                      fontsize=legend_fontsize, frameon=False, ncol=1, edgecolor=colors_google['red'])

                    plt.subplots_adjust(left=0.07, right=0.93, bottom=0.25, top=0.90) # Adjusted margins
                    
                    plt.savefig("optuna_parallel_coordinates_matplotlib.png", dpi=300, bbox_inches='tight')
                    plt.show()
            else:
                print("处理后没有足够的有效维度或数据来创建平行坐标图 (Matplotlib)。")




        # --- 3. 超参数重要性散点图 ---
        if hyperparameters:
            num_params = len(hyperparameters)
            # 确定子图网格大小
            cols = 3
            rows = int(np.ceil(num_params / cols))

            fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5), sharey=False)
            axes = axes.flatten() # 将axes展平为1D数组以便更容易迭代

            for i, param in enumerate(hyperparameters):
                if i < len(axes): # 确保我们不会尝试绘制超过可用子图数量的图
                    ax = axes[i]
                    # 设置子图背景为Google风格的浅灰色
                    ax.set_facecolor('#f8f9fa')
                    
                    # 使用感知均匀的色图如'viridis'或'plasma'
                    sc = ax.scatter(df_trials[param], df_trials['value'],
                                    c=df_trials['trial'], cmap='viridis', alpha=0.7, s=50,
                                    edgecolor='white')  # 添加白色边缘以提高可见性

                    ax.set_xlabel(param, fontsize=30, fontweight='bold', color='#424242')
                    ax.set_ylabel("AUC", fontsize=30, fontweight='bold', color='#424242')

                    # 使用Google风格的标题框
                    ax.set_title(f"AUC vs {param}", fontsize=30, fontweight='bold', color=colors_google['blue'],
                               bbox=dict(boxstyle="round,pad=0.3", fc='#E8F0FE', ec=colors_google['blue'], alpha=0.7))
                    
                    if param in log_params: # 对数参数使用对数刻度
                        ax.set_xscale('log')
                    
                    # 设置网格线
                    ax.grid(True, linestyle=':', alpha=0.6, color=colors_google['light_grey'])
                    
                    # 移除顶部和右侧边框
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # 使左侧和底部边框加粗
                    ax.spines['bottom'].set_linewidth(1.5)
                    ax.spines['left'].set_linewidth(1.5)
                    ax.spines['bottom'].set_color('#424242')
                    ax.spines['left'].set_color('#424242')

                    # 获取坐标轴范围以添加箭头
                    xlims = ax.get_xlim()
                    ylims = ax.get_ylim()
                    
                    # 添加坐标轴箭头
                    arrow_props = dict(facecolor=colors_google['blue'], edgecolor=colors_google['blue'],
                                      width=1.5, headwidth=8, headlength=10,
                                      shrinkA=0, shrinkB=0)
                    
                    # X轴箭头
                    ax.annotate('',
                                xy=(xlims[1], ylims[0]),  # 箭头尖端
                                xytext=(-arrow_props['headlength'], 0),  # 相对于尖端的箭头尾部偏移
                                textcoords='offset points',
                                arrowprops=arrow_props,
                                xycoords='data',
                                clip_on=False)
                    
                    # Y轴箭头
                    ax.annotate('',
                                xy=(xlims[0], ylims[1]),  # 箭头尖端
                                xytext=(0, -arrow_props['headlength']),  # 相对于尖端的箭头尾部偏移
                                textcoords='offset points',
                                arrowprops=arrow_props,
                                xycoords='data',
                                clip_on=False)

            # 为试验编号添加颜色条
            if num_params > 0: # 仅当有图时
                cbar = fig.colorbar(sc, ax=axes[:num_params], orientation='horizontal', fraction=0.05, pad=0.1)
                cbar.set_label('Trial Number', fontsize=30, fontweight='bold', color='#424242')
                cbar.ax.tick_params(colors='#424242')

            # 隐藏任何未使用的子图
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # 添加Google风格的总体标题
            plt.suptitle("AUC vs Hyperparameters", fontsize=22, fontweight='bold', y=1.03 if rows > 1 else 1.05,
                        color=colors_google['blue'],
                        bbox=dict(boxstyle="round,pad=0.6", fc='#E8F0FE', ec=colors_google['blue'], alpha=0.8))
            
            plt.tight_layout(rect=[0, 0, 1, 0.98 if rows > 1 else 0.95])
            plt.savefig("optuna_param_vs_value_scatter.png", dpi=300)
            plt.show()

        # --- 4. 等高线图（针对两个重要参数） ---
        if 'learning_rate' in df_trials.columns and 'n_estimators' in df_trials.columns:
            # 使用Plotly创建更好的交互式等高线图
            fig_contour = go.Figure(data =
                go.Contour(
                    z=df_trials['value'],
                    x=df_trials['learning_rate'],
                    y=df_trials['n_estimators'],
                    colorscale='viridis',  # 选择一个好的色标
                    colorbar=dict(title='AUC',
                                 titlefont=dict(color='#424242'),
                                 tickfont=dict(color='#424242')),
                    contours=dict(
                        coloring='heatmap',  # 或 'lines' 或 'fill'
                        showlabels=True,  # 在等高线上显示标签
                        labelfont=dict(  # 标签的字体属性
                            size=12,
                            color='white',
                        )
                    ),
                )
            )
            # 添加所有试验点以更好地理解分布
            fig_contour.add_trace(
                go.Scatter(
                    x=df_trials['learning_rate'],
                    y=df_trials['n_estimators'],
                    mode='markers',
                    marker=dict(
                        color=df_trials['value'],
                        colorscale='viridis',
                        size=8,
                        line=dict(color='white', width=1)
                    ),
                    showlegend=False,
                    name='Trial Points'
                )
            )
            # 标记最佳点
            fig_contour.add_trace(
                go.Scatter(
                    x=[best_trial_overall['learning_rate']],
                    y=[best_trial_overall['n_estimators']],
                    mode='markers',
                    marker=dict(
                        color=colors_google['yellow'],
                        size=15,
                        symbol='star',
                        line=dict(color='black', width=2)
                    ),
                    showlegend=True,
                    name=f'Best Trial ({best_trial_overall["trial"]:.0f})'
                )
            )
            # 使用Google风格更新布局
            fig_contour.update_layout(
                title={'text': 'Contour Plot: Learning Rate & n_estimators Effect on Objective',
                       'font': {'size': 20, 'color': colors_google['blue']},
                       'x': 0.5,
                       'y': 0.95},
                xaxis_title={'text': 'Learning Rate (log scale)', 'font': {'size': 16, 'color': '#424242'}},
                yaxis_title={'text': 'Number of Trees', 'font': {'size': 16, 'color': '#424242'}},
                xaxis_type='log',  # 对学习率重要的对数刻度
                plot_bgcolor='#f8f9fa',  # Google风格的浅灰色背景
                paper_bgcolor='white',
                xaxis=dict(
                    gridcolor=colors_google['light_grey'],
                    gridwidth=1,
                    zerolinecolor='#424242',
                    zerolinewidth=2,
                    tickfont=dict(color='#424242')
                ),
                yaxis=dict(
                    gridcolor=colors_google['light_grey'],
                    gridwidth=1,
                    zerolinecolor='#424242',
                    zerolinewidth=2,
                    tickfont=dict(color='#424242')
                ),
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#424242'
                )
            )
            fig_contour.show()
            fig_contour.write_image("optuna_contour_lr_n_estimators.png", scale=2)
        else:
            print("由于在解析数据中未找到'learning_rate'或'n_estimators'，跳过等高线图。")

    else:
        print("DataFrame为空，跳过绘图。")
