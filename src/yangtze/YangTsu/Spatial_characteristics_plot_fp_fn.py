from loaddata import mydata
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm # For custom fonts

# --- 字体配置 (强烈建议进行配置以达到出版质量) ---
# 请取消以下代码块的注释，并根据您的系统和偏好进行配置。
# 您可能需要安装字体，或者查找系统中已安装字体的准确名称。
try:
    # 查找系统可用字体 (如果需要)
    # available_fonts = sorted([f.name for f in fm.fontManager.ttflist])
    # print("系统中可用的字体:")
    # for font_name_option in available_fonts:
    #     print(font_name_option)
    
    # 设置首选字体，例如 'Arial' (无衬线) 或 'Times New Roman' (衬线)
    # preferred_font = 'Arial' # 修改为您希望使用的字体名称
    # if preferred_font in available_fonts:
    #     plt.rcParams['font.family'] = preferred_font
    # else:
    #     print(f"警告: 字体 '{preferred_font}' 未在系统中找到, 将使用 Matplotlib 默认字体。")
    #     # 可以设置一个备选的通用字体族
    #     # plt.rcParams['font.family'] = 'sans-serif'
    #     # plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] # 一个比较好的默认无衬线字体

    # 设置各部分的字体大小
    plt.rcParams['font.size'] = 10       # 基础字号 (用于刻度标签等)
    plt.rcParams['axes.titlesize'] = 12  # 子图标题字号 (此处为年份)
    plt.rcParams['axes.labelsize'] = 12  # 坐标轴标签字号 (此处为 POD, FAR, CSI)
    # plt.rcParams['xtick.labelsize'] = 10 # X轴刻度标签字号 (当前图中无刻度标签)
    # plt.rcParams['ytick.labelsize'] = 10 # Y轴刻度标签字号 (当前图中无刻度标签)
    # plt.rcParams['legend.fontsize'] = 10 # 图例字号 (当前图中无图例)
    plt.rcParams['figure.titlesize'] = 14 # 主图标题字号 (suptitle)
except Exception as e:
    print(f"字体设置过程中出现错误: {e}。将使用 Matplotlib 默认字体设置。")

mydata = mydata()
_, _, X, Y = mydata.yangtsu()
PRODUCTS = mydata.get_products()
#定义降雨分类阈值, 默认为0.1mm/d
rain_threshold = 0.1 
# days = 0 # This variable was unused

#生成从2016年-2020年5年的时间编码
time_pd_range = pd.date_range(start='2016-01-01', end='2020-12-31', freq='D')
time_y_full = np.array(time_pd_range.year)
# time_m_full = np.array(time_pd_range.month) # Unused if only selecting by year for X, Y
# time_d_indices_full = np.arange(len(time_pd_range)) # Unused

years = [2016, 2017, 2018, 2019, 2020]

# Define the cropping slice
# IMPORTANT: Ensure these slice indices are valid for your data dimensions after np.nansum
# The spatial dimensions of fp_map, etc., are (144, 256) based on README
# So, slice_y should be within 0-143 and slice_x within 0-255.
slice_y = slice(70, 131)
slice_x = slice(60, 201)

for i_prod, product_name in enumerate(PRODUCTS):
    # Create a figure with 3 rows and 5 columns of subplots
    fig, axes = plt.subplots(3, 5, figsize=(12, 7.5)) 
    # Increased height a bit for better spacing, sharex/sharey for consistent axes if we add them later
    fig.suptitle(f'Spatial Characteristics for {product_name} (Rain Threshold: {rain_threshold}mm/d)', fontsize=plt.rcParams['figure.titlesize'], fontweight='bold')

    metrics_data = {'POD': [], 'FAR': [], 'CSI': []} # To store data for common colorbar

    for i_year, year_val in enumerate(years):
        # Select data for the current product and year
        is_year = (time_y_full == year_val)
        
        X_data = X[i_prod, is_year, :, :]
        Y_data = Y[is_year, :, :] # Assuming Y corresponds to all products or is CHM

        X_is_rain = np.where(X_data > rain_threshold, 1, 0)
        Y_is_rain = np.where(Y_data > rain_threshold, 1, 0)
        
        # 计算fp, fn, tp, tn 的年度空间累积
        fp_map = np.nansum(((X_is_rain == 1) & (Y_is_rain == 0)), axis=0)
        fn_map = np.nansum(((X_is_rain == 0) & (Y_is_rain == 1)), axis=0)
        tp_map = np.nansum(((X_is_rain == 1) & (Y_is_rain == 1)), axis=0)
        # tn_map = np.nansum(((X_is_rain == 0) & (Y_is_rain == 0)), axis=0) # tn_map not directly used in POD, FAR, CSI

        # --- 安全地计算 POD, FAR, CSI ---
        # POD = TP / (TP + FN)
        pod_denominator = tp_map + fn_map
        POD_spatial = np.full_like(tp_map, np.nan, dtype=float)
        np.divide(tp_map, pod_denominator, out=POD_spatial, where=pod_denominator != 0)

        # FAR = FP / (TP + FP) --- Corrected FAR calculation
        far_denominator = tp_map + fp_map
        FAR_spatial = np.full_like(fp_map, np.nan, dtype=float) # Initialize with NaN
        # Conventionally, if TP+FP = 0 (no positive predictions), FAR = 0.
        # np.divide will leave NaN for 0/0, so we handle it:
        np.divide(fp_map, far_denominator, out=FAR_spatial, where=far_denominator != 0)
        FAR_spatial[far_denominator == 0] = 0.0 # Set FAR to 0 if no positive predictions

        # CSI = TP / (TP + FP + FN)
        csi_denominator = tp_map + fp_map + fn_map
        CSI_spatial = np.full_like(tp_map, np.nan, dtype=float)
        np.divide(tp_map, csi_denominator, out=CSI_spatial, where=csi_denominator != 0)
        
        # Crop the data
        POD_cropped = POD_spatial[slice_y, slice_x]
        FAR_cropped = FAR_spatial[slice_y, slice_x]
        CSI_cropped = CSI_spatial[slice_y, slice_x]


        metrics_data['POD'].append(POD_cropped)
        metrics_data['FAR'].append(FAR_cropped)
        metrics_data['CSI'].append(CSI_cropped)
        
        # --- 定义更美观的 Colormaps ---
        cmap_pod = 'YlGnBu' 
        cmap_far = 'YlOrRd'  
        cmap_csi = 'PuBuGn' 

        # POD (Row 0)
        ax_pod = axes[0, i_year]
        im_pod = ax_pod.imshow(POD_cropped, cmap=cmap_pod, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax_pod.set_title(f'{year_val}')
        if i_year == 0:
            ax_pod.set_ylabel('POD', fontweight='bold', labelpad=10) # Add Y-axis label for the first plot in the row

        # FAR (Row 1)
        ax_far = axes[1, i_year]
        im_far = ax_far.imshow(FAR_cropped, cmap=cmap_far, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        # ax_far.set_title(f'FAR {year_val}') # Title is just year now
        if i_year == 0:
            ax_far.set_ylabel('FAR', fontweight='bold', labelpad=10)

        # CSI (Row 2)
        ax_csi = axes[2, i_year]
        im_csi = ax_csi.imshow(CSI_cropped, cmap=cmap_csi, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        # ax_csi.set_title(f'CSI {year_val}') # Title is just year now
        if i_year == 0:
            ax_csi.set_ylabel('CSI', fontweight='bold', labelpad=10)

        # --- 清理子图坐标轴，实现更"扁平化"和简洁的外观 ---
        for ax_row_of_subplots in axes:
            for current_ax in ax_row_of_subplots:
                current_ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
                # 移除子图边框 (spines)
                current_ax.spines['top'].set_visible(False)
                current_ax.spines['right'].set_visible(False)
                current_ax.spines['bottom'].set_visible(False)
                current_ax.spines['left'].set_visible(False)
    
    # --- 为每行添加共享的颜色条 ---
    # 颜色条参数
    cbar_width = 0.015  # 颜色条宽度
    x_pos_cbar = 0.90   # 颜色条左侧的X轴位置 (相对于图窗)
    
    # POD Colorbar
    pos_pod_row_ax = axes[0, -1] 
    bbox_pod = pos_pod_row_ax.get_position() # 获取子图行位置信息
    cax_pod = fig.add_axes([x_pos_cbar, bbox_pod.y0, cbar_width, bbox_pod.height])
    cb_pod = fig.colorbar(im_pod, cax=cax_pod, orientation='vertical')
    # cb_pod.set_label('POD', size=plt.rcParams['axes.labelsize']-2) # 可选：为颜色条添加标签

    # FAR Colorbar
    pos_far_row_ax = axes[1, -1]
    bbox_far = pos_far_row_ax.get_position()
    cax_far = fig.add_axes([x_pos_cbar, bbox_far.y0, cbar_width, bbox_far.height])
    cb_far = fig.colorbar(im_far, cax=cax_far, orientation='vertical')
    # cb_far.set_label('FAR', size=plt.rcParams['axes.labelsize']-2)

    # CSI Colorbar
    pos_csi_row_ax = axes[2, -1]
    bbox_csi = pos_csi_row_ax.get_position()
    cax_csi = fig.add_axes([x_pos_cbar, bbox_csi.y0, cbar_width, bbox_csi.height])
    cb_csi = fig.colorbar(im_csi, cax=cax_csi, orientation='vertical')
    # cb_csi.set_label('CSI', size=plt.rcParams['axes.labelsize']-2)

    # --- 调整整体布局 ---
    # 使用 subplots_adjust 进行精细调整
    # 参数：left, right, bottom, top 定义了子图区域边界
    # wspace, hspace 定义了子图间的间距
    plt.subplots_adjust(left=0.08, right=x_pos_cbar - 0.02, bottom=0.05, top=0.92, wspace=0.05, hspace=0.12)
    
    plt.show()
    # To save the figure:
    # fig.savefig(f"{product_name}_spatial_metrics.png", dpi=300, bbox_inches='tight')

        




    
    
















