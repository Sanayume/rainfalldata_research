import numpy as np
import xarray as xr
import os
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
# from scale_utils import downscale, upscale # 这个脚本中似乎没用到，暂时注释
import sys
import io
from tqdm import tqdm
import cartopy.crs as ccrs # 导入 cartopy
import cartopy.feature as cfeature # 导入 cartopy feature
import matplotlib.ticker as mticker # 导入 ticker 用于精细控制刻度
from academic_plotter import AcademicStylePlotter
import matplotlib.colors as mcolors # 添加这一行

# 导入你的绘图类
from academic_plotter import AcademicStylePlotter # 假设类保存在 academic_plotter.py

# 设置UTF-8编码 (如果你的环境需要)
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# --- 路径设置 ---
# 获取当前脚本文件所在的目录
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# 构建基础路径 (向上三级)
basepath = os.path.join(os.path.dirname(os.path.dirname(current_script_dir)), "data")
# 保存路径
savepath_intermediate = os.path.join(basepath, "intermediate", "nationwide", "IMERGdata")
if not os.path.exists(savepath_intermediate):
    os.makedirs(savepath_intermediate)
# 图像保存路径
figure_savepath = os.path.join(current_script_dir, "figures") # 将图片保存在脚本同级的 figures 文件夹
if not os.path.exists(figure_savepath):
    os.makedirs(figure_savepath)

imergfolder = os.path.join(basepath, "raw", "nationwide", "IMERG")
if not os.path.exists(imergfolder):
    print(f"错误: IMERG 原始数据文件夹未找到: {imergfolder}")
    sys.exit()
imergfiles = sorted([f for f in os.listdir(imergfolder) if f.endswith('.nc') or f.endswith('.nc4')]) # 确保只读取nc文件并排序
print(f"找到 {len(imergfiles)} 个 IMERG 文件。")
if not imergfiles:
    print(f"错误: 在 {imergfolder} 中没有找到 IMERG 文件。请检查路径和文件名。")
    sys.exit()

# 定义年份信息
year_info = [
    (0, 366, 2016),     # 2016年（闰年）
    # (366, 731, 2017),   # 2017年 (暂时只处理2016年作为示例)
    # (731, 1096, 2018),  # 2018年
    # (1096, 1461, 2019), # 2019年
    # (1461, 1827, 2020)  # 2020年（闰年）
]

# --- 初始化绘图器 ---
# 你可以调整字体和基础字号以匹配你的目标风格
plotter = AcademicStylePlotter(font_family='Arial', base_font_size=9)


# 处理每一年的数据
for start_idx, end_idx, year in year_info:
    print(f"\n处理{year}年数据...")
    # print(f"从索引{start_idx}到{end_idx}")

    if end_idx > len(imergfiles):
        print(f"警告: 年份 {year} 的结束索引 {end_idx} 超出文件列表长度 {len(imergfiles)}。跳过此年份。")
        continue

    # 读取数据
    datalist = []
    # 使用tqdm显示进度条
    for count in tqdm(range(start_idx, end_idx), desc=f"读取 {year} 年数据"):
        # print(f"[{count+1}/{len(imergfiles)}] Processing {imergfiles[count]}") # tqdm会处理进度显示
        file_path = os.path.join(imergfolder, imergfiles[count])
        try:
            with xr.open_dataset(file_path) as data_xr:
                # 尝试获取经纬度信息，第一次获取即可
                if 'lon' not in locals() or 'lat' not in locals():
                    lon = data_xr['lon'].values
                    lat = data_xr['lat'].values
                    # IMERG 数据的经度可能是 0-360，需要转换为 -180 到 180 以便 Cartopy 绘制全球图
                    # 或者在绘制时使用 ccrs.Globe(semimajor_axis=6371007.18108238, semiminor_axis=6371007.18108238, ellipse=None)
                    # 并确保数据和经纬度对应正确
                    # 如果经度是 0-360，在 pcolormesh 中可以配合 central_longitude=180 的投影
                    # 这里我们假设经纬度是标准的 -180 to 180 和 -90 to 90
                    # 如果 IMERG 的经度是 0-360，需要做调整
                    # 例如： lon = np.where(lon > 180, lon - 360, lon)
                    # 或者在绘制时使用特定的投影

                precip_data = data_xr['precipitationCal'].values
                # IMERG 原始数据的维度顺序通常是 (time, lat, lon) 或 (time, lon, lat)
                # 从你的代码看，你期望 (lon, lat, time) 最终转置到 (lat, lon)
                # .values 之后，squeeze() 移除单维度
                # 假设原始是 (time, lat, lon)
                if precip_data.ndim == 3 and precip_data.shape[0] == 1: # 每日文件，时间维度为1
                    precip_data = np.squeeze(precip_data, axis=0) # 移除时间维度
                elif precip_data.ndim == 2: # 如果已经是二维（lat, lon）
                    pass
                else:
                    # 你的原始代码是 data = np.transpose(data, (1,2,0)); data = np.squeeze(data)
                    # 这意味着原始是 (time, dim1, dim2) -> (dim1, dim2, time) -> squeeze
                    # 假设原始是 (time, lat, lon)
                    if precip_data.shape[-1] == 1: # (lat, lon, time) 且 time=1
                         precip_data = np.squeeze(precip_data, axis=-1)
                    # 需要确认 'precipitationCal' 的原始维度顺序
                    # 假设squeeze后是 (lat, lon)
                datalist.append(precip_data)
        except Exception as e:
            print(f"读取文件 {file_path} 失败: {e}")
            continue

    if not datalist:
        print(f"未能成功读取 {year} 年的任何数据。")
        continue

    # 转换为numpy数组 (time, lat, lon)
    yearly_data_stack = np.array(datalist)

    # 计算年总降雨量 (沿时间轴求和)
    total_precipitation_year = np.nansum(yearly_data_stack, axis=0) # (lat, lon)

    # --- 进行绘图 ---
    fig = plt.figure(figsize=(8, 4.5)) # 调整图像大小以获得更好的全球图比例
    # 使用 PlateCarree 投影，但可以指定 central_longitude 以便更好地展示全球
    # 如果你的 lon 是 0-360, central_longitude=180 比较合适
    # 如果你的 lon 是 -180-180, central_longitude=0 比较合适
    # ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180 if np.any(lon > 180) else 0))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=150)) # 尝试Robinson投影

    # 格式化地图轴
    # 对于全球图，可能不需要太密集的刻度
    lon_ticks_global = [-180, -120, -60, 0, 60, 120, 180]
    lat_ticks_global = [-60, -30, 0, 30, 60]
    plotter.format_map_axes(ax, coast_lw=0.5, border_lw=0.3, grid_lw=0.3,
                            show_ocean=True, # 给海洋一个浅色背景
                            # lon_ticks=lon_ticks_global, # Robinson投影的网格线可能需要不同处理
                            # lat_ticks=lat_ticks_global
                            )
    ax.gridlines(color='gray', linestyle=':', linewidth=0.4) # Robinson投影需要这样加网格线

    # 选择一个合适的色板表示降雨量 (例如，从蓝到绿到黄到红)
    # 你可以定义一个新的 'annual_precip' 色板在你的类中
    # 或者使用现有的，例如 'viridis', 'YlGnBu', 'jet' (jet通常不推荐)
    # cmap_precip = plotter.get_colormap('viridis') # 示例
    precip_colors = ["#FFFFFF", "#CDEBFF", "#9AD8FF", "#64C8FF", "#32B4FF",  # very light to light blue
                     "#1E96FF", "#0078FF", "#0050E1", "#0028C8",  # medium to dark blue
                     "#00FF00", "#7FFF00", "#FFFF00", "#FF7F00",  # green, lime, yellow, orange
                     "#FF0000", "#B40000", "#800000"]             # red, dark red, maroon
    nodes_precip = np.array([0, 50, 100, 200, 300, 400, 500, 750, 1000,
                             1500, 2000, 2500, 3000, 4000, 5000, 7000]) / 7000 # 归一化
    cmap_precip = mcolors.LinearSegmentedColormap.from_list("custom_precip_annual",
                                                            list(zip(nodes_precip, precip_colors)))


    # 确定数据的最大值用于色条归一化，排除极端异常值
    max_val = np.nanpercentile(total_precipitation_year, 99) if np.any(total_precipitation_year > 0) else 1000
    min_val = 0

    # 绘制数据
    # 注意：pcolormesh 的 lon 和 lat 需要是格网的边界，或者数据中心对齐
    # 如果 lon, lat 是数据中心，total_precipitation_year 的shape应该是 (len(lat), len(lon))
    # Cartopy 的 pcolormesh 需要 transform=ccrs.PlateCarree() 来指明数据的原始坐标系
    mesh = ax.pcolormesh(lon, lat, total_precipitation_year,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap_precip,
                         vmin=min_val, vmax=max_val, shading='auto')

    # 添加色条
    cbar = plt.colorbar(mesh, ax=ax, orientation='horizontal', pad=0.05, shrink=0.7, aspect=30)
    cbar.set_label(f'Annual Total Precipitation (mm/year)', size=plotter.base_font_size)
    cbar.ax.tick_params(labelsize=plotter.base_font_size - 1)
    # cbar.set_ticks(np.linspace(min_val, max_val, 6).astype(int)) # 设置色条刻度

    # 添加标题和子图标签
    plotter.set_title_for_subplot(ax, f'IMERG Global Total Precipitation - {year}', fontsize_offset=1)
    # plotter.add_subplot_label(ax, f'({year})') # 如果有多个年份的子图

    # 保存图像
    output_filename = os.path.join(figure_savepath, f'IMERG_TotalPrecipitation_{year}_Global_Styled.png')
    # 使用类的保存函数
    plotter.save_figure(fig, output_filename, dpi=200) # 可以指定dpi
    print(f"图像已保存到: {output_filename}")
    plt.close(fig) # 关闭图像，释放内存

print("\n所有年份处理完毕。")