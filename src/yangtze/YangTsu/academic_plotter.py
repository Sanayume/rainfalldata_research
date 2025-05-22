import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np # For examples

class AcademicStylePlotter:
    """
    一个用于创建具有统一学术风格图表的类。
    进一步优化以模仿指定论文的图表风格。
    """
    def __init__(self, font_family='Arial', base_font_size=9): # 论文中字体看起来不大，尝试9pt
        """
        初始化绘图器并应用基础样式。

        Args:
            font_family (str): 全局字体家族。
            base_font_size (int): 基础字号。
        """
        self.font_family = font_family
        self.base_font_size = base_font_size
        self._apply_base_style()

    def _apply_base_style(self):
        """应用基础的 matplotlib rcParams 设置。"""
        # 尝试不基于seaborn风格，从更干净的起点开始，或使用'classic'
        # plt.style.use('classic') # 或者完全自定义
        plt.style.use('default') # 使用 Matplotlib 的默认风格作为起点，然后覆盖

        self.rcParams = {
            "font.family": self.font_family,
            "font.sans-serif": [self.font_family, "DejaVu Sans", "Helvetica", "Bitstream Vera Sans"],
            "font.serif": ["Times New Roman", "DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman"],
            "font.size": self.base_font_size, # 整体字号偏小
            "axes.labelsize": self.base_font_size,
            "axes.titlesize": self.base_font_size, # 子图标题字号与标签一致或稍大一点点
            "axes.labelweight": 'normal', # 标签不加粗
            "axes.titleweight": 'normal', # 子图标题不加粗（图1中产品名）
            "xtick.labelsize": self.base_font_size -1, # 刻度标签更小
            "ytick.labelsize": self.base_font_size -1,
            "legend.fontsize": self.base_font_size -1,
            "figure.titlesize": self.base_font_size + 2, # 主图标题
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.format": 'pdf',
            "savefig.bbox": 'tight',
            "axes.edgecolor": 'black',
            "axes.linewidth": 0.6, # 边框线更细
            "grid.color": "gainsboro", # 浅灰色网格
            "grid.linestyle": ":",   # 点状网格线
            "grid.linewidth": 0.5,
            "xtick.direction": 'in',
            "ytick.direction": 'in',
            "xtick.top": True, # 显示顶部刻度线
            "ytick.right": True, # 显示右侧刻度线
            "xtick.major.size": 3, # 刻度线长度
            "ytick.major.size": 3,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "lines.linewidth": 1.2, # 线条略细
            "lines.markersize": 4,
            "patch.edgecolor": 'black', # 柱状图等的边框
            "patch.linewidth": 0.5,
            "legend.frameon": False, # 图例无边框
            "legend.loc": "best",
            "image.cmap": "viridis", # 默认图像色板
            "figure.facecolor": "white", # 图形背景色
            "axes.facecolor": "white",   # 坐标轴背景色
        }
        plt.rcParams.update(self.rcParams)
        print(f"Academic style (v2) applied with base font size {self.base_font_size}pt.")

    def update_style(self, style_dict):
        plt.rcParams.update(style_dict)
        self.rcParams.update(style_dict)
        print("Style updated.")

    def get_colormap(self, name, n_colors=None, reverse=False):
        cmap = None
        colors_list = None

        if name == 'correlation' or name == 'kge' or name == 'sca':
            # 对应图1 CC, 图4 SCA, 图5a CC, 图5d KGE
            # 红色(低) - 橙 - 黄 - 浅绿 - 绿色(高)
            # 论文中 0.4-0.6 区域是黄绿色过渡，1是深绿，0是深红
            # SCA的图4，颜色过渡比较平滑
            nodes = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # 控制颜色点的位置
            colors_hex = ['#d7191c', '#fdae61', '#ffffbf', '# CEEFCE', '#a6d96a', '#1a9641', '#006837'] #深红, 橙, 浅黄, 极浅绿, 浅绿, 绿, 深绿
            # 微调：确保0.5附近是黄/浅黄，向两边过渡
            # 示例：图4 SCA 0.4(红)-0.6(黄)-0.8(浅绿)-1(深绿)
            # 在 AcademicStylePlotter 类的 get_colormap 方法内
            if name == 'correlation' or name == 'kge' or name == 'sca':
                # ...
                nodes = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
                colors_hex = ['#d7191c', '#fdae61', '#ffffbf', '#CEEFCE', '#a6d96a', '#1a9641', '#006837'] # 修正：移除了空格
                # ...
                if name == 'sca': # 图4 (假设这是你之前代码中的逻辑)
                    nodes = [0.0, 0.4, 0.5, 0.6, 0.8, 1.0]
                    # 确保这里的颜色也正确，如果它也使用了那个带空格的颜色
                    colors_hex = ['#d7191c', '#d7191c', '#ffffbf', '#a6d96a','#1a9641', '#006837']
                cmap = mcolors.LinearSegmentedColormap.from_list("custom_RdYlGn_Paper", list(zip(nodes, colors_hex)))
                # ...
        elif name == 'bias': # 对应图1 Relative Bias, 图5c RB
            # 蓝色(负) - 白色(零) - 红色(正)，论文中-40%是深蓝，40%是深红，0是白色
            nodes = [0.0, 0.45, 0.5, 0.55, 1.0] # 0.5 对应 0 bias
            colors_hex = ['#2166ac', '#f7f7f7', '#f7f7f7', '#f7f7f7', '#b2182b'] # 深蓝, 几乎白, 白, 几乎白, 深红
            # 或者更平滑的:
            # colors_hex = ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'] #蓝-浅蓝-白-浅红-红
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_CoolWarm_Paper", list(zip(nodes, colors_hex)))
        elif name == 'rmse': # 对应图1 RMSE, 图5b RMSE
            # 论文中绿色(低) - 黄 - 橙 - 红色(高)
            nodes = [0.0, 0.3, 0.6, 1.0]
            colors_hex = ['#1a9641', '#ffffbf', '#fdae61', '#d7191c'] # 绿, 浅黄, 橙, 红
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_GnYlRd_Paper", list(zip(nodes, colors_hex)))
        elif name == 'miss_detection': # 图7左侧
            # 5%, 15%, 25% -> 浅绿，中绿，深绿
            nodes = [0.0, 0.05, 0.15, 0.25, 1.0] # 以25%为上限演示
            colors_hex = ['#f7fcf5', '#c7e9c0', '#74c476', '#238b45', '#00441b'] # 从极浅绿到深绿 (Greens seq)
            cmap = mcolors.LinearSegmentedColormap.from_list("miss_detection_paper", colors_hex) # 可以调整节点
        elif name == 'false_detection': # 图7右侧
            # 5%, 15%, 25% -> 浅红，中红，深红
            nodes = [0.0, 0.05, 0.15, 0.25, 1.0]
            colors_hex = ['#fff5f0', '#fcbba1', '#fb6a4a', '#cb181d', '#67000d'] # 从极浅红到深红 (Reds seq)
            cmap = mcolors.LinearSegmentedColormap.from_list("false_detection_paper", colors_hex)

        elif name == 'qualitative_prod_lines': # 对应图3 的5条线
            # CHIRPS (blue), MSWEP (orange), CMADS (green), PERSIANN (red), ITPCAS (purple)
            # 尽量从论文中取色
            colors_list = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7'] # 类Okabe-Ito, 颜色区分明显
            # 或 plt.cm.tab10.colors[:5]
        elif name == 'error_types_stack': # 对应图2 Error Type 1,2,3 + 另一个(Days)
            # 黄, 绿, 蓝, (灰色/青色?)
            colors_list = ['#fee08b', '#a6d96a', '#66c2a5', '#3288bd'] # 黄, 浅绿, 浅青, 蓝 (示例)

        if cmap:
            if reverse: cmap = cmap.reversed()
            return cmap
        elif colors_list:
            return colors_list[:n_colors] if n_colors else colors_list
        else: # Fallback
            return plt.cm.get_cmap(name, n_colors) if n_colors else plt.cm.get_cmap(name)

    def format_map_axes(self, ax, extent=None, lon_ticks=None, lat_ticks=None,
                        coast_lw=0.6, border_lw=0.4, grid_lw=0.5,
                        show_ocean=False, show_land=False, land_color='#f0f0f0'):
        """
        格式化地理地图坐标轴，更接近论文风格。
        """
        if not hasattr(ax, 'projection'):
            raise ValueError("Axes must have a Cartopy projection.")

        ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=coast_lw, edgecolor='black', zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale('50m'), linestyle=':', linewidth=border_lw, edgecolor='black', zorder=3)

        if show_ocean:
            ax.add_feature(cfeature.OCEAN.with_scale('50m'), facecolor='aliceblue', zorder=0) # 论文中海洋是白色
        if show_land:
             ax.add_feature(cfeature.LAND.with_scale('50m'), facecolor=land_color, zorder=0) # 论文中陆地是白色或数据覆盖

        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=grid_lw, color='gainsboro', alpha=0.8, linestyle=':')
        gl.top_labels = False
        gl.right_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'size': self.base_font_size - 1.5, 'color': 'black'}
        gl.ylabel_style = {'size': self.base_font_size - 1.5, 'color': 'black'}

        if lon_ticks:
            gl.xlocator = mticker.FixedLocator(lon_ticks)
        if lat_ticks:
            gl.ylocator = mticker.FixedLocator(lat_ticks)
        return gl

    def add_subplot_label(self, ax, label, x=-0.09, y=1.03, fontsize_offset=-0.5, fontweight='normal', **kwargs):
        # 论文中的 (a), (1) 等标签字号略小，不加粗
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=self.base_font_size + fontsize_offset,
                fontweight=fontweight,
                va='bottom', ha='left', # 调整对齐可能更符合论文风格
                **kwargs)

    def set_title_for_subplot(self, ax, title_text, loc='center', fontsize_offset=0):
        # 论文子图标题 (如产品名) 字体大小与轴标签类似
        ax.set_title(title_text, loc=loc, fontsize=self.base_font_size + fontsize_offset, weight='normal')


    def create_heatmap_table(self, ax, data, row_labels, col_labels, cmap, vmin, vmax,
                             cbar_label="", val_fmt="{x:.2f}", text_colors=("black", "white"),
                             text_threshold_val=None):
        """
        创建类似图5的表格型热力图。

        Args:
            text_threshold_val: 切换文字颜色的数据阈值，如果 cmap 中间色浅，则取值小的用深色字，大的用浅色字
        """
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')

        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=self.base_font_size-1.5)
        ax.set_yticklabels(row_labels, fontsize=self.base_font_size-1.5)
        ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True) # X轴标签在下方

        # 根据背景色自动选择文字颜色
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                bgcolor = cmap(norm(val))
                # 简单的亮度判断选择文字颜色
                lum = 0.299*bgcolor[0] + 0.587*bgcolor[1] + 0.114*bgcolor[2] # 计算亮度
                text_color = text_colors[1] if lum < 0.5 else text_colors[0] # 亮度小于0.5用白色字

                if text_threshold_val is not None: # 如果提供了阈值判断
                     text_color = text_colors[1] if val > text_threshold_val else text_colors[0]

                ax.text(j, i, val_fmt.format(x=val),
                        ha="center", va="center", color=text_color,
                        fontsize=self.base_font_size - 2) # 数值字号更小

        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, size=self.base_font_size-1)
        cbar.ax.tick_params(labelsize=self.base_font_size-2)
        return im, cbar

    def format_time_series_axes(self, ax, y_label="", title="", major_x_locator=None, minor_x_locator=None):
        """ 格式化时间序列图的坐标轴 (如图4) """
        ax.set_ylabel(y_label)
        ax.set_title(title)
        if major_x_locator:
            ax.xaxis.set_major_locator(major_x_locator)
        if minor_x_locator:
            ax.xaxis.set_minor_locator(minor_x_locator)
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # 如果X是日期
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    def save_figure(self, fig, filename, **kwargs):
        save_kwargs = {
            'dpi': plt.rcParams['savefig.dpi'],
            'bbox_inches': plt.rcParams['savefig.bbox'],
            'facecolor': fig.get_facecolor(),
            'transparent': False # 通常学术图不需要透明背景
        }
        save_kwargs.update(kwargs)
        fig.savefig(filename, **save_kwargs)
        print(f"Figure saved to {filename}")

# --- 如何使用 (示例保持不变或略作调整) ---
if __name__ == '__main__':
    plotter = AcademicStylePlotter(font_family='Arial', base_font_size=9)

    # 示例1: 线图 (图3)
    fig1, ax1 = plt.subplots(figsize=(5, 3.5)) # 调整figsize以适应内容
    product_names = ['CHIRPS', 'MSWEP', 'CMADS', 'PERSIANN', 'ITPCAS']
    line_colors = plotter.get_colormap('qualitative_prod_lines')
    x_data = np.linspace(0, 10, 100)
    for i, name in enumerate(product_names):
        y_data = np.exp(-((x_data - (i+1)*0.8)**2) / (2 * (0.5 + i*0.1)**2)) * (2+i*0.5) # 模拟密度曲线
        ax1.plot(x_data, y_data, label=name, color=line_colors[i])
    ax1.set_xlabel('Precipitation intensity (mm/d)')
    ax1.set_ylabel('Probability density')
    # ax1.set_title('Precipitation Intensity Distribution') # 论文中图3无大标题
    ax1.legend(loc='upper right')
    plotter.add_subplot_label(ax1, '(a)')
    plt.tight_layout(pad=0.5)
    plotter.save_figure(fig1, 'demo_line_plot_v2.pdf')
    plt.show()


    # 示例2: 地图热力图 (图1 或 图7)
    fig2, ax2 = plt.subplots(figsize=(4, 3.5), subplot_kw={'projection': ccrs.PlateCarree()}) # 调整尺寸
    # 论文中图1的刻度比较少
    gl = plotter.format_map_axes(ax2, extent=[75, 135, 18, 55], lon_ticks=[80, 100, 120], lat_ticks=[20, 30, 40, 50])
    lons_map = np.array([75, 85, 95, 105, 115, 125, 135]) # 匹配图1的经度标签
    lats_map = np.array([20, 30, 40, 50]) # 匹配图1的纬度标签
    gl.xlocator = mticker.FixedLocator(lons_map)
    gl.ylocator = mticker.FixedLocator(lats_map)


    lons_data = np.linspace(75, 135, 50)
    lats_data = np.linspace(18, 55, 40)
    map_data = np.random.rand(len(lats_data), len(lons_data)) * 0.8 + 0.1 # Correlation 0.1 to 0.9

    cmap_corr = plotter.get_colormap('correlation')
    mesh = ax2.pcolormesh(lons_data, lats_data, map_data, transform=ccrs.PlateCarree(), cmap=cmap_corr, vmin=0, vmax=1)
    # 论文中图1色条在下方，水平
    cbar = plt.colorbar(mesh, ax=ax2, orientation='horizontal', pad=0.08, aspect=30, shrink=0.9)
    cbar.set_label('Correlation Coefficient', size=plotter.base_font_size-1)
    cbar.set_ticks(np.linspace(0, 1, 6)) # 例如 0, 0.2, 0.4, 0.6, 0.8, 1.0
    cbar.ax.tick_params(labelsize=plotter.base_font_size-2)
    plotter.set_title_for_subplot(ax2, 'CHIRPS') # 产品名作为子图标题
    plotter.add_subplot_label(ax2, '(1)', x=-0.12, y=1.0) # 调整标签位置

    plt.tight_layout(rect=[0, 0.05, 1, 0.98], h_pad=0.1, w_pad=0.1) # 调整间距
    plotter.save_figure(fig2, 'demo_map_plot_v2.pdf')
    plt.show()


    # 示例3: 表格型热力图 (图5a)
    fig3, ax3 = plt.subplots(figsize=(8, 3.5)) # 调整尺寸
    basins_short = ['Songliao', 'Haihe', 'Huaihe', 'Yellow', 'Yangtze', 'Pearl', 'Southeast', 'Southwest']
    products_table = ['CHIRPS', 'MSWEP', 'CMADS', 'PERSIANN', 'ITPCAS']
    cc_data_table = np.random.rand(len(products_table), len(basins_short)) * 0.5 + 0.4 # 示例数据 0.4-0.9

    plotter.create_heatmap_table(ax3, cc_data_table, products_table, basins_short,
                                 cmap=plotter.get_colormap('sca'), # 使用SCA色板更接近图5的颜色范围
                                 vmin=0.4, vmax=1.0, # 根据图5调整范围
                                 cbar_label="Correlation Coefficient (CC)",
                                 text_threshold_val=0.65) # 假设0.65以上用白色字
    plotter.set_title_for_subplot(ax3, "(a) Correlation Coefficient (CC)", loc='left', fontsize_offset=1) # 图5的子图标题
    plt.tight_layout(pad=1.0)
    plotter.save_figure(fig3, 'demo_table_heatmap_v2.pdf')
    plt.show()