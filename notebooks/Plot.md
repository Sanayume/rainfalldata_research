# Python 多库精细化作图教程 - 论文图表定制

本教程旨在帮助你掌握使用 Python 主流绘图库（Matplotlib, Seaborn, Plotly）制作高质量、复杂且美观的论文图表。我们将从各库的核心概念入手，结合实例，逐步深入，助你从入门到精通。

## 〇、核心绘图库概览与核心组件 (入门级速查与进阶提示)

在开始详细学习之前，我们先对即将涉及的几个核心绘图库及其最关键的组件有一个初步认识。这部分内容旨在提供一个高层概览，方便你理解各库的定位和基本操作逻辑。详细的参数和方法将在后续章节中结合实例展开。

### 库 1: Matplotlib - Python 绘图的基石

*   **定位**: 功能强大、灵活，提供底层的绘图API，几乎可以绘制任何类型的静态图表。是许多其他高级绘图库（如Seaborn）的基础。特别适合需要对图表每个细节进行精细控制的场景，是发表质量图表的最终选择之一。
*   **核心导入**: `import matplotlib.pyplot as plt`
*   **面向对象 vs. Pyplot 接口**: Matplotlib 有两种使用方式：
    1.  **面向对象 (OO) 风格**: 显式创建和操作 `Figure` 和 `Axes` 对象。这是推荐的、更强大和灵活的方式，尤其适用于复杂图表和自定义。
    2.  **Pyplot 风格**: 依赖 `pyplot` 模块维护内部状态，自动作用于“当前”的 Figure 和 Axes。对于快速、简单的绘图很方便。

#### Matplotlib 核心组件与概念:

1.  **`Figure` 对象 (`fig`)**:
    *   **概念**: 整个绘图区域的顶层容器，可以看作一张画布或一个窗口。一个 Figure 可以包含一个或多个 Axes (子图)、标题、图例等。
    *   **创建**:
        *   `fig = plt.figure(figsize=(width, height), dpi=val, facecolor='w', edgecolor='k')`
            *   `figsize`: (元组) 图像的宽度和高度，单位为英寸。例如 `(10, 6)`。
            *   `dpi`: (整数) 每英寸点数，影响图像的分辨率和屏幕显示大小。
            *   `facecolor`: Figure 的背景颜色。
            *   `edgecolor`: Figure 边框颜色。
    *   **常用属性/方法**:
        *   `fig.add_axes([left, bottom, width, height])`: 手动添加 Axes，坐标是相对于 Figure 的比例 (0到1)。例如 `[0.1, 0.1, 0.8, 0.8]` 表示一个占据 Figure 80% 区域的 Axes。
        *   `fig.add_subplot(nrows, ncols, index)` 或 `fig.add_subplot(xyz)`: 添加子图到规则网格中。`index` 从1开始。例如 `fig.add_subplot(2, 2, 1)` 或 `fig.add_subplot(221)`。
        *   `fig.subplots(nrows, ncols, ...)`: 更现代的方法，一次性创建 Figure 和一个或多个 Axes 对象，返回 `(fig, ax)` 或 `(fig, axs_array)`。
        *   `fig.suptitle(text, fontsize=16, fontweight='bold')`: 设置 Figure 的主标题。
        *   `fig.savefig(filename, dpi=300, bbox_inches='tight', transparent=False, ...)`: 保存图像。
            *   `bbox_inches='tight'`: 自动裁剪空白边缘。
            *   `transparent=True`: 保存为透明背景 (如果格式支持，如 PNG)。
        *   `fig.tight_layout()`: 自动调整子图参数，使其填充整个 Figure 区域，避免标签重叠。
        *   `fig.clf()`: 清除整个 Figure。

2.  **`Axes` 对象 (`ax` 或 `axs`)**:
    *   **概念**: 实际进行绘图的坐标系区域（子图）。一个 Figure 可以包含一个或多个 Axes。大部分绘图操作都是在 Axes 对象上进行的。
    *   **创建**:
        *   `ax = fig.add_subplot(1,1,1)` (在已有的 `fig` 上添加)
        *   `fig, ax = plt.subplots()` (同时创建 `fig` 和单个 `ax`)
        *   `fig, axs = plt.subplots(nrows, ncols, sharex=True, sharey=False)` (同时创建 `fig` 和多个 `ax` 组成的数组，可共享轴)。
    *   **常用绘图方法 (在 `ax` 对象上调用)**:
        *   `ax.plot(x, y, color='blue', linestyle='--', marker='o', label='Data1', ...)`: 线图。
        *   `ax.scatter(x, y, s=size_array, c=color_array, marker='^', alpha=0.7, label='Points', ...)`: 散点图。
        *   `ax.bar(x, height, width=0.8, color='green', label='Category A', ...)`: 垂直条形图。
        *   `ax.barh(y, width, height=0.8, color='purple', label='Category B', ...)`: 水平条形图。
        *   `ax.hist(x, bins=20, color='skyblue', edgecolor='black', density=False, ...)`: 直方图。
        *   `ax.boxplot(data, labels=['A', 'B'], showmeans=True, patch_artist=True, ...)`: 箱线图。
        *   `ax.imshow(Z, cmap='viridis', aspect='auto', interpolation='nearest', origin='lower', ...)`: 热力图或图像显示。
        *   `ax.contour(X, Y, Z, levels=10, colors='k', ...)` / `ax.contourf(X, Y, Z, levels=10, cmap='RdYlBu', ...)`: 等高线图 (线/填充)。
        *   `ax.errorbar(x, y, yerr=y_errors, xerr=x_errors, fmt='o', capsize=5, ecolor='gray', ...)`: 误差棒图。
        *   `ax.fill_between(x, y1, y2, color='lightgray', alpha=0.5, where=y1 > y2, interpolate=True, ...)`: 填充两条曲线之间的区域。
        *   `ax.text(x, y, string, fontsize=12, color='red', ha='center', va='bottom', ...)`: 添加文本。
        *   `ax.annotate(text, xy=(x_point, y_point), xytext=(x_text, y_text), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8), ...)`: 添加带箭头的标注。
        *   `ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_list, ...)`: 饼图。
    *   **常用设置方法 (在 `ax` 对象上调用，通常以 `set_` 开头)**:
        *   `ax.set_title(label, fontsize=14, loc='left', ...)`: 设置 Axes 标题。
        *   `ax.set_xlabel(label, fontsize=12, labelpad=10, ...)` / `ax.set_ylabel(label, fontsize=12, labelpad=10, ...)`: 设置轴标签。
        *   `ax.set_xlim(min_val, max_val)` / `ax.set_ylim(min_val, max_val)`: 设置轴范围。
        *   `ax.set_xticks(tick_positions)` / `ax.set_yticks(tick_positions)`: 设置轴刻度位置。
        *   `ax.set_xticklabels(labels_list, rotation=45, ha='right', fontsize=10, ...)` / `ax.set_yticklabels(labels_list, ...)`: 设置轴刻度标签文本。
        *   `ax.tick_params(axis='x', direction='in', length=6, width=1, colors='black', labelsize=10, labelrotation=0, ...)`: 精细控制刻度线和标签样式。
        *   `ax.legend(loc='best', title='Legend Title', fontsize=10, frameon=True, ncol=2, ...)`: 显示图例。
        *   `ax.grid(True/False, which='major', axis='both', linestyle=':', linewidth=0.5, color='gray', ...)`: 显示/隐藏网格线。
        *   `ax.spines[location].set_visible(False/True)`: 控制轴线（'top', 'bottom', 'left', 'right'）的显示。
        *   `ax.spines[location].set_linewidth(val)`: 设置轴线宽度。
        *   `ax.spines[location].set_color(color_str)`: 设置轴线颜色。
        *   `ax.set_aspect('equal')` 或 `ax.set_aspect(ratio)`: 设置x,y轴的缩放比例。
        *   `ax.invert_xaxis()` / `ax.invert_yaxis()`: 反转轴方向。
        *   `ax.twinx()` / `ax.twiny()`: 创建共享X轴/Y轴的次坐标轴。
        *   `ax.cla()`: 清除当前 Axes 上的内容。

3.  **`pyplot` 模块 (`plt`)**:
    *   **概念**: 提供了一个方便的函数式接口（状态机接口），很多函数会自动作用于“当前” Figure 和 Axes。也用于创建 Figure 和 Axes。对于快速探索性绘图非常有用。
    *   **常用函数**:
        *   `plt.figure()`: 创建 Figure。
        *   `plt.subplots()`: 创建 Figure 和 Axes。
        *   `plt.plot()`, `plt.scatter()`, etc.: (pyplot 风格) 直接在当前 Axes 上绘图。
        *   `plt.title()`, `plt.xlabel()`, etc.: (pyplot 风格) 设置当前 Axes 的属性。
        *   `plt.show()`: 显示所有已创建的图像。在 Jupyter Notebook 等环境中，通常不需要显式调用。
        *   `plt.style.use(style_name)`: 应用预定义的样式表 (如 'ggplot', 'seaborn-v0_8-whitegrid', 'fivethirtyeight')。
        *   `plt.rcParams`: 全局参数配置字典，可用于自定义默认样式。例如 `plt.rcParams['font.sans-serif'] = ['SimHei']` 用于显示中文。
        *   `plt.savefig()`: 保存当前 Figure。
        *   `plt.clf()`: 清除当前 Figure。
        *   `plt.cla()`: 清除当前 Axes。
        *   `plt.close()` 或 `plt.close(fig_or_name)` 或 `plt.close('all')`: 关闭 Figure 窗口。

### 库 2: Seaborn - 基于 Matplotlib 的统计数据可视化

*   **定位**: 构建在 Matplotlib 之上，提供更高级的接口，用于绘制美观且信息丰富的统计图形。简化了许多复杂图表的创建，并具有更优雅的默认样式。特别适合探索性数据分析 (EDA) 和展示统计关系。
*   **核心导入**: `import seaborn as sns`
*   **与 Matplotlib 的关系**: Seaborn 的函数通常返回 Matplotlib 的 `Axes` 对象 (对于轴级函数) 或 Seaborn 特有的 `FacetGrid`/`PairGrid`/`JointGrid` 对象 (对于图级函数，这些对象内部管理 Matplotlib Axes)。这意味着你可以使用 Matplotlib 的方法进一步精细调整 Seaborn 图表。

#### Seaborn 核心组件与概念:

1.  **绘图函数 (通常是模块级函数)**: Seaborn 函数通常分为 "轴级 (axes-level)" 和 "图级 (figure-level)"。
    *   **轴级函数**: 在指定的 Matplotlib `Axes` 上绘图 (通过 `ax=` 参数)。例如 `sns.scatterplot(..., ax=my_ax)`。
    *   **图级函数**: 内部创建自己的 Matplotlib Figure 和 Axes，通常提供更高级的功能，如自动分面。例如 `sns.displot()`, `sns.relplot()`, `sns.catplot()`, `sns.lmplot()`, `sns.jointplot()`, `sns.pairplot()`, `sns.clustermap()`.

    *   **关系图 (Relational plots)**: 展示两个变量之间的关系。
        *   `sns.scatterplot(data, x, y, hue, size, style, ax, ...)`: 散点图，可以根据其他变量映射颜色、大小、样式。
        *   `sns.lineplot(data, x, y, hue, size, style, units, estimator, ci, ax, ...)`: 线图，常用于显示趋势，默认会聚合数据并显示置信区间 (CI)。`units` 用于重复测量的长格式数据。
        *   `sns.relplot(data, x, y, hue, size, style, col, row, kind, ...)`: 图级接口，用于绘制 `scatterplot` 或 `lineplot`，并支持分面。`kind='scatter'` (默认) 或 `kind='line'`.
    *   **分布图 (Distribution plots)**: 可视化单个变量的分布或多个变量的联合分布。
        *   `sns.histplot(data, x, y, hue, stat, bins, kde, cumulative, multiple, element, ax, ...)`: 直方图，可以叠加核密度估计(KDE)。`multiple` 控制多组数据堆叠方式 ('layer', 'stack', 'dodge', 'fill')。
        *   `sns.kdeplot(data, x, y, hue, fill, levels, thresh, cumulative, bw_adjust, ax, ...)`: 核密度估计图，可用于一维或二维数据。
        *   `sns.ecdfplot(data, x, y, hue, stat, complementary, ax, ...)`: 经验累积分布函数图。
        *   `sns.rugplot(data, x, y, hue, height, expand_margins, ax, ...)`: 在轴上绘制小刻度线表示数据点。
        *   `sns.displot(data, x, y, hue, col, row, kind, ...)`: 图级接口，用于绘制分布图 (`histplot`, `kdeplot`, `ecdfplot`)，并支持分面。`kind='hist'` (默认), `'kde'`, `'ecdf'`.
    *   **分类图 (Categorical plots)**: 这些图通常有一个分类轴和一个数值轴。
        *   散点类 (显示每个观测点):
            *   `sns.stripplot(data, x, y, hue, jitter, dodge, ax, ...)`: 分类散点图 (带状)，`jitter` 避免点重叠。
            *   `sns.swarmplot(data, x, y, hue, size, dodge, ax, ...)`: 分类散点图 (蜂群状，避免重叠，不适合大数据集)。
        *   分布类 (显示每个类别的分布摘要):
            *   `sns.boxplot(data, x, y, hue, order, hue_order, orient, ax, ...)`: 箱线图。
            *   `sns.violinplot(data, x, y, hue, order, hue_order, split, inner, cut, scale, ax, ...)`: 小提琴图，结合了箱线图和核密度估计。`split=True` 可用于比较两个hue级别。
            *   `sns.boxenplot(data, x, y, hue, order, hue_order, k_depth, ax, ...)`: 增强型箱线图 (letter-value plot)，更适合大数据集。
        *   估计类 (显示集中趋势和置信区间):
            *   `sns.pointplot(data, x, y, hue, order, hue_order, estimator, ci, markers, linestyles, dodge, join, ax, ...)`: 点图 (显示均值或其他估计量和置信区间)。
            *   `sns.barplot(data, x, y, hue, order, hue_order, estimator, ci, errcolor, capsize, ax, ...)`: 条形图 (默认显示均值和置信区间)。
            *   `sns.countplot(data, x, y, hue, order, hue_order, stat, ax, ...)`: 计数条形图，统计每个类别的出现次数。
        *   `sns.catplot(data, x, y, hue, col, row, kind, ...)`: 图级接口，用于绘制各种分类图，并支持分面。`kind` 可以是 `'strip'`, `'swarm'`, `'box'`, `'violin'`, `'boxen'`, `'point'`, `'bar'`, `'count'`.
    *   **回归图 (Regression plots)**: 可视化线性关系。
        *   `sns.regplot(data, x, y, scatter, fit_reg, ci, order, logistic, lowess, robust, ax, ...)`: 绘制散点和线性回归模型拟合。
        *   `sns.lmplot(data, x, y, hue, col, row, fit_reg, ci, order, ...)`: `regplot` 的图级接口，更方便创建分面回归图。
    *   **矩阵图 (Matrix plots)**: 可视化矩阵数据。
        *   `sns.heatmap(data, vmin, vmax, cmap, center, annot, fmt, linewidths, linecolor, cbar, ax, ...)`: 热力图，常用于显示相关性矩阵或混淆矩阵。`annot=True` 显示数值。
        *   `sns.clustermap(data, method, metric, cmap, standard_scale, row_cluster, col_cluster, annot, ...)`: 层次聚类热力图，对行和/或列进行聚类并显示树状图。
    *   **多图网格 (Multi-plot grids)**: 用于创建基于数据子集的分面图。这些是图级对象。
        *   `sns.FacetGrid(data, row, col, hue, col_wrap, height, aspect, ...)`: 创建一个网格，然后用 `.map(plotting_func, 'x_var', 'y_var', ...)` 或 `.map_dataframe(plotting_func, x='x_col_name', y='y_col_name', ...)` 在每个子图上绘制。
        *   `sns.PairGrid(data, vars, hue, diag_sharey, corner, ...)`: 绘制数据集中变量两两关系的网格。使用 `.map_diag()`, `.map_lower()`, `.map_upper()` 指定不同区域的绘图函数。
        *   `sns.pairplot(data, vars, hue, kind, diag_kind, markers, corner, ...)`: `PairGrid` 的简化接口，绘制变量间的散点图 (默认 `kind='scatter'`) 和对角线上的分布图 (默认 `diag_kind='auto'`, 通常是 histplot 或 kdeplot)。
        *   `sns.jointplot(data, x, y, kind, color, hue, space, ratio, marginal_ticks, marginal_kws, ...)`: 绘制两个变量的联合分布和各自的边缘分布。
            *   `kind`: ('scatter', 'kde', 'hist', 'reg', 'resid')

2.  **样式与颜色控制**:
    *   `sns.set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None)`: 全局设置 Seaborn 主题。
        *   `context`: ('paper', 'notebook', 'talk', 'poster') 控制绘图元素的缩放比例，影响线宽、字体大小等。
        *   `style`: ('darkgrid', 'whitegrid', 'dark', 'white', 'ticks') 控制背景和网格。
        *   `palette`: 调色板名称 (如 'viridis', 'rocket', 'muted', 'pastel')、颜色列表或 `sns.color_palette()` 返回的对象。
        *   `rc`: 传递给 `matplotlib.rcParams` 的参数字典，用于更细致的 Matplotlib 样式控制。例如 `rc={'axes.labelsize': 12}`。
    *   `sns.set_style(style, rc)`: 只设置网格和背景样式。
    *   `sns.set_context(context, font_scale, rc)`: 只设置元素缩放。
    *   `sns.set_palette(palette, n_colors, desat, color_codes)`: 设置当前默认调色板。
    *   `sns.color_palette(palette=None, n_colors=None, desat=None, as_cmap=False)`: 返回颜色列表或 Colormap 对象。常用的 `palette` 包括：
        *   定性: 'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind'
        *   顺序: 'viridis', 'plasma', 'magma', 'cividis', 'Blues', 'Greens'
        *   发散: 'coolwarm', 'RdBu_r', 'BrBG'
    *   `sns.axes_style(style=None, rc=None)` / `sns.plotting_context(context=None, font_scale=1, rc=None)`: 用于 `with` 语句的上下文管理器，临时改变样式。
        ```python
        # Example of temporary style change
        # with sns.axes_style("white"):
        #     sns.histplot(data, x='value')
        ```
    *   `sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)`: 移除顶部和右侧的轴脊柱（spines），使图形更简洁。

### 库 3: Plotly (Plotly Express & Graph Objects) - 现代交互式图表

*   **定位**: 用于创建丰富的、交互式的、可发布到 Web 的图表。它有两个主要的API级别：
    *   **Plotly Express (`px`)**: 更高级、简洁的接口，通常只需要一行代码就能创建复杂的图表。是 Plotly 推荐的起点，特别适合快速制作和探索数据。数据通常需要是 "long-form" 或 "tidy" 格式的 Pandas DataFrame。
    *   **Plotly Graph Objects (`go`)**: 更底层、更灵活的接口，可以对图表的每个细节进行完全控制，类似于 Matplotlib 的面向对象接口。Plotly Express 内部也是使用 Graph Objects 构建图表。当需要高度自定义或 Plotly Express 不支持的图表类型时使用。
*   **核心导入**:
    *   `import plotly.express as px`
    *   `import plotly.graph_objects as go`
    *   `from plotly.subplots import make_subplots` (用于创建包含多个子图的 `go.Figure`)

#### Plotly Express (`px`) 核心组件与概念:

1.  **绘图函数 (返回 `plotly.graph_objects.Figure` 对象)**: 大多数函数都接受 Pandas DataFrame 作为第一个参数，然后通过列名指定 `x`, `y`, `color`, `size` 等。
    *   `px.scatter(data_frame, x, y, color, symbol, size, text, hover_name, hover_data, facet_row, facet_col, animation_frame, log_x, log_y, ...)`: 散点图。
    *   `px.line(data_frame, x, y, color, line_group, line_dash, symbol, text, hover_name, hover_data, facet_row, facet_col, ...)`: 线图。`line_group` 用于区分不应连接的线段。
    *   `px.bar(data_frame, x, y, color, orientation, barmode, text, hover_name, hover_data, facet_row, facet_col, ...)`: 条形图。
        *   `orientation`: ('v' 或 'h')
        *   `barmode`: ('group', 'overlay', 'relative'/'stack')
        *   `text_auto`: True 或格式字符串 (如 `'.2s'`)，用于在条形上显示数值。
    *   `px.histogram(data_frame, x, y, color, nbins, marginal, histfunc, histnorm, cumulative, barmode, ...)`: 直方图。
        *   `marginal`: ('rug', 'box', 'violin') 在边缘添加分布图。
        *   `histfunc`: ('count', 'sum', 'avg', 'min', 'max')
        *   `histnorm`: ('percent', 'probability', 'density', 'probability density')
    *   `px.box(data_frame, x, y, color, points, notched, hover_data, ...)`: 箱线图。`points` ('all', 'outliers', False).
    *   `px.violin(data_frame, x, y, color, box, points, hover_data, ...)`: 小提琴图。
    *   `px.density_heatmap(data_frame, x, y, z, histfunc, histnorm, color_continuous_scale, ...)`: 二维密度热力图 (类似2D直方图)。
    *   `px.density_contour(data_frame, x, y, z, histfunc, histnorm, ...)`: 二维密度等高线图。
    *   `px.imshow(img, color_continuous_scale, zmin, zmax, aspect, origin, binary_string, ...)`: 热力图/图像。`img` 可以是 NumPy 数组, PIL Image, xarray DataArray。
    *   `px.scatter_matrix(data_frame, dimensions, color, symbol, size, title, ...)`: 散点图矩阵 (类似 `sns.pairplot`)。
    *   `px.parallel_coordinates(data_frame, dimensions, color, color_continuous_scale, labels, ...)`: 平行坐标图，用于高维数据可视化。
    *   `px.parallel_categories(data_frame, dimensions, color, ...)`: 平行类别图 (类似平行坐标，但用于分类数据)。
    *   地理图: `px.scatter_geo()`, `px.line_geo()`, `px.choropleth()` (区域着色图), `px.scatter_mapbox()`, `px.line_mapbox()`, `px.choropleth_mapbox()`, `px.density_mapbox()` (基于 Mapbox 瓦片)。
    *   3D 图: `px.scatter_3d()`, `px.line_3d()`, `px.mesh_3d()`. (Plotly Express 没有直接的 `surface_3d`, 但 `go.Surface` 可以用)。
    *   树形图和旭日图: `px.treemap()`, `px.sunburst()`.
    *   漏斗图: `px.funnel()`, `px.funnel_area()`.

2.  **核心参数 (大部分 Plotly Express 函数通用)**:
    *   `data_frame`: Pandas DataFrame (虽然也接受类字典对象，但DataFrame最常用，推荐使用长格式数据)。
    *   `x`, `y`, `z`: DataFrame 中的列名，或直接是数据序列。
    *   `color`: 按此列的值为标记着色 (分类或连续)。
    *   `symbol`: 按此列的值为标记指定不同形状 (分类)。
    *   `size`: 按此列的值为标记指定不同大小 (连续)。
    *   `text`: 指定在标记旁显示的文本。
    *   `hover_name`, `hover_data`: 控制鼠标悬停时显示的信息。`hover_data` 可以是列名列表或字典。
    *   `facet_row`, `facet_col`: 按这些列的值创建分面（子图网格）。
    *   `facet_col_wrap`: 当只有 `facet_col` 时，指定每行多少个子图。
    *   `animation_frame`, `animation_group`: 创建动画。`animation_frame` 指定时间序列或类别，`animation_group` 用于在帧之间连接对象。
    *   `labels`: (字典) 自定义轴标签和图例标题，如 `labels={'csi_scores':'CSI Score', 'year_col':'Year'}`。
    *   `title`: 图表标题。
    *   `template`: Plotly 主题模板 ('plotly', 'plotly_white', 'plotly_dark', 'ggplot2', 'seaborn', 'simple_white', 'none')。
    *   `width`, `height`: 图表的宽度和高度（像素）。
    *   `color_discrete_sequence`, `color_continuous_scale`, `symbol_sequence`, `line_dash_sequence`: 自定义颜色、符号、线型序列。
        *   `color_discrete_map`: 字典，将特定值映射到特定颜色。
        *   `color_continuous_midpoint`: 连续色阶的中间点值。
    *   `category_orders`: 字典，指定分类变量中类别的顺序，如 `category_orders={'day': ['Thur', 'Fri', 'Sat', 'Sun']}`。
    *   `log_x`, `log_y`: (布尔) 设置对数轴。

#### Plotly Graph Objects (`go`) 核心组件与概念:

1.  **`go.Figure` 对象**:
    *   **概念**: Plotly 图表的容器，包含数据 (traces) 和布局 (layout)。
    *   **创建**:
        *   `fig = go.Figure()`
        *   `fig = go.Figure(data=trace_object_or_list, layout=layout_object_or_dict)`
        *   `fig = make_subplots(rows=r, cols=c, subplot_titles=[...], specs=[[{'type':'xy'}, {'type':'polar'}]], ...)`: 创建带子图的 Figure。
    *   **方法**:
        *   `fig.add_trace(trace_object, row=None, col=None)`: 添加一个“轨迹”(trace)，如 `go.Scatter`, `go.Bar`。如果 Figure 是用 `make_subplots` 创建的，可以用 `row` 和 `col` 指定子图位置。
        *   `fig.add_traces(list_of_trace_objects, rows=None, cols=None)`: 添加多个轨迹。
        *   `fig.update_layout(...)`: 更新图表的布局选项 (标题、轴、图例等)。参数是布局属性的键值对或一个 `go.Layout` 对象。
        *   `fig.update_xaxes(...)`, `fig.update_yaxes(...)`: 更新特定轴或所有匹配轴的属性。可以使用 `selector` 或 `row`/`col` 来定位轴。
        *   `fig.update_traces(...)`: 更新特定轨迹或所有匹配轨迹的属性。可以使用 `selector` 来定位轨迹。
        *   `fig.show(renderer=None, config=None)`: 显示图表 (在Jupyter环境中或打开浏览器)。`renderer` 可以指定渲染器 (如 'notebook', 'colab', 'browser')。`config` 字典可以控制 Plotly.js 的行为 (如禁用模式栏按钮)。
        *   `fig.write_html(filename, full_html=True, include_plotlyjs='cdn')`: 保存为 HTML 文件。
        *   `fig.write_image(filename_or_stream, format=None, width=None, height=None, scale=None)`: 保存为静态图片 (png, jpg, svg, pdf, eps - 需要安装 `kaleido` 包: `pip install -U kaleido`)。
        *   `fig.to_json()` / `fig.from_json()`: 序列化和反序列化 Figure。

2.  **Trace 对象 (例如 `go.Scatter`, `go.Bar`, `go.Heatmap`, etc.)**:
    *   **概念**: 代表图表中的一组数据及其可视化方式 (如一条线、一组条形、一个热力图等)。每个 trace 都是一个字典或 `go.TraceType` 的实例。
    *   **创建与属性 (以 `go.Scatter` 为例)**:
        *   `trace = go.Scatter(x=x_data, y=y_data, mode='lines+markers+text', name='Trace 1', text=text_labels, textposition='top center', line=dict(color='red', width=2, dash='dash'), marker=dict(symbol='circle', size=8, color='blue', colorscale='Viridis', showscale=True, colorbar=dict(title='Colorbar')), fill='tozeroy', fillcolor='rgba(0,100,80,0.2)')`
            *   `mode`: ('lines', 'markers', 'text', 'lines+markers', 'none', etc.)
            *   `name`: 图例中显示的名称。
            *   `line`: (字典) 控制线条属性 (`color`, `width`, `dash`).
            *   `marker`: (字典) 控制标记属性 (`symbol`, `size`, `color`, `opacity`, `colorscale`, `colorbar`).
            *   `fill`: ('none', 'tozeroy', 'tozerox', 'tonexty', 'tonextx', 'toself', 'tonext') 控制区域填充。
            *   `hoverinfo`: ('x', 'y', 'z', 'text', 'name', 'all', 'none', 'skip') 控制悬停信息。
            *   `customdata`: 附加数据，可在 `hovertemplate` 中使用。
            *   `hovertemplate`: 自定义悬停文本的 HTML 模板。
    *   **其他常用 Trace 类型**:
        *   `go.Bar(x, y, orientation, marker_color, text, textposition)`
        *   `go.Histogram(x, y, nbinsx, histnorm, cumulative_enabled)`
        *   `go.Box(y, x, name, boxpoints, jitter, pointpos, notched)`
        *   `go.Violin(y, x, name, side, box_visible, meanline_visible)`
        *   `go.Heatmap(z, x, y, colorscale, zmin, zmax, colorbar)`
        *   `go.Contour(z, x, y, contours_coloring, line_width, colorbar)`
        *   `go.Surface(z, x, y, colorscale, scene)` (用于3D曲面)
        *   `go.Choropleth(locations, z, geojson, locationmode)`
        *   `go.Scatter3d(x, y, z, mode, marker_size, line_color)`
        *   `go.Ohlc(open, high, low, close, x)` (金融K线图)
        *   `go.Candlestick(open, high, low, close, x)` (金融蜡烛图)
        *   `go.Table(header=dict(values=...), cells=dict(values=...))`

3.  **Layout 对象/字典 (`go.Layout` 或直接用字典)**:
    *   **概念**: 控制图表的整体外观和非数据元素 (标题、轴、图例、注释、形状、颜色、字体、边距等)。
    *   **常用属性 (作为 `go.Layout` 的参数或 `fig.update_layout` 的参数)**:
        *   `title_text` (或 `title=dict(text='My Title', x=0.5, y=0.9, xanchor='center', yanchor='top')`)
        *   `xaxis_title_text` (或 `xaxis=dict(title='X Axis Label', type='log', range=[min,max], showgrid=True, gridcolor='lightgrey', zeroline=False, tickvals=[...], ticktext=[...], tickangle=45)`)
        *   `yaxis_title_text` (或 `yaxis=dict(...)`)
        *   `xaxis_range`, `yaxis_range`: `[min, max]`
        *   `xaxis_type`, `yaxis_type`: ('linear', 'log', 'date', 'category', 'multicategory')
        *   `legend_title_text` (或 `legend=dict(title='Legend', orientation='h', x=0, y=1.1, bgcolor='rgba(255,255,255,0.5)', bordercolor='black', borderwidth=1)`)
        *   `font_family`, `font_size`, `font_color` (全局字体设置)
        *   `paper_bgcolor` (图表外部背景色), `plot_bgcolor` (绘图区域背景色)
        *   `margin`: (字典) `dict(l=left_margin, r=right_margin, t=top_margin, b=bottom_margin, pad=padding)`
        *   `annotations`: (列表 of 字典) 添加文本标注。每个字典定义一个标注 (`x`, `y`, `text`, `showarrow`, `arrowhead`, `ax`, `ay`).
        *   `shapes`: (列表 of 字典) 添加线条、矩形、圆形等形状。每个字典定义一个形状 (`type`, `x0`, `y0`, `x1`, `y1`, `line_color`).
        *   `images`: (列表 of 字典) 添加背景图片。
        *   `updatemenus`: (列表 of 字典) 添加下拉菜单等交互控件，用于动态修改图表。
        *   `sliders`: (列表 of 字典) 添加滑块控件，常用于动画或参数调整。
        *   `template`: (字符串或 `go.layout.Template` 对象) 应用预设或自定义主题。
        *   `hovermode`: ('x', 'y', 'closest', 'x unified', 'y unified', False) 控制悬停行为。
        *   `dragmode`: ('zoom', 'pan', 'select', 'lasso', 'orbit', False) 控制鼠标拖拽行为。
        *   `grid`: (在 `xaxis`, `yaxis` 内) `dict(rows=R, columns=C, pattern='independent')` 用于子图布局 (较少直接用，`make_subplots` 更方便)。

---

这个概览应该能让你对这三个库有一个初步的印象和核心组件的了解。**Matplotlib 是基础和精细控制，Seaborn 是统计美化与便捷，Plotly 是交互现代与Web友好。**

*   **选择 Matplotlib 当**:
    *   你需要对图表的每一个元素进行像素级的完美控制。
    *   你需要创建非常规或高度定制的图表类型。
    *   你主要关注静态图表的出版质量。
*   **选择 Seaborn 当**:
    *   你主要进行统计数据可视化和探索性数据分析。
    *   你想快速生成美观的、信息丰富的标准统计图表 (如分布图、关系图、分类图)。
    *   你喜欢它简洁的API和优雅的默认样式，同时仍希望能够用Matplotlib进行微调。
*   **选择 Plotly 当**:
    *   你需要创建交互式图表 (缩放、平移、悬停信息、动态更新)。
    *   你需要将图表嵌入网页或制作在线仪表盘。
    *   你喜欢 Plotly Express 的简洁语法，特别是处理 Pandas DataFrame 时。
    *   你需要 3D 图表或特定类型的图表 (如旭日图、树状图、地理空间图) 且希望有良好的交互性。

通常，这些库可以协同工作。例如，用 Seaborn 快速生成一个基本图表，然后用 Matplotlib 的方法进行精细调整。Plotly Express 生成的图表对象也可以通过其 `graph_objects` 结构进行深度定制。



## 第一部分：Matplotlib 精细化作图

Matplotlib 是 Python 科学计算生态中最基础也是最重要的绘图库。它提供了强大的底层绘图能力，几乎可以定制图表的每一个元素。对于需要发表的论文图，Matplotlib 的面向对象接口是实现精细控制的首选。

### 第1步：准备工作与绘图基础

#### 1.1 导入库

标准的导入方式如下：

```python
import matplotlib.pyplot as plt
import numpy as np # 通常用于生成示例数据或处理数值数据
```

在 Jupyter Notebook 或 Google Colab 中，通常还会加上一行魔法命令，让图像直接在输出单元格显示：

```python
%matplotlib inline
```
如果是在普通的 Python 脚本中运行，最后需要调用 `plt.show()` 来显示图像。

#### 1.2 理解 Figure 和 Axes (再次强调)

这是 Matplotlib 面向对象绘图的核心：

*   **Figure (`fig`)**: 整个图表的画布、顶层容器。你可以设置它的大小、背景色等。
*   **Axes (`ax`)**: 画布上的一个子区域，代表一个独立的坐标系（或子图）。实际的绘图操作（画线、散点、设置标签等）都是在 Axes 对象上进行的。一个 Figure 可以包含一个或多个 Axes。

#### 1.3 创建第一个简单的图表 (pyplot 风格 vs 面向对象风格)

让我们用一个简单的例子来对比你可能已经熟悉的 `pyplot` 风格和我们将要重点学习的面向对象 (OO) 风格。

**数据准备 (以你的项目为例，假设我们要画不同降雨产品的时间序列)**

由于你的项目数据比较复杂，我们先用模拟数据来演示基本绘图，后续再结合你的 `.mat` 文件数据。

```python
# 模拟时间数据 (例如，表示一个月的天数)
days = np.arange(1, 31)

# 模拟三种降雨产品在这些天内的日降雨量 (mm)
# 实际中，你会从你的 .mat 文件加载这些数据
product_A_rain = np.abs(np.random.normal(5, 5, 30)) # 平均5mm，标准差5，取绝对值确保非负
product_B_rain = np.abs(np.random.normal(6, 7, 30)) + np.sin(days/5) * 3
product_C_rain_truth = product_A_rain * 0.8 + np.random.normal(0, 2, 30) # 假设C是观测真值
product_C_rain_truth[product_C_rain_truth < 0] = 0 # 确保非负
```

**1.3.1 Pyplot 风格 (快速绘图，隐式管理 Figure 和 Axes)**

```python
plt.figure(figsize=(10, 5)) # 可选：创建一个新的Figure，并指定大小

plt.plot(days, product_A_train, label='Product A (Satellite)')
plt.plot(days, product_B_train, label='Product B (Satellite)', linestyle='--')
plt.plot(days, product_C_rain_truth, label='Product C (Ground Truth)', color='black', linewidth=2)

plt.title('Daily Rainfall Comparison (Pyplot Style)')
plt.xlabel('Day of Month')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True)

# plt.show() # 如果在脚本中运行
```

**代码解释 (Pyplot 风格):**

*   `plt.figure(figsize=(10, 5))`: `pyplot` 模块隐式创建了一个 Figure 对象。`figsize` 参数设置画布大小。
*   `plt.plot(...)`: 每次调用 `plt.plot` 都会在“当前” Axes 上绘制一条线。如果还没有 Axes，`pyplot` 会自动创建一个。
*   `plt.title()`, `plt.xlabel()`, `plt.ylabel()`, `plt.legend()`, `plt.grid()`: 这些函数都作用于“当前” Axes。

**这种方式的优点是代码简洁，适合快速探索。缺点是当图表变复杂或需要多个子图时，对特定元素的控制会变得不直观。**

**1.3.2 面向对象 (OO) 风格 (显式创建和管理 Figure 和 Axes)**

这是我们推荐的、用于精细化控制和论文图表的方式。

```python
# 1. 创建 Figure 对象 (画布)
fig = plt.figure(figsize=(10, 5)) # figsize设置画布大小为10x5英寸

# 2. 在 Figure 上添加 Axes 对象 (坐标系/子图)
# fig.add_subplot(1, 1, 1) 表示创建一个1行1列的子图网格，当前操作第1个子图。
ax = fig.add_subplot(1, 1, 1)

# 3. 使用 Axes 对象 (ax) 的方法进行绘图
ax.plot(days, product_A_train, label='Product A (Satellite)')
ax.plot(days, product_B_train, label='Product B (Satellite)', linestyle='--')
ax.plot(days, product_C_rain_truth, label='Product C (Ground Truth)', color='black', linewidth=2)

# 4. 使用 Axes 对象 (ax) 的方法设置属性
ax.set_title('Daily Rainfall Comparison (OO Style)') # 注意是 set_title
ax.set_xlabel('Day of Month')
ax.set_ylabel('Rainfall (mm)')
ax.legend()
ax.grid(True)

# plt.show() # 如果在脚本中运行
```

**代码解释 (OO 风格):**

*   `fig = plt.figure(figsize=(10, 5))`: 我们显式地创建了一个 `Figure` 对象，并将其赋值给变量 `fig`。现在我们可以完全控制这个画布。
*   `ax = fig.add_subplot(1, 1, 1)`: 我们在 `fig` 上显式地添加了一个 `Axes` 对象，并将其赋值给变量 `ax`。所有后续的绘图和设置都将针对这个 `ax`。
*   `ax.plot(...)`: 调用 `ax` 对象的 `plot` 方法。
*   `ax.set_title(...)`, `ax.set_xlabel(...)` 等: 调用 `ax` 对象的相应设置方法。注意很多方法名前面加了 `set_`。

**1.3.3 更简洁的 OO 风格创建方式: `plt.subplots()`**

`plt.subplots()` 是一个非常有用的便捷函数，它可以同时创建 Figure 和一个或多个 Axes 对象。

```python
# 同时创建 Figure 和一个 Axes 对象
# fig, ax = plt.subplots() # 默认创建一个1x1的子图
fig, ax = plt.subplots(figsize=(10, 5)) # 也可以直接指定figsize

# 后续操作与上面的 OO 风格完全一样
ax.plot(days, product_A_train, label='Product A (Satellite)')
ax.plot(days, product_B_train, label='Product B (Satellite)', linestyle='--')
ax.plot(days, product_C_rain_truth, label='Product C (Ground Truth)', color='black', linewidth=2)

ax.set_title('Daily Rainfall Comparison (subplots OO Style)')
ax.set_xlabel('Day of Month')
ax.set_ylabel('Rainfall (mm)')
ax.legend()
ax.grid(True)

# plt.show()
```
**`plt.subplots()` 是最推荐的开始面向对象绘图的方式。**

**你的任务与思考：**

1.  **运行以上三种方式的代码块**，确认你理解它们之间的区别和联系，以及输出结果是否一致。
2.  **重点理解 `fig` 和 `ax`**：`fig` 是画板，`ax` 是画板上的画框。我们所有的画画动作（`ax.plot`）和装饰（`ax.set_title`）都是在画框 `ax` 上进行的。
3.  **注意函数名的变化**: 从 `plt.title()` 到 `ax.set_title()`。这是一个常见的模式。

### 第2步：多子图 (`subplots`) 与布局管理

在科研论文中，我们经常需要将多个相关的图表并列展示，以便于比较。Matplotlib 的多子图功能使这变得容易。

#### 2.1 使用 `plt.subplots()` 创建多子图网格

`plt.subplots(nrows, ncols)` 可以创建一个包含 `nrows` 行 `ncols` 列的子图网格。

*   它返回一个 `Figure` 对象 (`fig`) 和一个包含所有 `Axes` 对象的 NumPy 数组 (`axs`)。
*   你可以通过索引访问 `axs` 中的每一个子图，例如 `axs[0, 0]` 表示第一行第一列的子图。
*   如果 `nrows` 或 `ncols` 为1，`axs` 可能是一维数组 (如 `axs[0]`) 或单个 Axes 对象 (如果 `squeeze=True`，这是默认行为)。

**示例：创建 2x2 的子图布局，分别展示不同产品的降雨量和一些统计信息**

```python
# 沿用之前的模拟数据
days = np.arange(1, 31)
product_A_rain = np.abs(np.random.normal(5, 5, 30))
product_B_rain = np.abs(np.random.normal(6, 7, 30)) + np.sin(days/5) * 3
product_C_rain_truth = product_A_rain * 0.8 + np.random.normal(0, 2, 30)
product_C_rain_truth[product_C_rain_truth < 0] = 0

# 1. 创建 Figure 和 2x2 的 Axes 数组
# figsize 控制整个 Figure 的大小
# sharex=True 和 sharey=True 可以让所有子图共享x轴和y轴的范围和刻度，
# 这在比较趋势时非常有用。也可以单独设置为 'col' 或 'row'。
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=False, sharey=False)
# axs 现在是一个 2x2 的 NumPy 数组:
# axs[0, 0]  axs[0, 1]
# axs[1, 0]  axs[1, 1]

# --- 子图 1: Product A 时间序列 ---
axs[0, 0].plot(days, product_A_rain, color='blue', label='Product A')
axs[0, 0].set_title('Product A Rainfall')
axs[0, 0].set_ylabel('Rainfall (mm)')
axs[0, 0].grid(True, linestyle=':')
axs[0, 0].legend(loc='upper right', fontsize='small') # loc设置图例位置，fontsize调小

# --- 子图 2: Product B 时间序列 ---
axs[0, 1].plot(days, product_B_rain, color='green', label='Product B')
axs[0, 1].set_title('Product B Rainfall')
axs[0, 1].grid(True, linestyle=':')
axs[0, 1].legend(loc='upper right', fontsize='small')

# --- 子图 3: Product C (Truth) 时间序列 ---
axs[1, 0].plot(days, product_C_rain_truth, color='black', label='Product C (Truth)')
axs[1, 0].set_title('Product C (Ground Truth) Rainfall')
axs[1, 0].set_xlabel('Day of Month')
axs[1, 0].set_ylabel('Rainfall (mm)')
axs[1, 0].grid(True, linestyle=':')
axs[1, 0].legend(loc='upper right', fontsize='small')

# --- 子图 4: 产品A vs 产品C 的散点图 (示例) ---
# 为了展示不同类型的图，我们在这里画一个散点图
axs[1, 1].scatter(product_A_rain, product_C_rain_truth, alpha=0.6, edgecolors='w', linewidth=0.5)
axs[1, 1].plot([0, max(product_A_rain.max(), product_C_rain_truth.max())], # 添加 y=x 参考线
             [0, max(product_A_rain.max(), product_C_rain_truth.max())],
             color='red', linestyle='--', linewidth=1)
axs[1, 1].set_title('Product A vs. Product C')
axs[1, 1].set_xlabel('Product A Rainfall (mm)')
axs[1, 1].set_ylabel('Product C Rainfall (mm)')
axs[1, 1].grid(True, linestyle=':')
axs[1, 1].axis('equal') # 设置x,y轴等比例，有助于观察偏差

# --- 调整整体布局 ---
# fig.suptitle('Multi-Product Rainfall Analysis', fontsize=16) # 给整个Figure一个主标题
# plt.tight_layout() 会自动调整子图参数，使它们适应Figure区域，防止重叠。
# rect=[0, 0, 1, 0.95] 可以为主标题留出空间 (上边距0.95)
fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # 稍微调整底部和顶部边距

# plt.show()
```

**代码解释与思考：**

*   `fig, axs = plt.subplots(2, 2, ...)`: 创建了一个 2x2 的子图网格。`axs` 是一个 2D NumPy 数组。
*   **访问子图**: 我们使用 `axs[row_index, col_index]` 来选择要操作的特定子图，例如 `axs[0, 0]` 是左上角的第一个子图。
*   **独立设置**: 每个 `Axes` 对象 (`axs[i, j]`) 都可以独立地调用 `plot`, `set_title`, `set_xlabel`, `legend` 等方法。
*   **`sharex` 和 `sharey`**:
    *   如果你设置 `sharex=True`，所有子图将共享相同的X轴范围和刻度。X轴的标签和刻度数字通常只会显示在最底部的子图上。
    *   如果你设置 `sharey=True`，所有子图将共享相同的Y轴范围和刻度。Y轴的标签和刻度数字通常只会显示在最左侧的子图上。
    *   这在你的项目中会非常有用，例如，当你比较不同模型（或特征版本V1-V6）在同一指标（如CSI）上的表现时，使用共享的Y轴能让比较更直观。
    *   **尝试修改**: 将 `sharex=False, sharey=False` 改为 `sharex=True, sharey=True`，然后观察X轴和Y轴标签和刻度的变化。
*   **`axs.flat`**: 如果你想用一个简单的 `for` 循环来遍历所有子图，可以使用 `axs.flat`。它会把多维的 `axs` 数组“展平”成一个一维的迭代器。

    ```python
    # 使用 axs.flat 迭代
    # fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    # for i, ax_single in enumerate(axs.flat):
    #     ax_single.plot(days, product_A_rain + i*2) # 简单画点东西
    #     ax_single.set_title(f'Subplot {i+1}')
    # fig.tight_layout()
    # plt.show()
    ```
    这在你需要对所有子图应用相似操作时非常方便。

#### 2.2 调整子图间距

*   **`plt.tight_layout()` 或 `fig.tight_layout()`**: 这是最常用的自动调整子图参数以适应画布的方法，它可以很好地处理标签重叠问题。
*   **`plt.subplots_adjust()` 或 `fig.subplots_adjust()`**: 提供更手动的控制。
    *   `fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)`
        *   `left`, `bottom`, `right`, `top`: 子图区域边界相对于 Figure 边缘的位置（0到1之间的分数）。
        *   `wspace`: 子图之间的宽度间距（以平均轴宽度的分数为单位）。
        *   `hspace`: 子图之间的高度间距（以平均轴高度的分数为单位）。
    *   **示例**: `fig.subplots_adjust(hspace=0.4, wspace=0.3)` 增加子图间的垂直和水平间距。
    *   **通常在 `tight_layout()` 效果不完美时使用。**

#### 2.3 针对你的项目 (思考与应用)

在你的 README 中，有很多地方可以用到多子图：

*   **2.3.2 节 (长江流域 XGBoost 模型性能迭代)**:
    *   你可以创建一个 2x3 (或 3x2) 的子图网格，每个子图展示一个特征版本 (V1-V6) 的关键指标随**预测概率阈值**变化的曲线（例如，POD, FAR, CSI vs. Threshold）。这样可以非常直观地比较不同特征集在不同决策点上的表现。
    *   或者，为每个特征版本 (V1-V6) 创建一个子图，子图内用不同颜色的线/点表示 POD, FAR, CSI 在某个固定概率阈值（如0.5）下的值。
*   **第 6 节 (长江流域多源降雨产品性能评估)** 和 **第 8 节 (全国范围)**:
    *   对于每个产品 (CMORPH, CHIRPS, ...)，可以创建一个子图，显示其 POD, FAR, CSI 随分类阈值变化的曲线。
    *   或者，创建一个 3xN (N为产品数量) 的网格，第一行是所有产品的 POD vs. 阈值，第二行是 FAR vs. 阈值，第三行是 CSI vs. 阈值。使用 `sharey='row'` 可以让同一行的Y轴共享，方便比较。
*   **第 10 节 (各降雨产品逐年性能指标的空间统计特征)**:
    *   例如，对于CMORPH产品的POD，你可以创建一个 1xN (N为统计量个数，如均值、标准差、偏度、峰度) 的子图，每个子图展示该统计量随年份和阈值变化的曲线或热力图。
    *   对于产品间的空间相关性，也可以用子图阵列来展示不同产品对之间的相关系数矩阵（或其对角线元素）。

**你的任务与思考：**

1.  **运行多子图示例代码**，理解 `axs` 数组的用法。
2.  **实验 `sharex` 和 `sharey` 参数**：设置为 `True`, `'col'`, `'row'`，观察效果。思考在你的项目中，什么时候应该共享轴。
3.  **尝试 `axs.flat`**：用它来给每个子图设置一个简单的标题或属性。
4.  **尝试 `fig.subplots_adjust()`**：在 `fig.tight_layout()` 之后（或之前）加入 `fig.subplots_adjust(hspace=0.5)`，看看子图垂直间距如何变化。
5.  **构思你的项目图表**:
    *   选择 README 中的一个表格数据（例如 2.3.2 节的 V1-V6 在 0.5 阈值下的 POD, FAR, CSI）。
    *   尝试用多子图（例如，每个指标一个子图，或者每个版本一个子图）来可视化这些数据。**不需要现在就完美实现，先有一个大致的草图或想法。**

当我们完成这部分后，你将对如何组织和布局多个相关的图表有一个很好的掌握。接下来，我们可以深入到对图表元素的精细控制，比如线条样式、颜色、字体、图例细节等，让你的图表更符合论文要求。

### 第3步：精细控制图表元素 (让图表“说话”)

现在我们已经掌握了如何创建单个和多个子图的框架。接下来的关键是精细控制图表中的每一个视觉元素，使其准确、清晰、专业地传达信息。这是制作论文级图表的核心。

我们将围绕一个**增强版的单子图**（基于之前展示V1-V6性能的图）来进行讲解，因为单子图的元素控制是多子图的基础。之后，这些技巧可以应用到你创建的每一个子图中。

#### 3.1 获取和修改线条属性 (`Line2D` 对象)

`ax.plot()` 函数返回一个包含 `Line2D` 对象的列表 (通常只有一个元素，除非一次 `plot` 画多条线)。我们可以获取这个对象，然后修改它的属性。

```python
# 假设我们回到之前那个展示V1-V6性能的单子图例子
# (数据定义部分省略，沿用之前的 feature_versions, csi_scores, far_scores, pod_scores)
# 我们只画CSI和FAR，简化一下

fig, ax1 = plt.subplots(figsize=(10, 6)) # 使用推荐的 plt.subplots()

# --- 绘制 CSI ---
color_csi = 'darkred' # 使用更深的红色
# line_csi 是一个 Line2D 对象 (因为 ax.plot 返回列表，我们用逗号解包)
line_csi, = ax1.plot(feature_versions, csi_scores,
                     color=color_csi,
                     marker='o', markersize=7, markerfacecolor='lightcoral', markeredgecolor=color_csi,
                     linestyle='-', linewidth=2,
                     label='CSI @0.5 Thr')
ax1.set_xlabel('Feature Set Version', fontsize=12, fontweight='bold')
ax1.set_ylabel('CSI (Critical Success Index)', color=color_csi, fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelcolor=color_csi, labelsize=10, direction='in')
ax1.tick_params(axis='x', rotation=30, labelsize=10, direction='in')
ax1.grid(True, linestyle=':', alpha=0.6, color='gray') # 更细的网格线

# --- 绘制 FAR (第二个Y轴) ---
ax2 = ax1.twinx()
color_far = 'darkblue' # 使用更深的蓝色
line_far, = ax2.plot(feature_versions, far_scores,
                     color=color_far,
                     marker='s', markersize=6, markerfacecolor='lightblue', markeredgecolor=color_far,
                     linestyle='--', linewidth=2,
                     label='FAR @0.5 Thr')
ax2.set_ylabel('FAR (False Alarm Ratio)', color=color_far, fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor=color_far, labelsize=10, direction='in')

# --- 修改已绘制线条的属性 (示例) ---
# 假设我们想让CSI的V6 (Optuna)那个点特别突出
# 'V6 (Optuna)' 是最后一个点，索引为 -1
# line_csi.set_markersize([7, 7, 7, 7, 7, 7, 12]) # 让最后一个点变大 (需要matplotlib 3.3+)
# 或者更通用的方式是重新绘制最后一个点，或者用 annotate 标记 (后面讲)

# 如果你想在绘制后改变颜色：
# line_csi.set_color('orange')

fig.suptitle('XGBoost Model Performance Comparison', fontsize=14, fontweight='bold')
lines = [line_csi, line_far]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=10, frameon=False)

fig.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()
```

**代码解释与可调参数 (`Line2D`):**

*   `line_csi, = ax1.plot(...)`: 我们获取了 `Line2D` 对象。
*   **颜色 (`color`)**: 如 `'red'`, `'#FF0000'`, `(1,0,0)`。
*   **标记 (`marker`)**:
    *   样式: `'.', ',', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_', 'None'`。
    *   大小: `markersize` 或 `ms` (浮点数)。
    *   填充色: `markerfacecolor` 或 `mfc`.
    *   边框色: `markeredgecolor` 或 `mec`.
    *   边框宽度: `markeredgewidth` 或 `mew`.
*   **线条 (`linestyle`, `linewidth`)**:
    *   样式 (`linestyle` or `ls`): `'-'`, `'--'`, `'-.'`, `':'`, `'None'` (不画线，只画标记), `'solid'`, `'dashed'`, `'dashdot'`, `'dotted'`。
    *   宽度 (`linewidth` or `lw`): 浮点数。
*   **透明度 (`alpha`)**: 0 (完全透明) 到 1 (完全不透明)。
*   **绘图顺序 (`zorder`)**: 整数，值越大，越晚绘制 (即在其他元素之上)。

#### 3.2 定制坐标轴 (Axis Customization)

这是论文图中非常重要的一环，包括范围、标签、刻度、轴线本身。

**3.2.1 轴范围 (`set_xlim`, `set_ylim`)**

```python
# 接上例
# 假设我们想让CSI的Y轴范围更紧凑一些
max_csi = np.max(csi_scores)
min_csi_display = 0.7 # 假设我们关注0.7以上的CSI
ax1.set_ylim(min_csi_display, max_csi + 0.05) # Y轴下限设为0.7，上限比最大值略高一点

# FAR 通常在 0 到一个较小的值之间，比如 0 到 0.1 或 0.15
max_far = np.max(far_scores)
ax2.set_ylim(0, max_far + 0.01)
# 也可以反转Y轴 (例如，有时错误率越低越好，希望它在上方)
# ax2.invert_yaxis() # 但FAR本身就是越小越好，所以通常不需要反转

# 对于X轴，如果feature_versions是数值，也可以设置xlim
# ax1.set_xlim(-0.5, len(feature_versions) - 0.5) # 给两边留点空隙
```

**3.2.2 轴标签 (`set_xlabel`, `set_ylabel`)**

*   `fontsize`: 字体大小。
*   `fontweight`: `'normal'`, `'bold'`, `'heavy'`, `'light'`。
*   `color`: 标签颜色。
*   `labelpad`: 标签与轴线的距离。

**3.2.3 刻度 (`set_xticks`, `set_yticks`, `set_xticklabels`, `set_yticklabels`, `tick_params`)**

```python
# --- 刻度设置 ---
# X轴刻度: feature_versions 已经是字符串列表，它们会自动作为刻度标签。
# 如果X是数值，我们可以手动设置刻度和标签：
# ax1.set_xticks([0, 1, 2, 3, 4, 5, 6])
# ax1.set_xticklabels(feature_versions, rotation=30, ha='right') # ha='right' 配合旋转标签

# Y轴刻度 (CSI - ax1)
csi_ticks = np.arange(0.7, max_csi + 0.05, 0.05) # 从0.7开始，每隔0.05一个刻度
ax1.set_yticks(csi_ticks)
ax1.set_yticklabels([f'{tick:.2f}' for tick in csi_ticks]) # 格式化刻度标签为两位小数

# Y轴刻度 (FAR - ax2)
far_ticks = np.arange(0, max_far + 0.01, 0.02) # 每隔0.02一个刻度
ax2.set_yticks(far_ticks)
ax2.set_yticklabels([f'{tick:.2f}' for tick in far_ticks])

# 全局刻度参数设置 (更精细)
# ax1.tick_params(axis='both', which='major', direction='in', length=6, width=1, colors='black', labelsize=10, pad=5)
# axis='both': 应用于x和y轴
# which='major': 应用于主刻度线 (还有 'minor' 和 'both')
# direction='in': 刻度线朝内 (还有 'out', 'inout')
# length, width: 刻度线长度和宽度
# colors: 刻度线和刻度标签的颜色 (如果labelcolor未单独设置)
# labelsize: 刻度标签字体大小
# pad: 刻度线与刻度标签之间的距离

# 如果要显示次刻度线 (minor ticks)
# ax1.minorticks_on()
# ax1.tick_params(which='minor', direction='in', length=3, width=0.5)
```
*   **`ax.tick_params()`** 是一个非常强大的方法，用于统一控制刻度线和刻度标签的各种属性。

**3.2.4 轴线 (`spines`)**

论文图通常更简洁，会隐藏顶部和右侧的轴线（除非有第二个Y轴）。

```python
# --- 轴线 (Spines) 设置 ---
# ax1 是主坐标系，它有左、下、上、右四条轴线
ax1.spines['top'].set_visible(False)    # 隐藏顶部的轴线
# ax1.spines['right'].set_visible(False)  # 如果没有ax2，也会隐藏右边的

# ax2 是第二个Y轴，它默认只显示右边的轴线。它的左、上、下轴线是和ax1共享且不可见的。
# ax3 (如果有的话) 也是类似，它有自己的右轴线。

# 可以改变轴线的颜色和宽度
ax1.spines['left'].set_color(color_csi)
ax1.spines['bottom'].set_color('black') # X轴通常是黑色
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)

ax2.spines['right'].set_color(color_far)
ax2.spines['right'].set_linewidth(1.5)

# (如果用了ax3，也类似设置 ax3.spines['right'])
```

#### 3.3 图例 (`legend`)

*   `loc`: 位置参数，如 `'best'`, `'upper right'`, `'lower left'`, `'center'`, `'upper center'` 等。
*   `bbox_to_anchor=(x, y, width, height)` 或 `bbox_to_anchor=(x, y)`: 精确控制图例位置。坐标通常是相对于 Axes 的 (0,0) 左下角，(1,1) 右上角。
*   `ncol`: 图例的列数。
*   `frameon`: (布尔值) 是否显示图例边框。论文图通常 `frameon=False` 更简洁。
*   `fontsize`: 图例字体大小。
*   `title`: 图例的标题。
*   `fancybox`: (布尔值) 是否用圆角边框。
*   `shadow`: (布尔值) 是否带阴影。

**在之前的例子中，我们已经用到了 `ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)`，这是一个很好的实践，将图例放在图表下方，避免遮挡数据。**

#### 3.4 网格线 (`grid`)

*   `ax.grid(True)` 或 `ax.grid(False)`: 显示或隐藏。
*   `linestyle` 或 `ls`: (`'-'`, `'--'`, `':'`, `'-.'`)。
*   `linewidth` 或 `lw`: 线宽。
*   `color`: 颜色。
*   `alpha`: 透明度。
*   `which`: (`'major'`, `'minor'`, `'both'`) 控制主网格还是次网格。
*   `axis`: (`'x'`, `'y'`, `'both'`) 控制哪个轴的网格。

**在之前的例子中 `ax1.grid(True, linestyle=':', alpha=0.6, color='gray')` 是一个不错的设置，使用细的点状灰色网格，不太抢眼。**

#### 3.5 文本与标注 (`text`, `annotate`)

**`ax.text(x, y, "string", ...)`**: 在图表的指定数据坐标 `(x,y)` 处添加文本。
    *   **`transform=ax.transAxes`**: 一个非常有用的参数。当设置了这个转换后，`(x,y)` 坐标将是相对于 `Axes` 本身的比例坐标，其中 `(0,0)` 是左下角，`(1,1)` 是右上角。这使得在固定位置（如角落）添加文本非常方便，不受数据范围变化的影响。

**`ax.annotate("text", xy=(point_x, point_y), xytext=(text_x, text_y), arrowprops=dict(...), ...)`**: 添加带箭头的标注，指向数据点 `xy`，文本位于 `xytext`。
    *   `arrowprops`: 一个字典，用于定义箭头的样式，例如 `dict(facecolor='black', shrink=0.05, width=1, headwidth=5, connectionstyle='arc3,rad=.2')`。
        *   `shrink`: 箭头尖端和尾端与目标点/文本的距离。
        *   `width`: 箭头主体宽度。
        *   `headwidth`: 箭头头部宽度。
        *   `connectionstyle`: 箭头路径样式，如 `'arc3,rad=.2'` (弧形)。

```python
# --- 添加文本和标注 (示例) ---
# 在图表右上角添加一些说明性文字 (使用Axes坐标)
ax1.text(0.98, 0.95, 'Data: Yangtze Basin\nModel: XGBoost\nProb. Thr: 0.5',
         transform=ax1.transAxes, # 使用Axes相对坐标
         fontsize=9,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round,pad=0.4', fc='whitesmoke', alpha=0.7, ec='gray')) # 带背景框

# 假设我们想高亮 V6 (Optuna) 的 CSI 值
idx_optuna = feature_versions.index('V6 (Optuna)')
csi_optuna = csi_scores[idx_optuna]

ax1.annotate(f'Best CSI: {csi_optuna:.4f}',
             xy=(idx_optuna, csi_optuna), # 箭头指向的点
             xytext=(idx_optuna - 1, csi_optuna + 0.03), # 文本放置的位置
             fontsize=10, color='darkred',
             arrowprops=dict(facecolor='darkred', shrink=0.05, width=0.5, headwidth=6,
                             connectionstyle="arc3,rad=.2"))

# plt.show() # 更新显示
```

**你的任务与思考：**

1.  **详细阅读每个部分的解释和代码示例。**
2.  **在你之前运行的代码基础上，逐步添加这些精细控制的设置：**
    *   修改线条的 `markerfacecolor`, `markeredgecolor`, `markeredgewidth`。
    *   精确设置 `ax1.set_ylim()` 和 `ax2.set_ylim()`，让Y轴范围更合适。
    *   自定义Y轴的刻度 `ax1.set_yticks()` 和 `ax2.set_yticks()`，并格式化标签。
    *   **重点练习** `ax1.tick_params()`，尝试不同的 `direction`, `length`, `pad`。
    *   确保顶部和右侧轴线（对于 `ax1`）被隐藏，并调整剩余轴线的颜色和宽度。
    *   尝试不同的图例参数，如 `frameon=False`，或改变 `bbox_to_anchor`。
    *   添加一个 `ax.text()` 元素，使用 `transform=ax.transAxes` 将其放置在某个角落。
    *   选择一个数据点，使用 `ax.annotate()` 为其添加带箭头的标注。
3.  **参考你的项目 README 中的数据和图表需求**:
    *   例如，在可视化你的多模型对比表 (2.3.1节) 时，你可能需要用到不同颜色和标记的散点图或条形图，并精确控制图例和标签。
    *   在展示特征重要性时 (2.3.2节)，水平条形图 (`ax.barh()`) 的标签、颜色、排序都需要仔细调整。
    *   在展示大量表格数据 (如第6、8、10节) 时，清晰的轴标签、刻度、图例和可能的标注至关重要。

这一步是精细化作图中最花时间但也最能提升图表质量的部分。请耐心实验每一个参数。当你对这些控制方法有了较好的掌握后，我们就可以进入下一个主题，比如颜色管理和Colormaps，这对于你项目中的降雨产品空间分布图或热力图会非常有用。


### 第4步：颜色 (Colors) 和颜色映射 (Colormaps)

颜色是数据可视化中一个极其重要的元素。恰当的颜色选择可以增强图表的可读性、突出重点、传递额外信息，而不当的颜色则可能误导读者或降低图表美感。对于论文图表，颜色的选择尤其需要考虑印刷效果（灰度）、色盲友好性以及期刊的特定要求。

#### 4.1 Matplotlib 中的颜色表示

Matplotlib 支持多种方式指定颜色：

1.  **名称 (Named Colors)**:
    *   基本颜色: `'b'` (blue), `'g'` (green), `'r'` (red), `'c'` (cyan), `'m'` (magenta), `'y'` (yellow), `'k'` (black), `'w'` (white)。
    *   X11/CSS4 颜色名称: `'skyblue'`, `'coral'`, `'darkgreen'`, `'slateblue'`, `'lightgray'` 等等。你可以搜索 "Matplotlib named colors" 查看完整列表。
    *   Tableau 调色板颜色 (推荐，对比度好): `'tab:blue'`, `'tab:orange'`, `'tab:green'`, `'tab:red'`, `'tab:purple'`, `'tab:brown'`, `'tab:pink'`, `'tab:gray'`, `'tab:olive'`, `'tab:cyan'`。
2.  **十六进制颜色码 (Hex Strings)**:
    *   例如：`'#FF5733'` (橙红色), `'#3375FF'` (蓝色)。
3.  **RGB 或 RGBA 元组 (浮点数，0-1范围)**:
    *   RGB: `(R, G, B)` 例如 `(0.2, 0.4, 0.8)`。
    *   RGBA: `(R, G, B, Alpha)` 例如 `(0.2, 0.4, 0.8, 0.5)`，Alpha 表示透明度。
4.  **灰度值 (字符串形式的浮点数，0-1范围)**:
    *   `'0.0'` (黑色) 到 `'1.0'` (白色)。例如 `'0.75'` (浅灰色)。

**示例：在之前的线图中使用不同的颜色表示**

```python
# 沿用之前的模拟数据和基本的双Y轴图结构
days = np.arange(1, 31)
csi_scores = np.random.rand(30) * 0.3 + 0.6 # CSI 在 0.6-0.9
far_scores = np.random.rand(30) * 0.05 + 0.01 # FAR 在 0.01-0.06

fig, ax1 = plt.subplots(figsize=(10, 5))

# CSI 使用 Tableau 橙色
line_csi, = ax1.plot(days, csi_scores, color='tab:orange', marker='o', label='CSI')
ax1.set_xlabel('Day')
ax1.set_ylabel('CSI Score', color='tab:orange')
ax1.tick_params(axis='y', labelcolor='tab:orange')
ax1.set_ylim(0.55, 0.95)

# FAR 使用十六进制的深灰色
ax2 = ax1.twinx()
line_far, = ax2.plot(days, far_scores, color='#555555', marker='x', linestyle='--', label='FAR') # 深灰色
ax2.set_ylabel('FAR Score', color='#555555')
ax2.tick_params(axis='y', labelcolor='#555555')
ax2.set_ylim(0, 0.07)

ax1.set_title('Daily CSI and FAR Scores')
ax1.legend(handles=[line_csi, line_far], loc='upper left')
fig.tight_layout()
# plt.show()
```

#### 4.2 颜色映射 (Colormaps)

当你的数据值需要用颜色强度或色调来表示时（例如在热力图、散点图的颜色维度、等高线图中），就需要使用颜色映射 (Colormap)。Colormap 将一个标量数据范围映射到一系列颜色。

Matplotlib 提供了大量的内置 Colormaps，可以分为几类：

1.  **顺序型 (Sequential)**: 通常用于表示有顺序的、从低到高的数据（如降雨强度、温度）。颜色从浅到深或从一种色调过渡到另一种。
    *   例如: `'viridis'`, `'plasma'`, `'inferno'`, `'magma'` (这四个是感知上均匀的，推荐), `'Greys'`, `'Blues'`, `'Reds'`, `'Greens'`。
2.  **发散型 (Diverging)**: 用于表示数据围绕一个中心值（通常是0）向两边发散的情况（如正负相关、高于或低于平均值）。颜色从一种极端过渡到中性色，再过渡到另一种极端色。
    *   例如: `'coolwarm'` (蓝-白-红), `'RdBu'` (红-白-蓝), `'seismic'`, `'PiYG'`。
3.  **定性型 (Qualitative/Categorical)**: 用于表示没有内在顺序的分类数据。颜色之间差异明显，易于区分。
    *   例如: `'Pastel1'`, `'Pastel2'`, `'Paired'`, `'Accent'`, `'Set1'`, `'Set2'`, `'Set3'`, `'tab10'`, `'tab20'`。
4.  **循环型 (Cyclic)**: 用于表示具有周期性的数据（如方向、相位）。颜色在两端平滑连接。
    *   例如: `'twilight'`, `'twilight_shifted'`, `'hsv'`。

**如何使用 Colormap?**

*   在支持 `cmap` 参数的绘图函数中直接指定，如 `ax.scatter(..., c=values, cmap='viridis')` 或 `ax.imshow(data, cmap='coolwarm')`。
*   通过 `plt.cm.get_cmap('colormap_name')` 获取 Colormap 对象，然后可以从中获取特定颜色。

**示例：使用 Colormap 绘制散点图，颜色表示第三个维度**

假设我们要绘制长江流域各个站点的多年平均降雨量，颜色表示该站点的海拔高度。

```python
# 模拟数据 (实际中你会从你的数据加载)
num_stations = 100
# 模拟经度 (longitude), 纬度 (latitude)
lon = np.random.rand(num_stations) * 15 + 105 # 长江流域大致经度范围
lat = np.random.rand(num_stations) * 8 + 28   # 长江流域大致纬度范围
# 模拟多年平均降雨量 (作为散点图的Y轴或大小)
avg_rainfall = np.random.rand(num_stations) * 1500 + 500 # 500-2000 mm
# 模拟海拔高度 (作为颜色维度)
elevation = np.random.rand(num_stations) * 3000 # 0-3000 m

fig, ax = plt.subplots(figsize=(10, 7))

# 绘制散点图
# c=elevation: 将海拔高度数据传递给 c 参数，用于颜色映射
# cmap='terrain': 选择 'terrain' Colormap (适合表示地形海拔)
# s=avg_rainfall/20: 用平均降雨量来控制点的大小 (除以一个系数使其不至于过大)
# alpha=0.8: 设置透明度
scatter_plot = ax.scatter(lon, lat, c=elevation, cmap='terrain', s=avg_rainfall/20, alpha=0.8, edgecolors='black', linewidth=0.5)

# 添加颜色条 (Colorbar)
# plt.colorbar() 或 fig.colorbar() 用于显示颜色映射的图例
cbar = fig.colorbar(scatter_plot, ax=ax, label='Elevation (m)', pad=0.02)
# label: 颜色条的标签
# pad: 颜色条与主图的间距

ax.set_xlabel('Longitude (°E)')
ax.set_ylabel('Latitude (°N)')
ax.set_title('Average Rainfall and Elevation of Stations in Yangtze River Basin (Simulated)')
ax.grid(True, linestyle='--', alpha=0.5)

# 可以在图上添加一些地理边界 (简化示意)
# from matplotlib.patches import Rectangle
# yangtze_boundary = Rectangle((105, 28), 15, 8, linewidth=1, edgecolor='r', facecolor='none', linestyle=':')
# ax.add_patch(yangtze_boundary)

fig.tight_layout()
# plt.show()
```

**代码解释与 Colormap 相关参数:**

*   `ax.scatter(..., c=values, cmap='colormap_name')`:
    *   `c`: 一个数值序列，其值将被映射到颜色。
    *   `cmap`: Colormap 的名称。
*   `fig.colorbar(mappable_object, ax, label, orientation, fraction, pad, shrink, aspect, ticks, format, ...)`:
    *   `mappable_object`: 进行了颜色映射的绘图对象，例如 `scatter_plot` (由 `ax.scatter` 返回) 或 `image_object` (由 `ax.imshow` 返回)。
    *   `ax`: 指定颜色条关联的 Axes 对象 (或 Axes 对象列表)。
    *   `label`: 颜色条的标签。
    *   `orientation`: (`'vertical'` 或 `'horizontal'`) 颜色条的方向。
    *   `fraction`: 颜色条相对于原始 Axes 的高度/宽度比例。
    *   `pad`: 颜色条与 Axes 之间的间距。
    *   `shrink`: 颜色条长度的缩放因子。
    *   `aspect`: 颜色条的宽高比。
    *   `ticks`: 自定义颜色条上的刻度位置。
    *   `format`: 刻度标签的格式化字符串。

**选择 Colormap 的原则 (尤其对论文):**

1.  **感知均匀性 (Perceptually Uniform)**: 对于顺序型数据，颜色变化应与数值变化在感知上保持一致。`viridis`, `plasma`, `inferno`, `magma`, `cividis` 是很好的选择。
2.  **色盲友好性 (Colorblind-Friendly)**: 确保色觉障碍者也能区分颜色代表的信息。很多感知均匀的 Colormap (如 `viridis`) 也是色盲友好的。避免使用纯红和纯绿的组合。
3.  **灰度可读性 (Grayscale Readability)**: 如果论文可能被黑白打印，确保 Colormap 在转换为灰度后仍能清晰地区分不同数值。
4.  **避免彩虹色 (Rainbow/Jet)**: `'jet'` Colormap 虽然曾经很流行，但它在感知上不均匀，容易产生视觉假象，且不色盲友好，**强烈不推荐在科研图表中使用**。
5.  **与数据类型匹配**:
    *   顺序数据 -> 顺序型 Colormap
    *   发散数据 -> 发散型 Colormap (确保中性色对应数据的中心点)
    *   分类数据 -> 定性型 Colormap

#### 4.3 在你的项目中应用颜色和 Colormaps

*   **线图/条形图对比**:
    *   当你对比不同降雨产品 (CMORPH, CHIRPS, ..., CHM) 或不同模型版本 (V1-V6) 的性能指标时，为每个产品/版本选择一个固定的、易于区分的颜色。`tab10` 或 `Set1`/`Set2` 等定性型 Colormap 中的颜色是不错的选择。
    *   你可以创建一个颜色字典：
        ```python
        product_colors = {
            'CMORPH': 'tab:blue',
            'CHIRPS': 'tab:orange',
            'GSMAP': 'tab:green',
            # ... 其他产品 ...
            'CHM (Truth)': 'black'
        }
        # 然后在绘图时 ax.plot(..., color=product_colors['CMORPH'])
        ```
*   **空间分布图 (如误差分布、产品性能指标的空间分布)**:
    *   你的 README 中提到“绘制预测误差的空间分布图”、“FP/FN 事件高发的热点区域”。这些非常适合用 `ax.imshow()` (如果数据是规则网格) 或 `ax.scatter()` (如果数据是站点) 配合 Colormap 来展示。
    *   例如，展示 CSI 的空间分布：CSI 值可以用一个顺序型 Colormap (如 `'viridis'` 或 `'RdYlGn'` 反转一下，让绿色表示高CSI) 来表示。
    *   展示误差（预测值 - 真实值）：可以用一个发散型 Colormap (如 `'coolwarm'` 或 `'RdBu_r'`)，中心点为0，红色表示高估，蓝色表示低估。
    *   **结合你的 `data/intermediate/` 中的 `.mat` 数据** (例如 `CMORPH_2016.mat` 是一个 144x256 的矩阵)，你可以用 `ax.imshow()` 来绘制某一天的降雨量空间分布图，颜色深浅代表降雨强度。
        ```python
        # 假设你已经加载了某天的降雨数据 a_day_rain (144x256的NumPy数组)
        # fig, ax = plt.subplots()
        # im = ax.imshow(a_day_rain, cmap='Blues', origin='lower', extent=[lon_min, lon_max, lat_min, lat_max])
        # fig.colorbar(im, label='Rainfall (mm)')
        # ax.set_title('Rainfall Distribution on YYYY-MM-DD')
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # origin='lower' 确保 (0,0) 索引在左下角 (地理坐标通常如此)
        # extent 定义图像的地理坐标范围
        ```
*   **热力图 (Heatmaps)**:
    *   例如，你的 "产品间性能指标空间相关性" 或 "和地面观测站(CHM)之间的相关系数" 表格，可以用热力图 (`ax.imshow()` 或 `sns.heatmap()`) 来可视化相关系数矩阵，颜色表示相关性强度 (通常用发散型 Colormap，如 `'RdBu_r'`，红色表示正相关，蓝色表示负相关)。

**你的任务与思考：**

1.  **修改之前的线图示例**:
    *   尝试为你绘制的 CSI 和 FAR 线条选择不同的命名颜色、十六进制颜色或 Tableau 颜色。
    *   思考哪种颜色组合在你的论文中看起来最专业和清晰。
2.  **运行散点图示例**:
    *   尝试不同的顺序型 Colormap (如 `'viridis'`, `'Blues'`, `'YlGnBu'`) 和发散型 Colormap (如 `'coolwarm'`, `'RdYlBu'`)，观察效果。
    *   修改颜色条的 `label`, `orientation` (`'horizontal'`)，以及 `pad` 和 `shrink` 参数。
3.  **思考你的项目数据**:
    *   **表格 6, 7, 8, 9 (产品性能和统计特征)**: 你会如何用颜色来区分不同的产品或指标？
    *   **README 中提到的空间分布图**: 你会为降雨强度、误差、CSI 等变量选择哪种类型的 Colormap？为什么？
    *   **误报/漏报 (FP/FN) 深度诊断**: 当你绘制 FP/FN 事件高发的热点区域图时，可以用颜色强度表示事件发生的频率或数量。

颜色和 Colormap 的选择对图表的最终呈现效果影响巨大。多尝试，多比较，并参考你领域内高质量论文的图表示例。


### 第5步：高级文本与标注技巧 (Annotations)

在图表中准确、清晰地添加文本和标注，可以极大地增强信息传递的效率，引导读者关注重点，解释特定现象。Matplotlib 提供了强大的工具来实现这一点。

#### 5.1 `ax.text()` 的进阶用法

我们之前用过 `ax.text()`，这里回顾并扩展一些常用参数：

*   **坐标系统 (`transform`)**:
    *   `ax.transData` (默认): 坐标是数据本身的坐标。文本会随数据缩放和平移。
    *   `ax.transAxes`: 坐标是相对于 Axes 区域的比例，`(0,0)` 是左下角，`(1,1)` 是右上角。非常适合在固定位置（如角落）添加版权信息、数据来源或通用标签。
    *   `fig.transFigure`: 坐标是相对于 Figure 画布的比例。适合在整个画布的固定位置添加文本。
*   **对齐 (`horizontalalignment` 或 `ha`, `verticalalignment` 或 `va`)**:
    *   `ha`: `'left'`, `'center'`, `'right'`
    *   `va`: `'bottom'`, `'center'`, `'top'`, `'baseline'`
    决定了文本框的哪个点与你指定的 `(x,y)` 坐标对齐。
*   **旋转 (`rotation`)**: 文本旋转角度（度）。
*   **字体属性 (`fontdict` 或单独参数)**:
    *   `fontsize`, `fontweight`, `fontfamily`, `color`。
    *   也可以用 `fontdict={'fontsize': 12, 'fontweight': 'bold', 'color': 'green'}`。
*   **背景框 (`bbox`)**: 给文本添加一个背景框。
    *   `bbox=dict(boxstyle='round,pad=0.5', fc='wheat', ec='black', alpha=0.8)`
        *   `boxstyle`: 框的形状 (`'square'`, `'round'`, `'roundtooth'`, `'sawtooth'`, `'arrow'`, `'larrow'`, `'rarrow'` 等) 和可选参数 (如 `pad` 内边距)。
        *   `fc`: (facecolor) 填充颜色。
        *   `ec`: (edgecolor) 边框颜色。
        *   `alpha`: 透明度。

**示例：在之前的多Y轴图上添加更丰富的文本信息**

```python
# 沿用V1-V6性能图的数据和基本结构
feature_versions = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6 (Default)', 'V6 (Optuna)']
csi_scores = [0.7820, 0.7841, 0.7859, 0.8101, 0.8115, 0.8228, 0.9147]
far_scores = [0.0823, 0.0687, 0.0681, 0.0572, 0.0564, 0.0819, 0.0335]
pod_scores = [0.8410, 0.8322, 0.8337, 0.8520, 0.8529, 0.8880, 0.9447] # POD数据

fig, ax1 = plt.subplots(figsize=(12, 7)) # 稍微调整画布大小

# --- 绘制 CSI ---
color_csi = 'tab:red'
line_csi, = ax1.plot(feature_versions, csi_scores, color=color_csi, marker='o', markersize=7, label='CSI @0.5 Thr')
ax1.set_xlabel('Feature Set Version', fontsize=12)
ax1.set_ylabel('CSI (Critical Success Index)', color=color_csi, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_csi, labelsize=10, direction='in')
ax1.tick_params(axis='x', rotation=30, labelsize=10, ha='right', direction='in') # ha='right' 配合旋转
ax1.grid(True, linestyle=':', alpha=0.6)

# --- 绘制 FAR ---
ax2 = ax1.twinx()
color_far = 'tab:blue'
line_far, = ax2.plot(feature_versions, far_scores, color=color_far, marker='s', markersize=6, linestyle='--', label='FAR @0.5 Thr')
ax2.set_ylabel('FAR (False Alarm Ratio)', color=color_far, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_far, labelsize=10, direction='in')

# --- 绘制 POD ---
ax3 = ax1.twinx()
ax3.spines["right"].set_position(("outward", 60)) # 调整第三个Y轴位置
color_pod = 'tab:green'
line_pod, = ax3.plot(feature_versions, pod_scores, color=color_pod, marker='^', markersize=7, linestyle=':', label='POD @0.5 Thr')
ax3.set_ylabel('POD (Probability of Detection)', color=color_pod, fontsize=12)
ax3.tick_params(axis='y', labelcolor=color_pod, labelsize=10, direction='in')

# --- 隐藏原始的顶部和右侧轴线 (对ax1) ---
ax1.spines['top'].set_visible(False)
# ax1的右轴线现在被ax2和ax3使用，所以我们不直接隐藏ax1的右轴线
# 而是控制ax2和ax3的轴线是否显示除了它们自己使用的那条
# (实际上 twinx() 会处理好这些，我们主要控制ax1的顶部)

# --- 添加文本信息 ---
# 1. 在图表左上角 (Axes坐标) 添加一个总体说明
ax1.text(0.02, 0.98, 'Model: XGBoost\nRegion: Yangtze River Basin\nProb. Threshold: 0.5',
         transform=ax1.transAxes, # 相对于ax1的坐标
         fontsize=9,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', ec='grey', alpha=0.8))

# 2. 在图表右下角 (Figure坐标) 添加数据来源或作者信息
fig.text(0.98, 0.02, 'Source: Project README Sec 2.3.2\nAuthor: Your Name',
         transform=fig.transFigure, # 相对于整个Figure的坐标
         fontsize=8, color='gray',
         verticalalignment='bottom', horizontalalignment='right')

# 3. 标记CSI最大值点 (数据坐标)
max_csi_val = np.max(csi_scores)
max_csi_idx = np.argmax(csi_scores)
ax1.text(feature_versions[max_csi_idx], max_csi_val + 0.01, # x, y 坐标略微偏移
         f'{max_csi_val:.4f}', # 要显示的文本
         fontsize=9, color=color_csi,
         ha='center', va='bottom') # 水平居中，垂直底部对齐

fig.suptitle('XGBoost Model Performance Evolution', fontsize=15, fontweight='bold')
lines = [line_csi, line_far, line_pod]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10, frameon=False)

fig.tight_layout(rect=[0, 0.08, 1, 0.93]) # 调整rect以适应图例和标题
# plt.show()
```

#### 5.2 `ax.annotate()` 进行精确标注

`annotate` 非常适合需要用箭头指向特定数据点并添加说明的情况。

**核心参数：**

*   `s` 或 `text`: 标注的文本内容。
*   `xy`: 元组 `(x, y)`，被标注的数据点坐标（箭头指向的点）。
*   `xytext`: 元组 `(x, y)`，标注文本的放置位置。
*   `xycoords`: `xy` 参数的坐标系。
    *   `'data'` (默认): 使用数据坐标系。
    *   `'axes fraction'`: 同 `ax.transAxes`。
    *   `'figure fraction'`: 同 `fig.transFigure`。
    *   也可以是其他 Artist 对象。
*   `textcoords`: `xytext` 参数的坐标系。选项同 `xycoords`。通常 `xycoords` 用 `'data'`，而 `textcoords` 可以用 `'data'`, `'offset points'` (相对于`xy`的像素偏移), `'axes fraction'` 等。
*   `arrowprops`: 一个字典，定义箭头的属性。
    *   `arrowstyle`: 箭头的样式，如 `'-'`, `'->'`, `'-|>'`, `'<-'`, `'<->'`, `'fancy'`, `'simple'`, `'wedge'` 等。
        *   也可以是更复杂的样式如 `'Arc3,rad=.2'` (弧形箭头), `'Angle,angleA=0,angleB=90,rad=10'`。
    *   `connectionstyle`: 连接 `xy` 和 `xytext` 的路径样式，如 `'arc3,rad=0.2'`, `'angle3,angleA=0,angleB=-90'`。
    *   `facecolor` (或 `fc`): 箭头填充色。
    *   `edgecolor` (或 `ec`): 箭头边框色。
    *   `linewidth` (或 `lw`): 箭头边框宽度。
    *   `width`: 箭头主体的宽度 (对于某些箭头样式)。
    *   `headwidth`: 箭头头部的宽度。
    *   `headlength`: 箭头头部的长度。
    *   `shrinkA`, `shrinkB`: 箭头两端距离 `xy` 和 `xytext` 的收缩量（以点为单位）。

**示例：高亮V6 (Optuna) 的 CSI 和 FAR 值**

```python
# ... (接上一个例子的绘图代码，在fig.tight_layout()之前) ...

# --- 使用 annotate 高亮 ---
# 高亮 V6 (Optuna) 的 CSI
optuna_idx = feature_versions.index('V6 (Optuna)')
csi_optuna_val = csi_scores[optuna_idx]
far_optuna_val = far_scores[optuna_idx]

# 标注CSI
ax1.annotate(f'Optimized CSI: {csi_optuna_val:.3f}',
             xy=(optuna_idx, csi_optuna_val),                 # 箭头指向的点 (数据坐标)
             xytext=(optuna_idx - 0.5, csi_optuna_val + 0.05), # 文本位置 (数据坐标)
             fontsize=10, color=color_csi,
             arrowprops=dict(arrowstyle="->", # 箭头样式
                             facecolor=color_csi,
                             edgecolor=color_csi,
                             linewidth=0.8,
                             connectionstyle="angle3,angleA=0,angleB=-90")) # 连接线样式

# 标注FAR (在ax2上)
# 注意 xytext 的坐标也需要考虑FAR的Y轴范围
ax2.annotate(f'Optimized FAR: {far_optuna_val:.3f}',
             xy=(optuna_idx, far_optuna_val),
             xytext=(optuna_idx - 1.5, far_optuna_val + 0.015), # 文本位置调整
             fontsize=10, color=color_far,
             arrowprops=dict(arrowstyle="-|>", # 另一种箭头样式
                             facecolor=color_far,
                             edgecolor=color_far,
                             linewidth=0.8,
                             connectionstyle="arc3,rad=-0.2")) # 弧形连接线

# fig.tight_layout(rect=[0, 0.08, 1, 0.93])
# plt.show()
```

**`annotate` 的坐标系选择非常灵活:**

*   你可以让箭头指向数据点 (`xycoords='data'`)，但文本位置相对于整个 Axes (`textcoords='axes fraction'`)，例如：
    `ax1.annotate("Peak", xy=(peak_x, peak_y), xycoords='data', xytext=(0.8, 0.8), textcoords='axes fraction', arrowprops=...)`

#### 5.3 LaTeX 支持 (数学公式和特殊符号)

Matplotlib 可以通过内置的 `mathtext` 引擎或完整的 LaTeX 系统（如果已安装）来渲染数学公式和特殊符号。

*   **`mathtext`**: 只需将文本字符串用 `r"$...$"` 包裹起来。
    *   例如: `ax.set_xlabel(r'Angle $\theta$ (radians)')`
    *   `ax.set_title(r'Function $f(x) = \sum_{i=0}^{N} \alpha_i x^i$')`
    *   常用的希腊字母 (`\alpha`, `\beta`, `\gamma`, `\delta`, `\omega`, `\Omega`, `\pi`, `\mu`, `\sigma` 等)，上下标 (`x^2`, `x_i`)，分数 (`\frac{a}{b}`)，根号 (`\sqrt{x}`) 等都支持。
*   **完整 LaTeX 系统**:
    *   需要设置 `plt.rcParams['text.usetex'] = True`。
    *   同时可能需要配置 `plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{amssymb}'` 等来导入 LaTeX 包。
    *   这种方式可以渲染更复杂的 LaTeX 命令，但配置相对麻烦，且需要本地安装 LaTeX。
    *   对于大多数论文图中的简单公式，`mathtext` 已经足够。

**示例：在标签中使用 LaTeX**

```python
# ... (假设有 ax 对象) ...
# ax.set_xlabel(r'Rainfall Intensity ($\text{mm} \cdot \text{day}^{-1}$)') # 使用了 \cdot 和上标
# ax.set_ylabel(r'Probability Density $P(\text{Rain})$')
# ax.text(0.5, 0.5, r'$\Delta R \approx \frac{\partial R}{\partial t} \Delta t$', transform=ax.transAxes, fontsize=14)
# plt.show()
```

#### 5.4 针对你的项目 (思考与应用)

*   **图表说明**:
    *   在你的对比图 (如不同特征版本性能、不同产品性能) 的角落，使用 `ax.text(..., transform=ax.transAxes, ...)` 添加固定的说明文字，如数据来源年份、研究区域 (长江/全国)、模型名称、关键参数设置 (如降雨阈值定义)。
    *   在Figure的角落，用 `fig.text(..., transform=fig.transFigure, ...)` 添加图表生成日期或版本号（如果需要）。
*   **高亮关键结果**:
    *   当展示不同特征版本 (V1-V6-Optuna) 的性能曲线时，用 `ax.annotate()` 明确标出 Optuna 优化后模型的最佳性能点 (如CSI最高点，FAR最低点)，并显示其数值。
    *   在比较多个原始降雨产品与你的模型性能时，用箭头和文本突出你的模型在哪些指标上取得了显著提升。
    *   在你的误差分析图 (FP/FN空间分布) 中，如果有一些特别值得关注的区域或模式，可以用 `annotate` 来标记并简要说明。
*   **数学符号与单位**:
    *   在轴标签或标题中，如果涉及到单位（如 mm/day, m³/s）或特定符号（如平均值 $\mu$, 标准差 $\sigma$），使用 LaTeX 格式 (`r"$...$"`) 可以使它们更专业。例如，降雨强度单位可以写成 `r'$\text{mm} \cdot \text{day}^{-1}$'`。
    *   你的 README 中有很多指标如 POD, FAR, CSI，它们的定义可能涉及分数或比率，如果需要在图的注释中解释，LaTeX 会很有用。

**你的任务与思考：**

1.  **练习 `ax.text()`**:
    *   在之前的多Y轴性能图上，使用 `ax.transAxes` 在左上角添加模型信息，使用 `fig.transFigure` 在右下角添加你的名字或日期。
    *   尝试不同的 `bbox` 样式和颜色。
2.  **练习 `ax.annotate()`**:
    *   选择一个性能曲线上的转折点或最优/最差点，用 `annotate` 添加带箭头的说明。
    *   尝试不同的 `arrowstyle` 和 `connectionstyle` (如 `'arc3,rad=0.3'`, `'angle3'`)。
    *   尝试让箭头指向数据点 (`xycoords='data'`)，但文本位置使用 `'offset points'` (如 `xytext=(20, -15), textcoords='offset points'`)，这样文本会相对于数据点偏移固定像素。
3.  **练习 LaTeX 文本**:
    *   修改一个轴标签，使其包含一个简单的希腊字母或上下标，例如 `ax1.set_xlabel(r'Feature Version ($\text{V}_i$)')`。
4.  **构思**:
    *   回顾你的 README，找出那些最需要通过文本或箭头标注来强调信息的地方。例如，在“2.3.2 长江流域 XGBoost 模型性能迭代与评估实例”的表格中，Trial 322 参数下的最优 FAR 和 CSI 是非常值得在图表中用 `annotate` 高亮的。
    *   在你的降雨产品统计特征表 (第7、9节) 中，如果绘制相关系数矩阵的热力图，可以用 `ax.text()` 在每个单元格中标注具体的数值。

精通文本和标注技巧能让你的图表信息更丰富，更易于理解。


### 第6步：保存高质量图像 (`fig.savefig()`)

当你花费了大量精力制作出一张精美的图表后，最后一步关键操作就是将其以合适的分辨率和格式保存下来，以便插入论文或报告中。Matplotlib 的 `fig.savefig()` 方法提供了丰富的选项来满足这些需求。

#### 6.1 `fig.savefig()` 的常用参数

`fig.savefig(fname, dpi=None, format=None, bbox_inches=None, pad_inches=0.1, transparent=False, facecolor='auto', edgecolor='auto', ...)`

*   **`fname` (字符串或类路径对象)**:
    *   文件名，包含路径和扩展名。扩展名通常决定了保存的格式 (如 `'my_plot.png'`, `'results/figure1.pdf'`)。
*   **`dpi` (浮点数或 `'figure'`)**:
    *   Dots Per Inch (每英寸点数)，即图像分辨率。
    *   对于论文发表，通常要求 **300 dpi 或更高**。对于矢量图 (如 PDF, SVG)，DPI 主要影响栅格化元素（如图像背景）的分辨率，对矢量元素（线条、文本）影响不大，但某些查看器可能仍会参考它。
    *   如果设为 `'figure'`，则使用 Figure 对象创建时指定的 `dpi` (即 `fig = plt.figure(dpi=...)`)。如果两者都未指定，会使用 `savefig.dpi` rcParam 的值 (默认通常是100)。
*   **`format` (字符串)**:
    *   显式指定输出格式，如 `'png'`, `'pdf'`, `'svg'`, `'eps'`, `'jpg'`, `'tif'`。
    *   如果未指定，Matplotlib 会尝试从 `fname` 的扩展名推断。
    *   **推荐格式**:
        *   **矢量图 (Vector Graphics)**: `.pdf`, `.svg`, `.eps`
            *   **优点**: 可以无限放大而不失真（不产生马赛克），线条和文本非常清晰。是论文出版的首选格式。
            *   `.pdf`: 通用性好，易于查看和嵌入。
            *   `.svg`: 可缩放矢量图形，适合网页和用矢量编辑软件（如 Inkscape, Adobe Illustrator）进一步编辑。
            *   `.eps`: 封装的 PostScript，常用于 LaTeX 系统，但逐渐被 PDF 取代。
        *   **位图/栅格图 (Raster Graphics)**: `.png`, `.jpg`, `.tif`
            *   **优点**: `.png` 支持透明背景，无损压缩（对于线条和文本较多的图）；`.jpg` 有损压缩，文件小，适合照片类图像。`.tif` 常用于高质量印刷，支持无损压缩。
            *   **缺点**: 放大后会失真（出现锯齿或模糊）。
            *   **建议**: 如果必须用位图，优先选择 `.png` (对于线条图) 或高质量 `.jpg` (对于包含大量平滑颜色过渡的图，如复杂的热力图)。`.tif` 文件通常较大。
*   **`bbox_inches` (字符串或 `Bbox` 对象)**:
    *   控制保存图像的边界框。
    *   **`'tight'`**: 强烈推荐使用！它会自动裁剪图像，去除画布周围多余的空白边缘，使图像紧凑。
    *   也可以传入一个 `matplotlib.transforms.Bbox` 对象来精确指定裁剪区域。
*   **`pad_inches` (浮点数)**:
    *   当 `bbox_inches='tight'` 时，在自动计算的紧凑边界框外额外添加的填充（padding），单位是英寸。默认是 `0.1`。有时设为 `0.05` 或更小可以使边缘更紧凑。
*   **`transparent` (布尔值)**:
    *   如果为 `True`，图像的背景（Figure 和 Axes 的 `facecolor`）将被设为透明。这主要对 `.png` 和 `.svg` 格式有效。
    *   **应用**: 当你需要将图表叠加到有色背景或其他图像上时很有用。
*   **`facecolor` (颜色或 `'auto'`)**:
    *   设置保存图像时 Figure 的背景色。如果为 `'auto'`，则使用当前 Figure 的 `facecolor`。
*   **`edgecolor` (颜色或 `'auto'`)**:
    *   设置保存图像时 Figure 的边框色。
*   **`metadata` (字典)**:
    *   可以为 PDF 等格式嵌入元数据，如 `{'Title': 'My Plot Title', 'Author': 'My Name'}`。

#### 6.2 保存示例

```python
# 沿用之前的多Y轴性能图
# ... (之前的绘图代码，到 fig.tight_layout(...) 结束) ...

# --- 保存图像 ---
# 获取当前脚本所在的目录 (如果是在Colab或Jupyter Notebook的根目录，则为当前工作目录)
import os
output_dir = 'plot_outputs' # 定义一个输出子目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir) # 如果目录不存在，则创建它

# 1. 保存为 PDF (推荐用于论文)
pdf_filename = os.path.join(output_dir, 'xgboost_performance_V6.pdf')
try:
    fig.savefig(pdf_filename,
                dpi=300,               # 对于PDF，DPI主要影响嵌入的栅格图像（如果有）
                bbox_inches='tight',   # 自动裁剪空白
                pad_inches=0.05,       # 裁剪后的边缘填充可以小一点
                # metadata={'Title': 'XGBoost Performance', 'Author': 'Your Name'} # 可选元数据
               )
    print(f"图像已保存为: {pdf_filename}")
except Exception as e:
    print(f"保存PDF失败: {e}")


# 2. 保存为 PNG (用于快速预览或网页)
png_filename = os.path.join(output_dir, 'xgboost_performance_V6.png')
try:
    fig.savefig(png_filename,
                dpi=300,               # PNG是位图，DPI直接影响其像素密度
                bbox_inches='tight',
                pad_inches=0.05,
                # transparent=True,    # 如果需要透明背景
                facecolor='white'      # 可以强制背景色为白色，即使样式是其他颜色
               )
    print(f"图像已保存为: {png_filename}")
except Exception as e:
    print(f"保存PNG失败: {e}")


# 3. 保存为 SVG (矢量图，适合编辑)
svg_filename = os.path.join(output_dir, 'xgboost_performance_V6.svg')
try:
    fig.savefig(svg_filename,
                format='svg',          # 可以显式指定格式
                bbox_inches='tight',
                pad_inches=0.05
               )
    print(f"图像已保存为: {svg_filename}")
except Exception as e:
    print(f"保存SVG失败: {e}")


# plt.show() # 在脚本末尾显示图像 (如果需要)
```

**注意事项与最佳实践：**

1.  **始终检查输出文件**: 保存后，务必打开文件检查图像是否符合预期，特别是边界、分辨率、字体和颜色。
2.  **文件名**: 使用描述性的文件名，可以包含关键参数或日期，方便管理。
3.  **目录管理**: 将生成的图表保存在专门的输出目录中，保持项目整洁。
4.  **矢量图优先**: 尽可能使用 PDF 或 SVG 格式提交给期刊。它们在缩放时保持清晰，且通常文件大小比高DPI的PNG小。
5.  **`bbox_inches='tight'`**: 几乎总是需要这个参数，除非你有特殊布局需求。
6.  **DPI**: 对于位图 (PNG, JPG, TIF)，300 DPI 是一个好的起点，有些期刊可能要求更高（如600 DPI）。对于矢量图，DPI 的意义不同，但设置一个合理的值（如300）通常无害。
7.  **字体嵌入**:
    *   当保存为 PDF 时，Matplotlib 默认会尝试嵌入图表中使用的字体子集。这通常能确保在不同计算机上正确显示。
    *   可以通过 `plt.rcParams['pdf.fonttype'] = 42` (TrueType) 或 `plt.rcParams['svg.fonttype'] = 'path'` (将文本转为路径) 来控制字体嵌入方式，以满足特定期刊的要求（例如，避免Type 3字体）。Type 42 (TrueType) 通常是较好的选择，能保持文本的可编辑性。
8.  **颜色一致性**: 如果你的图表将在不同媒介（屏幕、打印）上展示，注意颜色在不同色彩空间（RGB vs CMYK）下的表现可能会有差异。选择对转换鲁棒的颜色有帮助。
9.  **样式文件中的保存参数**: 可以在 `.mplstyle` 文件中设置默认的 `savefig.*` 参数，如：
    ```
    savefig.dpi         : 300
    savefig.format      : pdf
    savefig.bbox        : tight
    savefig.pad_inches  : 0.05
    # savefig.transparent : True
    # savefig.facecolor   : white
    ```
    这样每次调用 `fig.savefig('filename')` 都会使用这些默认值。

#### 6.3 针对你的项目 (思考与应用)

*   **所有最终图表**: 无论是对比模型性能的线图/柱状图，还是展示降雨产品空间分布的地图/热力图，都应该使用 `fig.savefig()` 保存。
*   **格式选择**:
    *   提交给期刊的图表：优先使用 `.pdf`。
    *   需要在Word/PowerPoint中插入并可能需要调整大小的图表：`.svg` (Office 近期版本支持良好) 或高 DPI `.png`。
    *   项目报告或网页展示：`.png` 或 `.svg`。
*   **文件名规范**:
    *   例如，`results/plots/yangtze_feature_comparison_csi_far_pod_thr0.5.pdf`
    *   `results/plots/nationwide_cmorph_spatial_csi_2016_threshold0.1.png`
*   **自动化保存**: 如果你的脚本会生成大量图表（例如，对多个产品、多个年份、多个指标进行可视化），可以在循环中自动生成文件名并保存。

**你的任务与思考：**

1.  **运行上面的保存示例代码**，检查 `plot_outputs` 目录下是否生成了 PDF, PNG, SVG 文件。
2.  **打开这些文件**，比较它们的清晰度，尝试放大 PDF 和 SVG 文件，观察矢量图的优势。
3.  **尝试不同的 `dpi` 值** (例如，对 PNG 使用 `dpi=72` 和 `dpi=600`)，观察文件大小和清晰度的变化。
4.  **实验 `transparent=True`** (对 PNG)，并用图像查看器打开，确认背景是否透明。
5.  **修改 `plt.rcParams['pdf.fonttype'] = 42`** (在脚本开头)，然后重新保存 PDF，虽然肉眼可能看不出区别，但这是确保字体正确嵌入的一个好习惯。
6.  **规划你的项目图表输出**: 考虑为你的项目中不同类型的分析结果（如模型性能、特征重要性、空间分布、时间序列）设计一套标准的命名和保存规范。

掌握了图表的精细控制和高质量保存后，你的 Matplotlib 技能已经达到了一个相当不错的水平，足以应对大多数论文图表的需求。

接下来，我们可以选择：

1.  **深入 Matplotlib 的特定图表类型**: 例如，如何更细致地绘制热力图 (`imshow`)、等高线图 (`contourf`)，或者地理相关的图表 (这可能需要 `cartopy` 库的配合，但 Matplotlib 本身也能做一些基础地图叠加)。
2.  **学习 Seaborn**: 看看 Seaborn 如何用更简洁的代码绘制更美观的统计图表，特别是对于分类数据、分布数据和回归关系的可视化。
3.  **学习 Plotly**: 探索如何制作交互式的图表，这对于在线补充材料或个人网站展示非常有用。

根据你的项目需求，"Seaborn" 可能会是一个很好的下一步，因为它可以帮助你快速生成一些常见的统计对比图，并且默认样式更现代。你的 README 中有很多统计表格，Seaborn 可能很适合可视化它们。

## 第二部分：Seaborn - 优雅的统计数据可视化

Seaborn 是一个基于 Matplotlib 的 Python 数据可视化库。它提供了一个高级接口，用于绘制引人入生且信息丰富的统计图形。Seaborn 的主要优点在于：

*   **更美观的默认样式**: Seaborn 的默认主题和调色板通常比 Matplotlib 的原生样式更现代、更易读。
*   **简化复杂图表的绘制**: 对于一些常见的统计图表（如箱线图、小提琴图、热力图、回归图、分面网格图等），Seaborn 提供了非常简洁的函数。
*   **与 Pandas DataFrame 的紧密集成**: Seaborn 的函数通常能很好地处理 Pandas DataFrame，使得从数据到可视化的过程更流畅。
*   **统计功能内置**: 许多 Seaborn 函数会自动执行必要的统计计算（如均值、置信区间、核密度估计）。
*   **仍然可以精细控制**: 由于 Seaborn 构建在 Matplotlib 之上，其函数通常返回 Matplotlib 的 `Axes` 对象，因此你可以继续使用 Matplotlib 的方法进行后续的精细调整。

### 第1步：准备工作与基本概念

#### 1.1 导入库

标准的导入方式：

```python
import seaborn as sns
import matplotlib.pyplot as plt # 通常也需要导入 plt 用于显示或进一步调整
import pandas as pd # Seaborn 非常喜欢 Pandas DataFrame
import numpy as np # 用于数据生成
```

#### 1.2 Seaborn 的全局样式设置

Seaborn 提供了方便的函数来设置全局的绘图主题和样式。这些通常在脚本的开头设置一次。

*   `sns.set_theme()`: 一个便捷的函数，可以一次性设置多个方面。
    *   `context`: `'paper'`, `'notebook'`, `'talk'`, `'poster'`。控制图表元素的相对大小（如字体、线条宽度）。`'paper'` 适合论文。
    *   `style`: `'darkgrid'`, `'whitegrid'`, `'dark'`, `'white'`, `'ticks'`。控制背景和网格。
    *   `palette`: 调色板名称或颜色列表。Seaborn 有很多优秀的内置调色板。
    *   `font_scale`: 字体大小的缩放因子。
    *   `rc`: 一个字典，可以直接传递给 `matplotlib.rcParams` 进行更底层的设置。

```python
# 设置Seaborn的主题 (在脚本开头执行一次)
sns.set_theme(context='paper', style='whitegrid', palette='muted', font_scale=1.1)
# 'muted' 是一个不错的默认调色板
# font_scale=1.1 会将所有字体稍微放大一点

# 你也可以分开设置：
# sns.set_style("ticks") # 例如，只保留刻度线，无网格背景
# sns.set_context("talk") # 例如，用于演讲的更大字体
# sns.set_palette("viridis") # 设置默认调色板
```

#### 1.3 Seaborn 函数的基本模式

大多数 Seaborn 绘图函数遵循类似的模式：

`sns.function_name(data=your_dataframe, x='x_column_name', y='y_column_name', hue='category_column_name', ...)`

*   `data`: 通常是一个 Pandas DataFrame。
*   `x`, `y`: DataFrame 中用于 x 轴和 y 轴的列名。
*   `hue`: DataFrame 中用于根据其值对图元（点、线、条）进行颜色编码的分类列名。会生成图例。
*   `size`: 类似于 `hue`，但用于控制大小。
*   `style`: 类似于 `hue`，但用于控制标记样式或线条样式。
*   `ax`: 可以传入一个 Matplotlib 的 `Axes` 对象，让 Seaborn 在指定的子图上绘图。这对于将 Seaborn 图集成到复杂的 Matplotlib 布局中非常重要。

#### 1.4 示例数据准备 (模拟你的项目场景)

假设我们有一些模型性能数据，类似于你的 README 2.3.1 节（基础模型选择与性能对比）或 2.3.2 节（长江流域 XGBoost 模型性能迭代）。

```python
# 模拟 README 2.3.1 节的数据
model_data = {
    'Model Name': ['KNN', 'SVM', 'Random Forest', 'LightGBM', 'Gaussian NB', 'XGBoost (Default)'],
    'Accuracy': [0.7917, 0.8021, 0.8408, 0.8366, 0.7019, 0.8819],
    'POD': [0.7839, 0.7496, 0.8378, 0.8221, 0.5799, 0.8880],
    'FAR': [0.1308, 0.0819, 0.1001, 0.0929, 0.0909, 0.0819],
    'CSI': [0.7012, 0.7026, 0.7665, 0.7582, 0.5481, 0.8228]
}
df_model_comparison = pd.DataFrame(model_data)

# 模拟 README 2.3.2 节部分数据 (V1, V4, V6 Optuna 在不同阈值下的CSI)
threshold_data = {
    'Feature Set': ['V1']*5 + ['V4']*5 + ['V6 Optuna']*5,
    'Threshold': [0.3, 0.4, 0.5, 0.6, 0.7] * 3,
    'CSI': [0.7975, 0.7941, 0.7820, 0.7615, 0.7321,  # V1
            0.8171, 0.8101, 0.8011, 0.78, 0.76,    # V4 (假设数据)
            0.9168, 0.9177, 0.9147, 0.9084, 0.8992], # V6 Optuna
    'FAR': [0.1280, 0.1030, 0.0823, 0.0646, 0.0479, # V1
            0.0675, 0.0572, 0.0482, 0.04, 0.03,     # V4 (假设数据)
            0.0526, 0.0416, 0.0335, 0.0270, 0.0210]  # V6 Optuna
}
df_threshold_perf = pd.DataFrame(threshold_data)
```

### 第2步：常用统计图表绘制

#### 2.1 分类图 (Categorical Plots) - 对比不同模型/版本的性能

当你的X轴是分类变量（如模型名称、特征集版本）时，Seaborn 的分类图非常有用。

**2.1.1 条形图 (`sns.barplot` 或 `sns.catplot(kind='bar')`)**

`sns.barplot` 默认会计算数值变量的均值，并显示置信区间（通常是95% CI的引导程序估计）。如果你的数据已经是聚合好的值（如你的性能指标），它会直接使用这些值。

```python
plt.figure(figsize=(10, 6)) # 创建Matplotlib Figure
ax_bar = sns.barplot(x='Model Name', y='CSI', data=df_model_comparison, palette='viridis')
# palette: 可以选择Seaborn的调色板，如 'viridis', 'rocket', 'muted', 'pastel'等

ax_bar.set_title('Comparison of Model Performance (CSI)', fontsize=15)
ax_bar.set_xlabel('Model', fontsize=12)
ax_bar.set_ylabel('CSI Score', fontsize=12)
ax_bar.tick_params(axis='x', rotation=30, ha='right') # 旋转X轴标签

# 在条形图上添加数值标签
for p in ax_bar.patches: # ax.patches 包含图中所有的 "patch" 对象 (条形就是patch)
    ax_bar.annotate(f"{p.get_height():.3f}",       # 获取条形的高度作为文本
                    (p.get_x() + p.get_width() / 2., p.get_height()), # 文本位置 (条形顶部中心)
                    ha='center', va='center',      # 对齐方式
                    xytext=(0, 5),                # 文本偏移量 (向上偏移5个点)
                    textcoords='offset points',   # 偏移坐标系
                    fontsize=9)

plt.ylim(0, df_model_comparison['CSI'].max() * 1.1) # 调整Y轴范围给标签留空间
plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   你的 **README 2.3.1 节** 的表格可以直接用这种方式可视化，比较不同模型的 Accuracy, POD, FAR, CSI。你可以为每个指标画一个条形图，或者使用后面会讲到的 `catplot` 创建分面图。
*   `palette` 参数可以让你轻松更换颜色方案。

**2.1.2 点图 (`sns.pointplot` 或 `sns.catplot(kind='point')`)**

点图用点的位置来表示估计值（默认均值），并用误差线表示置信区间。它更适合比较不同类别之间的趋势变化，特别是当 `hue` 参数被用来引入第二个分类变量时。

```python
# 比较不同特征集在不同阈值下的CSI (使用df_threshold_perf)
plt.figure(figsize=(10, 6))
ax_point = sns.pointplot(x='Threshold', y='CSI', hue='Feature Set',
                         data=df_threshold_perf,
                         markers=['o', 's', '^'],  # 为不同hue指定不同标记
                         linestyles=['-', '--', ':'], # 为不同hue指定不同线型
                         dodge=True) # dodge=True 使不同hue的线条和点稍微错开，避免重叠

ax_point.set_title('CSI vs. Prediction Threshold by Feature Set', fontsize=15)
ax_point.set_xlabel('Prediction Threshold', fontsize=12)
ax_point.set_ylabel('CSI Score', fontsize=12)
ax_point.legend(title='Feature Set', loc='lower left') # loc图例位置
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   你的 **README 2.3.2 节** 的多个表格展示了不同特征版本 (V1-V6) 在不同预测概率阈值下的性能。`sns.pointplot` 或 `sns.lineplot` (后面会讲) 非常适合可视化这种“指标 vs. 阈值，按特征版本分组”的数据。
*   `hue` 参数在这里是关键，它自动为你区分了不同的特征集。
*   `markers` 和 `linestyles` 允许你为不同的 `hue` 类别自定义外观，这在黑白打印的论文中非常重要。

**2.1.3 箱线图 (`sns.boxplot`) 和小提琴图 (`sns.violinplot`)**

如果你有每个类别下的原始数据分布（而不仅仅是聚合值），箱线图和小提琴图可以很好地展示这些分布的特性（中位数、四分位数、异常值等）。

**假设你的数据是这样的：** 对于每个特征版本V1-V6，你进行了多次（比如K-Fold交叉验证的K次）实验，得到了K个CSI值。

```python
# 模拟每个特征版本在K-Fold下的CSI得分数据
np.random.seed(42) # 为了结果可复现
data_for_boxplot = {
    'Feature Set': [],
    'CSI': []
}
sets = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6 Optuna']
base_csi = [0.78, 0.785, 0.79, 0.81, 0.815, 0.91]
for i, s_name in enumerate(sets):
    # 模拟5折的CSI得分
    scores = np.random.normal(loc=base_csi[i], scale=0.015, size=5)
    scores = np.clip(scores, base_csi[i]-0.03, base_csi[i]+0.03) # 限制范围
    data_for_boxplot['Feature Set'].extend([s_name] * 5)
    data_for_boxplot['CSI'].extend(scores)

df_boxplot_data = pd.DataFrame(data_for_boxplot)

plt.figure(figsize=(10, 6))
ax_box = sns.boxplot(x='Feature Set', y='CSI', data=df_boxplot_data, palette='Set2', width=0.6)
# palette='Set2' 是一个适合分类的调色板
# width 控制箱体的宽度

# 可以叠加 stripplot 或 swarmplot 来显示原始数据点
sns.stripplot(x='Feature Set', y='CSI', data=df_boxplot_data, color=".3", jitter=0.1, size=4, ax=ax_box)
# color=".3" 是一种指定灰度的方式 (0是黑，1是白)
# jitter=0.1 让点在x轴方向上稍微散开一点，避免完全重叠

ax_box.set_title('Distribution of CSI Scores by Feature Set (K-Fold CV)', fontsize=15)
ax_box.set_xlabel('Feature Set Version', fontsize=12)
ax_box.set_ylabel('CSI Score', fontsize=12)
ax_box.tick_params(axis='x', rotation=15, ha='right')
plt.tight_layout()
# plt.show()

# 小提琴图类似，但能更好地显示分布的形状
# plt.figure(figsize=(10, 6))
# sns.violinplot(x='Feature Set', y='CSI', data=df_boxplot_data, palette='pastel', inner='quartile')
# inner='quartile' 会在小提琴内部显示四分位数线，也可以是 'box', 'stick', 'point'
# plt.title('Violin Plot of CSI Scores by Feature Set', fontsize=15)
# plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   在你的 **2.3.2 节的 K-Fold 交叉验证结果** 部分，你有每个折的验证集 AUC。这非常适合用箱线图或小提琴图来展示不同超参数试验（如果 Trials 代表不同参数组）或不同折之间的性能稳定性。
*   如果你的某些分析涉及到比较不同区域/站点/时间的某个指标的分布，这些图也很有用。

#### 2.2 关系图 (`sns.lineplot`, `sns.scatterplot`) - 展示趋势和变量间关系

**2.2.1 线图 (`sns.lineplot`)**

`sns.lineplot` 非常适合展示一个数值变量随另一个数值变量（通常是时间或有序类别）变化的趋势。它可以自动计算并显示置信区间（如果数据中存在重复的x值或通过`estimator`参数聚合）。

```python
# 使用 df_threshold_perf 数据
plt.figure(figsize=(10, 6))
ax_line = sns.lineplot(x='Threshold', y='CSI', hue='Feature Set', style='Feature Set', # style也按Feature Set区分
                       data=df_threshold_perf,
                       markers=True, dashes=False, # markers=True显示数据点, dashes=False确保实线 (除非style覆盖)
                       linewidth=2)

# 使用你的项目指标：POD, FAR, CSI vs 阈值
# 先将数据从宽格式转为长格式，方便Seaborn的hue参数使用
df_threshold_perf_long = pd.melt(df_threshold_perf,
                                 id_vars=['Feature Set', 'Threshold'],
                                 value_vars=['CSI', 'FAR'], # 假设我们只看CSI和FAR
                                 var_name='Metric',
                                 value_name='Score')

plt.figure(figsize=(12, 7))
ax_line_metrics = sns.lineplot(x='Threshold', y='Score', hue='Feature Set', style='Metric',
                               data=df_threshold_perf_long[df_threshold_perf_long['Feature Set'] == 'V6 Optuna'], # 只看V6 Optuna
                               markers=True, dashes=False, linewidth=2, palette=['tab:red', 'tab:blue']) # 手动指定颜色

ax_line_metrics.set_title('Performance Metrics vs. Threshold (V6 Optuna)', fontsize=15)
ax_line_metrics.set_xlabel('Prediction Threshold', fontsize=12)
ax_line_metrics.set_ylabel('Score', fontsize=12)
ax_line_metrics.legend(title='Metric/Set', loc='center right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
# plt.show()
```
**思考与你的项目：**

*   你的 **README 2.3.2 节** 的表格数据（不同特征版本下，各项指标随预测概率阈值的变化）是 `sns.lineplot` 的完美应用场景。你可以用 `hue` 来区分特征版本，用 `style` 来区分不同的指标 (POD, FAR, CSI)。
*   `pd.melt` 是 Pandas 中一个非常有用的函数，可以将宽格式数据转换为长格式，这通常是 Seaborn (尤其是使用 `hue` 或 `style`) 更喜欢的数据组织方式。

**2.2.2 散点图 (`sns.scatterplot`)**

`sns.scatterplot` 允许你通过颜色 (`hue`)、大小 (`size`) 和形状 (`style`) 来编码额外的变量维度。

```python
# 模拟不同产品在多个站点的性能指标 (假设每个点是一个站点)
np.random.seed(10)
num_sites = 50
sites_data = {
    'Product': np.random.choice(['CMORPH', 'GSMAP', 'IMERG'], size=num_sites),
    'Region': np.random.choice(['East', 'West', 'Central'], size=num_sites),
    'POD': np.random.rand(num_sites) * 0.4 + 0.5, # 0.5-0.9
    'FAR': np.random.rand(num_sites) * 0.15 + 0.01, # 0.01-0.16
    'Mean Annual Precip (mm)': np.random.rand(num_sites) * 1000 + 800 # 800-1800
}
df_sites_perf = pd.DataFrame(sites_data)

plt.figure(figsize=(10, 7))
ax_scatter = sns.scatterplot(x='FAR', y='POD', hue='Product', size='Mean Annual Precip (mm)',
                             style='Region', data=df_sites_perf,
                             sizes=(50, 300), # 控制size映射的范围
                             alpha=0.7,
                             palette='Set1') # 'Set1' 是一个对比鲜明的定性调色板

ax_scatter.set_title('Site-Specific Product Performance (POD vs. FAR)', fontsize=15)
ax_scatter.set_xlabel('False Alarm Ratio (FAR)', fontsize=12)
ax_scatter.set_ylabel('Probability of Detection (POD)', fontsize=12)
ax_scatter.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.) # 图例放到图外
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # 为图例留出空间
# plt.show()
```
**思考与你的项目：**

*   如果你的 FP/FN 分析涉及到具体事件或站点的特征，可以用散点图来展示这些样本在不同特征维度上的分布，并用颜色/大小/形状编码其他属性（如误差大小、降雨强度、产品类型等）。
*   例如，绘制 FP 样本的“预测概率” vs “实际观测为0时某产品的原始值”，颜色可以代表区域或季节。

**你的任务与思考：**

1.  **运行本节的所有 Seaborn 示例代码**，理解每个函数的作用和常用参数。
2.  **尝试不同的 `palette`**: 例如，对条形图尝试 `'pastel'`, `'Set3'`；对线图尝试 `'colorblind'`, `'rocket_r'` (反转的rocket)。
3.  **修改 `sns.pointplot` 和 `sns.lineplot` 中的 `hue` 和 `style` 参数**: 看看如何用它们来展示更多维度的信息。
4.  **针对你的项目数据思考**:
    *   对于 **README 2.3.1 节** 的模型对比表，除了条形图，还可以用什么 Seaborn 图？（例如，如果每个模型有多次运行结果，可以用 `boxplot`）
    *   对于 **README 2.3.2 节** 的性能随阈值变化表，`sns.lineplot` 如何能帮你清晰地展示V1到V6 Optuna的演进过程？（提示：`hue_order` 和 `style_order` 参数可以控制类别顺序）
    *   对于 **README 第6、8、10节** 的大量产品性能数据和统计特征，Seaborn 能如何帮助你进行可视化对比和探索？（例如，用 `catplot(kind='bar')` 或 `catplot(kind='point')` 配合 `col` 参数按产品分面，展示多个指标）

Seaborn 提供了很多便捷的函数，可以大大提高你绘制统计图表的效率和美观度。关键在于理解你的数据结构以及你想通过图表传达什么信息，然后选择合适的 Seaborn 函数。

完成练习后，请告诉我你的进展和遇到的问题。接下来我们可以讨论 Seaborn 的分布图 (如直方图、KDE图) 和矩阵图 (如热力图)，或者学习如何将 Seaborn 图与 Matplotlib 的多子图布局结合起来。

#### 2.3 分布图 (Distribution Plots) - 理解数据分布特征

理解单个变量的分布或多个变量之间的联合分布，对于数据分析和模型诊断至关重要。Seaborn 提供了一系列强大的函数来可视化数据分布。

**2.3.1 直方图与核密度估计 (`sns.histplot`, `sns.kdeplot`)**

*   `sns.histplot(data, x, y, hue, stat, bins, kde, multiple, ...)`:
    *   `x` 或 `y`: 指定要绘制分布的变量。
    *   `hue`: 按分类变量对直方图进行分组和着色。
    *   `stat`: (`'count'`, `'frequency'`, `'density'`, `'probability'`) 统计量。
    *   `bins`: (`'auto'`, 整数, 或序列) 控制分箱数量或边界。
    *   `kde=True`: 同时绘制核密度估计曲线。
    *   `multiple`: (`'layer'`, `'stack'`, `'dodge'`, `'fill'`) 当使用 `hue` 时，不同组直方图的堆叠方式。
*   `sns.kdeplot(data, x, y, hue, fill, multiple, levels, thresh, bw_adjust, ...)`:
    *   `fill=True`: 填充KDE曲线下的区域。
    *   `multiple`: (`'layer'`, `'stack'`, `'fill'`) 当使用 `hue` 时，不同组KDE的堆叠方式。
    *   `levels`: (整数或序列) 控制等高线图的层级数或特定层级。
    *   `thresh`: (0-1) 低于此概率密度的区域不绘制。
    *   `bw_adjust`: (浮点数) 调整带宽，影响KDE曲线的平滑度。

**示例：分析不同降雨产品（或模型预测误差）的分布**

假设我们有模型预测的降雨量和对应的真实观测降雨量，我们可以分析误差的分布。

```python
# 模拟预测误差数据
np.random.seed(123)
prediction_errors_model_A = np.random.normal(loc=-0.5, scale=2, size=200) # 模型A倾向于低估
prediction_errors_model_B = np.random.normal(loc=0.2, scale=1.5, size=200)  # 模型B倾向于轻微高估，方差小

df_errors = pd.DataFrame({
    'Error': np.concatenate([prediction_errors_model_A, prediction_errors_model_B]),
    'Model': ['Model A'] * 200 + ['Model B'] * 200
})

# 1. 使用 histplot 显示误差分布
plt.figure(figsize=(10, 6))
sns.histplot(data=df_errors, x='Error', hue='Model', kde=True,
             multiple='layer', # 'layer'使不同组部分重叠，'stack'堆叠, 'dodge'并列
             palette={'Model A': 'skyblue', 'Model B': 'salmon'},
             alpha=0.6, element='step') # element='step' 只画轮廓，element='bars' (默认) 画条形

plt.title('Distribution of Prediction Errors for Two Models', fontsize=15)
plt.xlabel('Prediction Error (Predicted - True Rainfall, mm)', fontsize=12)
plt.ylabel('Density' if True else 'Count', fontsize=12) # 如果kde=True或stat='density'，Y轴是密度
plt.axvline(0, color='black', linestyle='--', linewidth=1) # 添加一条 x=0 的参考线
plt.legend(title='Model')
plt.tight_layout()
# plt.show()

# 2. 单独使用 kdeplot
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_errors, x='Error', hue='Model', fill=True, alpha=0.5,
            palette={'Model A': 'skyblue', 'Model B': 'salmon'},
            linewidth=2)
plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.title('Kernel Density Estimate of Prediction Errors', fontsize=15)
plt.xlabel('Prediction Error (mm)', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.legend(title='Model')
plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   **误差分析**: 你的项目关注 FP/FN (误报/漏报)。对于回归任务（预测降雨量），你可以绘制预测误差 (预测值 - 真实值) 的 `histplot` 或 `kdeplot`，按不同模型、不同特征集、不同区域或不同季节进行 `hue` 分组，观察误差分布的偏度、峰度和均值是否接近0。
*   **降雨产品原始值分布**: 你的 **第 7、9 节 (产品统计特征)** 包含了平均值、标准差、偏度、峰度等。你可以直接用 `histplot` 或 `kdeplot` 可视化每个原始降雨产品 (CMORPH, CHIRPS 等) 的日降雨量值的分布，并与 CHM (真值) 的分布进行对比。这能直观地看出哪些产品倾向于高估/低估，以及分布形状的差异。
    ```python
    # 假设 df_rain_values 有 'Rainfall' 列和 'Product' 列
    # sns.histplot(data=df_rain_values, x='Rainfall', hue='Product', log_scale=(False, True), element='step', fill=False)
    # log_scale=(False, True) 表示Y轴（计数）使用对数刻度，有助于观察稀有的大值降雨。
    # 或者只看kde:
    # sns.kdeplot(data=df_rain_values, x='Rainfall', hue='Product', log_scale=True, common_norm=False, cut=0)
    # log_scale=True 对X轴（降雨量）取对数，因为降雨分布通常是重尾的。
    # common_norm=False 让每个组的KDE独立归一化。cut=0 限制KDE在数据范围内。
    ```
*   **特征分布**: 在进行特征工程时，查看重要特征的分布，或者 FP/FN 样本在某些特征上的分布差异，可以使用这些图。

**2.3.2 联合与边缘分布 (`sns.jointplot`)**

`sns.jointplot(data, x, y, hue, kind, ...)` 用于可视化两个变量之间的联合分布以及它们各自的边缘分布。

*   `kind`:
    *   `'scatter'` (默认): 中间是散点图，边缘是直方图。
    *   `'kde'`: 中间是2D KDE图，边缘是1D KDE图。
    *   `'hist'`: 中间是2D直方图 (六边形分箱或矩形分箱)，边缘是直方图。
    *   `'reg'`: 中间是散点图加回归线，边缘是直方图加KDE。

```python
# 使用之前的站点性能数据 df_sites_perf
# 绘制 POD 和 FAR 的联合分布，并按产品类型着色
# 注意：hue 参数在 kind='scatter' 或 'kde' (较新版本Seaborn) 时效果较好
# 对于 'reg' 或 'hist'，hue 的支持可能有限或行为不同

# 示例1: kind='scatter'
g_scatter = sns.jointplot(data=df_sites_perf, x='FAR', y='POD', hue='Product',
                          palette='viridis', kind='scatter',  # kind='kde' 也会很不错
                          height=7, # 控制图像大小 (jointplot是Figure-level函数)
                          marginal_ticks=True) # 在边缘图上显示刻度
g_scatter.fig.suptitle('Joint Distribution of POD and FAR by Product (Scatter)', y=1.02, fontsize=15) # y调整标题位置
# plt.show()

# 示例2: kind='kde' (对于理解密度分布更好)
g_kde = sns.jointplot(data=df_sites_perf, x='FAR', y='POD', hue='Product',
                      kind='kde', fill=True,
                      levels=5, # KDE等高线的层级
                      thresh=0.1, # 忽略密度低于此值的区域
                      height=7,
                      palette='crest_r') # 'crest_r' 是一个不错的顺序调色板
g_kde.fig.suptitle('Joint Distribution of POD and FAR by Product (KDE)', y=1.02, fontsize=15)
# plt.show()
```

**思考与你的项目：**

*   **相关性分析**: 你的 **第7、9节** 有产品间的相关系数。你可以选择两个你认为可能相关的指标（例如，某个产品的POD和其平均降雨量，或者你的模型预测误差和某个输入特征的值），用 `jointplot` 来查看它们的联合分布和相关性。
*   **FP/FN特征分析**: 针对FP或FN样本，选择两个关键特征，用 `jointplot` 查看它们是否在这些错误样本中呈现特定的聚集模式。

#### 2.4 矩阵图 (`sns.heatmap`, `sns.clustermap`) - 可视化矩阵数据

**2.4.1 热力图 (`sns.heatmap`)**

`sns.heatmap(data, vmin, vmax, cmap, center, annot, fmt, linewidths, linecolor, cbar, ...)`
非常适合可视化矩阵数据，例如相关系数矩阵、混淆矩阵、或你的降雨产品性能指标表格。

*   `data`: 一个2D NumPy数组或Pandas DataFrame。
*   `vmin`, `vmax`: 颜色映射的最小值和最大值。
*   `cmap`: Colormap名称。对于相关系数，常用发散型如 `'RdBu_r'` (红-白-蓝，r表示反转，使正相关为暖色)。
*   `center`: 发散型Colormap的中心值 (例如，相关系数的0)。
*   `annot=True`: 在每个单元格中显示数值。
*   `fmt`: 数值格式化字符串 (如 `'.2f'` 表示两位小数)。
*   `linewidths`, `linecolor`: 单元格之间的分割线。
*   `cbar=True`: 是否显示颜色条。

**示例：可视化降雨产品间的相关系数矩阵 (来自你的第7节)**

```python
# 模拟第7节的相关系数矩阵数据 (部分)
corr_data = {
    'CMORPH': [1.00, 0.61, 0.50, 0.83, 0.66, 0.68],
    'CHIRPS': [0.61, 1.00, 0.43, 0.65, 0.58, 0.75],
    'SM2RAIN':[0.50, 0.43, 1.00, 0.54, 0.62, 0.46],
    'IMERG':  [0.83, 0.65, 0.54, 1.00, 0.73, 0.71],
    'GSMAP':  [0.66, 0.58, 0.62, 0.73, 1.00, 0.59],
    'PERSIANN':[0.68, 0.75, 0.46, 0.71, 0.59, 1.00]
}
index_names = ['CMORPH', 'CHIRPS', 'SM2RAIN', 'IMERG', 'GSMAP', 'PERSIANN']
df_corr = pd.DataFrame(corr_data, index=index_names, columns=index_names)

plt.figure(figsize=(8, 6.5))
sns.heatmap(df_corr, annot=True, fmt=".2f", cmap='coolwarm',
            linewidths=.5, linecolor='gray',
            cbar_kws={'label': 'Correlation Coefficient', 'shrink': 0.8}, # cbar_kws传递给colorbar的参数
            vmin=-1, vmax=1, center=0) # 确保颜色条对称且覆盖整个相关性范围

plt.title('Correlation Matrix of Rainfall Products (Yangtze Basin)', fontsize=15, pad=20) # pad增加标题与图的距离
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   你的 **第7节和第9节** 的相关系数表是 `sns.heatmap` 的直接应用场景。
*   如果你计算了不同特征之间的相关性矩阵 (如 README 2.4节提到的)，也可以用热力图可视化。
*   混淆矩阵 (TP, FP, FN, TN) 也可以用热力图展示，颜色表示数量或比例。

**2.4.2 层次聚类热力图 (`sns.clustermap`)**

`sns.clustermap(data, ...)` 与 `heatmap` 类似，但它会同时对数据的行和列进行层次聚类，并重新排序行和列，使得相似的行/列聚集在一起，有助于发现数据中的结构和模式。参数与 `heatmap` 大部分相似。

```python
# 使用同样的相关系数矩阵 df_corr
# clustermap 是 Figure-level 的，它会自己创建Figure
g_cluster = sns.clustermap(df_corr, annot=True, fmt=".2f", cmap='coolwarm_r', # _r 表示反转colormap
                           linewidths=.5, figsize=(8, 8),
                           cbar_kws={'label': 'Correlation Coefficient'},
                           vmin=-1, vmax=1, center=0,
                           # method='average', metric='euclidean' # 控制聚类算法和距离度量
                          )
g_cluster.fig.suptitle('Clustered Correlation Matrix of Rainfall Products', y=1.03, fontsize=15)
# clustermap 返回一个 ClusterGrid 对象，可以用 g_cluster.ax_heatmap 访问热力图的Axes
g_cluster.ax_heatmap.set_xticklabels(g_cluster.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
g_cluster.ax_heatmap.set_yticklabels(g_cluster.ax_heatmap.get_yticklabels(), rotation=0)
# plt.show()
```
**思考与你的项目：**
当你有很多产品或很多特征时，`clustermap` 可以帮助你自动发现哪些产品/特征在行为上更相似。

**你的任务与思考：**

1.  **运行本节的所有 Seaborn 示例代码。**
2.  **对于 `histplot` 和 `kdeplot`**:
    *   尝试不同的 `multiple` 参数 (`'layer'`, `'stack'`, `'dodge'`, `'fill'`)，观察 `hue` 分组的效果。
    *   尝试对降雨数据使用 `log_scale` (如 `log_scale=True` 或 `log_scale=(True, False)`)，看看对重尾分布的展示有何帮助。
3.  **对于 `jointplot`**:
    *   尝试不同的 `kind` 参数 (`'scatter'`, `'kde'`, `'hist'`, `'reg'`)。
    *   如果数据点很多，`kind='hist'` (六边形分箱) 或 `kind='kde'` 可能比 `'scatter'` 更能揭示密度。
4.  **对于 `heatmap`**:
    *   尝试不同的 `cmap` (如 `'viridis'`, `'Blues'`, `'RdYlGn'`)。
    *   修改 `annot`, `fmt`, `linewidths` 参数。
5.  **思考你的项目**:
    *   如何使用这些分布图和矩阵图来更深入地分析你的降雨产品数据、模型误差、特征之间的关系？
    *   例如，你的项目重点是降低误报率(FAR)。你可以绘制在发生误报(FP)时，各个原始降雨产品的值的分布 (`histplot` 或 `kdeplot`，以产品为 `hue`)，看看哪些产品在这些情况下更容易给出非零读数。
    *   对于你的特征工程迭代 (V1-V6)，如果能获得每次迭代后某些关键特征的值的分布，或者模型输出概率的分布，将有助于理解迭代的效果。

Seaborn 提供了绘制美观统计图的便捷途径。关键是理解每种图的适用场景，并学会结合 `hue`, `style`, `size` 等参数来展示多维信息。由于 Seaborn 基于 Matplotlib，你总是可以在 Seaborn 生成的图 (通常是 Matplotlib `Axes` 对象) 上使用 Matplotlib 的方法进行进一步的定制。

完成练习后，请告诉我你的进展。接下来我们可以讨论 Seaborn 的多图网格 (`FacetGrid`, `PairGrid`, `pairplot`, `jointplot` 本身也是一种网格)，或者如果你觉得 Matplotlib 和 Seaborn 的基础已经足够，我们可以开始了解 Plotly 用于交互式图表。

#### 2.5 多图网格 (Multi-plot Grids) - 分面可视化

当你需要按一个或多个分类变量的子集来重复绘制同一种类型的图表时，Seaborn 的多图网格功能非常强大。它们可以帮助你快速比较不同条件下的数据模式。这些通常是 "Figure-level" 的函数，它们会自己创建 Figure 和 Axes 网格。

**2.5.1 `sns.FacetGrid`**

`FacetGrid` 是一个通用的工具，用于创建按数据子集划分的绘图网格。你需要先初始化一个 `FacetGrid` 对象，然后使用 `.map()` 或 `.map_dataframe()` 方法将一个绘图函数应用于每个子图。

*   `FacetGrid(data, row, col, hue, col_wrap, height, aspect, palette, ...)`:
    *   `data`: Pandas DataFrame。
    *   `row`, `col`: DataFrame 中的列名，用于定义子图网格的行和列。
    *   `hue`: DataFrame 中的列名，用于在每个子图内部进行颜色编码。
    *   `col_wrap`: (整数) 当只使用 `col` 分面时，指定每行最多显示多少个子图，然后换行。
    *   `height`: 每个子图的高度 (英寸)。
    *   `aspect`: 每个子图的宽高比 (`width = height * aspect`)。
    *   `sharex`, `sharey`: (布尔值) 是否共享轴。

**示例：按特征集和指标类型展示性能随阈值的变化**

我们将使用之前创建的 `df_threshold_perf_long` 数据。

```python
# df_threshold_perf_long 包含 'Feature Set', 'Threshold', 'Metric', 'Score'
# 我们想按 'Feature Set' 分列，按 'Metric' 在图内用颜色区分

# 1. 初始化 FacetGrid
# col='Feature Set': 每个特征集一列
# hue='Metric': 在每个子图中，用颜色区分不同的Metric (CSI, FAR)
# col_wrap=3: 如果特征集很多，每行最多显示3个子图
g_facet = sns.FacetGrid(df_threshold_perf_long,
                        col='Feature Set',
                        hue='Metric',
                        col_wrap=3, # 每行最多3个子图
                        height=4, aspect=1.2, # 控制子图大小和形状
                        palette={'CSI': 'tab:green', 'FAR': 'tab:red'}, # 为Metric指定颜色
                        sharey=False, # Y轴不共享，因为CSI和FAR范围不同
                        legend_out=True) # 将图例画在图表外部

# 2. 使用 .map() 将绘图函数应用到每个子图
# 第一个参数是绘图函数 (例如 plt.plot, sns.lineplot, sns.scatterplot)
# 后续参数是传递给该绘图函数的列名 (来自data)
g_facet.map(sns.lineplot, 'Threshold', 'Score', marker='o', linewidth=1.5, markersize=5)
# 你也可以用 plt.plot
# g_facet.map(plt.plot, 'Threshold', 'Score', marker='o', linewidth=1.5, markersize=5)


# 3. 添加图例 (如果 map 中用的函数没有自动生成图例，或者想自定义)
g_facet.add_legend(title='Metric')

# 4. 设置整体标题和调整布局
g_facet.fig.suptitle('Performance Metrics vs. Threshold by Feature Set', fontsize=16, y=1.03) # y调整标题位置
g_facet.set_axis_labels('Prediction Threshold', 'Score') # 设置所有子图的轴标签
g_facet.set_titles(col_template="{col_name}") # 设置每个子图的标题，{col_name}会被替换为Feature Set的值

plt.tight_layout(rect=[0, 0, 1, 0.96]) # 为主标题调整布局
# plt.show()
```

**思考与你的项目：**

*   **多产品性能对比**: 你的 **第6、8、10节** 涉及多个降雨产品在不同阈值下的 POD, FAR, CSI。你可以使用 `FacetGrid`，例如 `col='Product'`，`hue='Metric'`，然后在每个子图上绘制 `sns.lineplot('Threshold', 'Score')`。
*   **模型误差分析**: 如果你有不同区域或不同季节的模型误差数据，可以用 `FacetGrid` 按区域/季节分面，展示误差分布 (`g.map(sns.histplot, 'Error')`)。
*   `col_wrap` 非常适合当你的分类变量有很多类别时，避免图表过宽。

**2.5.2 `sns.pairplot` 和 `sns.PairGrid`**

`pairplot` 用于可视化数据集中多个数值变量两两之间的关系。对角线上通常是每个变量自身的分布图，非对角线上是两个变量之间的散点图。

*   `sns.pairplot(data, hue, vars, kind, diag_kind, palette, markers, ...)`:
    *   `data`: Pandas DataFrame。
    *   `hue`: 分类列名，用于着色。
    *   `vars`: (列表) 选择要绘制的数值列名。如果为 `None`，则使用所有数值列。
    *   `kind`: (`'scatter'`, `'kde'`, `'hist'`, `'reg'`) 非对角线子图的类型。
    *   `diag_kind`: (`'auto'`, `'hist'`, `'kde'`, `None`) 对角线子图的类型。

`PairGrid` 是 `pairplot` 更底层的接口，允许你对上三角、下三角和对角线分别使用不同的绘图函数。

```python
# 使用站点性能数据 df_sites_perf (包含 POD, FAR, Mean Annual Precip)
# 我们只选择这三个数值变量进行两两对比

# 1. 使用 pairplot
g_pair = sns.pairplot(df_sites_perf,
                      vars=['POD', 'FAR', 'Mean Annual Precip (mm)'], # 选择要展示的变量
                      hue='Product', # 按产品类型着色
                      diag_kind='kde', # 对角线上画KDE图
                      kind='scatter', # 非对角线上画散点图 (也可以用 'reg' 来加回归线)
                      palette='husl', # 'husl' 是一个对色盲友好的定性调色板
                      markers=['o', 's', 'D'], # 为不同产品指定不同标记
                      plot_kws={'alpha': 0.7, 's': 60, 'edgecolor':'k', 'linewidth':0.5}, # 传递给散点图的参数
                      diag_kws={'fill': True, 'alpha': 0.5}) # 传递给KDE图的参数

g_pair.fig.suptitle('Pairwise Relationships of Performance Metrics and Precipitation', fontsize=16, y=1.02)
# plt.show()


# 2. 使用 PairGrid 进行更细致的控制 (示例)
# g = sns.PairGrid(df_sites_perf, vars=['POD', 'FAR'], hue='Region')
# g.map_upper(sns.scatterplot, s=50, alpha=0.7) # 上三角用散点图
# g.map_lower(sns.kdeplot, fill=True, thresh=0.1) # 下三角用KDE图
# g.map_diag(sns.histplot, kde=True) # 对角线用直方图+KDE
# g.add_legend()
# g.fig.suptitle('Custom PairGrid Example', y=1.02)
# plt.show()
```

**思考与你的项目：**

*   **特征相关性探索**: 如果你的特征工程产生了很多数值特征，可以用 `pairplot` 快速查看它们之间的两两关系以及各自的分布，有助于发现共线性或有趣的模式。
*   **多指标关联**: 例如，在你的站点数据中，可以探索 POD, FAR, CSI 以及一些地理/气候因素（如海拔、年均降雨量）之间的两两关系。
*   如果变量过多，`pairplot` 可能会变得非常拥挤。这时需要有选择性地使用 `vars` 参数。

**2.5.3 `sns.jointplot` (回顾)**

我们之前在分布图中提到过 `sns.jointplot`。它其实也是一种特殊的多图网格，专门用于展示两个变量的联合分布和各自的边缘分布。

**你的任务与思考：**

1.  **运行本节的 `FacetGrid` 和 `pairplot` 示例代码。**
2.  **对于 `FacetGrid`**:
    *   尝试将 `col='Feature Set'` 改为 `row='Metric'`，`col='Feature Set'`，观察布局如何变化。
    *   尝试在 `.map()` 中使用不同的绘图函数，例如 `g_facet.map(plt.scatter, 'Threshold', 'Score', s=30)`。
3.  **对于 `pairplot`**:
    *   尝试将 `kind` 改为 `'reg'`，观察非对角线上的变化。
    *   尝试将 `diag_kind` 改为 `'hist'`。
4.  **构思你的项目应用**:
    *   **特征工程迭代效果展示**: 你在 README 2.2 节详细描述了 V1 到 V6 的特征迭代。如果每个版本都有一些关键性能指标（如在特定测试集上的 CSI、FAR、POD），你可以使用 `FacetGrid`（`col='Feature Set Version'`）来并列展示这些指标的某个方面（例如，指标值随某个参数变化的曲线，或者指标的分布）。
    *   **多源数据产品对比 (第6, 8, 10节)**: 你的表格 1-6 和统计特征表 7, 9，包含大量产品和指标。
        *   **对于表格 1-6 (POD, FAR, CSI vs. 阈值)**: 可以用 `FacetGrid`，`col='Product'`, `row='Metric'` (POD/FAR/CSI)，然后在每个子图上画线图 (`sns.lineplot('Threshold', 'Value')`)。
        *   **对于表格 7, 9 (基本统计特征)**: 如果你想比较多个产品在多个统计特征上的值（如均值、标准差、偏度），可以先将表格整理成长格式，然后用 `sns.catplot(kind='bar', x='Statistic', y='Value', col='Product', sharey=False)`。
        *   **对于表格10 (逐年性能)**: 可以用 `FacetGrid`，`col='Product'`, `row='Year'`，然后在每个子图里画性能随阈值变化的曲线。或者，`col='Product'`, `hue='Year'`，在一个图里画多年曲线。

多图网格是探索和呈现多变量、多子集数据关系的强大工具。Seaborn 使得创建这些复杂的图表变得相对容易。

完成这些练习后，请告诉我你的感受和问题。
到此，我们已经覆盖了 Seaborn 的一些核心功能，包括样式设置、分类图、关系图、分布图、矩阵图和多图网格。你现在应该能用 Seaborn 绘制出很多美观且信息丰富的统计图表了。

接下来，我们可以选择：

1.  **深入 Seaborn 的特定图表或参数**: 如果你对某个 Seaborn 图表类型或参数有特别的疑问或想深入了解。
2.  **学习如何将 Seaborn 图与 Matplotlib 的 Figure 和 Axes 对象更灵活地结合**: 例如，在一个复杂的 Matplotlib 子图布局中嵌入 Seaborn 图。
3.  **开始学习 Plotly**: 用于创建交互式图表。

考虑到你的项目目标是“复杂美观的图”，并且你已经对 Matplotlib 有了基础，学习如何将 Seaborn 和 Matplotlib 更好地结合，以便在 Seaborn 的便捷性基础上进行 Matplotlib 级别的精细调整，可能会很有价值。

### 第3步：Seaborn 与 Matplotlib 的协同工作

Seaborn 的强大之处在于它不仅能快速生成美观的统计图，而且由于其构建在 Matplotlib 之上，它返回的通常是 Matplotlib 的 `Axes` 对象（对于 Figure-level 函数如 `jointplot`, `pairplot`, `FacetGrid`, `catplot` 等，它们返回的是 Seaborn 特定的 Grid 对象，但这些 Grid 对象内部也包含了 Matplotlib 的 Figure 和 Axes）。这意味着你可以利用所有 Matplotlib 的精细控制方法来进一步定制 Seaborn 生成的图表。

#### 3.1 获取 Seaborn 图返回的 Axes 对象

*   **Axes-level 函数**: 大多数基本的 Seaborn 绘图函数（如 `sns.barplot`, `sns.lineplot`, `sns.scatterplot`, `sns.histplot`, `sns.boxplot`, `sns.heatmap` 等）都接受一个 `ax` 参数，允许你指定在哪个 Matplotlib `Axes` 上绘图。如果未提供 `ax`，它们通常会作用于 `plt.gca()` (get current Axes)，并且函数本身会返回它所绘制的那个 `Axes` 对象。

    ```python
    # 示例：获取 Axes-level 函数返回的 Axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5)) # 创建一个1行2列的Matplotlib子图布局

    # 在 ax1 上用 Seaborn 画条形图
    sns.barplot(x='Model Name', y='CSI', data=df_model_comparison, ax=ax1, palette='GnBu_d')
    ax1.set_title('CSI Scores (Seaborn on ax1)')
    ax1.tick_params(axis='x', rotation=30, ha='right')

    # 在 ax2 上用 Seaborn 画线图，并获取返回的 Axes (虽然我们已经有ax2了)
    # 如果不指定 ax，sns.lineplot 会画在 plt.gca() 上，并返回它
    returned_ax = sns.lineplot(x='Threshold', y='CSI', hue='Feature Set',
                               data=df_threshold_perf, ax=ax2, markers=True)
    # returned_ax 和 ax2 是同一个对象
    # print(returned_ax is ax2) # 应输出 True
    ax2.set_title('CSI vs. Threshold (Seaborn on ax2)')
    ax2.legend(title='Feature Set')

    plt.tight_layout()
    # plt.show()
    ```

*   **Figure-level 函数**: 像 `sns.jointplot`, `sns.pairplot`, `sns.FacetGrid`, `sns.relplot`, `sns.displot`, `sns.catplot` 这类函数，它们自己管理 Figure 和 Axes 的创建。它们返回的是一个特定的 Grid 对象 (如 `JointGrid`, `PairGrid`, `FacetGrid`)。
    *   你可以通过 Grid 对象的属性访问其内部的 Matplotlib Figure (`.fig`) 和 Axes (通常是 `.ax` 对于单个主图，`.axes` 对于网格中的所有 Axes，或 `.ax_joint`, `.ax_marg_x`, `.ax_marg_y` 对于 `JointGrid`)。

    ```python
    # 示例：从 Figure-level 函数获取 Figure 和 Axes
    g = sns.jointplot(data=df_sites_perf, x='FAR', y='POD', kind='kde', height=6)

    # g 是一个 JointGrid 对象
    # g.fig 是 Matplotlib Figure 对象
    # g.ax_joint 是中间的联合分布图的 Axes 对象
    # g.ax_marg_x 是顶部边缘分布图的 Axes 对象
    # g.ax_marg_y 是右侧边缘分布图的 Axes 对象

    # 我们可以对这些 Axes 进行 Matplotlib 操作
    g.ax_joint.set_xlabel('FAR (Customized)', fontweight='bold')
    g.ax_marg_x.set_title('Marginal X Distribution', fontsize=10)
    g.fig.suptitle('Joint KDE Plot with Matplotlib Customizations', y=1.03, fontsize=14)

    # plt.show()
    ```
    对于 `FacetGrid` (以及基于它的 `relplot`, `displot`, `catplot`)：
    ```python
    # g_facet = sns.FacetGrid(...) # 如之前的例子
    # g_facet.fig  # Figure 对象
    # g_facet.axes # 一个包含所有子图 Axes 的 NumPy 数组
    # ax_single_subplot = g_facet.axes[0,0] # 获取第一个子图的 Axes
    # ax_single_subplot.set_facecolor('lightyellow') # 例如，改变第一个子图的背景色
    ```

#### 3.2 使用 Matplotlib 方法定制 Seaborn 图

一旦你获取了目标 `Axes` 对象，就可以使用所有前面学过的 Matplotlib 方法进行定制。

**示例：定制一个 Seaborn 条形图**

```python
# 沿用 df_model_comparison 数据
plt.style.use('seaborn-v0_8-whitegrid') # 使用一个Seaborn样式作为基础

fig, ax = plt.subplots(figsize=(10, 6))

# 1. 用 Seaborn 创建基础条形图
sns.barplot(x='Model Name', y='CSI', data=df_model_comparison, ax=ax,
            palette='Spectral', edgecolor='black', linewidth=1) # Spectral是一个发散调色板

# 2. 使用 Matplotlib 方法进行精细调整
ax.set_title('Model CSI Comparison (Seaborn + Matplotlib)', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Machine Learning Model', fontsize=13, labelpad=10)
ax.set_ylabel('Critical Success Index (CSI)', fontsize=13, labelpad=10)

ax.tick_params(axis='x', rotation=40, ha='right', labelsize=11)
ax.tick_params(axis='y', labelsize=11, direction='in', length=5)

# 设置Y轴范围和更精细的刻度
max_csi = df_model_comparison['CSI'].max()
ax.set_ylim(0, max_csi * 1.15)
ax.set_yticks(np.arange(0, max_csi * 1.1, 0.1)) # 每隔0.1一个刻度
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) # Y轴刻度标签格式为两位小数

# 隐藏顶部和右侧的轴线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.2)
ax.spines['bottom'].set_linewidth(1.2)

# 在每个条形上添加数值标签 (Matplotlib 方法)
for p in ax.patches:
    value = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., # x 位置
            value + 0.01,                  # y 位置 (在条形上方一点)
            f'{value:.3f}',                # 文本内容
            ha='center', va='bottom',      # 对齐
            fontsize=9, color='dimgray', fontweight='medium')

# 添加水平参考线
ax.axhline(y=0.75, color='gray', linestyle='--', linewidth=1, label='Target CSI (0.75)')
ax.legend(fontsize='small', loc='upper left', frameon=False)

# 添加一些注释文本
ax.text(0.98, 0.95, 'Data: README Sec 2.3.1',
         transform=ax.transAxes, ha='right', va='top', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.3', fc='whitesmoke', alpha=0.7))

plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   **统一风格**: 如果你的论文中同时使用了 Matplotlib 直接绘制的图和 Seaborn 绘制的图，确保它们的整体风格（字体、字号、颜色方案等）是一致的。你可以通过在 Seaborn 的 `set_theme` 中使用 `rc` 参数来传递 Matplotlib 的 `rcParams` 设置，或者在 Seaborn 绘图后，获取 `Axes` 对象再用 Matplotlib 方法统一调整。
*   **复杂布局**: 如果你需要一个非常复杂的布局（例如，一些子图是 Matplotlib 画的，一些是 Seaborn 画的，或者子图大小不一），通常的流程是：
    1.  用 Matplotlib 的 `plt.figure()` 和 `fig.add_subplot()` (或者 `GridSpec` 来创建更复杂的非均匀网格) 来构建整体的 Figure 和 Axes 框架。
    2.  然后将每个 `Axes` 对象传递给相应的 Seaborn 绘图函数 (使用 `ax=` 参数)。
    3.  最后，对每个 `Axes` 或整个 `Figure` 进行最终的 Matplotlib 调整。
*   **你的图表**:
    *   对于 **2.3.1 节的模型对比表**，你可以用 `sns.barplot` 生成基础图，然后用 Matplotlib 添加精确的数值标签、调整轴线、添加参考线等。
    *   对于 **2.3.2 节的性能迭代曲线**，可以用 `sns.lineplot` 或 `sns.pointplot`，然后用 Matplotlib 调整图例位置、字体、线条样式、标记细节等。
    *   对于 **第10节的逐年统计特征**，如果你用 `sns.catplot(kind='bar', col='Year', row='Statistic')` 来展示，你可以访问 `g.axes[i,j]` 来对每个子图进行特定调整（例如，如果某个子图的Y轴范围特别大或小，可以单独设置）。

#### 3.3 实例：结合 Matplotlib 和 Seaborn 绘制你的项目图

让我们尝试可视化你的 **README 第6/8节 (产品性能评估)** 中的数据，例如，绘制不同产品在特定阈值下的 POD, FAR, CSI 对比。

```python
# 模拟部分数据 (来自你的长江流域表1,2,3，假设阈值为0.1mm/d)
product_perf_data = {
    'Product': ['CMORPH', 'CHIRPS', 'GSMAP', 'IMERG', 'PERSIANN', 'SM2RAIN', 'XGBoost_Opt'], # 加入你的模型结果
    'POD':     [0.5460,   0.4383,   0.6085,  0.7053,  0.6368,    0.9030,    0.9447], # 假设XGBoost在0.5阈值，0.1mm/d定义下
    'FAR':     [0.1905,   0.1935,   0.0613,  0.1947,  0.2467,    0.3088,    0.0335],
    'CSI':     [0.4838,   0.3966,   0.5852,  0.6025,  0.5269,    0.6434,    0.9147]
}
df_product_perf = pd.DataFrame(product_perf_data)

# 我们想为 POD, FAR, CSI 分别绘制条形图，并排显示
fig, axs = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False) # 1行3列的子图
# sharey=False 因为各项指标的范围不同

metrics_to_plot = ['POD', 'FAR', 'CSI']
y_labels = ['Probability of Detection (POD)', 'False Alarm Ratio (FAR)', 'Critical Success Index (CSI)']
palettes = ['Greens_d', 'Reds_d', 'Blues_d'] # 为每个指标选择一个调色板系列

for i, metric in enumerate(metrics_to_plot):
    current_ax = axs[i]
    sns.barplot(x='Product', y=metric, data=df_product_perf, ax=current_ax,
                palette=palettes[i], edgecolor='black', linewidth=0.8)

    current_ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='medium')
    current_ax.set_xlabel('') # X轴标签在底部总的说明即可，或者每个子图都有
    if i == 0: # 只在最左边的图显示Y轴主标签
        current_ax.set_ylabel('Score', fontsize=12)
    else:
        current_ax.set_ylabel('') # 其他子图不显示Y轴标签，避免重复

    current_ax.tick_params(axis='x', rotation=45, ha='right', labelsize=10)
    current_ax.tick_params(axis='y', labelsize=10)
    current_ax.grid(axis='y', linestyle=':', alpha=0.7) # 只显示水平网格线

    # 添加数值标签
    for p in current_ax.patches:
        current_ax.annotate(f"{p.get_height():.3f}",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom',
                            xytext=(0, 3), textcoords='offset points', fontsize=8.5)
    # 调整Y轴范围以适应标签
    if metric == 'FAR': # FAR 通常范围较小
        current_ax.set_ylim(0, df_product_perf[metric].max() * 1.2 + 0.05)
    else: # POD 和 CSI 通常范围较大
        current_ax.set_ylim(0, 1.05) # 假设POD/CSI最大为1
        current_ax.set_yticks(np.arange(0, 1.1, 0.2))


fig.suptitle('Rainfall Product & Model Performance (Yangtze Basin @ 0.1mm/d Rain Threshold, 0.5 Prob. Thr. for Model)', fontsize=16, fontweight='bold', y=1.02)
fig.text(0.5, -0.02, 'Product / Model', ha='center', va='center', fontsize=12) # 底部共享X轴标签

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局
# plt.show()
```

**要点总结：**

*   **Seaborn 用于快速生成结构**: `sns.barplot` 轻松创建了基本的条形图。
*   **Matplotlib 用于细节调整**:
    *   `fig, axs = plt.subplots(...)` 创建了子图布局。
    *   `ax.set_title()`, `ax.set_xlabel()`, `ax.set_ylabel()`, `ax.tick_params()` 等用于定制每个子图的标签、标题、刻度。
    *   `ax.patches` 和 `ax.annotate()` 用于在条形上添加精确的数值。
    *   `fig.suptitle()`, `fig.text()` 用于添加整体标题和说明。
    *   `plt.tight_layout()` (或 `fig.tight_layout()`) 最终调整布局。

**你的任务与思考：**

1.  **运行本节的示例代码**，理解 Seaborn 函数如何与 Matplotlib 的 Figure 和 Axes 对象交互。
2.  **尝试修改定制 Seaborn 条形图的示例**:
    *   改变 `palette`。
    *   调整数值标签的 `fontsize`, `xytext`。
    *   修改Y轴的刻度生成方式 (`ax.set_yticks()`)。
3.  **思考如何将这种组合方法应用于你的项目中更复杂的图表**:
    *   **多指标对比 (如上例)**: 清晰展示你的模型 (`XGBoost_Opt`) 相对于其他基准产品在多个指标上的优势。
    *   **误差特征空间分布 (结合第10节的思路)**: 你可以用 `cartopy` 或 `geopandas` 绘制地理底图 (这是 Matplotlib 的功能)，然后在特定的 `Axes` 上用 `sns.scatterplot` 或 `sns.kdeplot` 叠加表示误差大小或FP/FN发生频率的点或等值线，颜色用 Seaborn 的调色板。
    *   **特征重要性图**: 可以用 `sns.barplot` 画水平条形图显示特征重要性，然后用 Matplotlib 添加数值、排序、调整颜色等。

通过结合 Seaborn 的便捷性和 Matplotlib 的灵活性，你可以创建出既美观又高度定制化的论文图表。

完成练习后，请告诉我你的想法。我们差不多完成了 Seaborn 的核心内容。接下来，我们可以：
1.  讨论一些更高级的 Seaborn 主题，比如更复杂的调色板使用、或者 `catplot`/`relplot`/`displot` 等 Figure-level 函数的更多用法。
2.  开始学习 Plotly，用于创建交互式图表。
3.  或者，如果你觉得对静态图表的掌握已经比较有信心，我们可以回头看看你项目中还有哪些具体的数据想可视化，我们可以一起构思并尝试实现。

### 第4步：Seaborn 高级主题与技巧

我们已经学习了 Seaborn 的基础绘图函数以及如何与 Matplotlib 协同工作。现在我们来探讨一些更高级的主题和技巧，让你的 Seaborn 图表更上一层楼。

#### 4.1 更深入的调色板 (Palettes) 使用

Seaborn 提供了丰富的调色板选项，并且可以轻松创建和使用自定义调色板。

*   **Seaborn 内置调色板**:
    *   **定性型 (Qualitative)**: 用于区分没有顺序的类别。例如：`'deep'`, `'muted'`, `'pastel'`, `'bright'`, `'dark'`, `'colorblind'`, `'Set1'`, `'Set2'`, `'Set3'`, `'Paired'`, `'Accent'`, `'tab10'`, `'tab20'`。
    *   **顺序型 (Sequential)**: 用于表示有顺序的数据。例如：`'Blues'`, `'Greens'`, `'Reds'`, `'Oranges'`, `'Purples'`, `'YlGnBu'`, `'viridis'`, `'plasma'`, `'inferno'`, `'magma'`, `'cividis'`, `'rocket'`, `'mako'`。通常名称后加 `_d` 表示深色版本，加 `_r` 表示反转顺序。
    *   **发散型 (Diverging)**: 用于表示围绕中心点发散的数据。例如：`'coolwarm'`, `'RdBu_r'`, `'PiYG'`, `'PRGn'`, `'BrBG'`, `'vlag'`, `'icefire'`。
*   **创建自定义调色板**:
    *   `sns.color_palette(color_list)`: 从一个颜色列表（如 Matplotlib 颜色名称、十六进制码）创建调色板。
    *   `sns.light_palette(color, n_colors, reverse=False, as_cmap=False)`: 创建一个从浅到指定颜色的顺序调色板。
    *   `sns.dark_palette(color, n_colors, reverse=False, as_cmap=False)`: 创建一个从深到指定颜色的顺序调色板。
    *   `sns.diverging_palette(h_neg, h_pos, s=75, l=50, sep=1, n=6, center='light', as_cmap=False)`: 创建一个自定义的发散调色板，通过色调 (hue) 定义两端颜色。
*   **使用调色板**:
    *   在绘图函数中通过 `palette=` 参数指定。
    *   用 `sns.set_palette(palette_name_or_list)` 设置全局默认调色板。

**示例：为你的模型对比图自定义调色板**

```python
# 沿用 df_model_comparison 数据

# 1. 定义一个自定义的颜色列表 (例如，与你的机构或项目主题色相关)
my_custom_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
# 这些是tab10的前几个颜色，你可以替换成你自己的

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Model Name', y='CSI', data=df_model_comparison,
                 palette=my_custom_palette, # 使用自定义颜色列表
                 edgecolor='black')
ax.set_title('Model CSI Comparison with Custom Palette', fontsize=15)
ax.tick_params(axis='x', rotation=30, ha='right')
plt.tight_layout()
# plt.show()

# 2. 创建一个基于某个核心色的顺序调色板 (例如，你的 XGBoost 是蓝色主题)
xgboost_color = 'royalblue'
# 创建一个从浅蓝到深蓝的调色板，假设有6个模型，最后一个是XGBoost，给它最深的颜色
# 但这里我们直接用一个能体现变化的调色板来展示你的特征版本演进
feature_palette = sns.light_palette("seagreen", n_colors=len(df_threshold_perf['Feature Set'].unique()), reverse=False)

plt.figure(figsize=(10, 6))
sns.lineplot(x='Threshold', y='CSI', hue='Feature Set',
             data=df_threshold_perf,
             palette=feature_palette, # 使用生成的顺序调色板
             linewidth=2, marker='o')
plt.title('CSI vs. Threshold (Sequential Palette for Feature Sets)', fontsize=15)
plt.legend(title='Feature Set')
plt.tight_layout()
# plt.show()
```

**思考与你的项目**:

*   为你的不同降雨产品、不同模型、不同特征版本设定一套固定的、有区分度的颜色方案，并在所有相关图表中保持一致。这能极大提升论文的可读性和专业性。
*   如果展示某个指标随某个连续变量（如年份、阈值）的变化趋势，顺序调色板可以很好地强调这种递进关系。
*   如果展示误差或差异（如预测-观测），发散调色板是首选。

#### 4.2 Figure-level 函数 (`catplot`, `relplot`, `displot`) 的威力

我们之前接触了 `FacetGrid`，而 `catplot` (分类图), `relplot` (关系图), `displot` (分布图) 是更高层次的接口，它们内部使用 `FacetGrid` 来轻松创建分面图。它们共享很多参数，如 `data`, `x`, `y`, `hue`, `row`, `col`, `col_wrap`, `kind`。

*   **`sns.catplot(..., kind=type, ...)`**:
    *   `kind`: `'strip'`, `'swarm'`, `'box'`, `'violin'`, `'boxen'`, `'point'`, `'bar'`, `'count'`。
*   **`sns.relplot(..., kind=type, ...)`**:
    *   `kind`: `'scatter'`, `'line'`。
*   **`sns.displot(..., kind=type, ...)`**:
    *   `kind`: `'hist'`, `'kde'`, `'ecdf'`。

**示例：使用 `catplot` 比较不同模型的多个性能指标**

首先，我们需要将 `df_model_comparison` 数据转换为更适合分面的长格式。

```python
df_model_comparison_long = pd.melt(df_model_comparison,
                                   id_vars=['Model Name'],
                                   value_vars=['Accuracy', 'POD', 'FAR', 'CSI'],
                                   var_name='Metric',
                                   value_name='Score')

# 使用 catplot 创建分面条形图
# col='Metric': 每个指标一列
# sharey=False: 不同指标的Y轴范围不同
g_cat = sns.catplot(x='Model Name', y='Score', col='Metric',
                    data=df_model_comparison_long,
                    kind='bar', # 指定图表类型为条形图
                    height=4, aspect=1, # 控制子图大小
                    palette='Set2',
                    sharey=False, # Y轴不共享
                    col_wrap=2) # 每行最多2个图

g_cat.set_axis_labels("Model", "Score Value")
g_cat.set_titles("{col_name}") # 设置每个子图的标题
g_cat.set_xticklabels(rotation=30, ha='right') # 对所有子图的X轴标签进行旋转
g_cat.fig.suptitle('Overall Model Performance Comparison', fontsize=16, y=1.03)
plt.tight_layout(rect=[0,0,1,0.95])
# plt.show()
```

**思考与你的项目**:

*   **你的 README 第6、8节的表格 (产品性能随阈值变化)**:
    *   可以先将表格数据整理成长格式 (列：Product, Threshold, Metric, Value)。
    *   然后使用 `sns.relplot(data=df_long, x='Threshold', y='Value', hue='Product', col='Metric', kind='line', facet_kws={'sharey': False})`。这样可以为每个指标 (POD, FAR, CSI) 生成一个子图，子图内用不同颜色的线表示不同产品随阈值的变化。
*   **你的 README 第10节 (逐年性能)**:
    *   可以 `sns.relplot(data=df_yearly_long, x='Threshold', y='Value', hue='Year', col='Product', row='Metric', kind='line', facet_kws={'sharey': False})`，这将创建一个非常全面的视图。

#### 4.3 临时样式控制 (`with sns.axes_style()` 和 `with sns.plotting_context()`)

有时你只想为一个特定的图表临时改变样式或绘图上下文，而不影响全局设置。可以使用 `with` 语句：

```python
# 假设全局样式是'whitegrid'
# sns.set_theme(style='whitegrid')

fig, ax1 = plt.subplots()
ax1.plot(days, csi_scores, label='CSI (Global Style)')

# 创建第二个图，临时使用不同样式
with sns.axes_style("darkgrid"): # 临时的背景和网格样式
    with sns.plotting_context("poster"): # 临时的字体和元素大小 (海报大小)
        fig2, ax2 = plt.subplots()
        sns.lineplot(x='Threshold', y='CSI', hue='Feature Set', data=df_threshold_perf, ax=ax2, palette='flare')
        ax2.set_title('CSI Plot (Temporary Darkgrid & Poster Context)')
        ax2.legend()

# 第三个图会回到全局样式
fig3, ax3 = plt.subplots()
ax3.plot(days, far_scores, label='FAR (Global Style)')

# plt.show()
```

#### 4.4 与 Matplotlib 对象更深度的交互

*   **修改图例**: `FacetGrid` 等返回的 Grid 对象有 `.legend` 属性，但有时你可能想用 Matplotlib 的方式更精细地控制图例。
    *   `g.add_legend()`: Seaborn 的标准图例。
    *   如果你想用 Matplotlib 的 `fig.legend()` 或 `ax.legend()`，你可能需要从 `g.legend_data` (一个字典，键是标签，值是Matplotlib的handle) 中获取handles和labels。

    ```python
    # g_facet = sns.FacetGrid(...)
    # g_facet.map(...)
    # # g_facet.add_legend() # Seaborn默认图例

    # # 如果想用Matplotlib自定义图例 (更复杂，但更灵活)
    # handles = [plt.Line2D([0], [0], marker='o', color=color, linestyle='') for color in g_facet.palette.values()] # 创建handles
    # labels = g_facet.hue_names # 获取hue的类别名
    # g_facet.fig.legend(handles, labels, title=g_facet.hue_kws.get('title', 'Hue'), loc='upper right', bbox_to_anchor=(1,1))
    ```

*   **遍历 `FacetGrid` 的 Axes**:
    ```python
    # g_facet = sns.FacetGrid(...)
    # g_facet.map(...)
    # for ax_sub in g_facet.axes.flat: # 遍历所有子图的Axes
    #     ax_sub.axhline(0.5, ls='--', color='grey') # 例如，在每个子图上画一条水平线
    #     if ax_sub.get_ylabel() == "Score": # 例如，只修改名为"Score"的Y轴标签
    #         ax_sub.set_ylabel("Performance Score", style='italic')
    ```

**你的任务与思考：**

1.  **练习自定义调色板**:
    *   为你项目中的不同降雨产品或模型版本设计一套你认为合适的颜色方案。
    *   尝试使用 `sns.light_palette` 或 `sns.dark_palette` 创建一个基于单一核心色的顺序调色板。
2.  **熟练使用 `catplot`, `relplot`, `displot`**:
    *   选择你 README 中的一个复杂表格 (如第6、8、10节的多产品多指标性能数据)。
    *   尝试将其整理成长格式 Pandas DataFrame。
    *   使用 `catplot` (如果X轴是分类，如产品名) 或 `relplot` (如果X轴是连续，如阈值) 进行分面可视化。实验 `row`, `col`, `hue`, `col_wrap` 等参数。
3.  **练习 `with sns.axes_style()`**: 创建一个图，然后在其下方用 `with` 语句创建一个使用完全不同风格的图，以体会临时样式的效果。
4.  **尝试修改 `FacetGrid` 子图的属性**: 在 `FacetGrid` 绘图后，遍历 `g.axes.flat`，尝试修改每个子图的标题、Y轴标签或添加一条参考线。

掌握了这些高级技巧后，你将能更灵活地运用 Seaborn 来满足你论文中多样化的可视化需求，并能确保图表风格的统一性和专业性。

至此，关于 Seaborn 的核心内容我们已经讲得比较全面了。你现在应该能够：
*   设置 Seaborn 的全局样式和调色板。
*   使用 Seaborn 的主要绘图函数（分类、关系、分布、矩阵）来可视化你的数据。
*   创建复杂的分面图表。
*   结合 Matplotlib 对 Seaborn 生成的图表进行精细调整。

接下来，我们是时候转向另一个强大的绘图库 Plotly 了，它专注于创建交互式图表。这对于在线补充材料、项目演示或个人网站展示你的研究成果会非常有用。

## 第三部分：Plotly - 现代交互式图表

Plotly 是一个功能强大的 Python 绘图库，以创建美观、交互式的图表而闻名。这些图表可以轻松地嵌入到网页、Jupyter Notebooks 中，或者导出为静态图片。Plotly 主要有两个 API 级别：

1.  **Plotly Express (`px`)**: 一个高级、简洁的接口，非常易于上手，几行代码就能创建出复杂的图表。它大大简化了 Plotly 的使用，是 Plotly 官方推荐的起点。
2.  **Plotly Graph Objects (`go`)**: 一个更底层、更灵活的接口，允许你对图表的每一个组件（数据轨迹、布局、轴、标注等）进行完全的、精细的控制。Plotly Express 实际上是在后台使用 Graph Objects 来构建图表的。

对于论文，虽然最终提交的通常是静态图片，但 Plotly 在以下方面对你的项目可能非常有价值：

*   **探索性数据分析 (EDA)**: 交互性（如悬停提示、缩放、平移、点选高亮）可以帮助你更深入地理解数据模式和异常点。
*   **在线补充材料/演示**: 可以将交互式 Plotly 图表导出为 HTML 文件，作为论文的在线补充材料，或者在演示文稿中嵌入，让观众能与数据互动。
*   **某些复杂图表的便捷创建**: Plotly Express 在创建某些类型的图表（如动画、地理图、平行坐标图、三维图）时，语法可能比 Matplotlib/Seaborn 更简洁。

### 第1步：准备工作与基本概念

#### 1.1 导入库

```python
import plotly.express as px  # Plotly Express, 高级接口
import plotly.graph_objects as go # Graph Objects, 底层接口
import pandas as pd
import numpy as np

# 在Jupyter Notebook/Lab或Google Colab中，Plotly图表通常会直接渲染出来。
# 如果在某些环境中不显示，可能需要设置渲染器：
# import plotly.io as pio
# pio.renderers.default = "notebook_connected" # or "colab", "jupyterlab", "iframe", "browser"
# 在 Colab 中，通常会自动选择 "colab" 渲染器。
```

#### 1.2 Plotly Express (`px`) 的核心思想

Plotly Express 的函数通常接受一个 Pandas DataFrame 作为 `data_frame` 参数，然后通过列名来指定 `x`, `y`, `color`, `size`, `symbol`, `facet_row`, `facet_col` 等视觉编码。这使得从数据到图表非常直观。

每个 `px` 函数都会返回一个 `plotly.graph_objects.Figure` 对象。

#### 1.3 Plotly Graph Objects (`go`) 的核心思想

使用 `go` 时，你需要更明确地构建图表的各个部分：
*   **Traces (轨迹)**: 代表图表中的一组数据及其可视化方式（如 `go.Scatter` 代表散点或线，`go.Bar` 代表条形等）。每个 Trace 都有很多属性可以配置。
*   **Layout (布局)**: 控制图表的整体外观和非数据元素（如标题、轴标签、图例、颜色轴、形状、标注等）。
*   **Figure**: `go.Figure` 对象是 Traces 和 Layout 的容器。你可以创建一个空的 Figure，然后用 `fig.add_trace()` 添加轨迹，用 `fig.update_layout()` 修改布局。

#### 1.4 示例数据 (沿用之前的模拟数据)

```python
# 模型对比数据
model_data = {
    'Model Name': ['KNN', 'SVM', 'Random Forest', 'LightGBM', 'Gaussian NB', 'XGBoost (Default)', 'XGBoost (Optuna)'], # 添加一个Optuna结果
    'Accuracy': [0.7917, 0.8021, 0.8408, 0.8366, 0.7019, 0.8819, 0.9456],
    'POD': [0.7839, 0.7496, 0.8378, 0.8221, 0.5799, 0.8880, 0.9447],
    'FAR': [0.1308, 0.0819, 0.1001, 0.0929, 0.0909, 0.0819, 0.0335],
    'CSI': [0.7012, 0.7026, 0.7665, 0.7582, 0.5481, 0.8228, 0.9147]
}
df_model_comparison_px = pd.DataFrame(model_data)

# 阈值性能数据
threshold_data = {
    'Feature Set': ['V1']*5 + ['V4']*5 + ['V6 Optuna']*5,
    'Threshold': [0.3, 0.4, 0.5, 0.6, 0.7] * 3,
    'CSI': [0.7975, 0.7941, 0.7820, 0.7615, 0.7321,
            0.8171, 0.8101, 0.8011, 0.7800, 0.7600,
            0.9168, 0.9177, 0.9147, 0.9084, 0.8992],
    'FAR': [0.1280, 0.1030, 0.0823, 0.0646, 0.0479,
            0.0675, 0.0572, 0.0482, 0.0400, 0.0300,
            0.0526, 0.0416, 0.0335, 0.0270, 0.0210],
    'POD': [0.9032, 0.8738, 0.8410, 0.8037, 0.7601, # V1 POD (假设)
            0.9100, 0.8800, 0.8520, 0.8200, 0.7900, # V4 POD (假设)
            0.9660, 0.9558, 0.9447, 0.9319, 0.9169]  # V6 Optuna POD
}
df_threshold_perf_px = pd.DataFrame(threshold_data)
```

### 第2步：使用 Plotly Express (`px`) 快速创建交互图表

#### 2.1 条形图 (`px.bar`)

```python
# 使用 Plotly Express 绘制条形图对比模型CSI
fig_bar_px = px.bar(df_model_comparison_px,
                    x='Model Name',
                    y='CSI',
                    color='Model Name', # 每个条形用不同颜色，并自动生成图例
                    labels={'CSI': 'Critical Success Index (CSI)', 'Model Name': 'Model'}, # 自定义标签
                    title='Model Performance Comparison (CSI Scores)',
                    text_auto='.3f', # 自动在条形上显示数值，格式为3位小数
                    height=500)     # 设置图像高度

# 进一步定制布局 (虽然px已经创建了Figure，但我们可以用 .update_layout() 修改)
fig_bar_px.update_layout(
    xaxis_title='Machine Learning Model',
    yaxis_title='CSI Score',
    title_x=0.5, # 标题居中
    legend_title_text='Models',
    # xaxis_tickangle=-30 # 旋转X轴标签
)
fig_bar_px.update_xaxes(tickangle=-30) # 另一种旋转X轴标签的方式

fig_bar_px.show()
```

**交互性**:

*   **悬停**: 鼠标悬停在条形上会显示详细数据。
*   **图例交互**: 点击图例中的项可以显示/隐藏对应的条形。
*   **工具栏**: 图表右上角有缩放、平移、下载等工具。

#### 2.2 线图 (`px.line`)

```python
# 使用 Plotly Express 绘制不同特征集下CSI随阈值变化的线图
fig_line_px = px.line(df_threshold_perf_px,
                      x='Threshold',
                      y='CSI',
                      color='Feature Set',      # 按 'Feature Set' 用不同颜色绘制线条
                      symbol='Feature Set',     # 按 'Feature Set' 用不同标记 (可选)
                      markers=True,             # 显示数据点标记
                      labels={'CSI': 'CSI Score', 'Threshold': 'Prediction Threshold'},
                      title='CSI vs. Prediction Threshold by Feature Set',
                      height=500)

fig_line_px.update_layout(
    legend_title_text='Feature Set',
    title_x=0.5,
    yaxis_range=[df_threshold_perf_px['CSI'].min() * 0.95, df_threshold_perf_px['CSI'].max() * 1.05] # 设置Y轴范围
)
fig_line_px.update_traces(linewidth=2.5) # 更新所有轨迹的线宽

fig_line_px.show()
```

**思考与你的项目**:

*   你的 **README 2.3.2 节** 中的多表格数据（性能 vs 阈值，按特征集）非常适合用 `px.line` 绘制。`color` 和 `symbol` (或 `line_dash`) 参数可以轻松区分不同组。
*   **悬停提示 (`hover_data`)**: 你可以在 `px.line` 或 `px.bar` 中加入 `hover_data=['POD', 'FAR']` (假设这些列在DataFrame中)，这样鼠标悬停时除了x,y值，还会显示POD和FAR的值。

#### 2.3 散点图 (`px.scatter`)

```python
# 使用之前的站点性能数据 (df_sites_perf，需要重新生成或加载)
np.random.seed(10)
num_sites = 50
sites_data_px = {
    'Product': np.random.choice(['CMORPH', 'GSMAP', 'IMERG'], size=num_sites),
    'Region': np.random.choice(['East', 'West', 'Central'], size=num_sites),
    'POD': np.random.rand(num_sites) * 0.4 + 0.5,
    'FAR': np.random.rand(num_sites) * 0.15 + 0.01,
    'Mean Annual Precip (mm)': np.random.rand(num_sites) * 1000 + 800,
    'Elevation (m)': np.random.randint(100, 3000, size=num_sites) # 添加一个Elevation列
}
df_sites_perf_px = pd.DataFrame(sites_data_px)


fig_scatter_px = px.scatter(df_sites_perf_px,
                            x='FAR',
                            y='POD',
                            color='Product',  # 按产品着色
                            size='Mean Annual Precip (mm)', # 点的大小表示年均降雨量
                            symbol='Region',  # 点的形状表示区域
                            hover_name='Product', # 悬停时显示的名称
                            hover_data={'FAR':':.3f', 'POD':':.3f', 'Elevation (m)':True}, # 控制悬停信息和格式
                            title='Site-Specific Product Performance (POD vs. FAR)',
                            labels={'FAR':'False Alarm Ratio', 'POD':'Probability of Detection'},
                            color_discrete_sequence=px.colors.qualitative.Plotly, # 使用Plotly的定性调色板
                            # symbol_sequence=px.colors.symbol_sequences. khác nhau # 可以选择不同符号序列
                            height=600)

fig_scatter_px.update_layout(title_x=0.5, legend_tracegroup_general_attrs_visible=False) # 简化图例分组
# legend_tracegroup_general_attrs_visible=False 避免图例中出现 (Product, Region) 这样的组合标题

fig_scatter_px.show()
```

**思考与你的项目**:

*   **多维数据探索**: 如果你有站点级别的数据，包含多个特征（如经纬度、海拔、不同产品/模型的性能指标），`px.scatter` 可以让你通过颜色、大小、形状同时可视化多个维度。
*   **你的降雨产品原始统计特征 (第7, 9节)**: 例如，你可以画一个散点图，X轴是产品的平均值，Y轴是标准差，颜色代表产品类型，点的大小代表与CHM的相关系数。

#### 2.4 热力图 (`px.imshow`)

Plotly Express 的 `px.imshow` 主要用于显示图像或矩阵数据，可以作为热力图使用。对于相关系数矩阵等，它也非常方便。

```python
# 使用之前的相关系数矩阵 df_corr
fig_heatmap_px = px.imshow(df_corr,
                           text_auto='.2f', # 在单元格中显示数值，格式为两位小数
                           aspect="auto",   # 'auto' 使单元格为方形，'equal'保持数据比例
                           color_continuous_scale='RdBu_r', # 发散型调色板，_r表示反转
                           zmin=-1, zmax=1, # 确保颜色映射对称于0
                           labels=dict(color="Correlation"), # 颜色条标签
                           title='Correlation Matrix of Rainfall Products (Plotly Express)',
                           height=600)

fig_heatmap_px.update_layout(title_x=0.5)
fig_heatmap_px.update_xaxes(side="bottom", tickangle=-45) # 将X轴刻度标签放到下方并旋转

fig_heatmap_px.show()
```

**思考与你的项目**:

*   你的 **第7节和第9节** 的相关系数矩阵是直接的应用场景。
*   你的 **`data/intermediate/` 下的 `.mat` 文件** 存储了格点降雨数据 (如CMORPH_2016.mat 是 144x256 矩阵)。你可以选择某一天或某个月的平均降雨量，用 `px.imshow` 绘制其空间分布热力图。
    ```python
    # 假设 a_day_rain 是一个 144x256 的 NumPy 数组
    # fig_rain_map = px.imshow(a_day_rain,
    #                          color_continuous_scale='Blues', # 顺序型调色板
    #                          labels=dict(color="Rainfall (mm)"),
    #                          title='Rainfall Distribution on YYYY-MM-DD')
    # fig_rain_map.show()
    ```
*   **混淆矩阵** 也可以用 `px.imshow` 可视化。

#### 2.5 分面图 (Faceting)

Plotly Express 的大部分绘图函数都支持 `facet_row` 和 `facet_col` 参数，用于轻松创建分面图。

```python
# 使用 df_threshold_perf_px 数据，按 Feature Set 分面，每行一个 Feature Set
fig_facet_line_px = px.line(df_threshold_perf_px,
                            x='Threshold',
                            y='CSI',
                            color='Feature Set', # 颜色也按Feature Set (可选，如果分面了，颜色可能不需要再区分)
                            facet_row='Feature Set', # 按 Feature Set 分行
                            markers=True,
                            height=600, # 总高度
                            title='CSI vs. Threshold (Faceted by Feature Set)')

fig_facet_line_px.update_layout(title_x=0.5)
# 由于每个子图的Y轴标签都是 "CSI"，Plotly Express 可能会自动隐藏重复的轴标签
# 你可以用 fig.for_each_yaxis(lambda yaxis: yaxis.update(title_text='')) 来清空所有Y轴标签（如果需要）
# 或 fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1])) 来简化子图标题

fig_facet_line_px.show()

# 另一个例子：模型性能对比，按指标分列
df_model_comp_long_px = pd.melt(df_model_comparison_px, id_vars=['Model Name'],
                                value_vars=['POD', 'FAR', 'CSI'],
                                var_name='Metric', value_name='Score')

fig_facet_bar_px = px.bar(df_model_comp_long_px,
                          x='Model Name',
                          y='Score',
                          color='Model Name',
                          facet_col='Metric', # 按 Metric 分列
                          facet_col_wrap=2,   # 每行最多2个图
                          height=700,
                          labels={'Score': 'Metric Score'},
                          title='Model Performance Metrics Comparison')

fig_facet_bar_px.update_layout(title_x=0.5)
fig_facet_bar_px.update_xaxes(matches=None, showticklabels=True, tickangle=-45) # 确保所有X轴标签显示并旋转
fig_facet_bar_px.for_each_yaxis(lambda y: y.update(matches=None, showticklabels=True)) # 确保所有Y轴标签显示
fig_facet_bar_px.update_traces(texttemplate='%{y:.3f}', textposition='outside') # 条形上的数值

fig_facet_bar_px.show()
```

**你的任务与思考：**

1.  **运行本节的所有 Plotly Express 示例代码**，体验其交互性（悬停、缩放、图例点击）。
2.  **对于 `px.bar` 和 `px.line`**:
    *   尝试修改 `color` 参数，使用不同的列名或不使用它。
    *   添加 `hover_data` 参数，包含你数据中其他相关的列。
    *   对于线图，尝试 `line_dash='Feature Set'` 参数，看看线条样式如何变化。
3.  **对于 `px.scatter`**:
    *   尝试将 `size` 或 `symbol` 映射到不同的列，观察效果。
    *   修改 `color_discrete_sequence` 或 `color_continuous_scale`。
4.  **对于 `px.imshow`**:
    *   如果你的相关系数矩阵是对称的，颜色条的中心应该在0。确保 `zmin`, `zmax` 和 `color_continuous_scale` (如 `'RdBu_r'`) 设置正确。
5.  **对于分面图**:
    *   尝试 `facet_row` 和 `facet_col` 的不同组合。
    *   使用 `facet_col_wrap` 控制换行。
    *   用 `fig.for_each_annotation(...)` 或 `fig.for_each_xaxis(...)` / `fig.for_each_yaxis(...)` 来统一修改所有子图的标题或轴属性。
6.  **思考你的项目**:
    *   **交互式探索**: 哪些数据最适合用 Plotly Express 进行交互式探索？例如，站点级别的多变量数据、时间序列数据随阈值变化的曲线等。
    *   **在线补充材料**: 哪些图表如果做成交互式的 HTML 文件，能更好地辅助你的论文读者理解结果？（例如，可以交互选择不同产品/模型/特征集，查看其性能曲线）。
    *   **复杂分面**: 你的项目中有很多可以按产品、按指标、按年份、按区域进行分面的数据，Plotly Express 可以很方便地实现。

Plotly Express 是一个非常强大的工具，尤其适合快速创建具有丰富视觉编码和交互性的图表。它的语法简洁，与 Pandas DataFrame 结合得很好。

完成练习后，请告诉我你的感受。接下来，我们将简要介绍 Plotly Graph Objects (`go`)，了解如何用它来进行更底层的、更精细的图表控制（类似于 Matplotlib 的面向对象接口），以及如何将 Plotly Express 生成的 Figure 对象转换为 Graph Objects 进行修改，最后是如何保存 Plotly 图表为静态图片和 HTML。

### 第3步：使用 Plotly Graph Objects (`go`) 进行精细控制与定制

虽然 Plotly Express (`px`) 非常便捷，但有时你可能需要对图表的某些特定细节进行更深层次的控制，这时就需要用到 Plotly Graph Objects (`go`)。实际上，`px` 生成的图表背后就是由 `go` 的组件构成的。

一个 `go.Figure` 对象主要由两部分组成：

1.  **`data`**: 一个包含一个或多个 "trace" 对象的列表。每个 trace 代表图上的一组数据及其可视化方式（如散点、线、条形等）。常见的 trace 类型有 `go.Scatter`, `go.Bar`, `go.Histogram`, `go.Heatmap` 等。
2.  **`layout`**: 一个 `go.Layout` 对象 (或其属性构成的字典)，描述了图表的非数据部分，如标题、轴、图例、颜色、形状、标注等。

#### 3.1 创建一个简单的 `go.Figure`

```python
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# 沿用之前的模拟数据
days = np.arange(1, 31)
csi_scores_go = np.random.rand(30) * 0.3 + 0.6
far_scores_go = np.random.rand(30) * 0.05 + 0.01

# 1. 创建 Figure 对象
fig_go_simple = go.Figure()

# 2. 添加第一个 Trace (CSI)
fig_go_simple.add_trace(go.Scatter(
    x=days,
    y=csi_scores_go,
    mode='lines+markers', # 'lines', 'markers', 'text', 'lines+markers'
    name='CSI Score',      # 图例中显示的名称
    line=dict(color='firebrick', width=2), # 线条属性
    marker=dict(symbol='circle', size=8, color='firebrick', line=dict(width=1, color='darkred')) # 标记属性
))

# 3. 添加第二个 Trace (FAR) - 假设我们想让它在第二个Y轴上
# 要创建第二个Y轴，我们需要在layout中定义它，并在这里指定 trace 使用哪个Y轴
fig_go_simple.add_trace(go.Scatter(
    x=days,
    y=far_scores_go,
    mode='lines+markers',
    name='FAR Score',
    yaxis='y2', # 指定使用名为 'y2' 的Y轴
    line=dict(color='royalblue', width=2, dash='dash'), # dash: 'solid', 'dot', 'dash', 'longdash', ...
    marker=dict(symbol='square', size=7, color='royalblue')
))

# 4. 更新 Layout
fig_go_simple.update_layout(
    title_text='Daily CSI and FAR Scores (Graph Objects)',
    title_x=0.5,
    xaxis_title_text='Day of Month',
    yaxis_title_text='CSI Score', # 第一个Y轴的标题
    yaxis_color='firebrick',       # 第一个Y轴的颜色

    # 定义第二个Y轴 (y2)
    yaxis2=dict(
        title='FAR Score',
        titlefont=dict(color='royalblue'),
        tickfont=dict(color='royalblue'),
        overlaying='y', # 关键：让这个Y轴覆盖在第一个Y轴 'y' 之上
        side='right',   # 将这个Y轴放在右边
        anchor='x',     # 确保它与主X轴对齐
        range=[0, far_scores_go.max()*1.2] # 设置范围
    ),

    legend_title_text='Metrics',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.7)'), # 图例位置和背景
    height=500,
    # template='plotly_white' # 可以应用模板
)

fig_go_simple.show()
```

**代码解释与 `go` 的常用属性:**

*   **`go.Scatter(...)`**:
    *   `mode`: 控制是画线、点还是都有。
    *   `name`: 图例名称。
    *   `line`: 一个字典，包含如 `color`, `width`, `dash` (线条样式) 等属性。
    *   `marker`: 一个字典，包含如 `symbol` (标记形状: `'circle'`, `'square'`, `'diamond'`, `'cross'`, `'x'`, `'triangle-up'`, etc.), `size`, `color`, `opacity`, `line` (标记边框线) 等属性。
    *   `yaxis='y2'` (或 `'y3'`, etc.): 指定该 trace 使用哪个 Y 轴。对应的 Y 轴需要在 `layout` 中定义。
*   **`fig.update_layout(...)`**:
    *   `title_text`, `xaxis_title_text`, `yaxis_title_text`: 设置标题和轴标签。
    *   `yaxis_color`, `yaxis_tickfont_color`, `yaxis_titlefont_color`: 控制轴及标签颜色。
    *   **多Y轴定义**: 通过 `yaxis2=dict(...)`, `yaxis3=dict(...)` 来定义额外的Y轴。
        *   `overlaying='y'`: (对于 `yaxis2`) 表示 `y2` 覆盖在主Y轴 `y` 上（共享X轴）。
        *   `side='right'`: 将该轴放在右侧。
        *   `anchor='x'`: 如果 `overlaying='y'`，通常 `anchor='x'`。如果 `overlaying='free'`，则可以设置 `position`。
        *   `position` (0-1): 如果 `overlaying='free'`，设置轴相对于绘图区域的位置。
    *   `legend`: 一个字典，控制图例的位置 (`x`, `y`, `xanchor`, `yanchor`), 方向 (`orientation`), 背景色 (`bgcolor`), 边框 (`bordercolor`, `borderwidth`) 等。
    *   `template`: 可以应用预设的 Plotly 主题模板，如 `'plotly'`, `'plotly_white'`, `'ggplot2'`, `'seaborn'`, `'plotly_dark'`。
    *   `margin`: `dict(l=px, r=px, t=px, b=px)` 控制图表边缘的留白。
    *   `paper_bgcolor`, `plot_bgcolor`: 画布和绘图区域的背景色。

#### 3.2 修改 Plotly Express 生成的 Figure 对象

Plotly Express 生成的 `fig` 对象就是一个 `go.Figure` 对象。因此，你可以先用 `px` 快速创建图表，然后用 `go` 的方法进行精细调整。

```python
# 1. 使用 Plotly Express 创建基础图表
fig_px_to_go = px.scatter(df_sites_perf_px,
                          x='FAR', y='POD', color='Product', size='Mean Annual Precip (mm)',
                          title='Site Performance (px base, go customized)')

# 2. 使用 Graph Objects 方法进行修改
fig_px_to_go.update_layout(
    legend_orientation='h', # 图例水平排列
    legend_yanchor='bottom', legend_y=1.02,
    legend_xanchor='right', legend_x=1,
    xaxis=dict(
        title_text='False Alarm Ratio (Custom)',
        showgrid=True, gridwidth=1, gridcolor='LightPink',
        zeroline=True, zerolinewidth=2, zerolinecolor='Red' # 强调x=0线
    ),
    yaxis_title_font_size=15,
    plot_bgcolor='lightyellow'
)

# 修改特定轨迹的属性 (Plotly Express通常会将每个hue类别创建为一个trace)
# fig_px_to_go.data 是一个包含所有trace的元组
# 假设第一个trace是CMORPH
if len(fig_px_to_go.data) > 0:
    fig_px_to_go.data[0].marker.symbol = 'diamond' # 将CMORPH的标记改为菱形
    fig_px_to_go.data[0].marker.size = df_sites_perf_px.loc[df_sites_perf_px['Product']=='CMORPH', 'Mean Annual Precip (mm)'] / 50 + 5 # 调整大小逻辑

# 添加标注
fig_px_to_go.add_annotation(
    x=0.1, y=0.6, # 数据坐标
    text="Low FAR, Moderate POD",
    showarrow=True, arrowhead=1, arrowcolor='black',
    font=dict(size=10, color="black"),
    align="left",
    bordercolor="black", borderwidth=1, borderpad=4,
    bgcolor="rgba(255, 255, 100, 0.7)", # 淡黄色背景
    opacity=0.8
)

# 添加形状 (例如，一个矩形区域)
fig_px_to_go.add_shape(
    type="rect",
    x0=0.05, y0=0.8, x1=0.15, y1=0.95, # 矩形的对角线坐标
    line=dict(color="RoyalBlue", width=2, dash="dot"),
    fillcolor="LightSkyBlue", opacity=0.3,
    layer="below" # 将形状放在数据点下方
)

fig_px_to_go.show()
```

**要点：**

*   `fig.update_layout()` 用于修改图表整体和轴、图例等布局属性。
*   `fig.update_xaxes()`, `fig.update_yaxes()` 可以更具体地针对某个轴进行修改，支持 `selector` 参数来选择特定轴。
*   `fig.data` 是一个包含所有轨迹的元组。你可以通过索引访问每个轨迹 (`fig.data[0]`, `fig.data[1]`, ...) 并修改其属性 (如 `marker`, `line`)。
*   `fig.add_annotation()` 用于添加文本标注，支持箭头。
*   `fig.add_shape()` 用于在图表上绘制线条、矩形、圆形等。

#### 3.3 保存 Plotly 图表

*   **交互式 HTML**:
    `fig.write_html("path/to/your_figure.html")`
    这会生成一个独立的 HTML 文件，可以在浏览器中打开并保持交互性。非常适合作为论文的在线补充材料。
*   **静态图片 (PNG, JPG, SVG, PDF)**:
    需要安装 `kaleido` 包: `pip install -U kaleido`
    `fig.write_image("path/to/your_figure.png", width=1000, height=600, scale=2)`
    *   `width`, `height`: 输出图片的像素宽度和高度。
    *   `scale`: 缩放因子。`scale=2` 会使图片分辨率更高 (例如，如果width=800, scale=2, 实际输出1600px宽)。对于论文，可能需要 `scale=3` 或更高以达到 300 DPI 的效果。
    *   支持的格式: `'png'`, `'jpeg'`, `'webp'`, `'svg'`, `'pdf'`, `'eps'`。
    *   **注意**: 保存为 PDF 或 SVG 时，Plotly 生成的文本通常会被转换为路径，这意味着文本在矢量编辑软件中可能不再是可编辑的文本对象。如果需要可编辑文本的矢量图，Matplotlib 的 PDF/SVG 输出通常更好。

```python
# 假设 fig_bar_px 是我们之前用 Plotly Express 创建的条形图

# 保存为HTML
# html_path = "plotly_barplot.html"
# fig_bar_px.write_html(html_path)
# print(f"交互式图表已保存为: {html_path}")

# 保存为PNG (确保已安装 kaleido: !pip install -U kaleido)
# png_path_plotly = "plotly_barplot.png"
# try:
#     fig_bar_px.write_image(png_path_plotly, width=800, height=500, scale=3) # scale=3 尝试提高分辨率
#     print(f"静态图表已保存为: {png_path_plotly}")
# except ValueError as e:
#     print(f"保存为静态图片失败，请确保已安装kaleido: {e}")
# except Exception as e_gen:
#     print(f"保存静态图片时发生其他错误: {e_gen}")
```

#### 3.4 针对你的项目 (Plotly 的应用场景)

*   **交互式探索你的多维数据**:
    *   **站点性能数据 (模拟的 `df_sites_perf_px`)**: 用 `px.scatter` 或 `px.scatter_3d` (如果还有第三个数值维度如海拔)，通过悬停、缩放、筛选来探索不同产品、区域、降雨量下的 POD/FAR 关系。
    *   **时间序列数据**: 你的降雨产品或模型预测的时间序列，用 `px.line` 绘制，可以交互地查看特定日期的数值，或放大某个时间段。
*   **在线补充材料**:
    *   **模型性能对比**: 将对比不同模型/特征集的条形图或线图 (`px.bar`, `px.line`) 导出为 HTML，读者可以交互地查看具体数值、隐藏/显示某些组。
    *   **空间分布图**: 如果你用 `px.imshow` 绘制了降雨量的空间分布，或者用 `px.scatter_geo` (需要地理坐标) 绘制了站点性能，交互式版本能让读者更好地探索空间模式。
    *   **参数敏感性分析**: 如果你分析了某个模型参数对性能的影响，可以制作一个包含滑块 (slider) 的交互图，让读者拖动滑块查看不同参数值下的性能曲线。
*   **复杂图表**:
    *   **动画**: 如果你有逐年或逐月的数据，可以用 `animation_frame` 参数 (例如 `px.bar(..., animation_frame='Year')`) 轻松创建动画条形图或散点图，展示动态变化。
    *   **平行坐标图 (`px.parallel_coordinates`)**: 如果你想比较多个模型/产品在多个性能指标上的表现，平行坐标图是一个不错的选择。
*   **定制主题**: `px.defaults.template = "plotly_white"` 或在绘图函数中指定 `template=`。

**你的任务与思考：**

1.  **运行本节的 `go.Figure` 示例**，理解手动添加 trace 和更新 layout 的过程。尝试修改 `yaxis2` 的属性，如 `side='left'` (虽然会重叠，但可以观察效果)，或改变颜色。
2.  **练习修改 `px` 生成的图**:
    *   用 `px.line` 创建一个图。
    *   然后用 `fig.update_layout()` 改变标题字体、轴标签字体、添加背景色 (`plot_bgcolor`)。
    *   用 `fig.data[i].line.color = 'new_color'` 修改某条线的颜色。
    *   用 `fig.add_annotation()` 和 `fig.add_shape()` 添加自定义元素。
3.  **练习保存 Plotly 图表**:
    *   将你创建的一个 Plotly 图表保存为 HTML 文件，并在浏览器中打开它。
    *   安装 `kaleido` (`pip install -U kaleido`)，然后尝试将图表保存为 PNG 和 PDF 文件。调整 `width`, `height`, `scale` 参数，观察输出图片的分辨率。
4.  **思考 Plotly 在你的项目中的具体应用场景**:
    *   对于你的那些包含大量数据的表格（如第6、8、10节），如果做成交互式的 Plotly 图表（例如，下拉菜单选择产品/指标/年份，然后动态更新图表），会不会更有利于探索和展示？
    *   你的特征重要性结果，除了静态条形图，用 Plotly 的交互式条形图展示（悬停显示具体数值）会不会更好？

Plotly 提供了与 Matplotlib/Seaborn 不同的图表制作和消费体验。虽然论文主要用静态图，但 Plotly 在数据探索、动态展示和在线分享方面具有独特优势。

到此，我们已经概览了 Matplotlib, Seaborn, 和 Plotly 这三个核心的 Python 绘图库。你现在对它们各自的特点、核心用法以及如何让它们协同工作应该有了一个比较全面的认识。

接下来，我们可以：
1.  针对你项目中某个具体的数据集或图表需求，一起设计并用合适的库（或组合）来实现它。
2.  讨论更高级的 Matplotlib 主题，如自定义 Artist、动画、或与 Cartopy/GeoPandas 结合进行地理绘图（如果你的降雨数据需要精确的地图背景）。
3.  更深入 Plotly 的某个高级功能，如回调 (callbacks) 以创建更复杂的交互应用 (这通常需要 Dash 框架)。
4.  
好的，既然你对各个方向都持开放态度，那我们来选择一个既能巩固已有知识，又能引入一些新技巧，并且与你的项目紧密相关的方向。

考虑到你的项目涉及大量的**空间数据** (如降雨产品的格点数据、站点数据) 和**多产品/多模型对比**，我们将结合 Matplotlib 和 Seaborn (如果适用) 来探讨如何更有效地可视化这些类型的数据，并引入一些使图表更符合地理信息展示习惯的技巧。我们暂时不引入新的库如 Cartopy，而是先用 Matplotlib 本身的能力来处理。

```markdown
## 第四部分：高级可视化技巧与项目应用案例

在掌握了 Matplotlib、Seaborn 和 Plotly 的基础之后，我们现在可以将这些知识融会贯通，应用到你项目中更具体的、可能更复杂的绘图需求上。我们将重点关注如何清晰地展示空间数据、多变量比较，并让图表在视觉上传达更丰富的信息。

### 4.1 可视化空间数据 (如降雨场或性能指标的空间分布)

你的项目中有很多空间数据，例如 `data/intermediate/` 下的格点降雨产品数据 (144x256 矩阵)，或者模型性能指标在不同站点/区域的分布。

#### 4.1.1 使用 `imshow` 可视化规则网格数据

`ax.imshow()` 非常适合展示二维数组（如图像、矩阵）。

**示例：绘制某一天的模拟降雨量空间分布图**

```python
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors # 用于颜色相关的操作

# 模拟一个144x256的降雨场数据 (0-50mm)
np.random.seed(42)
simulated_rainfall_field = np.random.rand(144, 256) * 50
# 制造一些空间结构，例如，一个高值区域
simulated_rainfall_field[50:90, 100:150] += np.random.rand(40, 50) * 30
simulated_rainfall_field = np.clip(simulated_rainfall_field, 0, 80) # 限制最大值

# 假设我们知道这个区域的经纬度范围
# (这些值需要根据你的实际数据进行调整)
lon_min, lon_max = 70, 140 # 示例经度范围
lat_min, lat_max = 15, 55  # 示例纬度范围

fig, ax = plt.subplots(figsize=(10, 7))

# 1. 使用 imshow 绘制
# cmap='Blues': 使用蓝色系的顺序调色板，颜色越深表示降雨量越大
# origin='lower': 将数组的 (0,0) 索引放在图像的左下角 (地理坐标的习惯)
# extent=[lon_min, lon_max, lat_min, lat_max]: 定义图像的坐标轴范围，使其对应经纬度
# aspect='auto': 自动调整宽高比以适应Axes。对于地理图，有时需要 'equal' 或手动设置。
img = ax.imshow(simulated_rainfall_field, cmap='Blues', origin='lower',
                extent=[lon_min, lon_max, lat_min, lat_max],
                aspect='auto',
                interpolation='nearest') # 'nearest' 使像素边界清晰, 'bilinear'更平滑

# 2. 添加颜色条 (Colorbar)
cbar = fig.colorbar(img, ax=ax, label='Rainfall (mm)', pad=0.03, shrink=0.85)
# shrink: 颜色条相对于Axes高度的比例
cbar.ax.tick_params(labelsize=9) # 调整颜色条刻度字体大小
cbar.set_label('Rainfall (mm)', size=11, weight='bold') # 调整颜色条标签

# 3. 设置轴标签和标题
ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Simulated Daily Rainfall Distribution', fontsize=14, fontweight='bold', pad=10)

# 4. 调整刻度
ax.tick_params(axis='both', labelsize=10, direction='in')
# 可以更精细地设置经纬度刻度
ax.set_xticks(np.arange(lon_min, lon_max + 10, 10)) # 每隔10度一个主刻度
ax.set_yticks(np.arange(lat_min, lat_max + 10, 10))

# 5. 添加网格线 (可选，对于地理图有时不需要内部网格)
# ax.grid(True, linestyle=':', color='gray', alpha=0.5)

# (可选) 添加海岸线或国界等地理边界
# 这通常需要额外的地理数据和库 (如 cartopy, geopandas)，
# 但我们可以用简单的线条模拟一下，例如一个矩形代表研究区域
# from matplotlib.patches import Rectangle
# study_area = Rectangle((lon_min_study, lat_min_study), width, height,
#                        edgecolor='black', facecolor='none', linewidth=1.5, linestyle='--')
# ax.add_patch(study_area)

plt.tight_layout()
# plt.show()
```

**关键参数和技巧 (`imshow`):**

*   `cmap`: Colormap 的选择至关重要。对于降雨量这类顺序数据，`'Blues'`, `'YlGnBu'`, `'viridis'`, `'GnBu'` 等都是不错的选择。
*   `origin='lower'` 或 `'upper'`: 决定数组索引 `(0,0)` 在图像的左下角还是左上角。对于地理数据，通常原点在左下角。
*   `extent=[left, right, bottom, top]`: 将图像的像素坐标映射到数据坐标（如经纬度）。这是使其具有地理意义的关键。
*   `aspect`: 控制像素的宽高比。`'auto'` 会拉伸以填充 Axes，`'equal'` 会保持像素为正方形（如果数据单位一致）。
*   `interpolation`: 控制像素间的插值方法。`'nearest'` 显示原始像素块，`'bilinear'`, `'bicubic'` 会进行平滑处理。对于科学数据，有时 `'nearest'` 更能反映原始分辨率。
*   **颜色映射的归一化 (`norm`)**:
    *   默认情况下，`imshow` 会将数据的最小值和最大值线性映射到 Colormap 的0到1范围。
    *   如果你的数据分布非常不均匀（例如，大部分是小雨，少数极端暴雨），线性映射可能导致大部分区域颜色相近，无法区分细节。这时可以使用非线性归一化，如对数归一化：
        `norm=mcolors.LogNorm(vmin=simulated_rainfall_field[simulated_rainfall_field > 0].min(), vmax=simulated_rainfall_field.max())` (vmin设为正数的最小值以避免log(0))
    *   或者分段归一化 `mcolors.BoundaryNorm(boundaries, cmap.N)`，可以为特定的数值区间指定不同的颜色。

**思考与你的项目：**

*   **可视化你的 `.mat` 降雨产品数据**: 你可以直接加载 `.mat` 文件中的二维数组，然后用 `imshow` 展示。确保 `extent` 参数与你的数据实际经纬度范围匹配。
*   **误差空间分布**: 计算你的模型预测与CHM真值之间的误差场 (Prediction - Truth)，然后用 `imshow` 配合一个**发散型 Colormap** (如 `'RdBu_r'`, `center=0`) 来可视化。
*   **性能指标空间分布**: 如果你计算了模型在每个格点上的CSI、POD或FAR，也可以用 `imshow` 将这些指标场可视化，使用合适的顺序型或发散型Colormap。

#### 4.1.2 使用 `pcolormesh` 可视化非规则网格或强调格点边界

与 `imshow` 类似，但 `ax.pcolormesh(X, Y, C, cmap, norm, shading, ...)` 在处理非规则网格或需要明确显示每个格元边界时更灵活。

*   `X`, `Y`: 定义网格单元角点的2D数组。如果 `C` 是 `(M, N)`，则 `X`, `Y` 通常是 `(M+1, N+1)`。
*   `C`: (M, N) 的2D数组，表示每个格元的值。
*   `shading`:
    *   `'flat'` (默认如果 X,Y 是 M+1,N+1): 每个四边形用 `C[i,j]` 的颜色填充。
    *   `'gouraud'` 或 `'auto'`: 进行颜色插值，边界会平滑。
    *   `'nearest'` (如果 X,Y 与 C 维度相同): 每个 `C[i,j]` 的颜色会围绕 `(X[i,j], Y[i,j])` 点。

对于你的规则0.25°网格数据，`imshow` 通常足够且效率更高。但如果你的数据来自不规则网格，或者你想非常强调每个网格单元的独立性，`pcolormesh` 是一个选择。

#### 4.1.3 在地理图上添加散点数据 (站点数据)

如果你的数据是站点数据（如你的README中提到的长江流域点位数据），你可以将它们作为散点叠加到地理背景图上。

```python
# 假设我们有一个简化的长江流域底图 (用imshow模拟)
# 实际中，你可能会用cartopy加载更真实的地理边界

fig, ax = plt.subplots(figsize=(10, 7))
# 模拟一个非常模糊的背景地形图
background_terrain = np.random.rand(100,100)*1000
ax.imshow(background_terrain, cmap='Greys', origin='lower',
          extent=[lon_min, lon_max, lat_min, lat_max], aspect='auto', alpha=0.3)

# 沿用之前的站点数据 df_sites_perf_px
# 颜色表示CSI (假设我们为站点计算了CSI)
np.random.seed(12)
df_sites_perf_px['CSI'] = np.random.rand(len(df_sites_perf_px)) * 0.5 + 0.4 # 0.4-0.9
df_sites_perf_px['longitude'] = np.random.rand(len(df_sites_perf_px)) * (120-90) + 90 # 假设站点在90-120E
df_sites_perf_px['latitude'] = np.random.rand(len(df_sites_perf_px)) * (35-25) + 25   # 假设站点在25-35N


scatter_sites = ax.scatter(df_sites_perf_px['longitude'], df_sites_perf_px['latitude'],
                           s=70, # 点的大小
                           c=df_sites_perf_px['CSI'], # 颜色由CSI值决定
                           cmap='RdYlGn', # 红-黄-绿 Colormap，绿色表示高CSI
                           vmin=0.4, vmax=0.9, # 固定颜色映射范围
                           edgecolors='black', linewidth=0.7, alpha=0.85,
                           label='Weather Stations')

cbar_sites = fig.colorbar(scatter_sites, ax=ax, label='Site CSI Score', shrink=0.7)
cbar_sites.set_label('Site CSI Score', size=11, weight='bold')

ax.set_xlabel('Longitude (°E)', fontsize=12)
ax.set_ylabel('Latitude (°N)', fontsize=12)
ax.set_title('Station CSI Scores over Yangtze Basin (Simulated)', fontsize=14, fontweight='bold')
ax.set_xlim(lon_min, lon_max) # 确保散点图的范围与背景图一致
ax.set_ylim(lat_min, lat_max)
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
# plt.show()
```

**思考与你的项目：**

*   你的项目中有大量站点数据，例如用 CHM 作为真值。你可以计算每个站点的模型性能指标（POD, FAR, CSI），然后用这种方式可视化它们的空间分布，颜色代表性能好坏，点的大小可以代表其他信息（如站点年均降雨量、误差大小等）。
*   **FP/FN 事件的地理分布**: 如果你的FP/FN事件与特定站点或格点关联，可以将它们标记在地图上，颜色或符号表示事件类型或频率。

### 4.2 对比多个指标或多个产品/模型的进阶图表

#### 4.2.1 分组条形图 (Matplotlib 或 Seaborn 的 `catplot`)

我们之前用 Seaborn 的 `catplot(kind='bar')` 做过多指标分面。如果想在一张图上直接比较多个模型在多个指标上的表现，分组条形图是常用方法。

# 使用 df_model_comparison 数据
df_melted = pd.melt(df_model_comparison_px, id_vars=['Model Name'],
                    value_vars=['POD', 'FAR', 'CSI'], # 选择要比较的指标
                    var_name='Metric', value_name='Score')

plt.figure(figsize=(12, 7))
# Seaborn 的 catplot (或直接 barplot 如果数据已聚合好且想手动分组)
ax_grouped_bar = sns.barplot(x='Model Name', y='Score', hue='Metric',
                             data=df_melted,
                             palette={'POD':'#5cb85c', 'FAR':'#d9534f', 'CSI':'#4682b4'}, # 自定义颜色
                             edgecolor='black', linewidth=0.7)

ax_grouped_bar.set_title('Multi-metric Model Performance Comparison', fontsize=15, fontweight='bold')
ax_grouped_bar.set_xlabel('Model', fontsize=12)
ax_grouped_bar.set_ylabel('Score', fontsize=12)
ax_grouped_bar.tick_params(axis='x', rotation=30, ha='right')
ax_grouped_bar.legend(title='Metric', loc='upper right', frameon=True, facecolor='whitesmoke', framealpha=0.7)
ax_grouped_bar.grid(axis='y', linestyle='--', alpha=0.6)

# 添加数值标签 (这个比较复杂，因为需要知道每个hue分组的条形位置)
# 通常对于分组条形图，直接在条形上标数会很拥挤，可以考虑只标某个重点指标
# 或者用表格补充具体数值。
# 一个简化的方式是只标出某个模型某个指标的值，或者用annotate
for i, model in enumerate(df_model_comparison_px['Model Name']):
    csi_val = df_model_comparison_px.loc[df_model_comparison_px['Model Name']==model, 'CSI'].iloc[0]
    # 这个定位比较粗略，需要根据条形宽度和分组情况精确计算
    # ax_grouped_bar.text(i + 0.2, csi_val + 0.01, f'{csi_val:.3f}', color='#4682b4', ha='center', va='bottom', fontsize=8)

plt.ylim(0, 1.1) # 假设指标最大为1
plt.tight_layout()
# plt.show()
```

**手动用 Matplotlib 实现分组条形图会更灵活，但代码也更复杂，需要计算每个条形组内每个条形的位置。**

#### 4.2.2 雷达图 (Radar Chart / Spider Plot) - 多指标综合评估

雷达图适合比较多个实体在多个定量指标上的表现。每个指标是一个轴，从中心点向外辐射。

```python
from math import pi

# 使用 df_model_comparison 数据，选择几个模型和指标
# 我们选择 'Random Forest', 'LightGBM', 'XGBoost (Optuna)'
# 指标: 'Accuracy', 'POD', 'CSI' (FAR越小越好，需要转换或单独表示)
labels = ['Accuracy', 'POD', 'CSI']
num_vars = len(labels)

# 数据准备 (确保所有指标都是越大越好，如果FAR要加入，可以考虑用 1-FAR)
models_to_plot = ['Random Forest', 'LightGBM', 'XGBoost (Optuna)']
data_for_radar = df_model_comparison_px[df_model_comparison_px['Model Name'].isin(models_to_plot)][['Model Name'] + labels]

# 计算角度
angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
angles += angles[:1] # 闭合雷达图

# 创建 Figure 和极坐标 Axes
fig, ax_radar = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True)) # subplot_kw很重要

colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # 为每个模型选个颜色

for i, model_name in enumerate(models_to_plot):
    values = data_for_radar[data_for_radar['Model Name'] == model_name][labels].values.flatten().tolist()
    values += values[:1] # 闭合数据
    ax_radar.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i], marker='o')
    ax_radar.fill(angles, values, color=colors[i], alpha=0.25) # 填充区域

# 设置雷达图的轴标签和刻度
ax_radar.set_thetagrids(np.array(angles[:-1]) * 180/pi, labels) # 设置角度网格线和标签
ax_radar.set_yticks(np.arange(0.5, 1.1, 0.1)) # 设置径向刻度
ax_radar.set_yticklabels([f"{tick:.1f}" for tick in np.arange(0.5, 1.1, 0.1)], fontsize=9)
ax_radar.set_rlabel_position(30) # 径向刻度标签的位置 (角度)
ax_radar.set_ylim(0.5, 1.0) # 根据你的数据范围调整

plt.title('Model Performance Radar Chart (Normalized Metrics)', size=14, y=1.1, fontweight='bold')
ax_radar.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True, facecolor='white', framealpha=0.8)
# plt.show()
```
**雷达图注意事项**:
*   所有指标的尺度最好相似，或者进行归一化处理，否则某个尺度大的指标会主导图形。
*   指标数量不宜过多（通常3-8个为佳），否则图会很乱。
*   确保所有指标都是“越大越好”或“越小越好”的同向性，如果不是，需要转换（如用 1-FAR）。

#### 4.2.3 堆叠条形图或面积图 - 展示组成或累积效应

*   **堆叠条形图**: 如果你想展示某个总量是如何由不同部分组成的，或者多个类别的贡献。
*   **堆叠面积图**: 类似于堆叠条形图，但用于连续的X轴（如时间），展示不同组分随时间变化的累积趋势。

这些图在你的项目中可能用于：
*   如果你的模型预测的是不同降雨等级的概率，可以用堆叠条形图展示某个时刻各等级概率的分布。
*   如果分析不同降雨产品对总降雨量的贡献（在融合模型中），可以用堆叠面积图展示逐日或逐月的贡献变化。

**你的任务与思考：**

1.  **运行本节所有示例代码。**
2.  **对于 `imshow`**:
    *   尝试不同的 `cmap`，特别是 `'viridis'`, `'plasma'`, `'coolwarm'`, `'RdBu_r'`。
    *   尝试 `interpolation='bilinear'` 和 `'nearest'` 的区别。
    *   如果你的数据中有0或负值，而你用了 `LogNorm`，确保正确处理（如 `vmin` 设置为数据的最小正值）。
3.  **对于分组条形图 (Seaborn)**:
    *   尝试修改 `palette`，或者为 `hue` 的每个类别手动指定颜色列表。
    *   思考如何更有效地添加数值标签，避免拥挤。
4.  **对于雷达图**:
    *   选择你项目中3-5个关键的、同向性的性能指标，为你的几个核心模型（或特征版本）绘制雷达图。
    *   调整 `ax_radar.set_ylim()` 以更好地匹配你的数据范围。
5.  **构思你的项目图表**:
    *   **空间性能对比**: 如何在一张图上并列展示多个降雨产品（或你的模型）的CSI空间分布图？(提示: Matplotlib 的 `subplots` 结合 `imshow`)。
    *   **误差分析**: 你的 README 2.4 节提到了“FP/FN 事件高发的热点区域”和“专门分析FP/FN样本对应的特征分布”。
        *   热点区域图可以用 `imshow` 或带地理信息的 `scatter`。
        *   特征分布对比可以用 `sns.histplot` 或 `sns.kdeplot`，`hue` 分为 FP, FN, TP, TN 组，`x` 是某个你怀疑与误差相关的特征。

我们已经探索了很多 Matplotlib 和 Seaborn 的高级绘图技巧。这些技巧的组合应该能让你应对绝大多数论文图表的需求。

接下来，我们可以：
1.  更深入地探讨某个特定的图表类型或技巧（例如，更复杂的 `GridSpec` 布局，动画，或地理绘图入门）。
2.  总结一下 Matplotlib 和 Seaborn 的最佳实践和常见问题。
3.  如果你对交互式图表仍然感兴趣，我们可以开始 Plotly 的 Graph Objects 部分或更高级的 Plotly Express 用法。
