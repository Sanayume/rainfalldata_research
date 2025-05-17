from loaddata import mydata
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

data = mydata()
# Use get_basin_point_data to reduce the number of points for the scatter plot
# X will have shape (n_products, time, n_points)
# Y will have shape (time, n_points)
X, Y = data.get_basin_point_data(2) 
PRODUCT = data.get_products()

for i in range(X.shape[0]):
    # X_product will have shape (time, n_points)
    X_product = X[i, :, :] 
    product = PRODUCT[i]
    print(f"现在开始画第{i}个{product}的散点图")

    # Flatten data: Y is (time, n_points), X_product is (time, n_points)
    # y_flat and x_product_flat will have (time * n_points) elements
    y_flat = Y.flatten()
    x_product_flat = X_product.flatten()

    # 移除 NaN 值，确保数据清洁
    valid_indices = ~np.isnan(y_flat) & ~np.isnan(x_product_flat)
    y_flat = y_flat[valid_indices]
    x_product_flat = x_product_flat[valid_indices]

    if len(y_flat) == 0 or len(x_product_flat) == 0:
        print(f"产品 {product} 没有有效数据点，跳过绘图。")
        continue

    # 计算点密度
    xy = np.vstack([y_flat, x_product_flat])
    try:
        z = gaussian_kde(xy)(xy)
    except (np.linalg.LinAlgError, ValueError) as e: # Catch ValueError too for singular matrix
        # 如果数据点太少或共线性太强，KDE可能会失败，此时使用单一颜色
        print(f"产品 {product} 的KDE计算失败 ({e})，将使用单一颜色绘制。")
        z = np.ones_like(y_flat)

    # 根据密度对点进行排序，以便更密集区域的点绘制在顶部（可选，但有时效果更好）
    idx = z.argsort()
    y_flat, x_product_flat, z = y_flat[idx], x_product_flat[idx], z[idx]

    fig = plt.figure(figsize=(10, 8)) # Assign figure to a variable
    scatter = plt.scatter(y_flat, x_product_flat, c=z, s=10, cmap='viridis', alpha=0.7)

    # 添加 y=x 参考线
    # Recalculate min_val and max_val in case of empty arrays after NaN removal
    if len(y_flat) > 0 and len(x_product_flat) > 0:
        min_val = min(np.min(y_flat), np.min(x_product_flat))
        max_val = max(np.max(y_flat), np.max(x_product_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='y=x line')
    else: # Fallback if all data was NaN
        plt.plot([0,1], [0,1], 'r--', lw=2, label='y=x line (default)')


    plt.xlabel('Observed Rainfall (Y)', fontsize=14)
    plt.ylabel(f'{product} Rainfall (X)', fontsize=14)
    plt.title(f'Scatter Plot: Observed vs {product} Rainfall', fontsize=16)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, label='Point Density')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Point Density', size=14)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout() # 调整布局以防止标签重叠
    
    plt.show()  # Show the current plot
    plt.close(fig) # Close the figure to free memory and allow loop to continue

