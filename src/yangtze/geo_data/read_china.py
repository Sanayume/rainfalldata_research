import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import io
import os

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'Microsoft YaHei', 'Arial', 'sans-serif']
# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 1. 全国 GeoJSON 数据接口 URL
china_file = f"{current_dir}/china_provinces.geojson"

china = gpd.read_file(china_file)
print("数据读取成功！")
print(china.head())
print(f"包含 {len(china)} 个地理特征（省份/直辖市）")


# 2. 简单的地图可视化
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
china.plot(ax=ax, color='lightgray', edgecolor='black')
plt.title("中国各省边界 (GeoJSON)")
plt.xlabel("经度")
plt.ylabel("纬度")
plt.show()
