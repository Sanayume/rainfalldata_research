import osgeo.gdal as gdal
import geopandas as gpd
import matplotlib.pyplot as plt
file_china = "china_provinces.geojson"
file_world = "world.geojson"
file_yangtze = "yangtze.shp"
# 第4步：打印一下 world_data，看看它是什么

data1_yangtze = gpd.read_file(file_yangtze)
print(data1_yangtze.head())
data_yangtze = data1_yangtze.copy()
data1_yangtze.to_file("yangtze.geojson", driver='GeoJSON')
print("-"*100)

china_data = gpd.read_file(file_china)
print(china_data.head())
print("-"*100)
world_data = gpd.read_file(file_world)
print(world_data.head())
fig, ax  = plt.subplots(3, 1, figsize=(10, 15))
world_data.plot(ax=ax[0], color='lightblue', edgecolor='black')
# 设置标题
ax[0].set_title('World Countries', fontsize=15)
# 显示图形
china_data.plot(ax=ax[1], color='lightgreen', edgecolor='black')
# 设置标题
ax[1].set_title('China Provinces', fontsize=15)
data_yangtze.plot(ax=ax[2], color='salmon', edgecolor='black')
ax[2].set_title('Yangtze River', fontsize=15)
data_yangtze.boundary.plot(ax=ax[2], color='black', linewidth=1)
plt.tight_layout()
plt.show()