import geopandas as gpd
import matplotlib.pyplot as plt
import requests
import io

# 1. 全国 GeoJSON 数据接口 URL
geojson_url = "https://geojson.cn/api/china/1.6.2/100000.json"

try:
    # 使用 requests 库下载数据，并明确设置 verify=False 来禁用SSL验证
    response = requests.get(geojson_url, verify=False) # <--- 重点在这里！
    response.raise_for_status() # 检查HTTP请求是否成功

    # 将文本内容读入内存中的文件对象
    geojson_text = response.text
    geojson_data = io.StringIO(geojson_text)
    
    # gpd.read_file() 可以从文件路径或文件对象读取
    china_gdf = gpd.read_file(geojson_data)

    # 保存 GeoJSON 数据到本地文件
    output_geojson_path = "china_provinces.geojson"
    try:
        with open(output_geojson_path, 'w', encoding='utf-8') as f:
            f.write(geojson_text)
        print(f"GeoJSON 数据已成功保存到 {output_geojson_path}")
    except Exception as e:
        print(f"保存 GeoJSON 文件失败：{e}")

    print("数据读取成功！")
    print(china_gdf.head())
    print(f"包含 {len(china_gdf)} 个地理特征（省份/直辖市）")


    # 2. 简单的地图可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    china_gdf.plot(ax=ax, color='lightgray', edgecolor='black')
    plt.title("中国各省边界 (GeoJSON)")
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.show()

except requests.exceptions.SSLError as e:
    print(f"SSL证书验证失败：{e}")
    print("已尝试禁用SSL验证。如果问题依然存在，请检查URL或网络环境。")
except Exception as e:
    print(f"读取数据失败：{e}")
    print("请检查URL是否正确，网络是否畅通，或数据接口是否有效。")