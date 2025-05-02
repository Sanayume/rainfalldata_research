import numpy as np
from scipy.io import loadmat
import os

class mydata:
    def __init__(self):
        # 使用 os.path.join 确保路径分隔符正确，并使用绝对路径
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "intermediate", "nationwide")
        mask_base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "data", "processed", "nationwide", "masks")

        self.DATAFILE ={
            "CMORPH": os.path.join(base_path, "CMORPHdata", "CMORPH_2016_2020.mat"), # (144, 256, 1827)
            "CHIRPS": os.path.join(base_path, "CHIRPSdata", "chirps_2016_2020.mat"), # (144, 256, 1827)
            "SM2RAIN": os.path.join(base_path, "SM2RAINDATA", "sm2rain_2016_2020.mat"), # (144, 256, 1827)
            "IMERG": os.path.join(base_path, "IMERGdata", "IMERG_2016_2020.mat"), # (144, 256, 1827)
            "GSMAP": os.path.join(base_path, "GSMAPdata", "GSMAP_2016_2020.mat"), # (144, 256, 1827)
            "PERSIANN": os.path.join(base_path, "PERSIANNdata", "PERSIANN_2016_2020.mat"), # (144, 256, 1827)
            "CHM": os.path.join(base_path, "CHMdata", "CHM_2016_2020.mat"), # (144, 256, 1827) - Target
            "MASK": os.path.join(mask_base_path, "combined_china_basin_mask.mat"), # (144, 256)
        }
        # 定义切片参数 - 移除空间切片
        # lat_slice = slice(50, 100) # Removed
        # lon_slice = slice(100, 150) # Removed
        time_slice = slice(0, 1827) # 5 years

        # 加载并处理 X 数据
        self.X = []
        print("Loading full spatial data for X...")
        for key in ["CMORPH", "CHIRPS", "SM2RAIN", "IMERG", "GSMAP", "PERSIANN"]:
            print(f"  Loading {key}...")
            # 加载 .mat 文件，提取 'data' 字段，应用时间切片，然后转置
            # 原始 shape: (lat, lon, time) -> 时间切片后 -> 转置后 shape: (time, lat, lon)
            data = loadmat(self.DATAFILE[key])['data'][:, :, time_slice] # No spatial slice
            self.X.append(np.transpose(data, (2, 0, 1)))
            # 每个元素的 shape 应该是 (1827, 144, 256)
        self.X = np.array(self.X, dtype=np.float32) # Convert list to numpy array (n_products, time, lat, lon)
        print(f"  Finished loading X. Shape: {self.X.shape}")

        self.features = ["CMORPH", "CHIRPS", "SM2RAIN", "IMERG", "GSMAP", "PERSIANN"]

        # 加载并处理 Y 数据 (目标)
        print("Loading full spatial data for Y...")
        y_data = loadmat(self.DATAFILE["CHM"])['data'][:, :, time_slice] # No spatial slice
        self.Y = np.transpose(y_data, (2, 0, 1)).astype(np.float32) # Shape: (1827, 144, 256)
        # Squeeze into (time, lat, lon)
        self.Y = np.squeeze(self.Y)
        print(f"  Finished loading Y. Shape: {self.Y.shape}")


        # 加载 Mask (保持不变，因为 mask 本身就是全区域的)
        print("Loading mask...")
        self.MASK = loadmat(self.DATAFILE["MASK"])['mask'] # Shape: (144, 256)
        print(f"  Finished loading mask. Shape: {self.MASK.shape}")

    def get_x(self):
        # print("type(self.X):", type(self.X), "shape:", self.X.shape)
        return self.X
    def get_y(self):
        # print("type(self.Y):", type(self.Y), "shape:", self.Y.shape)
        return self.Y
    def get_mask(self):
        # print("type(self.MASK):", type(self.MASK), "shape:", self.MASK.shape)
        return self.MASK



