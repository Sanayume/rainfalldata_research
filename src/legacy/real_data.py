import xarray as xr
import numpy as np
from scipy.io import savemat, loadmat
import matplotlib.pylab as plt
filename = 'CHM_PRE_0.1dg_19612022.nc'

# 使用 open_dataset 分批读取数据，并选择 2016 年的数据
data = xr.open_dataset(filename, chunks={'time': 365})  # 按照时间维度分块，每块大约一年
# 优化分块方案
data = data.chunk({'time': -1, 'latitude': 'auto', 'longitude': 'auto'})
# 选择 2016 年的数据
data_2016 = data.sel(time=slice('2016-01-01', '2016-12-31'))

data = data_2016.variables['pre']
data = data.values
data[data == -99.9] = np.nan
np.transpose(data,(2,1,0))
print(data.shape)
chm2016 = {'data':data}
savemat('chm2016new.mat',chm2016)
data = np.nanmean(data, axis=0)
print(data)
print(data.shape)
data = np.flipud(data)
plt.figure(figsize=(5,3))
plt.imshow(data)
plt.colorbar()
plt.title('data')
plt.show()


