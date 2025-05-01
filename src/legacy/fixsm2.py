import numpy as np
import matplotlib.pyplot as plt
import io
import sys
from scipy.io import loadmat, savemat

sm2rain = loadmat('sm2raindata/sm2rain_2016_2020.mat')['data']
imerg =  loadmat('IMERGdata/IMERG_2016_2020.mat')['data']

missdata = sm2rain[:,:,1185]

replacedata =   imerg[:,:,1185]

sm2rain[:,:,1185] = replacedata

plt.imshow(sm2rain[:,:,1185])
plt.show()

savemat('sm2raindata/sm2rain_2016_2020.mat',{'data':sm2rain})
# 以上代码片段的作用是将sm2rain数据集中的第1185天的数据使用IMERG数据集中的数据进行填充，然后保存到原文件中。请补全代码片段中缺失的部分。