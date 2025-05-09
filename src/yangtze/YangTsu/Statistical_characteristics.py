from loaddata import mydata
import numpy as np

mydata = mydata()
_, _, X, Y = mydata.yangtsu()
PRODUCTS = mydata.get_products()
print(X.shape)
print(Y.shape)

for i in range(X.shape[0]):
    product = PRODUCTS[i]
    print(product)
    #分析第i个产品的统计特征,包括平均值、标准差、最大值、最小值、中位数、众数、方差、偏度、峰度以及和Y之间的统计特征
    print(f"平均值: {np.mean(X[i, :, :])}")
    print(f"标准差: {np.std(X[i, :, :])}")
    print(f"最大值: {np.max(X[i, :, :])}")
    print(f"最小值: {np.min(X[i, :, :])}")
    print(f"中位数: {np.median(X[i, :, :])}")
    print(f"众数: {np.mode(X[i, :, :])}")
    print(f"方差: {np.var(X[i, :, :])}")
    print(f"偏度: {np.skewness(X[i, :, :])}")
    print(f"峰度: {np.kurtosis(X[i, :, :])}")
    print(f"和Y之间的统计特征: {np.corrcoef(X[i, :, :], Y)}")

    for j in range(i, X.shape[0]):
        product2 = PRODUCTS[j]
        #分析第i个产品和第j个产品之间的统计特征: 相关系数
        print(f"和{product2}之间的相关系数: {np.corrcoef(X[i, :, :], X[j, :, :])}")







