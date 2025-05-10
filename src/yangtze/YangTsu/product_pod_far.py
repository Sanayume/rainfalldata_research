from loaddata import mydata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
data = mydata()
X, Y = data.get_basin_spatial_data(1)

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
'''
  X_spatial shape: (6, 1827, 144, 256)
  Y_spatial shape: (1827, 144, 256)
'''
product_names = data.PRODUCTS
# 定义分类阈值
thresholds = np.arange(0.0, 1.0, 0.05)

for i in range(X.shape[0]):
    print(f"\n--- 产品 {product_names[i]} ---")
    # 展平原始预测数据和真实数据
    X_pred_flat = X[i, :, :, :].flatten()
    Y_true_flat = Y.flatten()

    # 找出X_pred_flat和Y_true_flat中共同非NaN的索引
    valid_indices = ~np.isnan(X_pred_flat) & ~np.isnan(Y_true_flat)
    
    X_pred_processed = X_pred_flat[valid_indices]
    Y_true_processed = Y_true_flat[valid_indices]

    if X_pred_processed.size == 0 or Y_true_processed.size == 0:
        print("处理后数据为空，跳过此产品。")
        continue

    for threshold in thresholds:
        print(f"  阈值: {threshold}")
        
        # 根据阈值将连续值转换为二分类标签 (0 或 1)
        # 假设大于等于阈值为1 (有事件)，小于阈值为0 (无事件)
        Y_true_categorical = (Y_true_processed > threshold).astype(int)
        X_pred_categorical = (X_pred_processed > threshold).astype(int)

        # 确保转换后的数据长度一致
        if Y_true_categorical.shape[0] != X_pred_categorical.shape[0]:
            print(f"    错误: Y_true_categorical 和 X_pred_categorical 长度不一致。跳过此阈值。")
            print(f"    Y_true_categorical shape: {Y_true_categorical.shape}, X_pred_categorical shape: {X_pred_categorical.shape}")
            continue
        
        if Y_true_categorical.size == 0:
             print(f"    警告: 阈值 {threshold} 下没有有效的分类数据点。")
             tn, fp, fn, tp = 0, 0, 0, 0
        else:
            cm = confusion_matrix(Y_true_categorical, X_pred_categorical, labels=[0, 1])
            # 如果某个类别不存在，confusion_matrix可能返回2x2，但对应行列可能全0
            # 我们需要确保tn, fp, fn, tp的正确解析
            if cm.shape == (1,1): # 只有一个类别出现
                if np.unique(Y_true_categorical).item() == 0 and np.unique(X_pred_categorical).item() == 0: # 全是TN
                    tn = cm[0,0]
                    fp, fn, tp = 0,0,0
                elif np.unique(Y_true_categorical).item() == 1 and np.unique(X_pred_categorical).item() == 1: # 全是TP
                    tp = cm[0,0]
                    tn, fp, fn = 0,0,0
                # 其他单类别情况，例如预测全0但实际有1，或反之，需要更细致处理或依赖sklearn的labels参数
                # 这里简单处理，若cm不是2x2则认为可能存在问题或特殊情况
                else: # 无法简单解析单维度cm，默认全0
                    tn, fp, fn, tp = 0,0,0,0
                    print(f"    警告: 混淆矩阵维度为 {cm.shape}, 可能导致指标计算不准确。检查数据分布。")

            elif cm.shape == (2,2):
                 tn, fp, fn, tp = cm.ravel()
            else: # 理论上labels=[0,1]应该保证2x2，除非输入为空或只有一类且sklearn行为异常
                print(f"    警告: 混淆矩阵维度异常 {cm.shape}。标签Y: {np.unique(Y_true_categorical)}, 标签X: {np.unique(X_pred_categorical)}")
                tn, fp, fn, tp = 0,0,0,0


        pod = tp / (tp + fn) if (tp + fn) > 0 else 0
        far = fp / (tp + fp) if (tp + fp) > 0 else 0
        csi = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        
        print(f"    POD: {pod:.4f}")
        print(f"    FAR: {far:.4f}")
        print(f"    CSI: {csi:.4f}")
        print(f"    TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")








