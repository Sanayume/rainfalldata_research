"""
使用前三天数据的XGBoost降水预测训练脚本
包含PCA降维、特征工程、交互特征和早停策略
"""

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
import os
from datetime import datetime
import warnings
from sklearn.metrics import classification_report, accuracy_score, f1_score
import gc

# 基础设置
os.makedirs('f:/rainfalldata/figures', exist_ok=True)
os.makedirs('f:/rainfalldata/results', exist_ok=True)
os.makedirs('f:/rainfalldata/models', exist_ok=True)
warnings.filterwarnings('ignore')

class RainfallPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = None
        self.xgb_model = None
        self.feature_names = []
        self.feature_importance = None
        self.dtype = np.float32  # 添加数据类型属性
    
    def load_data(self):
        """加载数据并转换为float32以节省内存"""
        print("加载数据...")
        DATAFILE = {
            "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
            "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
            "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat",
            "IMERG": "IMERGdata/IMERG_2016_2020.mat",
            "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
            "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
            "CHM": "CHMdata/CHM_2016_2020.mat",
            "MASK": "mask.mat",
        }
        
        DATAS = {}
        for key, filepath in DATAFILE.items():
            try:
                if key == "MASK":
                    DATAS[key] = loadmat(filepath)["mask"]
                else:
                    DATAS[key] = loadmat(filepath)["data"].astype(np.float32)
                print(f"成功加载 {key}: 形状 {DATAS[key].shape}")
            except Exception as e:
                print(f"加载 {key} 失败: {str(e)}")
        
        return DATAS

    def create_nonlinear_features(self, X):
        """创建非线性特征，使用float32"""
        nonlinear_features = []
        
        # 添加平方特征
        squared = np.square(X).astype(self.dtype)
        nonlinear_features.append(squared)
        
        # 添加立方特征
        cubed = np.power(X, 3).astype(self.dtype)
        nonlinear_features.append(cubed)
        
        # 添加对数特征
        log_features = np.log1p(np.abs(X)).astype(self.dtype)
        nonlinear_features.append(log_features)
        
        # 添加指数特征（归一化后防止溢出）
        X_normalized = (X / X.std(axis=0, keepdims=True)).clip(-3, 3)
        exp_features = np.exp(X_normalized).astype(self.dtype)
        nonlinear_features.append(exp_features)
        
        # 组合所有特征
        X_nonlinear = np.hstack([X] + nonlinear_features).astype(self.dtype)
        del nonlinear_features  # 释放内存
        gc.collect()
        
        print(f"非线性特征变换后维度: {X_nonlinear.shape}")
        return X_nonlinear

    def create_cross_features(self, X, n_products):
        """创建交叉特征，使用float32"""
        n_days = 3
        cross_features = []
        batch_size = min(10000, X.shape[0])  # 分批处理来节省内存
        
        for start_idx in range(0, X.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, X.shape[0])
            batch = X[start_idx:end_idx]
            batch_cross = []
            
            # 1. 同一产品不同天之间的两两交叉
            for p in range(n_products):
                for i in range(n_days):
                    for j in range(i+1, n_days):
                        idx1 = p + i * n_products
                        idx2 = p + j * n_products
                        cross = (batch[:, idx1] * batch[:, idx2]).reshape(-1, 1)
                        batch_cross.append(cross)
            
            # 2. 同一天不同产品之间的交叉
            for day in range(n_days):
                start_idx_day = day * n_products
                for i in range(n_products):
                    for j in range(i+1, n_products):
                        idx1 = start_idx_day + i
                        idx2 = start_idx_day + j
                        cross = (batch[:, idx1] * batch[:, idx2]).reshape(-1, 1)
                        batch_cross.append(cross)
            
            # 3. 三天之间的交叉特征
            for p in range(n_products):
                idx1, idx2, idx3 = p, p + n_products, p + 2 * n_products
                triple_cross = (batch[:, idx1] * batch[:, idx2] * batch[:, idx3]).reshape(-1, 1)
                batch_cross.append(triple_cross)
            
            # 合并批次特征
            batch_cross = np.hstack(batch_cross).astype(self.dtype)
            cross_features.append(np.hstack([batch, batch_cross]))
            
            # 清理内存
            del batch_cross
            gc.collect()
        
        X_cross = np.vstack(cross_features)
        print(f"交叉特征后维度: {X_cross.shape}")
        return X_cross

    def prepare_features(self, DATAS, batch_size=1000):
        """分批处理准备特征"""
        MASK = DATAS["MASK"]
        PRODUCT = DATAS.copy()
        PRODUCT.pop("MASK")
        CHM_DATA = PRODUCT.pop("CHM")
        
        nlat, nlon, ntime = CHM_DATA.shape
        valid_point = MASK == 1
        n_valid = np.sum(valid_point)
        
        # 从第三天开始，因为需要前两天的数据
        trainsample = n_valid * (ntime-3-366)  # 训练集
        testsample = n_valid * 366             # 测试集
        n_features_per_day = len(PRODUCT)
        total_features = n_features_per_day * 3  # 三天的特征
        
        print(f"准备训练数据: {trainsample} 样本, 每个样本 {total_features} 特征")
        
        # 分批生成特征
        def generate_features():
            for t in range(3, ntime):  # 从第三天开始
                is_train = t < (ntime - 366)
                
                features_batch = []
                labels_batch = []
                
                for i in range(nlat):
                    for j in range(nlon):
                        if MASK[i,j] == 1:
                            # 收集三天的特征
                            feature = []
                            for day_offset in range(3):
                                current_day = t - day_offset
                                for product in PRODUCT:
                                    value = DATAS[product][i,j,current_day]
                                    feature.append(float(value) if not np.isnan(value) else 0.0)
                            
                            features_batch.append(feature)
                            labels_batch.append(float(CHM_DATA[i,j,t] > 0))
                            
                            if len(features_batch) >= batch_size:
                                # 特征工程
                                X_batch = np.array(features_batch, dtype=self.dtype)
                                # 1. 添加非线性特征
                                X_batch = self.create_nonlinear_features(X_batch)
                                # 2. 添加交叉特征
                                X_batch = self.create_cross_features(X_batch, len(PRODUCT))
                                
                                yield X_batch, np.array(labels_batch, dtype=self.dtype), is_train
                                features_batch = []
                                labels_batch = []
                
                if features_batch:  # 处理剩余的数据
                    X_batch = np.array(features_batch, dtype=self.dtype)
                    X_batch = self.create_nonlinear_features(X_batch)
                    X_batch = self.create_cross_features(X_batch, len(PRODUCT))
                    yield X_batch, np.array(labels_batch, dtype=self.dtype), is_train
        
        return generate_features, trainsample, testsample, total_features

    def apply_pca(self, X, n_components=None):
        """应用PCA降维，使用float32"""
        X = X.astype(self.dtype)
        if self.pca is None:
            if n_components is None:
                n_components = min(X.shape[0], X.shape[1])
            self.pca = PCA(n_components=n_components)
            X_scaled = self.scaler.fit_transform(X)
            X_pca = self.pca.fit_transform(X_scaled).astype(self.dtype)
            explained_var = np.cumsum(self.pca.explained_variance_ratio_)
            
            # 选择解释95%方差所需的组件数
            n_components_95 = np.argmax(explained_var >= 0.95) + 1
            print(f"95%方差解释需要的组件数: {n_components_95}")
            
            # 重新拟合PCA
            self.pca = PCA(n_components=n_components_95)
            X_pca = self.pca.fit_transform(X_scaled).astype(self.dtype)
        else:
            X_scaled = self.scaler.transform(X)
            X_pca = self.pca.transform(X_scaled).astype(self.dtype)
        
        return X_pca

    def train_model(self, X_train, y_train, X_val, y_val):
        """训练XGBoost模型"""
        print("\n训练XGBoost模型...")
        
        # 计算类别权重
        neg_pos_ratio = np.sum(y_train == 0) / np.sum(y_train == 1)
        adjusted_ratio = min(neg_pos_ratio, 1.5)
        
        self.xgb_model = xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=500,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=adjusted_ratio,
            objective='binary:logistic',
            early_stopping_rounds=20,
            eval_metric=['logloss', 'auc']
        )
        
        eval_set = [(X_val, y_val)]
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=True
        )
        
        return self.xgb_model.best_iteration

    def optimize_threshold(self, y_true, y_pred_proba):
        """优化分类阈值"""
        thresholds = np.arange(0.2, 0.8, 0.05)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, pos_label=1)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return best_threshold, best_f1

    def analyze_feature_importance(self, feature_names):
        """分析特征重要性"""
        if self.xgb_model is None:
            return
        
        importance = self.xgb_model.feature_importances_
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # 绘制特征重要性图
        plt.figure(figsize=(15, 8))
        plt.bar(range(30), self.feature_importance['importance'][:30])
        plt.xticks(range(30), self.feature_importance['feature'][:30], rotation=45, ha='right')
        plt.title('Top 30 Most Important Features')
        plt.tight_layout()
        plt.savefig('f:/rainfalldata/figures/feature_importance.png')
        plt.close()
        
        return self.feature_importance

def main():
    predictor = RainfallPredictor()
    
    # 加载数据
    DATAS = predictor.load_data()
    
    # 准备特征生成器
    generator, trainsample, testsample, n_features = predictor.prepare_features(DATAS)
    
    # 收集训练数据
    print("\n收集训练数据...")
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []
    
    # 在收集数据时使用float32
    for X_batch, y_batch, is_train in generator():
        if is_train:
            X_train_list.append(X_batch.astype(np.float32))
            y_train_list.append(y_batch.astype(np.float32))
        else:
            X_test_list.append(X_batch.astype(np.float32))
            y_test_list.append(y_batch.astype(np.float32))
        
        # 定期清理内存
        if len(X_train_list) % 10 == 0:
            gc.collect()
    
    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test = np.vstack(X_test_list)
    y_test = np.concatenate(y_test_list)
    
    # 应用PCA降维
    print("\n应用PCA降维...")
    X_train_pca = predictor.apply_pca(X_train)
    X_test_pca = predictor.apply_pca(X_test)
    
    # 分割验证集
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train_pca, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # 训练模型
    best_iteration = predictor.train_model(X_train_part, y_train_part, X_val, y_val)
    
    # 预测和评估
    print("\n模型评估...")
    y_pred_proba = predictor.xgb_model.predict_proba(X_test_pca)[:, 1]
    best_threshold, best_f1 = predictor.optimize_threshold(y_test, y_pred_proba)
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    print(f"\n最佳阈值: {best_threshold:.2f}, F1分数: {best_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 保存模型和结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存模型
    model_path = f'f:/rainfalldata/models/xgboost_3days_pca_{timestamp}.json'
    predictor.xgb_model.save_model(model_path)
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    })
    results_path = f'f:/rainfalldata/results/predictions_3days_pca_{timestamp}.csv'
    results_df.to_csv(results_path, index=False)
    
    print(f"\n模型已保存至: {model_path}")
    print(f"预测结果已保存至: {results_path}")

    # 训练模型后添加特征重要性分析
    feature_importance = predictor.analyze_feature_importance(predictor.feature_names)
    print("\nTop 10 最重要特征:")
    print(feature_importance.head(10))

if __name__ == "__main__":
    main()
