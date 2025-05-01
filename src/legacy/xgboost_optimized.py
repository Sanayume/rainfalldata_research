import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
from datetime import datetime
from tqdm import tqdm
import sys
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*Parameters.*are not used.*")

# 配置matplotlib支持中文显示
def setup_chinese_font():
    """配置matplotlib以正确显示中文"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 测试中文显示是否正常
    plt.figure(figsize=(6, 1))
    plt.text(0.5, 0.5, '中文测试 - Chinese text test', 
            fontsize=14, ha='center', va='center')
    plt.title('中文字体测试')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/figures/chinese_test.png', dpi=200)
    plt.close()
    print("字体测试图片已保存到 'f:/rainfalldata/figures/chinese_test.png'")

# 创建保存图形的目录
os.makedirs('f:/rainfalldata/figures', exist_ok=True)
os.makedirs('f:/rainfalldata/models', exist_ok=True)

# 设置中文字体
setup_chinese_font()

# GPU可用性检测
def check_gpu_availability():
    """检查CUDA是否可用于XGBoost并打印GPU信息"""
    try:
        import cupy as cp
        
        # 尝试简单的cupy操作以验证GPU可用性
        test_array = cp.array([1, 2, 3])
        cp.sum(test_array)
        
        # 尝试创建XGBoost GPU训练器
        test_params = {
            'objective': 'binary:logistic',
            'tree_method': 'gpu_hist',
            'device': 'cuda:0',
            'predictor': 'gpu_predictor'
        }
        test_model = xgb.XGBClassifier(**test_params)
        print("XGBoost GPU支持已确认")
        
        # 检查NVIDIA GPU信息
        try:
            import subprocess
            nvidia_smi_output = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader'],
                stderr=subprocess.STDOUT
            ).decode('utf-8')
            print("可用的GPU设备:")
            print(nvidia_smi_output)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            print("警告: 无法获取GPU信息，但XGBoost GPU支持仍可用")
            return True
            
    except Exception as e:
        print(f"GPU不可用，将使用CPU: {str(e)}")
        return False

# 优化的XGBoost参数配置
def get_xgb_params(use_gpu=False, random_state=42):
    """返回基于GPU可用性的XGBoost参数"""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'logloss', 'auc'],
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'learning_rate': 0.01,
        'gamma': 0.1,
        'random_state': random_state,
        'scale_pos_weight': 1,
        'n_estimators': 1000,  # 使用较大值，依靠早停机制
        'verbosity': 1
    }
    
    # 基于设备选择合适的tree_method和predictor
    if use_gpu:
        params.update({
            'tree_method': 'gpu_hist',
            'device': 'cuda:0',
            'predictor': 'gpu_predictor',
            'sampling_method': 'gradient_based'
        })
    else:
        params.update({
            'tree_method': 'hist',
            'device': 'cpu',
            'predictor': 'cpu_predictor'
        })
    
    return params

# 加载和预处理数据
def load_and_preprocess_data():
    """加载数据并进行预处理"""
    print("加载数据...")
    
    DATAFILE = {
        "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat", 
        "IMERG": "IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "CHMdata/CHM_2016_2020.mat",
    }
    
    MASK = loadmat("mask.mat")["mask"]
    
    # 加载所有数据集
    DATAS = {}
    for key, filepath in DATAFILE.items():
        try:
            DATAS[key] = loadmat(filepath)["data"]
            print(f"成功加载 {key}: 形状 {DATAS[key].shape}")
        except Exception as e:
            print(f"加载 {key} 失败: {str(e)}")
    
    # 创建产品数据字典（不包含CHM）
    PRODUCT = {k: v for k, v in DATAS.items() if k != "CHM"}
    print(f"产品数据: {list(PRODUCT.keys())}")
    
    # 准备训练和测试数据
    nlat, nlon, ntime = DATAS["CHM"].shape
    valid_point = MASK == 1
    
    # 计算每年的样本数量
    days_per_year = [366, 365, 365, 365, 366]  # 2016-2020
    points_per_day = np.sum(MASK == 1)
    samples_per_year = [days * points_per_day for days in days_per_year]
    
    # 初始化训练集和测试集
    n_train_samples = sum(samples_per_year[:-1])  # 前四年的样本数
    n_test_samples = samples_per_year[-1]         # 最后一年的样本数
    
    X_train = np.zeros((n_train_samples, len(PRODUCT)))
    y_train = np.zeros(n_train_samples)
    X_test = np.zeros((n_test_samples, len(PRODUCT)))
    y_test = np.zeros(n_test_samples)
    
    # 处理数据
    train_idx = 0
    test_idx = 0
    last_year_start = sum(days_per_year[:-1])  # 最后一年的起始天数
    
    print("处理数据中...")
    for t in tqdm(range(ntime), desc="处理时间步骤"):
        # 判断当前是训练集还是测试集
        is_train = t < last_year_start
        current_idx = train_idx if is_train else test_idx
        
        for i in range(nlat):
            for j in range(nlon):
                if MASK[i,j] == 1:
                    # 收集特征
                    features = []
                    for product in PRODUCT.keys():
                        value = DATAS[product][i,j,t]
                        features.append(value if not np.isnan(value) else 0)
                    
                    # 根据年份分配到训练集或测试集
                    if is_train:
                        X_train[current_idx] = features
                        y_train[current_idx] = 1 if DATAS["CHM"][i,j,t] > 0 else 0
                        train_idx += 1
                    else:
                        X_test[current_idx] = features
                        y_test[current_idx] = 1 if DATAS["CHM"][i,j,t] > 0 else 0
                        test_idx += 1
    
    print(f"\n数据集划分信息:")
    print(f"训练集(2016-2019): {X_train.shape}")
    print(f"测试集(2020): {X_test.shape}")
    
    # 创建验证集
    val_size = samples_per_year[3]  # 使用2019年作为验证集
    X_train_final, X_val = X_train[:-val_size], X_train[-val_size:]
    y_train_final, y_val = y_train[:-val_size], y_train[-val_size:]
    
    print(f"最终训练集大小: {X_train_final.shape}")
    print(f"验证集大小: {X_val.shape}")
    
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test, list(PRODUCT.keys())

# 训练XGBoost模型
def train_xgboost_model(X_train, X_val, y_train, y_val, feature_names, use_gpu=False):
    """训练XGBoost模型并返回结果"""
    random_seed = np.random.randint(1, 10000)
    print(f"使用随机种子: {random_seed}")
    
    # 计算正负样本比例
    pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    print(f"正负样本比例: 1:{pos_weight:.2f}")
    
    # 获取优化过的参数
    model_params = get_xgb_params(use_gpu=use_gpu, random_state=random_seed)
    model_params['scale_pos_weight'] = pos_weight
    
    # 使用正确的数据类型创建DMatrix
    if use_gpu:
        try:
            import cupy as cp
            dtrain = xgb.DMatrix(data=cp.array(X_train), label=cp.array(y_train), feature_names=feature_names)
            dval = xgb.DMatrix(data=cp.array(X_val), label=cp.array(y_val), feature_names=feature_names)
        except Exception as e:
            print(f"创建GPU DMatrix失败: {str(e)}，转为CPU模式")
            dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
            dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=feature_names)
    else:
        dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=feature_names)
    
    # 训练参数和回调
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    num_round = 1000  # 最大迭代次数
    
    # 添加早停机制
    early_stopping_rounds = 20
    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds, 
            metric_name='auc',
            data_name='eval',
            maximize=True,
            save_best=True
        )
    ]
    
    # 记录训练开始时间
    start_time = time.time()
    
    # 训练模型
    print("开始训练XGBoost模型...")
    bst = xgb.train(
        model_params,
        dtrain,
        num_boost_round=num_round,
        evals=evallist,
        callbacks=callbacks,
        verbose_eval=50  # 每50次迭代显示一次结果
    )
    
    # 记录训练结束时间
    train_time = time.time() - start_time
    print(f"训练完成，耗时 {train_time:.2f} 秒")
    
    # 返回训练好的模型和日志
    return bst, bst.get_score(importance_type='gain')

# 评估模型
def evaluate_model(model, X_test, y_test, feature_names, use_gpu=False):
    """评估模型性能并返回各种指标"""
    print("评估模型...")
    
    # 创建测试集DMatrix
    if use_gpu:
        try:
            import cupy as cp
            dtest = xgb.DMatrix(data=cp.array(X_test), label=cp.array(y_test), feature_names=feature_names)
        except Exception as e:
            print(f"创建GPU测试集DMatrix失败: {str(e)}，转为CPU模式")
            dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)
    else:
        dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)
    
    # 预测概率和类别
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 计算各种评估指标
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    # 计算精确率-召回率曲线和AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # 打印评估结果
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n混淆矩阵:")
    print(cm)
    
    print(f"\nROC AUC: {auc_score:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # 返回评估指标
    return {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': auc_score,
        'pr_auc': pr_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# 分析和可视化特征重要性
def analyze_feature_importance(model, feature_names):
    """分析和可视化特征重要性"""
    # 获取特征重要性
    importance_scores = []
    importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']
    
    # 对每种重要性类型获取得分
    importance_by_type = {}
    for imp_type in importance_types:
        try:
            # 有些重要性类型可能不可用，使用try-except处理
            scores = model.get_score(importance_type=imp_type)
            # 转换为完整的特征列表（处理缺失的特征）
            full_scores = []
            for feature in feature_names:
                full_scores.append(scores.get(feature, 0))
            importance_by_type[imp_type] = full_scores
        except Exception as e:
            print(f"获取{imp_type}类型的重要性失败: {str(e)}")
    
    # 创建特征重要性DataFrame
    importance_data = {'特征': feature_names}
    for imp_type, scores in importance_by_type.items():
        if scores:
            importance_data[imp_type] = scores
    
    importance_df = pd.DataFrame(importance_data)
    
    # 如果gain可用，按gain排序
    if 'gain' in importance_df.columns:
        importance_df = importance_df.sort_values('gain', ascending=False)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    if 'gain' in importance_df.columns:
        sns.barplot(data=importance_df, y='特征', x='gain', palette='viridis')
        plt.title('特征重要性 (gain)')
        plt.xlabel('重要性得分')
        plt.tight_layout()
        plt.savefig('f:/rainfalldata/figures/feature_importance_gain.png', dpi=300)
    
    # 打印特征重要性
    print("\n特征重要性排名:")
    print(importance_df)
    
    return importance_df

# 保存模型和结果
def save_model_and_results(model, evaluation_metrics, importance_df, run_id=None):
    """保存模型和评估结果"""
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建保存路径
    model_dir = f'f:/rainfalldata/models/{run_id}'
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = f'{model_dir}/xgboost_model.json'
    model.save_model(model_path)
    print(f"\n模型已保存至: {model_path}")
    
    # 保存评估指标
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'PR AUC', 
                  'Precision (Class 1)', 'Recall (Class 1)', 'F1 (Class 1)',
                  'Precision (Class 0)', 'Recall (Class 0)', 'F1 (Class 0)'],
        'Value': [
            evaluation_metrics['roc_auc'],
            evaluation_metrics['pr_auc'],
            evaluation_metrics['classification_report']['1']['precision'],
            evaluation_metrics['classification_report']['1']['recall'],
            evaluation_metrics['classification_report']['1']['f1-score'],
            evaluation_metrics['classification_report']['0']['precision'],
            evaluation_metrics['classification_report']['0']['recall'],
            evaluation_metrics['classification_report']['0']['f1-score']
        ]
    })
    
    metrics_df.to_csv(f'{model_dir}/evaluation_metrics.csv', index=False)
    
    # 保存特征重要性
    importance_df.to_csv(f'{model_dir}/feature_importance.csv', index=False)
    
    # 保存混淆矩阵为CSV
    cm_df = pd.DataFrame(
        evaluation_metrics['confusion_matrix'],
        index=['实际: 无降雨', '实际: 有降雨'],
        columns=['预测: 无降雨', '预测: 有降雨']
    )
    cm_df.to_csv(f'{model_dir}/confusion_matrix.csv')
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/confusion_matrix.png', dpi=300)
    
    print(f"评估指标和可视化结果已保存至: {model_dir}")
    
    return run_id

# 交叉验证
def perform_cross_validation(X, y, feature_names, use_gpu=False, n_splits=5):
    """执行交叉验证并返回结果"""
    print(f"\n执行{n_splits}折交叉验证...")
    
    # 获取模型参数
    model_params = get_xgb_params(use_gpu=use_gpu)
    
    # 计算类别权重
    pos_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)
    model_params['scale_pos_weight'] = pos_weight
    
    # 创建StratifiedKFold对象
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储每次交叉验证的结果
    cv_results = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # 开始交叉验证
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n训练折 {fold+1}/{n_splits}")
        
        # 分割数据
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        # 创建DMatrix
        if use_gpu:
            try:
                import cupy as cp
                dtrain = xgb.DMatrix(data=cp.array(X_train_cv), label=cp.array(y_train_cv), feature_names=feature_names)
                dtest = xgb.DMatrix(data=cp.array(X_test_cv), label=cp.array(y_test_cv), feature_names=feature_names)
            except Exception as e:
                print(f"创建GPU DMatrix失败: {str(e)}，转为CPU模式")
                dtrain = xgb.DMatrix(data=X_train_cv, label=y_train_cv, feature_names=feature_names)
                dtest = xgb.DMatrix(data=X_test_cv, label=y_test_cv, feature_names=feature_names)
        else:
            dtrain = xgb.DMatrix(data=X_train_cv, label=y_train_cv, feature_names=feature_names)
            dtest = xgb.DMatrix(data=X_test_cv, label=y_test_cv, feature_names=feature_names)
        
        # 训练模型
        bst = xgb.train(
            model_params,
            dtrain,
            num_boost_round=200
        )
        
        # 预测和评估
        y_pred_proba = bst.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算评估指标
        auc = roc_auc_score(y_test_cv, y_pred_proba)
        report = classification_report(y_test_cv, y_pred, output_dict=True, zero_division=0)
        
        # 存储结果
        cv_results['auc'].append(auc)
        cv_results['accuracy'].append(report['accuracy'])
        cv_results['precision'].append(report['1']['precision'])
        cv_results['recall'].append(report['1']['recall'])
        cv_results['f1'].append(report['1']['f1-score'])
        
        print(f"折 {fold+1} - AUC: {auc:.4f}, 准确率: {report['accuracy']:.4f}, 精确率: {report['1']['precision']:.4f}, 召回率: {report['1']['recall']:.4f}")
    
    # 计算平均值和标准差
    for metric in cv_results:
        mean_val = np.mean(cv_results[metric])
        std_val = np.std(cv_results[metric])
        print(f"平均{metric}: {mean_val:.4f} (±{std_val:.4f})")
    
    return cv_results

# 主函数
def main():
    """主函数，执行完整的训练和评估流程"""
    print("XGBoost降水分类优化版本")
    print("=" * 50)
    
    # 检查GPU可用性
    USE_GPU = check_gpu_availability()
    
    # 加载和预处理数据
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = load_and_preprocess_data()
    
    # 执行交叉验证
    cv_results = perform_cross_validation(
        np.vstack((X_train, X_val)), 
        np.hstack((y_train, y_val)), 
        feature_names, 
        use_gpu=USE_GPU
    )
    
    # 训练最终模型
    model, importance = train_xgboost_model(X_train, X_val, y_train, y_val, feature_names, use_gpu=USE_GPU)
    
    # 评估模型
    evaluation = evaluate_model(model, X_test, y_test, feature_names, use_gpu=USE_GPU)
    
    # 分析特征重要性
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 保存模型和结果
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_model_and_results(model, evaluation, importance_df, run_id)
    
    # 绘制学习曲线
    results = model.eval_set
    if results:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(model.eval_result['train']['error'], label='训练')
        plt.plot(model.eval_result['eval']['error'], label='验证')
        plt.title('错误率曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('错误率')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(model.eval_result['train']['auc'], label='训练')
        plt.plot(model.eval_result['eval']['auc'], label='验证')
        plt.title('AUC曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('AUC')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'f:/rainfalldata/figures/learning_curves_{run_id}.png', dpi=300)
    
    print("\n训练和评估流程已完成!")
    print(f"所有结果已保存在 f:/rainfalldata/models/{run_id}/")

if __name__ == "__main__":
    main()
