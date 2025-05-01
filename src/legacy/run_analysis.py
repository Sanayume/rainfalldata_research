"""
降水数据分析示例程序
演示如何使用辅助模块进行完整的数据处理、模型训练和评估流程
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
from scipy.io import loadmat

# 添加utils目录到Python路径
sys.path.append('/f:/rainfalldata/utils')

# 导入自定义辅助模块
try:
    from utils.font_helper import setup_chinese_matplotlib
    from utils.gpu_utils import check_gpu_availability, print_gpu_status, to_device
    from utils.visualization_helper import (
        plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
        plot_feature_importance, plot_learning_curve
    )
    
    # 设置中文字体
    setup_chinese_matplotlib(test=False)
except ImportError as e:
    print(f"导入辅助模块出错: {str(e)}")
    print("请确保已创建所有必要的辅助模块文件")
    # 使用基本设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

# 导入机器学习相关库
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix


def create_output_dirs():
    """创建输出目录"""
    output_dirs = [
        'f:/rainfalldata/figures',
        'f:/rainfalldata/results',
        'f:/rainfalldata/models'
    ]
    for directory in output_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录: {directory}")


def load_rainfall_data():
    """加载降水数据"""
    print("正在加载数据...")
    
    # 数据文件路径
    data_files = {
        "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat", 
        "IMERG": "IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "CHMdata/CHM_2016_2020.mat",
    }
    
    # 加载掩膜数据
    try:
        mask = loadmat("mask.mat")["mask"]
        print(f"成功加载掩膜数据，形状: {mask.shape}")
    except Exception as e:
        print(f"加载掩膜数据失败: {str(e)}")
        print("将创建默认掩膜...")
        # 创建默认掩膜(如果无法加载)
        first_file = list(data_files.values())[0]
        try:
            first_data = loadmat(first_file)["data"]
            mask = np.ones((first_data.shape[0], first_data.shape[1]), dtype=np.int8)
            print(f"已创建默认掩膜，形状: {mask.shape}")
        except Exception:
            print("无法创建默认掩膜，程序终止")
            sys.exit(1)
    
    # 加载所有数据集
    datasets = {}
    for key, filepath in data_files.items():
        try:
            data = loadmat(filepath)["data"]
            datasets[key] = data
            print(f"成功加载 {key}, 形状: {data.shape}")
        except Exception as e:
            print(f"加载 {key} 失败: {str(e)}")
    
    # 检查是否所有必要的数据都已加载
    if "CHM" not in datasets:
        print("错误: 未找到参考数据集(CHM)")
        sys.exit(1)
    
    return datasets, mask


def prepare_training_data(datasets, mask):
    """准备训练和测试数据"""
    print("\n正在准备训练和测试数据...")
    
    # 创建产品数据字典（不包含CHM）
    products = {k: v for k, v in datasets.items() if k != "CHM"}
    feature_names = list(products.keys())
    print(f"特征列表: {feature_names}")
    
    # 提取维度
    chm_data = datasets["CHM"]
    nlat, nlon, ntime = chm_data.shape
    
    # 计算每年的样本数量
    days_per_year = [366, 365, 365, 365, 366]  # 2016-2020
    points_per_day = np.sum(mask == 1)
    samples_per_year = [days * points_per_day for days in days_per_year]
    
    # 分配训练集和测试集
    n_train_samples = sum(samples_per_year[:-1])  # 前四年的样本数
    n_test_samples = samples_per_year[-1]         # 最后一年的样本数
    
    X_train = np.zeros((n_train_samples, len(products)))
    y_train = np.zeros(n_train_samples)
    X_test = np.zeros((n_test_samples, len(products)))
    y_test = np.zeros(n_test_samples)
    
    # 处理数据
    train_idx = 0
    test_idx = 0
    last_year_start = sum(days_per_year[:-1])  # 最后一年的起始天数
    
    print("处理数据中...")
    from tqdm import tqdm
    
    for t in tqdm(range(ntime), desc="处理时间步骤"):
        # 判断当前是训练集还是测试集
        is_train = t < last_year_start
        current_idx = train_idx if is_train else test_idx
        
        for i in range(nlat):
            for j in range(nlon):
                if mask[i,j] == 1:
                    # 收集特征
                    features = []
                    for product in products.keys():
                        value = datasets[product][i,j,t]
                        features.append(value if not np.isnan(value) else 0)
                    
                    # 根据年份分配到训练集或测试集
                    if is_train:
                        X_train[current_idx] = features
                        y_train[current_idx] = 1 if chm_data[i,j,t] > 0 else 0
                        train_idx += 1
                    else:
                        X_test[current_idx] = features
                        y_test[current_idx] = 1 if chm_data[i,j,t] > 0 else 0
                        test_idx += 1
    
    print(f"\n数据集信息:")
    print(f"训练集 (2016-2019): {X_train.shape}, 正样本比例: {np.mean(y_train):.4f}")
    print(f"测试集 (2020): {X_test.shape}, 正样本比例: {np.mean(y_test):.4f}")
    
    # 创建验证集（使用最后一年的训练数据）
    val_size = samples_per_year[3]  # 2019年
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    print(f"最终训练集: {X_train.shape}, 验证集: {X_val.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def train_model(X_train, y_train, X_val, y_val, feature_names, use_gpu=False):
    """训练XGBoost模型"""
    print("\n开始训练模型...")
    
    # 设置随机种子
    random_seed = np.random.randint(1, 10000)
    print(f"随机种子: {random_seed}")
    
    # 计算类别权重
    pos_weight = np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
    print(f"正样本权重: {pos_weight:.2f}")
    
    # 设置模型参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'logloss', 'auc'],
        'max_depth': 6,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'learning_rate': 0.01,
        'gamma': 0.1,
        'random_state': random_seed,
        'scale_pos_weight': pos_weight,
        'verbosity': 1
    }
    
    # 根据GPU可用性设置树方法
    if use_gpu:
        params.update({
            'tree_method': 'gpu_hist',
            'device': 'cuda:0',
            'predictor': 'gpu_predictor',
        })
    else:
        params.update({
            'tree_method': 'hist',
            'device': 'cpu',
            'predictor': 'cpu_predictor',
        })
    
    # 创建DMatrix对象
    try:
        if use_gpu:
            import cupy as cp
            dtrain = xgb.DMatrix(data=to_device(X_train, 'cuda'), 
                                label=to_device(y_train, 'cuda'),
                                feature_names=feature_names)
            dval = xgb.DMatrix(data=to_device(X_val, 'cuda'),
                              label=to_device(y_val, 'cuda'),
                              feature_names=feature_names)
        else:
            dtrain = xgb.DMatrix(data=X_train, label=y_train,
                                feature_names=feature_names)
            dval = xgb.DMatrix(data=X_val, label=y_val,
                              feature_names=feature_names)
    except Exception as e:
        print(f"创建DMatrix时出错: {str(e)}")
        print("回退到CPU模式...")
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
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    print("开始训练XGBoost模型...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evallist,
        callbacks=callbacks,
        verbose_eval=50  # 每50次迭代显示一次结果
    )
    
    train_time = time.time() - start_time
    print(f"模型训练完成，耗时: {train_time:.2f} 秒")
    
    return model


def evaluate_model(model, X_test, y_test, feature_names, use_gpu=False):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    # 创建测试集DMatrix
    try:
        if use_gpu:
            import cupy as cp
            dtest = xgb.DMatrix(data=to_device(X_test, 'cuda'), 
                              feature_names=feature_names)
        else:
            dtest = xgb.DMatrix(data=X_test, feature_names=feature_names)
    except Exception as e:
        print(f"创建测试DMatrix失败: {str(e)}")
        dtest = xgb.DMatrix(data=X_test, feature_names=feature_names)
    
    # 预测概率和类别
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算各种评估指标
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 打印评估结果
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n混淆矩阵:")
    print(cm)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # 保存混淆矩阵可视化
    plt.figure(figsize=(8, 6))
    plot_confusion_matrix(y_test, y_pred, 
                        title='降水预测混淆矩阵',
                        labels=['无降雨', '有降雨'])
    plt.savefig('f:/rainfalldata/figures/confusion_matrix.png', dpi=300)
    plt.close()
    
    # 保存ROC曲线
    plt.figure(figsize=(8, 6))
    plot_roc_curve(y_test, y_pred_proba, 
                 title=f'ROC曲线 (AUC={roc_auc:.4f})')
    plt.savefig('f:/rainfalldata/figures/roc_curve.png', dpi=300)
    plt.close()
    
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    # 获取特征重要性
    try:
        importance_gain = model.get_score(importance_type='gain')
        importance_weight = model.get_score(importance_type='weight')
        
        # 创建完整的特征列表
        importance_data = {'特征': feature_names}
        
        # 转换为完整的特征列表（处理缺失的特征）
        gain_values = []
        weight_values = []
        for feature in feature_names:
            gain_values.append(importance_gain.get(feature, 0))
            weight_values.append(importance_weight.get(feature, 0))
        
        importance_data['gain'] = gain_values
        importance_data['weight'] = weight_values
        
        # 创建DataFrame
        importance_df = pd.DataFrame(importance_data)
        
        # 按gain排序
        importance_df = importance_df.sort_values('gain', ascending=False)
        
        # 可视化特征重要性
        plt.figure(figsize=(10, 6))
        plot_feature_importance(
            importance_df['gain'].values, 
            feature_names=importance_df['特征'].values,
            title='特征重要性 (Gain)'
        )
        plt.savefig('f:/rainfalldata/figures/feature_importance.png', dpi=300)
        plt.close()
        
        # 打印特征重要性
        print("\n特征重要性排名:")
        print(importance_df)
        
        return importance_df
    except Exception as e:
        print(f"分析特征重要性失败: {str(e)}")
        return None


def save_model_results(model, evaluation_metrics, importance_df=None):
    """保存模型和评估结果"""
    # 创建保存目录
    models_dir = 'f:/rainfalldata/models'
    results_dir = 'f:/rainfalldata/results'
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存模型
    model_path = f'{models_dir}/xgboost_model_{timestamp}.json'
    model.save_model(model_path)
    print(f"\n模型已保存至: {model_path}")
    
    # 保存混淆矩阵为CSV
    cm_df = pd.DataFrame(
        evaluation_metrics['confusion_matrix'],
        index=['实际: 无降雨', '实际: 有降雨'],
        columns=['预测: 无降雨', '预测: 有降雨']
    )
    cm_df.to_csv(f'{results_dir}/confusion_matrix_{timestamp}.csv')
    
    # 保存预测结果
    pred_df = pd.DataFrame({
        'y_true': evaluation_metrics['y_true'],
        'y_pred': evaluation_metrics['y_pred'],
        'y_pred_proba': evaluation_metrics['y_pred_proba']
    })
    pred_df.to_csv(f'{results_dir}/predictions_{timestamp}.csv', index=False)
    
    # 保存特征重要性
    if importance_df is not None:
        importance_df.to_csv(f'{results_dir}/feature_importance_{timestamp}.csv', index=False)
    
    # 保存评估指标
    with open(f'{results_dir}/metrics_{timestamp}.txt', 'w') as f:
        f.write(f"ROC AUC: {evaluation_metrics['roc_auc']:.4f}\n\n")
        f.write("混淆矩阵:\n")
        f.write(str(evaluation_metrics['confusion_matrix']))
        f.write("\n\n分类报告:\n")
        f.write(classification_report(
            evaluation_metrics['y_true'], 
            evaluation_metrics['y_pred'], 
            zero_division=0
        ))
    
    print(f"评估结果已保存至: {results_dir}")


def main():
    """主函数"""
    print("降水数据XGBoost分析程序")
    print("=" * 50)
    
    # 创建输出目录
    create_output_dirs()
    
    # 检查GPU可用性
    gpu_info = check_gpu_availability()
    
    if gpu_info['cuda_available']:
        print("发现GPU支持，将尝试使用GPU加速")
        use_gpu = True
    else:
        print("未发现GPU支持，将使用CPU")
        use_gpu = False
    
    # 加载数据
    datasets, mask = load_rainfall_data()
    
    # 准备训练和测试数据
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = prepare_training_data(datasets, mask)
    
    # 训练模型
    model = train_model(X_train, y_train, X_val, y_val, feature_names, use_gpu)
    
    # 评估模型
    evaluation = evaluate_model(model, X_test, y_test, feature_names, use_gpu)
    evaluation['y_true'] = y_test  # 添加真实标签
    
    # 分析特征重要性
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 保存模型和结果
    save_model_results(model, evaluation, importance_df)
    
    print("\n分析完成!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"发生错误: {str(e)}")
        import traceback
        traceback.print_exc()