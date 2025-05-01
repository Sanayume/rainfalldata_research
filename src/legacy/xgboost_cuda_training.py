"""
CUDA加速版降水预测训练脚本
通过利用GPU加速实现高效训练
"""

import os
import sys
import time
import json
import argparse
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, f1_score
)

# 确保输出目录存在
os.makedirs('f:/rainfalldata/models', exist_ok=True)
os.makedirs('f:/rainfalldata/results', exist_ok=True)
os.makedirs('f:/rainfalldata/figures', exist_ok=True)
os.makedirs('f:/rainfalldata/logs', exist_ok=True)

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 检测CuPy和RAPIDS是否可用
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("CuPy已加载，GPU加速可用")
except ImportError:
    CUPY_AVAILABLE = False
    print("CuPy未安装，将使用CPU模式")
    cp = np

try:
    import xgboost as xgb
    from xgboost import XGBClassifier
except ImportError:
    print("XGBoost未安装，请安装后再运行")
    sys.exit(1)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CUDA加速降水预测训练')
    parser.add_argument('--batch_size', type=int, default=200000, help='GPU批处理大小')
    parser.add_argument('--no_gpu', action='store_true', help='禁用GPU，使用CPU')
    parser.add_argument('--early_stopping', type=int, default=20, help='早停轮数')
    parser.add_argument('--output_dir', type=str, default='f:/rainfalldata/results', help='结果输出目录')
    parser.add_argument('--iterations', type=int, default=500, help='最大迭代次数')
    return parser.parse_args()

def check_gpu_status():
    """检查GPU状态并返回详细信息"""
    gpu_info = {
        'cuda_available': False,
        'gpu_name': 'Unknown',
        'memory_total': 0,
        'memory_free': 0,
        'cuda_version': 'Unknown'
    }
    
    if not CUPY_AVAILABLE:
        return gpu_info
    
    try:
        # 使用CuPy检查CUDA可用性
        gpu_info['cuda_available'] = cp.cuda.is_available()
        if gpu_info['cuda_available']:
            # 获取设备属性
            device_props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_info['gpu_name'] = device_props.get('name', 'Unknown GPU')
            gpu_info['cuda_version'] = '{}.{}'.format(
                cp.cuda.runtime.runtimeGetVersion() // 1000,
                (cp.cuda.runtime.runtimeGetVersion() % 1000) // 10
            )
            
            # 获取内存信息
            meminfo = cp.cuda.runtime.memGetInfo()
            gpu_info['memory_free'] = meminfo[0] / 1024**3  # GB
            gpu_info['memory_total'] = meminfo[1] / 1024**3  # GB
            
            # 尝试简单的GPU操作以确认工作正常
            a = cp.array([1, 2, 3])
            b = cp.array([4, 5, 6])
            cp.add(a, b)  # 验证基本操作
    except Exception as e:
        print(f"GPU检查时出错: {str(e)}")
        gpu_info['cuda_available'] = False
    
    return gpu_info

def print_gpu_info(gpu_info):
    """打印GPU信息"""
    print("\n===== GPU信息 =====")
    print(f"CUDA可用: {gpu_info['cuda_available']}")
    
    if gpu_info['cuda_available']:
        print(f"GPU: {gpu_info['gpu_name']}")
        print(f"CUDA版本: {gpu_info['cuda_version']}")
        print(f"总内存: {gpu_info['memory_total']:.2f} GB")
        print(f"可用内存: {gpu_info['memory_free']:.2f} GB")
    print("===================\n")

def to_device(data, use_gpu=True):
    """将数据转移到相应的设备（GPU或CPU）"""
    if use_gpu and CUPY_AVAILABLE:
        try:
            # 确保数据是numpy格式
            if not isinstance(data, (np.ndarray, cp.ndarray)):
                data = np.array(data)
            # 转移到GPU
            if isinstance(data, np.ndarray):
                return cp.array(data)
            return data  # 已经在GPU上
        except Exception as e:
            print(f"转移数据到GPU失败: {str(e)}")
            return data  # 失败则返回原始数据
    else:
        # 如果是cupy数组，转回numpy
        if CUPY_AVAILABLE and isinstance(data, cp.ndarray):
            return cp.asnumpy(data)
        return data

def batch_processing(func, data, batch_size=100000, use_gpu=True, *args, **kwargs):
    """批处理大型数据集，避免GPU内存不足"""
    if len(data) <= batch_size:
        # 单批处理
        data_device = to_device(data, use_gpu) if use_gpu and CUPY_AVAILABLE else data
        result = func(data_device, use_gpu=use_gpu, *args, **kwargs)
        # 确保返回的是CPU数据
        if use_gpu and CUPY_AVAILABLE and isinstance(result, cp.ndarray):
            return cp.asnumpy(result)
        return result
    
    # 多批处理
    result_parts = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_device = to_device(batch, use_gpu) if use_gpu and CUPY_AVAILABLE else batch
        batch_result = func(batch_device, use_gpu=use_gpu, *args, **kwargs)
        # 确保返回的是CPU数据
        if use_gpu and CUPY_AVAILABLE and isinstance(batch_result, cp.ndarray):
            batch_result = cp.asnumpy(batch_result)
        result_parts.append(batch_result)
    
    return np.vstack(result_parts)

def load_rainfall_data(use_gpu=True):
    """加载降水数据"""
    print("加载降水数据...")
    
    # 定义数据文件路径
    data_files = {
        "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat",
        "IMERG": "IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "CHMdata/CHM_2016_2020.mat",
        "MASK": "mask.mat",
    }
    
    # 加载数据
    datasets = {}
    for key, filepath in data_files.items():
        try:
            if key == "MASK":
                datasets[key] = loadmat(filepath)["mask"]
            else:
                datasets[key] = loadmat(filepath)["data"]
            print(f"成功加载 {key}: 形状 {datasets[key].shape}")
        except Exception as e:
            print(f"加载 {key} 失败: {str(e)}")
    
    return datasets

def prepare_cuda_data(datasets, batch_size=200000, use_gpu=True):
    """准备用于CUDA加速的训练和测试数据"""
    print("\n准备训练和测试数据...")
    
    # 提取必要数据
    mask = datasets["MASK"]
    chm_data = datasets["CHM"]
    product_data = {k: v for k, v in datasets.items() if k not in ["MASK", "CHM"]}
    feature_names = list(product_data.keys())
    
    # 计算形状
    nlat, nlon, ntime = chm_data.shape
    valid_points = np.sum(mask == 1)
    
    # 计算样本索引
    train_days = ntime - 366  # 前几年作为训练集
    test_days = 366         # 最后一年作为测试集
    
    train_samples = valid_points * train_days
    test_samples = valid_points * test_days
    
    print(f"训练样本数: {train_samples}, 测试样本数: {test_samples}")
    
    # 初始化数组
    X_train = np.zeros((train_samples, len(product_data)), dtype=np.float32)
    y_train = np.zeros(train_samples, dtype=np.int8)
    X_test = np.zeros((test_samples, len(product_data)), dtype=np.float32)
    y_test = np.zeros(test_samples, dtype=np.int8)
    
    # 有效地处理数据
    train_idx = 0
    test_idx = 0
    
    print("处理特征数据...")
    for t in range(ntime):
        is_train = t < train_days  # 前几年用作训练
        
        for i in range(nlat):
            for j in range(nlon):
                if mask[i, j] == 1:
                    # 收集特征
                    features = []
                    for product in product_data.keys():
                        val = datasets[product][i, j, t]
                        features.append(0.0 if np.isnan(val) else float(val))
                    
                    # 根据是训练还是测试分配数据
                    if is_train:
                        X_train[train_idx, :] = features
                        y_train[train_idx] = 1 if chm_data[i, j, t] > 0 else 0
                        train_idx += 1
                    else:
                        X_test[test_idx, :] = features
                        y_test[test_idx] = 1 if chm_data[i, j, t] > 0 else 0
                        test_idx += 1
    
    # 创建验证集
    val_ratio = 0.1
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, random_state=42, stratify=y_train
    )
    
    # 特征扩展和标准化
    print("进行特征工程和标准化...")
    
    # 1. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_final)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 添加交互特征（修改后的函数，处理GPU/CPU兼容性）
    def create_interactions(X, use_gpu=False):
        """创建交互特征，支持GPU和CPU"""
        n_features = X.shape[1]
        n_interactions = n_features * (n_features - 1) // 2
        total_features = n_features + n_interactions
        
        # 根据输入数据类型选择正确的数组库
        if use_gpu and CUPY_AVAILABLE and isinstance(X, cp.ndarray):
            xp = cp
        else:
            xp = np
            if use_gpu and CUPY_AVAILABLE and isinstance(X, cp.ndarray):
                X = cp.asnumpy(X)  # 如果X是CuPy数组但我们要用NumPy，先转换
        
        # 创建输出数组
        result = xp.zeros((X.shape[0], total_features), dtype=xp.float32)
        
        # 复制原始特征
        result[:, :n_features] = X
        
        # 添加交互特征
        idx = n_features
        for i in range(n_features):
            for j in range(i+1, n_features):
                if idx < total_features:
                    result[:, idx] = X[:, i] * X[:, j]
                    idx += 1
        
        return result
    
    # 使用修改后的批处理和create_interactions函数
    X_train_final = batch_processing(create_interactions, X_train_scaled, batch_size, use_gpu)
    X_val_final = batch_processing(create_interactions, X_val_scaled, batch_size, use_gpu)
    X_test_final = batch_processing(create_interactions, X_test_scaled, batch_size, use_gpu)
    
    # 打印最终形状
    print(f"最终特征形状: X_train: {X_train_final.shape}, X_val: {X_val_final.shape}, X_test: {X_test_final.shape}")
    
    # 计算类别分布信息
    pos_count = np.sum(y_train_final == 1)
    neg_count = np.sum(y_train_final == 0)
    ratio = neg_count / pos_count
    print(f"训练集正样本比例: {pos_count/(pos_count+neg_count):.4f} (正样本数: {pos_count}, 负样本数: {neg_count}, 比例: 1:{ratio:.2f})")
    
    return {
        'X_train': X_train_final,
        'y_train': y_train_final,
        'X_val': X_val_final,
        'y_val': y_val,
        'X_test': X_test_final,
        'y_test': y_test,
        'feature_names': feature_names,
        'class_ratio': ratio,
        'scaler': scaler
    }

def train_cuda_model(data, args, gpu_info):
    """使用CUDA加速训练XGBoost模型"""
    use_gpu = not args.no_gpu and gpu_info['cuda_available']
    
    print(f"\n开始{'GPU' if use_gpu else 'CPU'}模式训练...")
    
    # 设置XGBoost参数
    train_params = {
        'objective': 'binary:logistic',
        'eval_metric': ['error', 'logloss', 'auc'],
        'max_depth': 6,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.3,
        'reg_lambda': 1.0,
        'scale_pos_weight': data['class_ratio'],  # 使用正负样本比例
        'random_state': 42,
        'verbosity': 1
    }
    
    # 根据GPU可用性配置树方法
    if use_gpu:
        train_params.update({
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor',
            'sampling_method': 'gradient_based',
        })
    else:
        train_params.update({
            'tree_method': 'hist',
            'predictor': 'cpu_predictor',
        })
    
    # 准备DMatrix数据
    if use_gpu:
        dtrain = xgb.DMatrix(
            data=to_device(data['X_train'], use_gpu),
            label=to_device(data['y_train'], use_gpu)
        )
        
        dval = xgb.DMatrix(
            data=to_device(data['X_val'], use_gpu),
            label=to_device(data['y_val'], use_gpu)
        )
    else:
        dtrain = xgb.DMatrix(data=data['X_train'], label=data['y_train'])
        dval = xgb.DMatrix(data=data['X_val'], label=data['y_val'])
    
    # 设置评估集合和回调
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=args.early_stopping, 
            metric_name='auc',
            data_name='eval',
            maximize=True,
            save_best=True
        )
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model = xgb.train(
        params=train_params,
        dtrain=dtrain,
        num_boost_round=args.iterations,
        evals=evallist,
        callbacks=callbacks,
        verbose_eval=50  # 每50次迭代显示结果
    )
    
    # 记录训练时间
    train_time = time.time() - start_time
    print(f"\n训练完成，耗时: {train_time:.2f} 秒")
    
    return model, train_time

def evaluate_model(model, data, args, gpu_info):
    """评估模型性能"""
    use_gpu = not args.no_gpu and gpu_info['cuda_available']
    
    print("\n评估模型性能...")
    
    # 创建测试DMatrix
    if use_gpu:
        dtest = xgb.DMatrix(
            data=to_device(data['X_test'], use_gpu),
            label=to_device(data['y_test'], use_gpu)
        )
    else:
        dtest = xgb.DMatrix(data=data['X_test'], label=data['y_test'])
    
    # 预测
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算评估指标
    cm = confusion_matrix(data['y_test'], y_pred)
    roc_auc = roc_auc_score(data['y_test'], y_pred_proba)
    
    # 计算PR-AUC
    precision, recall, _ = precision_recall_curve(data['y_test'], y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(data['y_test'], y_pred))
    
    print("\n混淆矩阵:")
    print(cm)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # 返回评估结果
    return {
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_true': data['y_test']
    }

def optimize_threshold(evaluation):
    """优化分类阈值"""
    print("\n优化分类阈值...")
    
    thresholds = np.arange(0.2, 0.8, 0.05)
    results = {}
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (evaluation['y_pred_proba'] >= threshold).astype(int)
        
        # 计算正负类的性能指标
        rain_precision = precision_score(evaluation['y_true'], y_pred, pos_label=1)
        rain_recall = recall_score(evaluation['y_true'], y_pred, pos_label=1)
        rain_f1 = f1_score(evaluation['y_true'], y_pred, pos_label=1)
        
        no_rain_precision = precision_score(evaluation['y_true'], y_pred, pos_label=0)
        no_rain_recall = recall_score(evaluation['y_true'], y_pred, pos_label=0)
        no_rain_f1 = f1_score(evaluation['y_true'], y_pred, pos_label=0)
        
        # 使用加权F1作为优化目标
        weighted_f1 = 0.6 * rain_f1 + 0.4 * no_rain_f1
        
        results[threshold] = {
            'rain_f1': rain_f1,
            'rain_precision': rain_precision,
            'rain_recall': rain_recall,
            'no_rain_f1': no_rain_f1,
            'no_rain_precision': no_rain_precision,
            'no_rain_recall': no_rain_recall,
            'weighted_f1': weighted_f1
        }
        
        print(f"阈值 {threshold:.2f}: 雨天F1={rain_f1:.4f}, 召回率={rain_recall:.4f}, "
              f"无雨精确率={no_rain_precision:.4f}, 加权F1={weighted_f1:.4f}")
        
        if weighted_f1 > best_f1:
            best_f1 = weighted_f1
            best_threshold = threshold
    
    print(f"\n最佳阈值: {best_threshold:.2f}, 加权F1值: {best_f1:.4f}")
    
    # 重新计算最佳阈值的预测
    best_pred = (evaluation['y_pred_proba'] >= best_threshold).astype(int)
    
    # 可视化阈值调整
    plt.figure(figsize=(10, 6))
    x = list(results.keys())
    plt.plot(x, [results[t]['rain_recall'] for t in x], 'b-', label='雨天召回率')
    plt.plot(x, [results[t]['no_rain_precision'] for t in x], 'r-', label='无雨精确度')
    plt.plot(x, [results[t]['weighted_f1'] for t in x], 'g-', label='加权F1')
    plt.axvline(x=best_threshold, color='k', linestyle='--', label=f'最佳阈值={best_threshold:.2f}')
    plt.xlabel('阈值')
    plt.ylabel('性能指标')
    plt.title('分类阈值优化')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('f:/rainfalldata/figures/threshold_optimization.png', dpi=300)
    
    return {
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_pred': best_pred,
        'threshold_results': results
    }

def analyze_feature_importance(model, data):
    """分析特征重要性"""
    print("\n分析特征重要性...")
    
    try:
        # 获取特征重要性
        importance_dict = model.get_score(importance_type='gain')
        
        # 创建特征重要性DataFrame
        importance_df = pd.DataFrame([
            {'特征': feature, '重要性': importance_dict.get(str(i), 0)} 
            for i, feature in enumerate(data['feature_names'])
        ])
        
        # 排序
        importance_df = importance_df.sort_values('重要性', ascending=False)
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['特征'], importance_df['重要性'])
        plt.title('特征重要性 (gain)')
        plt.xlabel('重要性')
        plt.tight_layout()
        plt.savefig('f:/rainfalldata/figures/feature_importance.png', dpi=300)
        
        print("特征重要性排名:")
        print(importance_df)
        
        return importance_df
    except Exception as e:
        print(f"分析特征重要性时出错: {str(e)}")
        return None

def save_results(model, evaluation, threshold_results, feature_importance, train_time, args):
    """保存模型和评估结果"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = f"{args.output_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # 保存模型
    model_path = f"{run_dir}/model.json"
    model.save_model(model_path)
    
    # 保存评估结果
    with open(f"{run_dir}/metrics.json", "w") as f:
        metrics = {
            'roc_auc': float(evaluation['roc_auc']),
            'pr_auc': float(evaluation['pr_auc']),
            'best_threshold': float(threshold_results['best_threshold']),
            'best_f1': float(threshold_results['best_f1']),
            'train_time': train_time,
            'timestamp': timestamp
        }
        json.dump(metrics, f, indent=4)
    
    # 保存混淆矩阵
    np.savetxt(f"{run_dir}/confusion_matrix.csv", evaluation['confusion_matrix'], delimiter=',')
    
    # 保存特征重要性
    if feature_importance is not None:
        feature_importance.to_csv(f"{run_dir}/feature_importance.csv", index=False)
    
    # 保存配置信息
    with open(f"{run_dir}/config.json", "w") as f:
        config = {
            'batch_size': args.batch_size,
            'use_gpu': not args.no_gpu,
            'early_stopping': args.early_stopping,
            'iterations': args.iterations
        }
        json.dump(config, f, indent=4)
    
    # 复制图表
    import shutil
    for fig_file in ['threshold_optimization.png', 'feature_importance.png']:
        src_path = f'f:/rainfalldata/figures/{fig_file}'
        if os.path.exists(src_path):
            shutil.copy(src_path, f"{run_dir}/{fig_file}")
    
    print(f"\n结果已保存至: {run_dir}")
    return run_dir

def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 开始计时
    total_start_time = time.time()
    
    # 输出标题
    print("=" * 70)
    print("CUDA加速降水预测XGBoost训练程序")
    print("=" * 70)
    
    # 检查GPU状态
    gpu_info = check_gpu_status()
    print_gpu_info(gpu_info)
    
    # 决定使用GPU还是CPU
    use_gpu = not args.no_gpu and gpu_info['cuda_available']
    if args.no_gpu:
        print("已请求使用CPU模式")
    elif not gpu_info['cuda_available']:
        print("未检测到可用GPU，将使用CPU模式")
    else:
        print(f"将使用GPU加速: {gpu_info['gpu_name']}")
    
    # 加载数据
    datasets = load_rainfall_data(use_gpu)
    
    # 准备训练数据
    data = prepare_cuda_data(datasets, batch_size=args.batch_size, use_gpu=use_gpu)
    
    # 训练模型
    model, train_time = train_cuda_model(data, args, gpu_info)
    
    # 评估模型
    evaluation = evaluate_model(model, data, args, gpu_info)
    
    # 优化阈值
    threshold_results = optimize_threshold(evaluation)
    
    # 分析特征重要性
    feature_importance = analyze_feature_importance(model, data)
    
    # 保存结果
    output_dir = save_results(model, evaluation, threshold_results, 
                            feature_importance, train_time, args)
    
    # 总结
    total_time = time.time() - total_start_time
    print(f"\n总运行时间: {total_time:.2f} 秒")
    print(f"模型训练时间: {train_time:.2f} 秒 ({train_time/total_time*100:.1f}%)")
    print(f"ROC AUC: {evaluation['roc_auc']:.4f}")
    print(f"最佳阈值: {threshold_results['best_threshold']:.2f}")
    print(f"可视化和结果已保存至: {output_dir}")
    print("\nCUDA加速训练完成！")

if __name__ == "__main__":
    # 捕获异常，确保平滑退出
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
