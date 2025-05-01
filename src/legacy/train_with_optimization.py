"""
整合了所有优化模块的降水数据训练脚本
使用GPU加速、中文字体支持和高级可视化功能
"""

import os
import sys
import numpy as np
import pandas as pd
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.io import loadmat
import warnings
import argparse
import json

# 添加utils目录到Python路径
if not os.path.exists('f:/rainfalldata/utils'):
    os.makedirs('f:/rainfalldata/utils', exist_ok=True)

sys.path.append('f:/rainfalldata')
sys.path.append('f:/rainfalldata/utils')

# 导入自定义辅助模块
try:
    from utils.font_helper import setup_chinese_matplotlib
    from utils.gpu_utils import check_gpu_availability, to_device, print_gpu_status
    from utils.visualization_helper import (
        plot_confusion_matrix, plot_roc_curve, plot_pr_curve,
        plot_feature_importance, plot_learning_curve, plot_multiple_learning_curves
    )
    
    # 设置中文字体
    setup_chinese_matplotlib(test=False)
    print("成功导入辅助模块")
except ImportError as e:
    print(f"导入辅助模块失败: {str(e)}")
    print("将使用基本设置")
    # 使用基本设置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

# 导入机器学习库
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, auc
)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='优化版降水数据训练脚本')
    
    parser.add_argument('--gpu', action='store_true', help='是否使用GPU训练')
    parser.add_argument('--cv', type=int, default=0, help='交叉验证折数，0表示不使用交叉验证')
    parser.add_argument('--output_dir', type=str, default='f:/rainfalldata/results', help='结果输出目录')
    parser.add_argument('--batch_size', type=int, default=50000, help='GPU批处理大小')
    parser.add_argument('--early_stopping', type=int, default=20, help='早停轮数')
    parser.add_argument('--config', type=str, default='', help='配置文件路径')
    
    return parser.parse_args()

def load_config(config_path):
    """加载配置文件"""
    if not config_path or not os.path.exists(config_path):
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载配置文件失败: {str(e)}")
        return {}

def setup_directories():
    """创建必要的目录"""
    dirs = [
        'f:/rainfalldata/figures',
        'f:/rainfalldata/models',
        'f:/rainfalldata/results',
        'f:/rainfalldata/logs'
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"已创建目录: {directory}")

def load_rainfall_data():
    """加载降水数据"""
    print("\n加载降水数据...")
    
    data_files = {
        "CMORPH": "CMORPHdata/CMORPH_2016_2020.mat",
        "CHIRPS": "CHIRPSdata/chirps_2016_2020.mat",
        "SM2RAIN": "sm2raindata/sm2rain_2016_2020.mat", 
        "IMERG": "IMERGdata/IMERG_2016_2020.mat",
        "GSMAP": "GSMAPdata/GSMAP_2016_2020.mat",
        "PERSIANN": "PERSIANNdata/PERSIANN_2016_2020.mat",
        "CHM": "CHMdata/CHM_2016_2020.mat",
    }
    
    try:
        mask = loadmat("mask.mat")["mask"]
        print(f"成功加载掩膜数据，形状: {mask.shape}")
    except Exception as e:
        print(f"加载掩膜数据失败: {str(e)}")
        sys.exit(1)
    
    # 加载所有数据集
    datasets = {}
    for key, filepath in data_files.items():
        try:
            datasets[key] = loadmat(filepath)["data"]
            print(f"成功加载 {key}: 形状 {datasets[key].shape}")
        except Exception as e:
            print(f"加载 {key} 失败: {str(e)}")
            if key == "CHM":  # CHM是必需的
                sys.exit(1)
    
    return datasets, mask
                                                                                                                                                                                                                                                                                                                                                                                    
def prepare_data(datasets, mask):
    """准备训练和测试数据"""
    print("\n准备训练和测试数据...")
    
    # 创建产品数据字典（不包含CHM）
    products = {k: v for k, v in datasets.items() if k != "CHM"}
    feature_names = list(products.keys())
    print(f"特征列表: {feature_names}")
    
    # 提取维度
    nlat, nlon, ntime = datasets["CHM"].shape
    
    # 计算每年的样本数量
    days_per_year = [366, 365, 365, 365, 366]  # 2016-2020
    points_per_day = np.sum(mask == 1)
    print(f"每天的有效点数: {points_per_day}")
    samples_per_year = [days * points_per_day for days in days_per_year]
    print(f"每年样本数: {samples_per_year}")
    
    # 分配训练集和测试集
    n_train_samples = sum(samples_per_year[:-1])  # 前四年的样本数
    n_test_samples = samples_per_year[-1]         # 最后一年的样本数
    
    print(f"初始化数据数组 - 训练集: {n_train_samples} 样本, 测试集: {n_test_samples} 样本")
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
        
        for i in range(nlat):
            for j in range(nlon):
                if mask[i,j] == 1:
                    # 收集特征
                    features = []
                    for product in products.keys():
                        value = datasets[product][i,j,t]
                        features.append(0.0 if np.isnan(value) else float(value))
                    
                    # 根据年份分配到训练集或测试集
                    if is_train:
                        X_train[train_idx] = features
                        y_train[train_idx] = 1 if datasets["CHM"][i,j,t] > 0 else 0
                        train_idx += 1
                    else:
                        X_test[test_idx] = features
                        y_test[test_idx] = 1 if datasets["CHM"][i,j,t] > 0 else 0
                        test_idx += 1
    
    print(f"\n数据集信息:")
    print(f"训练集 (2016-2019): {X_train.shape}, 正样本比例: {np.mean(y_train):.4f}")
    print(f"测试集 (2020): {X_test.shape}, 正样本比例: {np.mean(y_test):.4f}")
    
    # 创建验证集（使用最后一年的训练数据）
    val_size = samples_per_year[3]  # 2019年
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train[:-val_size]
    y_train_final = y_train[:-val_size]
    
    print(f"最终训练集: {X_train_final.shape}, 验证集: {X_val.shape}")
    
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test, feature_names

def get_xgb_params(use_gpu=False, random_state=42):
    """返回XGBoost参数"""
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
        'verbosity': 1
    }
    
    # 根据GPU可用性设置树方法
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

def train_model(X_train, X_val, y_train, y_val, feature_names, use_gpu=False, early_stopping_rounds=20):
    """训练XGBoost模型"""
    print("\n开始训练XGBoost模型...")
    
    # 设置随机种子并计算类别权重
    random_seed = np.random.randint(1, 10000)
    print(f"随机种子: {random_seed}")
    
    pos_count = np.sum(y_train == 1)
    neg_count = np.sum(y_train == 0)
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"正样本权重: {pos_weight:.2f}  (正样本/负样本: {pos_count}/{neg_count})")
    
    # 获取模型参数
    params = get_xgb_params(use_gpu=use_gpu, random_state=random_seed)
    params['scale_pos_weight'] = pos_weight
    
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
            dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
            dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=feature_names)
    except Exception as e:
        print(f"创建DMatrix失败: {str(e)}, 将使用CPU模式")
        dtrain = xgb.DMatrix(data=X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(data=X_val, label=y_val, feature_names=feature_names)
    
    # 设置评估列表和早停回调
    evallist = [(dtrain, 'train'), (dval, 'eval')]
    callbacks = [
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name='auc',
            data_name='eval',
            maximize=True,
            save_best=True
        )
    ]
    
    # 训练模型
    start_time = time.time()
    print("开始训练...")
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,  # 最大迭代次数
        evals=evallist,
        callbacks=callbacks,
        verbose_eval=50  # 每50次迭代显示一次结果
    )
    
    train_time = time.time() - start_time
    print(f"训练完成，耗时: {train_time:.2f} 秒")
    
    return model

def evaluate_model(model, X_test, y_test, feature_names, use_gpu=False):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    # 创建测试集DMatrix
    try:
        if use_gpu:
            import cupy as cp
            dtest = xgb.DMatrix(data=to_device(X_test, 'cuda'), 
                              label=to_device(y_test, 'cuda'),
                              feature_names=feature_names)
        else:
            dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)
    except Exception as e:
        print(f"创建测试DMatrix失败: {str(e)}, 将使用CPU模式")
        dtest = xgb.DMatrix(data=X_test, label=y_test, feature_names=feature_names)
    
    # 预测概率和类别
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 计算各种评估指标
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # 计算精确率-召回率曲线和AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    # 打印评估结果
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print("\n混淆矩阵:")
    print(cm)
    
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    # 构造指标字典
    metrics = {
        'confusion_matrix': cm,
        'classification_report': report,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    return metrics

def perform_cross_validation(X, y, feature_names, use_gpu=False, n_splits=5):
    """执行交叉验证"""
    if n_splits < 2:
        return None
        
    print(f"\n执行{n_splits}折交叉验证...")
    
    # 获取模型参数
    params = get_xgb_params(use_gpu=use_gpu)
    
    # 计算类别权重
    pos_weight = np.sum(y == 0) / max(np.sum(y == 1), 1)
    params['scale_pos_weight'] = pos_weight
    
    # 创建StratifiedKFold对象
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 存储每次验证的结果
    cv_results = {
        'auc': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'models': [],
        'fold_metrics': []
    }
    
    # 开始交叉验证
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"\n训练折 {fold+1}/{n_splits}")
        
        # 分割数据
        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
        
        # 创建DMatrix
        try:
            if use_gpu:
                import cupy as cp
                dtrain = xgb.DMatrix(data=to_device(X_train_cv, 'cuda'), 
                                   label=to_device(y_train_cv, 'cuda'), 
                                   feature_names=feature_names)
                dtest = xgb.DMatrix(data=to_device(X_test_cv, 'cuda'),
                                  label=to_device(y_test_cv, 'cuda'),
                                  feature_names=feature_names)
            else:
                dtrain = xgb.DMatrix(data=X_train_cv, label=y_train_cv, feature_names=feature_names)
                dtest = xgb.DMatrix(data=X_test_cv, label=y_test_cv, feature_names=feature_names)
        except Exception as e:
            print(f"创建DMatrix失败: {str(e)}, 将使用CPU模式")
            dtrain = xgb.DMatrix(data=X_train_cv, label=y_train_cv, feature_names=feature_names)
            dtest = xgb.DMatrix(data=X_test_cv, label=y_test_cv, feature_names=feature_names)
        
        # 训练模型
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            early_stopping_rounds=20,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            verbose_eval=100
        )
        
        # 预测和评估
        y_pred_proba = model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # 计算评估指标
        auc_score = roc_auc_score(y_test_cv, y_pred_proba)
        report = classification_report(y_test_cv, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test_cv, y_pred)
        
        # 存储结果
        cv_results['auc'].append(auc_score)
        cv_results['accuracy'].append(report['accuracy'])
        cv_results['precision'].append(report['1']['precision'])
        cv_results['recall'].append(report['1']['recall'])
        cv_results['f1'].append(report['1']['f1-score'])
        cv_results['models'].append(model)
        cv_results['fold_metrics'].append({
            'confusion_matrix': cm,
            'classification_report': report,
            'auc': auc_score
        })
        
        print(f"折 {fold+1} - AUC: {auc_score:.4f}, 准确率: {report['accuracy']:.4f}, F1: {report['1']['f1-score']:.4f}")
    
    # 计算平均值和标准差
    for metric in ['auc', 'accuracy', 'precision', 'recall', 'f1']:
        mean_val = np.mean(cv_results[metric])
        std_val = np.std(cv_results[metric])
        print(f"平均{metric}: {mean_val:.4f} (±{std_val:.4f})")
    
    return cv_results

def visualize_results(model, evaluation, feature_names, output_dir):
    """可视化模型结果"""
    print("\n创建可视化结果...")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. 混淆矩阵
    plot_confusion_matrix(
        evaluation['y_true'], 
        evaluation['y_pred'],
        title='降水预测混淆矩阵',
        filepath=os.path.join(figures_dir, "confusion_matrix.png")
    )
    
    # 2. ROC曲线
    plot_roc_curve(
        evaluation['y_true'], 
        evaluation['y_pred_proba'],
        title='ROC曲线 (AUC = {:.4f})'.format(evaluation['roc_auc']),
        filepath=os.path.join(figures_dir, "roc_curve.png")
    )
    
    # 3. PR曲线
    plot_pr_curve(
        evaluation['y_true'], 
        evaluation['y_pred_proba'],
        title='精确率-召回率曲线 (AUC = {:.4f})'.format(evaluation['pr_auc']),
        filepath=os.path.join(figures_dir, "pr_curve.png")
    )
    
    # 4. 特征重要性
    importance_dict = model.get_score(importance_type='gain')
    plot_feature_importance(
        importance_dict,
        feature_names=feature_names,
        title='特征重要性 (gain)',
        filepath=os.path.join(figures_dir, "feature_importance.png"),
        sort=True
    )
    
    # 5. 学习曲线（如果可用）
    if hasattr(model, 'eval_result'):
        metrics = {}
        for metric in model.eval_result['train'].keys():
            metrics[metric] = {
                'train': model.eval_result['train'][metric],
                'eval': model.eval_result['eval'][metric]
            }
        
        plot_multiple_learning_curves(
            metrics,
            title='模型学习曲线',
            filepath=os.path.join(figures_dir, "learning_curves.png")
        )
    
    print(f"可视化结果已保存至: {figures_dir}")

def save_results(model, evaluation, feature_importance, output_dir, run_id=None):
    """保存模型和评估结果"""
    if run_id is None:
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建保存路径
    results_dir = os.path.join(output_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(results_dir, 'model.json')
    model.save_model(model_path)
    print(f"\n模型已保存至: {model_path}")
    
    # 保存评估指标
    metrics_df = pd.DataFrame({
        'Metric': ['ROC AUC', 'PR AUC', 
                  'Precision (Class 1)', 'Recall (Class 1)', 'F1 (Class 1)',
                  'Precision (Class 0)', 'Recall (Class 0)', 'F1 (Class 0)'],
        'Value': [
            evaluation['roc_auc'],
            evaluation['pr_auc'],
            evaluation['classification_report']['1']['precision'],
            evaluation['classification_report']['1']['recall'],
            evaluation['classification_report']['1']['f1-score'],
            evaluation['classification_report']['0']['precision'],
            evaluation['classification_report']['0']['recall'],
            evaluation['classification_report']['0']['f1-score']
        ]
    })
    
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)
    
    # 保存特征重要性
    feature_importance.to_csv(os.path.join(results_dir, 'feature_importance.csv'), index=False)
    
    # 保存混淆矩阵
    cm_df = pd.DataFrame(
        evaluation['confusion_matrix'],
        index=['实际: 无降雨', '实际: 有降雨'],
        columns=['预测: 无降雨', '预测: 有降雨']
    )
    cm_df.to_csv(os.path.join(results_dir, 'confusion_matrix.csv'))
    
    # 保存完整的分类报告
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(evaluation['y_true'], evaluation['y_pred'], zero_division=0))
    
    # 保存运行配置信息
    config = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_id': run_id,
        'roc_auc': float(evaluation['roc_auc']),
        'pr_auc': float(evaluation['pr_auc']),
        'positive_samples': int(np.sum(evaluation['y_true'] == 1)),
        'negative_samples': int(np.sum(evaluation['y_true'] == 0)),
        'true_positives': int(evaluation['confusion_matrix'][1, 1]),
        'false_positives': int(evaluation['confusion_matrix'][0, 1]),
        'true_negatives': int(evaluation['confusion_matrix'][0, 0]),
        'false_negatives': int(evaluation['confusion_matrix'][1, 0])
    }
    
    with open(os.path.join(results_dir, 'run_info.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"评估指标和结果已保存至: {results_dir}")
    
    return results_dir

def analyze_feature_importance(model, feature_names):
    """分析特征重要性"""
    # 获取不同类型的特征重要性
    importance_types = ['weight', 'gain', 'cover']
    importance_data = {'特征': feature_names}
    
    for imp_type in importance_types:
        try:
            importance = model.get_score(importance_type=imp_type)
            values = []
            for feature in feature_names:
                values.append(importance.get(feature, 0))
            importance_data[imp_type] = values
        except Exception as e:
            print(f"无法获取 {imp_type} 类型的重要性: {str(e)}")
    
    # 创建DataFrame并按gain排序
    importance_df = pd.DataFrame(importance_data)
    if 'gain' in importance_df.columns:
        importance_df = importance_df.sort_values('gain', ascending=False)
    
    # 打印特征重要性
    print("\n特征重要性排名:")
    print(importance_df)
    
    return importance_df

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建输出目录
    setup_directories()
    
    print("=" * 50)
    print("XGBoost降水预测优化训练脚本")
    print("=" * 50)
    
    # 检查GPU可用性
    gpu_info = check_gpu_availability()
    print_gpu_status()
    
    # 决定是否使用GPU
    use_gpu = args.gpu and gpu_info['cuda_available']
    if args.gpu and not gpu_info['cuda_available']:
        print("警告: 请求了GPU但未发现可用GPU，将使用CPU")
    
    if use_gpu:
        print("将使用GPU进行训练")
    else:
        print("将使用CPU进行训练")
    
    # 加载数据
    datasets, mask = load_rainfall_data()
    
    # 准备训练和测试数据
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = prepare_data(datasets, mask)
    
    # 执行交叉验证(如果需要)
    if args.cv > 1:
        cv_results = perform_cross_validation(
            np.vstack((X_train, X_val)),
            np.hstack((y_train, y_val)),
            feature_names,
            use_gpu=use_gpu,
            n_splits=args.cv
        )
    
    # 训练最终模型
    model = train_model(
        X_train, X_val, y_train, y_val,
        feature_names, 
        use_gpu=use_gpu,
        early_stopping_rounds=args.early_stopping
    )
    
    # 评估模型
    evaluation = evaluate_model(model, X_test, y_test, feature_names, use_gpu=use_gpu)
    evaluation['y_true'] = y_test  # 添加真实值到评估字典
    
    # 分析特征重要性
    importance_df = analyze_feature_importance(model, feature_names)
    
    # 创建运行ID
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存结果
    results_dir = save_results(
        model, evaluation, importance_df, 
        args.output_dir, run_id=run_id
    )
    
    # 可视化结果
    visualize_results(
        model, evaluation, feature_names, results_dir
    )
    
    print("\n训练和评估流程已完成!")
    print(f"所有结果已保存在 {results_dir}")

if __name__ == "__main__":
    # 捕获所有异常以确保干净退出
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()