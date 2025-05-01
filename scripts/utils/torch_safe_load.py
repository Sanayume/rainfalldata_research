"""
安全加载PyTorch模型的实用函数
"""
import torch
import logging
import os
import warnings

logger = logging.getLogger(__name__)

def safe_load_model(model_path, device=None, show_warnings=True, weights_only=False):
    """
    安全地加载PyTorch模型，处理各种异常情况
    
    Args:
        model_path: 模型文件的路径
        device: 要加载到哪个设备（默认为当前设备）
        show_warnings: 是否显示警告信息
        weights_only: 是否仅加载权重（True避免pickle安全警告）
    
    Returns:
        加载的模型数据，如果出错则为None
    """
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return None
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 如果不需要显示警告，临时禁用它们
    if not show_warnings:
        warnings.filterwarnings('ignore', category=UserWarning)
        warnings.filterwarnings('ignore', category=FutureWarning)
    
    try:
        # 尝试加载模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=weights_only)
        logger.info(f"成功加载模型: {model_path}")
        return checkpoint
    except RuntimeError as e:
        logger.error(f"加载模型时出现运行时错误: {e}")
    except Exception as e:
        logger.error(f"加载模型时出现异常: {type(e).__name__}: {e}")
    finally:
        # 恢复警告设置
        if not show_warnings:
            warnings.resetwarnings()
    
    return None

def extract_checkpoint_info(checkpoint, detailed=False):
    """
    提取检查点的关键信息
    
    Args:
        checkpoint: 加载的检查点数据
        detailed: 是否显示详细信息
    
    Returns:
        包含检查点关键信息的字典
    """
    info = {}
    
    if checkpoint is None:
        return {'status': 'empty'}
    
    # 提取基本信息
    if isinstance(checkpoint, dict):
        info['type'] = 'dict'
        
        # 提取常见键
        for key in ['epoch', 'train_loss', 'val_loss', 'metrics']:
            if key in checkpoint:
                info[key] = checkpoint[key]
        
        # 检查模型状态
        if 'model_state_dict' in checkpoint:
            info['has_model_state'] = True
            if detailed:
                try:
                    # 计算参数数量
                    param_count = sum(p.numel() for tensor_name, p in checkpoint['model_state_dict'].items())
                    info['parameter_count'] = param_count
                    info['layers'] = list(checkpoint['model_state_dict'].keys())
                except:
                    info['parameter_count'] = 'unknown'
        
        # 检查其他组件
        for component in ['optimizer_state_dict', 'scheduler_state_dict', 'scaler_X', 'scaler_y', 'config']:
            info[f'has_{component}'] = component in checkpoint
    
    elif isinstance(checkpoint, torch.nn.Module):
        info['type'] = 'model'
        info['parameter_count'] = sum(p.numel() for p in checkpoint.parameters())
    
    else:
        info['type'] = type(checkpoint).__name__
    
    info['status'] = 'valid'
    return info

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 测试函数
    print("\n== PyTorch 安全加载测试 ==\n")
    
    # 当前目录下寻找pth文件
    import glob
    model_files = glob.glob("models/*.pth")
    
    if not model_files:
        print("当前目录下没有找到模型文件")
    else:
        for model_file in model_files:
            print(f"\n检查模型文件: {model_file}")
            checkpoint = safe_load_model(model_file)
            info = extract_checkpoint_info(checkpoint, detailed=True)
            
            print(f"模型状态: {info['status']}")
            if 'epoch' in info:
                print(f"训练轮次: {info['epoch'] + 1}")
            if 'val_loss' in info:
                print(f"验证损失: {info['val_loss']:.4f}")
            if 'parameter_count' in info:
                print(f"参数数量: {info.get('parameter_count', 'unknown'):,}")
