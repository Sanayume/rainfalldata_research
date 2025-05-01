"""
中文字体处理辅助模块，解决matplotlib和其他图形库中文显示乱码问题
"""

import os
import sys
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.font_manager import FontProperties
import warnings

def get_system_fonts():
    """获取系统中安装的所有字体"""
    font_paths = fm.findSystemFonts()
    font_names = []
    
    for font_path in font_paths:
        try:
            font = fm.FontProperties(fname=font_path)
            font_name = font.get_name()
            if font_name:
                font_names.append((font_name, font_path))
        except Exception:
            pass
    
    return font_names

def find_chinese_fonts():
    """查找可能支持中文的字体"""
    all_fonts = get_system_fonts()
    chinese_fonts = []
    
    # 常见的中文字体名称
    chinese_font_names = [
        'SimHei', 'SimSun', 'Microsoft YaHei', 'NSimSun', 'FangSong', 'KaiTi',
        'Source Han Sans CN', 'Source Han Serif CN', 'WenQuanYi Micro Hei',
        'DengXian', '微软雅黑', '宋体', '黑体', '仿宋', '楷体', '华文细黑'
    ]
    
    # 查找包含中文字体名称的字体
    for name, path in all_fonts:
        for cn_name in chinese_font_names:
            if cn_name.lower() in name.lower() or cn_name.lower() in path.lower():
                chinese_fonts.append((name, path))
                break
    
    return chinese_fonts

def setup_chinese_font(font_name=None, test=True, save_path=None):
    """
    配置matplotlib以正确显示中文
    
    参数:
        font_name: 指定字体名称，若不指定则自动查找
        test: 是否测试字体
        save_path: 测试图像保存路径
    
    返回:
        成功设置的字体名称，若失败则返回None
    """
    # 配置matplotlib的默认参数
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 如果没有指定字体，尝试查找中文字体
    if font_name is None:
        chinese_fonts = find_chinese_fonts()
        if chinese_fonts:
            # 使用找到的第一个中文字体
            font_name = chinese_fonts[0][0]
            print(f"自动选择字体: {font_name}")
        else:
            # 如果找不到中文字体，使用默认的sans-serif
            print("警告：未找到中文字体，将尝试使用内置字体")
            font_name = 'sans-serif'
    
    # 设置全局字体
    plt.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    
    # 测试中文显示
    if test:
        if save_path is None:
            save_path = 'f:/rainfalldata/figures/font_test.png'
        
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 测试常见中文字符和标点
        test_text = (
            '中文字体测试 - ABCabc123\n'
            '特征重要性、迭代次数和错误率\n'
            '混淆矩阵，精确率与召回率\n'
        )
        
        plt.figure(figsize=(8, 3))
        plt.text(0.5, 0.5, test_text, 
                fontsize=14, ha='center', va='center')
        plt.title('中文字体支持测试')
        plt.axis('off')
        plt.tight_layout()
        
        try:
            plt.savefig(save_path, dpi=200)
            plt.close()
            print(f"字体测试成功，图片已保存到: {save_path}")
            return font_name
        except Exception as e:
            plt.close()
            print(f"字体测试失败: {str(e)}")
            return None
    
    return font_name

def list_available_chinese_fonts():
    """列出系统中所有可能的中文字体"""
    chinese_fonts = find_chinese_fonts()
    
    if not chinese_fonts:
        print("未找到支持中文的字体")
        return []
    
    print(f"找到 {len(chinese_fonts)} 个可能支持中文的字体:")
    for i, (name, path) in enumerate(chinese_fonts, 1):
        print(f"{i}. {name} - {path}")
    
    return [name for name, _ in chinese_fonts]

def create_font_config_file(font_name=None):
    """创建matplotlib字体配置文件"""
    if font_name is None:
        chinese_fonts = find_chinese_fonts()
        if chinese_fonts:
            font_name = chinese_fonts[0][0]
        else:
            print("警告：未找到中文字体，将使用默认字体")
            font_name = 'sans-serif'
    
    # 获取matplotlib配置目录
    import matplotlib
    config_dir = matplotlib.get_configdir()
    os.makedirs(config_dir, exist_ok=True)
    
    # 创建配置文件
    with open(os.path.join(config_dir, 'matplotlibrc'), 'w', encoding='utf-8') as f:
        f.write(f"font.family: sans-serif\n")
        f.write(f"font.sans-serif: {font_name}, DejaVu Sans, Arial Unicode MS, sans-serif\n")
        f.write(f"axes.unicode_minus: False\n")
    
    print(f"matplotlib配置文件已创建: {os.path.join(config_dir, 'matplotlibrc')}")
    print(f"设置的默认字体: {font_name}")
    return True

def test_all_chinese_fonts(output_dir='f:/rainfalldata/figures/font_tests'):
    """测试所有中文字体并生成示例图像"""
    chinese_fonts = find_chinese_fonts()
    
    if not chinese_fonts:
        print("未找到支持中文的字体")
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 测试每个中文字体
    test_text = '中文字体测试 - 特征重要性、AUC曲线和混淆矩阵'
    
    plt.figure(figsize=(15, len(chinese_fonts)*0.8))
    plt.axis('off')
    plt.title('中文字体对比')
    
    for i, (name, path) in enumerate(chinese_fonts):
        try:
            font_prop = FontProperties(fname=path)
            plt.text(0.1, 1-0.05*(i+1), f"{name}: {test_text}", 
                    fontproperties=font_prop, fontsize=12)
        except Exception as e:
            plt.text(0.1, 1-0.05*(i+1), f"{name}: 加载失败 - {str(e)}", 
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_chinese_fonts.png'), dpi=200)
    plt.close()
    
    print(f"中文字体对比图已保存至: {os.path.join(output_dir, 'all_chinese_fonts.png')}")
    return True

# 快速设置函数，可以从其他模块直接调用
def setup_chinese_matplotlib(test=True):
    """
    快速设置matplotlib支持中文显示
    
    返回:
        True: 设置成功
        False: 设置失败
    """
    try:
        # 首先尝试使用SimHei
        if setup_chinese_font('SimHei', test=test) is not None:
            return True
            
        # 如果失败，尝试使用Microsoft YaHei
        if setup_chinese_font('Microsoft YaHei', test=test) is not None:
            return True
            
        # 再次尝试使用SimSun
        if setup_chinese_font('SimSun', test=test) is not None:
            return True
        
        # 尝试自动选择字体
        if setup_chinese_font(None, test=test) is not None:
            return True
            
        # 如果所有尝试都失败，返回False
        print("警告：无法设置中文字体，图形中的中文可能显示为乱码")
        return False
    except Exception as e:
        print(f"设置中文字体时出错: {str(e)}")
        return False

if __name__ == "__main__":
    # 测试模块功能
    print("测试中文字体支持...")
    setup_chinese_matplotlib()
    list_available_chinese_fonts()
    test_all_chinese_fonts()
