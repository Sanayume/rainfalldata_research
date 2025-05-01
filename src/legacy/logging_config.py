"""
日志配置模块 - 为所有脚本提供统一的日志设置
"""
import logging
import sys
import os
from datetime import datetime
import re

def sanitize_text(text):
    """净化文本，删除或替换可能导致编码问题的字符"""
    # 替换特殊Unicode字符为ASCII等效字符的字典
    replacements = {
        '²': '^2',
        '³': '^3',
        '✓': '*',
        '✗': 'X',
        '✘': 'X',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '♥': '<3',
        '★': '*',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '≈': '~=',
        'π': 'pi',
        'μ': 'u',
        '°': ' deg',
        '±': '+/-'
    }
    
    # 使用替换字典
    for char, replacement in replacements.items():
        if char in text:
            text = text.replace(char, replacement)
    
    return text

class SafeFormatter(logging.Formatter):
    """安全的日志格式化程序，处理所有可能的编码问题"""
    def format(self, record):
        # 保存原始消息
        original_msg = record.msg
        
        try:
            # 如果消息是字符串，尝试净化它
            if isinstance(record.msg, str):
                record.msg = sanitize_text(record.msg)
                
            # 调用原始format方法
            result = super().format(record)
            return result
        except Exception as e:
            # 如果格式化失败，回退到简单的ASCII编码
            record.msg = f"[格式化错误: {type(e).__name__}] " + str(original_msg).encode('ascii', 'replace').decode('ascii')
            return super().format(record)
        finally:
            # 恢复原始消息以便其他处理程序使用
            record.msg = original_msg

class SafeStreamHandler(logging.StreamHandler):
    """安全的流处理程序，处理编码错误"""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # 无论如何都要输出一些内容
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                # 如果编码失败，使用ASCII并替换不可编码的字符
                stream.write(msg.encode('ascii', 'replace').decode('ascii') + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(
    name="rainfall_predictor",
    level=logging.INFO,
    log_to_file=True,
    log_dir="logs",
    console=True
):
    """设置日志系统"""
    # 创建日志目录
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 获取根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除可能的现有处理程序
    logger.handlers.clear()
    
    # 创建安全的格式化程序
    formatter = SafeFormatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 添加文件处理程序
    if log_to_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理程序
    if console:
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 屏蔽某些模块的调试日志
    for module in ['matplotlib', 'PIL', 'urllib3', 'torch.utils.data']:
        logging.getLogger(module).setLevel(logging.WARNING)
    
    return logger

if __name__ == "__main__":
    # 测试日志系统
    logger = setup_logging(name="test_logger")
    
    logger.info("日志系统测试")
    logger.info("这是正常的ASCII文本")
    logger.info("这包含中文字符 和 特殊字符 R² 值和 ✓ 勾号")
    
    try:
        1/0
    except Exception as e:
        logger.exception("测试异常记录")
    
    logger.info("测试完成")
