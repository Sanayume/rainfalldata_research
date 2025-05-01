"""
安全处理程序模块 - 为控制台和日志提供安全的输出处理
"""
import sys
import os
import logging
from datetime import datetime
import re

class SafeStreamHandler(logging.StreamHandler):
    """安全的流处理程序，能够处理编码错误"""
    def __init__(self, stream=None, encoding='utf-8'):
        super().__init__(stream)
        self.encoding = encoding
        
        # 重写terminator使用ASCII换行符
        self.terminator = '\n'
    
    def emit(self, record):
        """重写emit方法以安全处理编码错误"""
        try:
            msg = self.format(record)
            try:
                if isinstance(msg, str):
                    # 尝试以当前控制台编码写入
                    self.stream.write(msg + self.terminator)
                else:
                    # 如果不是字符串，转换为字符串
                    self.stream.write(str(msg) + self.terminator)
            except UnicodeEncodeError:
                # 如果发生编码错误，使用ASCII编码并替换不可编码字符
                safe_msg = msg.encode('ascii', 'replace').decode('ascii')
                self.stream.write(safe_msg + self.terminator)
            except Exception as e:
                # 最后的保底方案：使用纯ASCII和英文描述错误
                fallback_msg = f"[Error displaying log: {type(e).__name__}]"
                self.stream.write(fallback_msg + self.terminator)
            
            self.flush()
        except Exception:
            self.handleError(record)

class SafeFileHandler(logging.FileHandler):
    """安全的文件处理程序，能够处理文件写入错误"""
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        super().__init__(filename, mode, encoding, delay)
    
    def emit(self, record):
        """重写emit方法以安全处理文件写入错误"""
        try:
            super().emit(record)
        except Exception as e:
            # 如果写入失败，尝试写入一个错误标记
            try:
                with open(self.baseFilename, 'a', encoding='utf-8') as f:
                    f.write(f"[日志写入错误: {datetime.now().isoformat()}] {type(e).__name__}\n")
            except:
                # 如果还是失败，无能为力，交给handleError
                self.handleError(record)

class SafeFormatter(logging.Formatter):
    """安全的格式化程序，能够处理任何格式化错误"""
    def format(self, record):
        """重写format方法以安全处理格式化错误"""
        # 特殊字符替换表
        replacements = {
            '²': '^2',
            '³': '^3',
            '✓': '*',
            '✗': 'X',
            '→': '->',
            '←': '<-',
            '♥': '<3',
            '★': '*',
            '≤': '<=',
            '≥': '>=',
            '≠': '!=',
            '≈': '~='
        }
        
        # 保存原始消息
        original_msg = record.msg
        
        try:
            # 如果可能，替换特殊字符
            if isinstance(record.msg, str):
                for char, replacement in replacements.items():
                    if char in record.msg:
                        record.msg = record.msg.replace(char, replacement)
            
            # 调用父类的format方法
            return super().format(record)
        except Exception as e:
            # 如果格式化失败，使用简单的替代格式
            timestamp = self.formatTime(record)
            level = record.levelname
            
            # 确保消息是字符串并替换特殊字符
            if hasattr(original_msg, '__str__'):
                msg = str(original_msg)
                # 移除所有非ASCII字符
                msg = ''.join(c if ord(c) < 128 else '_' for c in msg)
            else:
                msg = f"[无法显示的消息类型: {type(original_msg).__name__}]"
                
            return f"{timestamp} - {level} - 格式化错误 - {msg}"
        finally:
            # 恢复原始消息以免影响其他处理程序
            record.msg = original_msg

def setup_safe_console():
    """配置安全的控制台输出"""
    # 获取根日志记录器
    root_logger = logging.getLogger()
    
    # 创建安全的格式化程序
    formatter = SafeFormatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 替换所有StreamHandler为SafeStreamHandler
    for i, handler in enumerate(root_logger.handlers):
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, SafeStreamHandler):
            # 创建安全处理程序并保留原始流
            safe_handler = SafeStreamHandler(handler.stream)
            safe_handler.setLevel(handler.level)
            safe_handler.setFormatter(formatter)
            
            # 移除旧的处理程序并添加新的
            root_logger.removeHandler(handler)
            root_logger.addHandler(safe_handler)
    
    # 如果没有任何StreamHandler，添加一个到标准输出
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 替换sys.stdout和sys.stderr的write方法
    original_stdout_write = sys.stdout.write
    original_stderr_write = sys.stderr.write
    
    def safe_write(stream_name, original_write):
        def _safe_write(text):
            try:
                return original_write(text)
            except UnicodeEncodeError:
                # 如果编码错误，替换为ASCII
                return original_write(text.encode('ascii', 'replace').decode('ascii'))
            except Exception as e:
                # 发生其他错误时记录到日志
                logging.error(f"{stream_name} 写入错误: {type(e).__name__}")
                return original_write("[无法显示的文本]")
        return _safe_write
    
    sys.stdout.write = safe_write("stdout", original_stdout_write)
    sys.stderr.write = safe_write("stderr", original_stderr_write)
    
    return True

if __name__ == "__main__":
    # 测试安全处理程序
    setup_safe_console()
    
    # 配置基本日志
    logging.basicConfig(level=logging.INFO)
    
    # 测试各种消息
    logging.info("普通ASCII文本")
    logging.info("中文字符: 你好，世界！")
    logging.info("特殊符号: R² 值为 0.95")
    logging.info("包含勾号: ✓ 成功, ✗ 失败")
    
    # 测试异常处理
    try:
        1/0
    except Exception as e:
        logging.exception("测试异常处理")
    
    print("测试完成!")
