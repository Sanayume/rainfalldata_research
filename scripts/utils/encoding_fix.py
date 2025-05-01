"""
编码修复工具 - 处理Windows控制台编码问题
"""
import sys
import os
import locale
import ctypes
import codecs
import re
import logging

def set_console_utf8():
    """
    尝试将Windows控制台设置为UTF-8模式
    
    注意：这只对当前进程有效，并且可能需要管理员权限
    """
    if os.name == 'nt':  # Windows操作系统
        try:
            # 获取控制台输出的代码页
            console_output_cp = ctypes.windll.kernel32.GetConsoleOutputCP()
            print(f"当前控制台代码页: {console_output_cp}")
            
            # 设置控制台代码页为UTF-8 (65001)
            if console_output_cp != 65001:
                if ctypes.windll.kernel32.SetConsoleOutputCP(65001):
                    print("成功将控制台输出代码页设置为UTF-8 (65001)")
                else:
                    print("无法设置控制台输出代码页")
                
                if ctypes.windll.kernel32.SetConsoleCP(65001):
                    print("成功将控制台输入代码页设置为UTF-8 (65001)")
                else:
                    print("无法设置控制台输入代码页")
        except Exception as e:
            print(f"设置控制台代码页时出错: {e}")
    else:
        print("非Windows系统，无需设置控制台代码页")

def patch_logging_handlers():
    """
    修补日志处理程序，防止UnicodeEncodeError异常
    """
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
    
    # 编译正则表达式
    pattern = re.compile('|'.join(re.escape(key) for key in replacements.keys()))
    
    def replace_unicode(text):
        return pattern.sub(lambda x: replacements[x.group(0)], text)
    
    # 获取根日志记录器的所有处理程序
    root_logger = logging.getLogger()
    
    for handler in root_logger.handlers:
        # 检查是否是StreamHandler且使用标准输出或错误
        if isinstance(handler, logging.StreamHandler):
            # 创建一个安全的write方法
            original_stream = handler.stream
            original_write = original_stream.write
            
            def safe_write(text):
                try:
                    # 首先尝试替换已知的特殊字符
                    safe_text = replace_unicode(text)
                    return original_write(safe_text)
                except UnicodeEncodeError:
                    try:
                        # 如果失败，尝试使用ASCII编码并替换不可编码的字符
                        encoded_text = text.encode(original_stream.encoding or 'ascii', 'replace').decode(original_stream.encoding or 'ascii')
                        return original_write(encoded_text)
                    except Exception as e:
                        # 最后的后备方法：移除所有非ASCII字符
                        ascii_only = ''.join(c if ord(c) < 128 else '_' for c in text)
                        return original_write(ascii_only)
            
            # 替换写入方法
            handler.stream.write = safe_write
            print(f"已强化修补日志处理程序以处理编码错误")
            
            # 特别处理终端输出的格式化程序，确保它安全处理特殊字符
            if hasattr(handler, 'formatter'):
                original_format = handler.formatter.format
                
                def safe_format(record):
                    try:
                        msg = original_format(record)
                        return replace_unicode(msg)
                    except:
                        # 如果格式化失败，尝试使用基本格式化
                        if hasattr(record, 'msg'):
                            safe_msg = str(record.msg).encode('ascii', 'replace').decode('ascii')
                            record.msg = safe_msg
                        return original_format(record)
                
                handler.formatter.format = safe_format

def patch_sys_excepthook():
    """修补sys.excepthook以安全处理异常中的Unicode字符"""
    original_excepthook = sys.excepthook
    
    def safe_excepthook(exc_type, exc_value, exc_traceback):
        try:
            # 尝试使用原始异常钩子
            return original_excepthook(exc_type, exc_value, exc_traceback)
        except UnicodeEncodeError:
            # 如果发生编码错误，使用ASCII编码处理异常消息
            try:
                exc_value = type(exc_value)(str(exc_value).encode('ascii', 'replace').decode('ascii'))
                return original_excepthook(exc_type, exc_value, exc_traceback)
            except:
                # 最后的后备方法：直接打印基本错误信息
                print(f"错误: {exc_type.__name__}: {str(exc_value).encode('ascii', 'replace').decode('ascii')}")
    
    sys.excepthook = safe_excepthook

def setup_environment(safe_logging=True):
    """设置环境以支持UTF-8编码"""
    # 检查Python解释器的默认编码
    print(f"Python默认编码: {sys.getdefaultencoding()}")
    
    # 检查当前区域设置
    print(f"当前区域设置: {locale.getpreferredencoding()}")
    
    # 设置环境变量以支持UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # 尝试设置控制台为UTF-8模式
    set_console_utf8()
    
    # 修补sys.excepthook以安全处理异常
    patch_sys_excepthook()
    
    # 修补日志处理程序以处理编码错误
    if safe_logging:
        patch_logging_handlers()
    
    # 重新配置stdout和stderr，使用更强大的错误处理
    try:
        # 为stdout设置UTF-8编码，使用backslashreplace错误处理器
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'backslashreplace')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'backslashreplace')
        print("已将标准输出/错误流重新配置为UTF-8，错误字符将使用转义序列表示")
    except Exception as e:
        print(f"重新配置输出流时出错: {e}")
    
    # 设置默认编码为UTF-8 (这对文件I/O有影响)
    if sys.getdefaultencoding() != 'utf-8':
        print("警告: 无法在运行时更改Python默认编码，默认编码仍为", sys.getdefaultencoding())

def test_unicode_output():
    """测试Unicode字符输出"""
    test_strings = [
        "普通ASCII字符",
        "带有重音的字符: é è à",
        "中文字符: 你好，世界！",
        "特殊符号: © ® ™ ± ² ³ ½",
        "R² 值是 0.95",
        "✓ 完成，✗ 失败",
        "表情符号: 😀 🚀 🐍 🔥"
    ]
    
    print("\n=== Unicode输出测试 ===")
    for i, s in enumerate(test_strings, 1):
        print(f"{i}. {s}")
    print("======================\n")
    
    # 测试日志输出
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    
    print("测试日志输出:")
    for s in test_strings:
        logger.info(s)

if __name__ == "__main__":
    # 显示横幅
    print("=" * 60)
    print(" Windows控制台编码修复工具 ".center(60))
    print("=" * 60)
    
    # 设置环境
    setup_environment()
    
    # 测试Unicode输出
    test_unicode_output()
    
    print("\n使用方法:")
    print("1. 在运行其他Python脚本前先运行此脚本")
    print("2. 或者在其他脚本开头添加以下代码:")
    print("   import encoding_fix")
    print("   encoding_fix.setup_environment()")
    print("\n" + "=" * 60)
