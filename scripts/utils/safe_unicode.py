"""
安全的Unicode字符处理工具
"""
import sys
import re

def safe_string(text, replacement='?'):
    """
    将字符串中的可能无法在当前控制台显示的Unicode字符替换为安全字符
    
    Args:
        text: 输入文本
        replacement: 替换字符，默认为问号
    
    Returns:
        安全的字符串
    """
    if text is None:
        return ""
    
    try:
        # 尝试使用控制台编码编码，不能编码的字符用替换字符代替
        console_encoding = sys.stdout.encoding or 'utf-8'
        return text.encode(console_encoding, 'replace').decode(console_encoding)
    except:
        # 如果出错，回退到ASCII编码
        return text.encode('ascii', 'replace').decode('ascii')

def replace_special_chars(text):
    """
    替换特殊Unicode字符为ASCII等效字符
    
    Args:
        text: 输入文本
    
    Returns:
        替换后的文本
    """
    replacements = {
        '²': '^2',
        '³': '^3',
        '±': '+/-',
        '×': 'x',
        '÷': '/',
        '√': 'sqrt',
        '∑': 'sum',
        '∞': 'inf',
        '≤': '<=',
        '≥': '>=',
        '≠': '!=',
        '≈': '~=',
        'π': 'pi',
        'µ': 'u',
        '°': ' deg',
        '£': 'GBP',
        '€': 'EUR',
        '¥': 'JPY',
        '©': '(c)',
        '®': '(R)',
        '™': '(TM)',
        # 常见的表情符号和特殊符号
        '✓': '[OK]',
        '✗': '[X]',
        '✘': '[X]',
        '⚠': '[!]',
        '⚡': '[!]',
        '→': '->',
        '←': '<-',
        '↑': '^',
        '↓': 'v',
        '♥': '<3',
        '★': '*',
        '☆': '*',
        '☑': '[v]',
        '☒': '[x]',
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    return text

def make_filename_safe(filename):
    """
    创建适合作为文件名的安全字符串
    
    Args:
        filename: 原始文件名
        
    Returns:
        安全的文件名
    """
    # 移除不允许用作文件名的字符
    safe_name = re.sub(r'[\\/*?:"<>|]', '_', filename)
    # 将多个连续下划线替换为单个下划线
    safe_name = re.sub(r'_+', '_', safe_name)
    return safe_name

if __name__ == "__main__":
    # 测试函数
    test_strings = [
        "Hello, World!",
        "The equation is: E=mc²",
        "Temperature: 25°C ± 0.5°C",
        "Please check: ✓ Done, ✗ Not done",
        "→ Next step",
        "I ♥ Python!",
        "Copyright © 2023",
        "R² value is 0.987",
        "π ≈ 3.14159",
        "£10.99 + €15.00",
        "File: my/file/path<with>invalid:chars?.txt",
        "😀 😎 🐍 🚀",  # 表情符号
    ]
    
    print("=== 原始字符串 vs 安全字符串 ===")
    print("-" * 50)
    
    for s in test_strings:
        safe = safe_string(s)
        replaced = replace_special_chars(s)
        filename = make_filename_safe(s)
        
        print(f"原始: {s}")
        print(f"安全: {safe}")
        print(f"替换: {replaced}")
        print(f"文件名: {filename}")
        print("-" * 50)
