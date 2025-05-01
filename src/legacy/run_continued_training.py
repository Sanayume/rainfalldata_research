"""
启动继续训练脚本 - 先修复编码问题，然后启动继续训练
"""
import os
import sys
import subprocess
import encoding_fix

# 在程序开始时设置Windows控制台编码为UTF-8
print("\n=== Windows控制台编码修复 ===")
# 确保完全修复编码问题
encoding_fix.setup_environment(safe_logging=True)

print("\n启动继续训练...\n" + "="*40)

# 定义需要运行的脚本路径
script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "continue_training.py")

# 设置环境变量确保子进程使用UTF-8编码
my_env = os.environ.copy()
my_env["PYTHONIOENCODING"] = "utf-8"
my_env["PYTHONLEGACYWINDOWSSTDIO"] = "utf-8"  # 用于旧版Windows处理方式

# 使用Python子进程运行训练脚本，继承设置好的环境
try:
    # 获取当前Python解释器路径
    python_exe = sys.executable
    
    # 显示将要运行的命令
    print(f"执行: {python_exe} {script_path}")
    print("="*40 + "\n")
    
    # 使用subprocess运行训练脚本
    result = subprocess.run(
        [python_exe, script_path], 
        check=True, 
        env=my_env,
        encoding='utf-8',  # 确保子进程使用UTF-8编码
        errors='backslashreplace',  # 处理无法编码的字符
        bufsize=1  # 行缓冲，立即显示输出
    )
    
    # 打印结果代码
    print("\n" + "="*40)
    print(f"继续训练完成，退出代码: {result.returncode}")
    print("="*40)
    
except subprocess.CalledProcessError as e:
    print("\n" + "="*40)
    print(f"继续训练过程中出现错误，退出代码: {e.returncode}")
    print(f"错误信息: {e}")
    print("="*40)
except Exception as e:
    print("\n" + "="*40)
    print(f"运行继续训练脚本时出现异常: {type(e).__name__}: {e}")
    print("="*40)
