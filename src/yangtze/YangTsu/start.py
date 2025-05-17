import subprocess
import sys
import os

def run_script(script_name):
    """
    执行指定的 Python 脚本。

    参数:
    script_name (str): 要执行的脚本文件名。

    返回:
    bool: 如果脚本成功执行 (返回码为 0)，则为 True，否则为 False。
    """
    if not os.path.exists(script_name):
        print(f"错误: 脚本 '{script_name}' 未找到。请确保它与本启动脚本在同一目录下，或者提供正确路径。")
        return False

    print(f"\n--- 开始执行脚本: {script_name} ---")
    try:
        # 使用 sys.executable 来确保使用的是当前 Python 环境的解释器
        # check=True 会在脚本返回非零退出码时抛出 CalledProcessError 异常
        process = subprocess.run([sys.executable, script_name], check=True, text=True)
        print(f"--- 脚本: {script_name} 执行完毕，返回码: {process.returncode} ---")
        return True
    except subprocess.CalledProcessError as e:
        print(f"--- 脚本: {script_name} 执行失败 ---")
        print(f"错误信息: {e}")
        print(f"返回码: {e.returncode}")
        if e.stdout:
            print(f"标准输出:\n{e.stdout}")
        if e.stderr:
            print(f"标准错误:\n{e.stderr}")
        return False
    except FileNotFoundError:
        print(f"错误: Python 解释器 '{sys.executable}' 或脚本 '{script_name}' 未找到。")
        return False
    except Exception as e:
        print(f"执行脚本 {script_name} 时发生未知错误: {e}")
        return False

if __name__ == "__main__":
    scripts_to_run = ["xgboost1.py", "other_model.py"]
    
    print(">>> 启动训练流程...")

    # 确保脚本都存在
    all_scripts_exist = True
    for script in scripts_to_run:
        if not os.path.exists(script):
            print(f"错误: 必要的脚本 '{script}' 未找到。流程中止。")
            all_scripts_exist = False
            break
    
    if not all_scripts_exist:
        sys.exit(1) # 以错误码退出

    # 依次执行脚本
    if run_script(scripts_to_run[0]): # 执行 xgboost1.py
        # 如果 xgboost1.py 成功，则执行 other_model.py
        if run_script(scripts_to_run[1]): # 执行 other_model.py
            print("\n>>> 所有脚本成功执行完毕！训练流程结束。")
        else:
            print(f"\n>>> 脚本 '{scripts_to_run[1]}' 执行失败。训练流程中止。")
    else:
        print(f"\n>>> 脚本 '{scripts_to_run[0]}' 执行失败。后续脚本将不会执行。训练流程中止。")