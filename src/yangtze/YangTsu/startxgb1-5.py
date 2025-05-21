import subprocess
import os

# 获取当前脚本所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
log_file_path = os.path.join(current_dir, "log_xgboost1_5.txt")

# 要按顺序执行的脚本列表
scripts_to_run = [
    "xgboost_best.py",
    "xgboost_optimization_main.py"
]

# 打开日志文件以追加模式写入
with open(log_file_path, "w", encoding="utf-8") as log_file:
    for script_name in scripts_to_run:
        script_path = os.path.join(current_dir, script_name)
        log_file.write(f"--- Starting execution of {script_name} ---\n\n")
        print(f"Running {script_name}...")
        try:
            # 执行脚本并捕获输出
            # 使用 python 而不是 sys.executable 来确保即使在 venv 外运行也能找到 python
            process = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                check=False, # 设置为 False，这样即使脚本出错也不会抛出 CalledProcessError
                encoding="utf-8",
                errors="replace" # 替换无法解码的字符
            )

            # 将标准输出写入日志
            if process.stdout:
                log_file.write("Standard Output:\n")
                log_file.write(process.stdout)
                log_file.write("\n")

            # 将标准错误写入日志
            if process.stderr:
                log_file.write("Standard Error:\n")
                log_file.write(process.stderr)
                log_file.write("\n")

            if process.returncode != 0:
                log_file.write(f"--- {script_name} finished with errors (return code: {process.returncode}) ---\n\n")
                print(f"{script_name} finished with errors. Check log for details.")
            else:
                log_file.write(f"--- Finished execution of {script_name} (return code: {process.returncode}) ---\n\n")
                print(f"{script_name} finished successfully.")

        except FileNotFoundError:
            error_message = f"Error: Script {script_path} not found.\n"
            log_file.write(error_message)
            log_file.write(f"--- Execution of {script_name} failed ---\n\n")
            print(error_message)
        except Exception as e:
            error_message = f"An unexpected error occurred while running {script_name}: {e}\n"
            log_file.write(error_message)
            log_file.write(f"--- Execution of {script_name} failed ---\n\n")
            print(error_message)
        
        # 确保每次写入后都刷新到磁盘
        log_file.flush()

print(f"All scripts processed. Output logged to {log_file_path}")