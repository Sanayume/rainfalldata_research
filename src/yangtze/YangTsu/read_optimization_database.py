import sqlite3
import pandas as pd
import os

# --- 配置数据库文件路径 ---
db_file_name = "my_optimization_history.db"
# 假设脚本和数据库文件在同一个目录下
# F:\rainfalldata\src\yangtze\YangTsu\
current_script_dir = os.path.dirname(os.path.abspath(__file__))
db_file_path = os.path.join(current_script_dir, db_file_name)

# 如果你知道数据库文件的确切位置，可以取消注释并修改下面这行：
db_file_path = r"F:\rainfalldata\my_optimization_history.db" # 或者其他你存放 .db 文件的路径

# --- 配置输出文本文件路径 ---
output_txt_file_name = "database_content_summary_full.txt" # 修改文件名以区分
output_txt_file_path = os.path.join(current_script_dir, output_txt_file_name)

if not os.path.exists(db_file_path):
    print(f"错误：数据库文件 '{db_file_path}' 未找到。")
    print("请确保文件名和路径正确。")
else:
    conn = None  # 初始化 conn
    try:
        with open(output_txt_file_path, 'w', encoding='utf-8') as outfile:
            outfile.write(f"--- 正在尝试连接到数据库: {db_file_path} ---\n")
            print(f"--- 正在尝试连接到数据库: {db_file_path} ---")
            print(f"--- 输出将保存到: {output_txt_file_path} ---")

            # 连接到 SQLite 数据库
            conn = sqlite3.connect(db_file_path)
            cursor = conn.cursor()

            # 1. 列出数据库中的所有表
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            outfile.write("\n数据库中的表:\n")
            if not tables:
                outfile.write("  未找到任何表。\n")
            else:
                for table in tables:
                    outfile.write(f"  - {table[0]}\n")

            # 2. 查看 'studies' 表的内容 (如果存在)
            if ('studies',) in tables:
                outfile.write("\n--- 'studies' 表内容 ---\n")
                try:
                    studies_df = pd.read_sql_query("SELECT * FROM studies", conn)
                    if studies_df.empty:
                        outfile.write("  'studies' 表为空。\n")
                    else:
                        outfile.write(studies_df.to_string(index=True) + "\n")
                except Exception as e:
                    outfile.write(f"  读取 'studies' 表时出错: {e}\n")
            else:
                outfile.write("\n未找到 'studies' 表。\n")

            # 3. 查看 'trials' 表的内容 (如果存在) - 输出全部内容
            if ('trials',) in tables:
                outfile.write("\n--- 'trials' 表内容 ---\n")
                try:
                    trials_df = pd.read_sql_query("SELECT * FROM trials", conn)
                    if trials_df.empty:
                        outfile.write("  'trials' 表为空。\n")
                    else:
                        outfile.write(trials_df.to_string(index=True) + "\n") # 写入所有行
                        outfile.write(f"\n'trials' 表共有 {len(trials_df)} 行。\n")
                except Exception as e:
                    outfile.write(f"  读取 'trials' 表时出错: {e}\n")
            else:
                outfile.write("\n未找到 'trials' 表。\n")

            # 4. 查看 'trial_params' 表的内容 (如果存在) - 输出全部内容
            if ('trial_params',) in tables:
                outfile.write("\n--- 'trial_params' 表内容 ---\n")
                try:
                    trial_params_df = pd.read_sql_query("SELECT * FROM trial_params", conn)
                    if trial_params_df.empty:
                        outfile.write("  'trial_params' 表为空。\n")
                    else:
                        outfile.write(trial_params_df.to_string(index=True) + "\n") # 写入所有行
                        outfile.write(f"\n'trial_params' 表共有 {len(trial_params_df)} 行。\n")
                except Exception as e:
                    outfile.write(f"  读取 'trial_params' 表时出错: {e}\n")
            else:
                outfile.write("\n未找到 'trial_params' 表。\n")
            
            # 5. 查看 'trial_values' 表的内容 (如果存在) - 输出全部内容
            if ('trial_values',) in tables:
                outfile.write("\n--- 'trial_values' 表内容 ---\n")
                try:
                    trial_values_df = pd.read_sql_query("SELECT * FROM trial_values", conn)
                    if trial_values_df.empty:
                        outfile.write("  'trial_values' 表为空。\n")
                    else:
                        outfile.write(trial_values_df.to_string(index=True) + "\n") # 写入所有行
                        outfile.write(f"\n'trial_values' 表共有 {len(trial_values_df)} 行。\n")
                except Exception as e:
                    outfile.write(f"  读取 'trial_values' 表时出错: {e}\n")
            else:
                outfile.write("\n未找到 'trial_values' 表。\n")

            # 你可以按照相同的模式添加其他表的完整输出
            # 例如 'trial_intermediate_values', 'study_user_attributes' 等
            # if ('trial_intermediate_values',) in tables:
            #     outfile.write("\n--- 'trial_intermediate_values' 表内容 ---\n")
            #     try:
            #         df = pd.read_sql_query("SELECT * FROM trial_intermediate_values", conn)
            #         if df.empty:
            #             outfile.write("  'trial_intermediate_values' 表为空。\n")
            #         else:
            #             outfile.write(df.to_string(index=True) + "\n")
            #             outfile.write(f"\n'trial_intermediate_values' 表共有 {len(df)} 行。\n")
            #     except Exception as e:
            #         outfile.write(f"  读取 'trial_intermediate_values' 表时出错: {e}\n")
            # else:
            #     outfile.write("\n未找到 'trial_intermediate_values' 表。\n")


            outfile.write("\n--- 脚本执行完毕 ---\n")
            print(f"--- 内容已写入到: {output_txt_file_path} ---")

    except sqlite3.Error as e:
        print(f"SQLite 错误: {e}")
        if 'outfile' in locals() and not outfile.closed:
             outfile.write(f"SQLite 错误: {e}\n")
    except Exception as e:
        print(f"发生错误: {e}")
        if 'outfile' in locals() and not outfile.closed:
            outfile.write(f"发生错误: {e}\n")
    finally:
        # 关闭数据库连接
        if conn:
            conn.close()
            print("--- 数据库连接已关闭 ---")