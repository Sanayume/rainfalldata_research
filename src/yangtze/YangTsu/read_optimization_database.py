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
            tables_in_db = cursor.fetchall() # tables_in_db is a list of tuples, e.g., [('studies',), ('trials',)]
            outfile.write("\n数据库中的表:\n")
            if not tables_in_db:
                outfile.write("  未找到任何表。\n")
            else:
                for table_tuple in tables_in_db:
                    outfile.write(f"  - {table_tuple[0]}\n")

            # Define the list of tables to process based on user request
            TABLES_TO_PROCESS = [
                "studies", "version_info", "study_directions", "study_user_attributes",
                "study_system_attributes", "trials", "trial_user_attributes",
                "trial_system_attributes", "trial_params", "trial_values",
                "trial_intermediate_values", "trial_heartbeats", "alembic_version"
            ]

            # 2. Process each table in the list
            for table_name in TABLES_TO_PROCESS:
                # Check if the table exists in the database
                if (table_name,) in tables_in_db:
                    outfile.write(f"\n--- '{table_name}' 表内容 ---\n")
                    try:
                        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        if df.empty:
                            outfile.write(f"  '{table_name}' 表为空。\n")
                        else:
                            outfile.write(df.to_string(index=True) + "\n")
                            outfile.write(f"\n'{table_name}' 表共有 {len(df)} 行。\n")
                    except Exception as e:
                        outfile.write(f"  读取 '{table_name}' 表时出错: {e}\n")
                else:
                    outfile.write(f"\n未找到 '{table_name}' 表。\n")
            
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
