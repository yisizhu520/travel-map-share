import csv

def swap_csv_columns(file_path, col1_name, col2_name):
    """Reads a CSV file, swaps two specified columns, and writes the changes back.

    Args:
        file_path (str): The path to the CSV file.
        col1_name (str): The name of the first column to swap.
        col2_name (str): The name of the second column to swap.
    """
    rows = []
    header = []

    try:
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader) # Read the header
            if col1_name not in header or col2_name not in header:
                print(f"错误：列 '{col1_name}' 或 '{col2_name}' 在文件中未找到。")
                return

            col1_index = header.index(col1_name)
            col2_index = header.index(col2_name)

            for row in reader:
                if len(row) > max(col1_index, col2_index):
                    # Swap the column values
                    row[col1_index], row[col2_index] = row[col2_index], row[col1_index]
                rows.append(row)

        with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header) # Write the header back
            writer.writerows(rows) # Write the modified rows

        print(f"文件 '{file_path}' 中的列 '{col1_name}' 和 '{col2_name}' 已成功交换。")

    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
    except Exception as e:
        print(f"处理文件时发生错误：{e}")

if __name__ == '__main__':
    # 指定CSV文件路径和要交换的列名
    csv_file_to_modify = 'test_data.csv'  # 假设脚本和CSV文件在同一目录下
    # 如果不在同一目录，请使用绝对路径，例如：
    # csv_file_to_modify = r'c:\workspace\private\travel-map-share\test_data_2.csv'
    column1 = 'eflag'
    column2 = 'dflag'

    # 获取脚本所在的目录
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    absolute_csv_path = os.path.join(script_dir, csv_file_to_modify)

    swap_csv_columns(absolute_csv_path, column1, column2)