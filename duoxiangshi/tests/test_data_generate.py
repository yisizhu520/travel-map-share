import csv
import random

def generate_data(output_csv_file, num_records=10000):
    """生成数据并写入CSV文件"""
    fieldnames = ['a', 'b', 'c', 'result']
    
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)  # 写入表头
        
        for _ in range(num_records):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = random.randint(1, 100)
            result = 2 * a + 3 * b - c
            writer.writerow([a, b, c, result])
            
    print(f"已生成 {num_records} 条数据，并保存到 {output_csv_file}")

if __name__ == '__main__':
    # 定义输出CSV文件的路径
    # 脚本和CSV文件将在同一个目录下 'c:\workspace\private\travel-map-share\duoxiangshi\'
    csv_file_path = 'duoxiangshi.csv' 
    generate_data(csv_file_path, num_records=10000)