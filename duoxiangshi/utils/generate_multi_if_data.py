import csv
import random

def generate_if_data(output_csv_file, num_records=10000):
    """根据特定条件生成数据并写入CSV文件"""
    fieldnames = ['x', 'y', 'a', 'b', 'c', 'result']
    
    with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)  # 写入表头
        
        for _ in range(num_records):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = random.randint(1, 100)
            x = random.randint(20, 60)
            
            # y 取值是 y1, y2 中的一个
            y = random.choice(['y1', 'y2'])
            result = 0
            # result 结果根据 x 和 y 的组合条件判断
            if y == 'y1':
                if x < 30:
                    result = a
                elif 30 <= x < 40:
                    result = a + b
                elif x >= 40:
                    result = a + b + c

            elif y == 'y2':
                if x < 30:
                    result = 2*a
                elif 30 <= x < 40:
                    result = 2*a + b
                elif x >= 40:
                    result = 2*a + b + c


                    
            writer.writerow([x, y, a, b, c, result])
            
    print(f"已生成 {num_records} 条数据，并保存到 {output_csv_file}")

if __name__ == '__main__':
    # 定义输出CSV文件的路径
    csv_file_path = 'multi_if_duoxiangshi.csv' 
    # 确保脚本和CSV文件在同一个目录下 'c:\workspace\private\travel-map-share\duoxiangshi\'
    # 如果需要指定绝对路径，可以取消下面的注释并修改
    # csv_file_path = 'c:\\workspace\\private\\travel-map-share\\duoxiangshi\\if_duoxiangshi.csv'
    generate_if_data(csv_file_path, num_records=10000)