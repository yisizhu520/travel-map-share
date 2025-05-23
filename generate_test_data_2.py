import csv
import random

def generate_test_data(output_file_path, num_records=100):
    """生成测试数据并写入CSV文件"""
    fieldnames = ['aflag', 'eflag', 'cflag', 'dflag', 'bflag', '操作名称', 'flag']
    with open(output_file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for _ in range(num_records):
            aflag = random.choice(['a', 'b'])
            eflag = random.choice([1, 2])
            bflag = random.randint(1, 9)
            cflag = random.randint(1, 9)
            dflag = random.randint(1, 9)
            operator_name = 'widgetName'
            flag_value = ''

            if aflag == 'a' and eflag == 1:
                flag_value = cflag + bflag
            elif aflag == 'b' and eflag == 1:
                flag_value = dflag + bflag
            elif aflag == 'a' and eflag == 2:
                flag_value = cflag + dflag
            elif aflag == 'b' and eflag == 2:
                flag_value = dflag + dflag

            writer.writerow({
                'aflag': aflag,
                'eflag': eflag,
                'cflag': cflag,
                'dflag': dflag,
                'bflag': bflag,
                '操作名称': operator_name,
                'flag': flag_value
            })
    print(f"已生成 {num_records} 条测试数据，并保存为 {output_file_path}")

if __name__ == '__main__':
    # 确保CSV文件与脚本在同一目录下，或者提供完整路径
    csv_file_path = 'test_data_2.csv' 
    # 如果脚本和csv不在同一目录，需要修改为绝对路径，例如：
    # csv_file_path = 'c:\\workspace\\private\\travel-map-share\\test_data_2.csv'
    generate_test_data(csv_file_path, num_records=10000) # 生成20条数据作为示例