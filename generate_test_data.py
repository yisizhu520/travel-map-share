import pandas as pd
import random
import string

def generate_random_letter():
    """生成随机字母"""
    return random.choice(string.ascii_lowercase)

def generate_test_data(num_records=10000):
    """生成测试数据"""
    data = []
    for _ in range(num_records):
        # 随机生成eflag (1, 2, 3)
        eflag = random.randint(1, 3)
        
        # 生成随机字母作为cflag, dflag, bflag
        cflag = generate_random_letter()
        dflag = generate_random_letter()
        bflag = generate_random_letter()
        
        # 根据eflag设置flag值
        if eflag == 1:
            flag = cflag
        elif eflag == 2:
            flag = dflag
        else:  # eflag == 3
            flag = bflag
        
        # 添加记录
        data.append({
            'eflag': eflag,
            'cflag': cflag,
            'dflag': dflag,
            'bflag': bflag,
            '操作名称': 'widgetName',
            'flag': flag
        })
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 保存为CSV文件
    df.to_csv('test_data.csv', index=False)
    print(f'已生成{num_records}条测试数据，并保存为test_data.csv')

if __name__ == '__main__':
    generate_test_data()