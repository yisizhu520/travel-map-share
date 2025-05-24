#!/usr/bin/env python3
"""
生成lisan.csv测试数据集

数据规则：
1. x 取值为 x1, x2, x3
2. a 取值为 a1, a2, a3
3. b 取值为 b1, b2, b3
4. c 取值为 [1,9] 范围内的整数
5. 结果规则：
   - 当 x = x1 时，result = a 列的值
   - 当 x = x2 时，result = b 列的值
   - 当 x = x3 时，result = c 列的值
"""

import pandas as pd
import numpy as np
import itertools
import os

def generate_lisan_data():
    print("🔧 === 生成 lisan.csv 测试数据集 === 🔧")
    
    # 定义取值范围
    x_values = ['x1', 'x2', 'x3']
    a_values = ['a1', 'a2', 'a3']
    b_values = ['b1', 'b2', 'b3']
    c_values = list(range(1, 10))  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    print(f"📊 数据取值范围:")
    print(f"   x: {x_values}")
    print(f"   a: {a_values}")
    print(f"   b: {b_values}")
    print(f"   c: {c_values}")
    
    # 生成所有可能的组合
    all_combinations = list(itertools.product(x_values, a_values, b_values, c_values))
    print(f"   总组合数: {len(all_combinations)}")
    
    # 为了让数据更丰富，每种组合生成多个样本
    samples_per_combination = 3  # 每种组合生成3个样本
    
    data = []
    
    for x, a, b, c in all_combinations:
        for _ in range(samples_per_combination):
            # 根据规则计算result
            if x == 'x1':
                result = a
            elif x == 'x2':
                result = b
            elif x == 'x3':
                result = c
            else:
                result = None  # 不应该发生
            
            data.append({
                'x': x,
                'a': a,
                'b': b,
                'c': c,
                'result': result
            })
    
    # 随机打乱数据顺序
    np.random.seed(42)  # 设置随机种子以便结果可重现
    np.random.shuffle(data)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n📈 生成的数据统计:")
    print(f"   总样本数: {len(df)}")
    print(f"   每种x值的样本数:")
    for x_val in x_values:
        count = len(df[df['x'] == x_val])
        print(f"     {x_val}: {count}")
    
    print(f"\n📋 各列取值分布:")
    for col in ['x', 'a', 'b']:
        if col in df.columns:
            print(f"   {col}: {sorted(df[col].unique())}")
    
    # 单独处理result列，因为它包含字符串和数字
    result_values = df['result'].unique()
    str_values = [v for v in result_values if isinstance(v, str)]
    num_values = [v for v in result_values if isinstance(v, (int, float))]
    all_result_values = sorted(str_values) + sorted(num_values)
    print(f"   result: {all_result_values}")
    
    print(f"   c: min={df['c'].min()}, max={df['c'].max()}")
    
    # 保存到CSV文件
    output_file = '../data/lisan.csv'
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ 数据已保存到: {output_file}")
    
    # 显示前几行数据作为示例
    print(f"\n📖 数据示例 (前10行):")
    print(df.head(10).to_string(index=False))
    
    # 验证规则正确性
    print(f"\n🔍 验证规则正确性:")
    error_count = 0
    
    for index, row in df.head(20).iterrows():  # 验证前20行
        x, a, b, c, result = row['x'], row['a'], row['b'], row['c'], row['result']
        
        expected_result = None
        if x == 'x1':
            expected_result = a
        elif x == 'x2':
            expected_result = b
        elif x == 'x3':
            expected_result = c
        
        if result == expected_result:
            status = "✅"
        else:
            status = "❌"
            error_count += 1
        
        print(f"   行{index+1}: x={x}, a={a}, b={b}, c={c} → result={result} (期望={expected_result}) {status}")
    
    if error_count == 0:
        print(f"\n🎉 规则验证通过！所有验证样本都符合预期规则")
    else:
        print(f"\n⚠️ 发现 {error_count} 个错误样本")
    
    return df

if __name__ == "__main__":
    generate_lisan_data() 