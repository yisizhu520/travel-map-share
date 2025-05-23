import pandas as pd
import os
import numpy as np
from demo import build_decision_tree, traverse_decision_tree

"""
这个脚本演示了如何使用traverse_decision_tree方法来遍历决策树并匹配符合条件的数据。

主要功能：
1. 构建决策树模型
2. 定义不同类型的匹配条件函数
3. 使用不同的匹配条件遍历决策树
4. 展示匹配结果
"""

# 辅助函数：打印遍历结果
def print_results(results, title):
    print(f"\n{title}:")
    if not results:
        print("未找到匹配结果")
        return
    
    for i, (path, rule, match) in enumerate(results, 1):
        print(f"匹配结果 {i}:")
        print(f"路径: {' -> '.join(path)}")
        print(f"规则: {rule}")
        print(f"匹配: {match}")
        print("---")

# 主函数
def main():
    # 创建输出目录
    output_dir = 'tree_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建决策树模型
    print("正在构建决策树模型...")
    model, feature_names, encoders = build_decision_tree('data.csv', target_column='flag', output_dir=output_dir)
    print(f"决策树已生成，请查看 {output_dir}/tree_closest.txt 以获取包含最接近离散值的决策树")
    
    # 读取原始数据用于遍历示例
    df = pd.read_csv('data.csv')
    print(f"\n数据集概览:\n{df.head()}")
    print(f"\n数据集统计:\n{df.describe()}")
    
    # 示例1：基于价格的匹配条件
    def price_condition(input_data):
        if input_data.empty:
            return False, ""
        
        # 如果数据集中包含价格低于5000的记录
        low_price_mask = input_data['price'] < 5000
        if low_price_mask.any():
            low_price_count = low_price_mask.sum()
            low_price_cities = input_data.loc[low_price_mask, 'city'].tolist()
            return True, f"找到{low_price_count}个价格低于5000的旅行目的地: {', '.join(low_price_cities)}"
        
        return False, "不匹配价格条件"
    
    # 示例2：基于地理位置和季节的匹配条件
    def location_season_condition(input_data):
        if input_data.empty:
            return False, ""
        
        # 亚洲国家列表
        asia_countries = ['中国', '日本', '韩国', '泰国']
        
        # 如果数据集中包含亚洲国家的夏季记录
        mask = (input_data['country'].isin(asia_countries)) & (input_data['season'] == '夏季')
        if mask.any():
            matched_cities = input_data.loc[mask, 'city'].tolist()
            return True, f"找到亚洲夏季旅行目的地: {', '.join(matched_cities)}"
        
        return False, "不匹配亚洲夏季条件"
    
    # 示例3：基于数据集大小的匹配条件
    def dataset_size_condition(input_data):
        size = input_data.shape[0]
        if size < 3:
            cities = input_data['city'].tolist()
            return True, f"数据集记录数小于3（当前记录数：{size}，城市：{', '.join(cities)}）"
        
        return False, f"数据集记录数不小于3（当前记录数：{size}）"
    
    # 示例4：基于目标变量分布的匹配条件
    def target_distribution_condition(input_data):
        if input_data.empty:
            return False, ""
        
        # 计算目标变量的分布
        flag_counts = input_data['flag'].value_counts()
        
        # 如果数据集中包含至少两种不同的flag值
        if len(flag_counts) >= 2:
            flag_info = ', '.join([f"{flag}: {count}条记录" for flag, count in flag_counts.items()])
            return True, f"包含多种flag值的数据集: {flag_info}"
        
        return False, "数据集中flag值分布不够多样"
    
    # 使用不同的匹配条件函数遍历决策树
    results1 = traverse_decision_tree(model, df, feature_names, encoders, price_condition)
    print_results(results1, "示例1：匹配价格低于5000的旅行目的地")
    
    results2 = traverse_decision_tree(model, df, feature_names, encoders, location_season_condition)
    print_results(results2, "示例2：匹配亚洲国家的夏季旅行目的地")
    
    results3 = traverse_decision_tree(model, df, feature_names, encoders, dataset_size_condition)
    print_results(results3, "示例3：匹配数据集中记录数小于3的叶节点")
    
    results4 = traverse_decision_tree(model, df, feature_names, encoders, target_distribution_condition)
    print_results(results4, "示例4：匹配包含多种flag值的数据集")
    
    # 示例5：不提供匹配条件函数，使用默认的匹配条件（始终返回不匹配）
    results5 = traverse_decision_tree(model, df, feature_names, encoders)
    print("\n示例5：使用默认匹配条件（不提供匹配条件函数）:")
    print("结果列表长度:", len(results5))
    print("由于默认匹配条件始终返回False，因此结果列表为空")

# 如果直接运行此脚本
if __name__ == "__main__":
    main()