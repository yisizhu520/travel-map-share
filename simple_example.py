# 简单示例：演示traverse_decision_tree方法的使用
import pandas as pd
from demo import build_decision_tree, traverse_decision_tree
import os

# 确保输出目录存在
output_dir = 'tree_output'
os.makedirs(output_dir, exist_ok=True)

# 步骤1：构建决策树模型
print("正在构建决策树模型...")
model, feature_names, encoders = build_decision_tree('data.csv', target_column='flag', output_dir=output_dir)
print(f"决策树已生成，请查看 {output_dir}/tree_closest.txt 以获取包含最接近离散值的决策树")

# 步骤2：读取原始数据
df = pd.read_csv('data.csv')
print("\n原始数据集:")
print(df)

# 步骤3：定义一个简单的匹配条件函数
def simple_match_condition(input_data):
    """简单的匹配条件：当数据集中包含中国城市时返回匹配"""
    if input_data.empty:
        return False, ""
    
    # 检查是否包含中国城市
    if (input_data['country'] == '中国').any():
        # 获取匹配的城市名称
        matched_cities = input_data.loc[input_data['country'] == '中国', 'city'].tolist()
        return True, f"包含中国城市: {', '.join(matched_cities)}"
    
    return False, "不包含中国城市"

# 步骤4：使用匹配条件函数遍历决策树
print("\n开始遍历决策树...")
results = traverse_decision_tree(model, df, feature_names, encoders, simple_match_condition)

# 步骤5：打印遍历结果
print("\n遍历决策树匹配结果:")
if not results:
    print("未找到匹配结果")
else:
    for i, (path, rule, match) in enumerate(results, 1):
        print(f"\n匹配结果 {i}:")
        print(f"路径: {' -> '.join(path)}")
        print(f"规则: {rule}")
        print(f"匹配结果: {match}")
        print("---")