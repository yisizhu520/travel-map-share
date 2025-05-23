import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import tree

# 导入自定义模块
from demo import build_decision_tree, traverse_decision_tree

def visualize_decision_tree_to_text(model, feature_names, encoder, X, output_file=None):
    """
    将决策树可视化为文本格式，并保存到文件
    
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        encoder: OneHot编码器
        X: 原始特征数据
        output_file: 输出文件路径
    """
    from sklearn.tree import export_text
    
    # 获取树的文本表示
    tree_text = export_text(model, feature_names=feature_names)
    
    # 按行分割文本
    lines = tree_text.split('\n')
    enhanced_lines = []
    
    for line in lines:
        # 查找包含条件的行
        for feature in feature_names:
            if feature in line and '<=' in line:
                # 提取阈值
                parts = line.split('<=')
                threshold_part = parts[1].strip()
                try:
                    threshold = float(threshold_part)
                    # 查找阈值对应的最接近原始值
                    if '_' in feature:
                        feature_parts = feature.split('_')
                        original_feature = feature_parts[0]
                        feature_value = '_'.join(feature_parts[1:])
                        if threshold >= 0.5:
                            line = f"{line} [最接近值: {feature_value}]"
                except ValueError:
                    pass
            elif feature in line and '>' in line:
                # 提取阈值
                parts = line.split('>')
                threshold_part = parts[1].strip()
                try:
                    threshold = float(threshold_part)
                    # 查找阈值对应的最接近原始值
                    if '_' in feature:
                        feature_parts = feature.split('_')
                        original_feature = feature_parts[0]
                        feature_value = '_'.join(feature_parts[1:])
                        if threshold < 0.5:
                            line = f"{line} [最接近值: {feature_value}]"
                except ValueError:
                    pass
        
        enhanced_lines.append(line)
    
    enhanced_tree_text = '\n'.join(enhanced_lines)
    
    # 如果提供了输出文件路径，则保存文本
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_tree_text)
        print(f"决策树文本已保存到 {output_file}")
    
    return enhanced_tree_text

def visualize_decision_tree_to_image(model, feature_names, class_names, output_file=None):
    """
    使用matplotlib可视化决策树
    
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        class_names: 类别名称列表
        output_file: 输出文件路径
    """
    plt.figure(figsize=(20, 10))
    tree.plot_tree(model, 
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True, 
                  rounded=True,
                  fontsize=10)
    
    # 添加标题
    plt.title('决策树可视化', fontsize=16)
    
    # 如果提供了输出文件路径，则保存图形
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"决策树图形已保存到 {output_file}")
    
    plt.close()

def visualize_paths_to_text(paths, output_file=None):
    """
    将决策路径可视化为文本格式，并保存到文件
    
    参数:
        paths: 路径列表，每个元素为(路径, 规则, 匹配结果)的元组
        output_file: 输出文件路径
    """
    if not paths:
        print("没有找到匹配的路径")
        return ""
    
    text_lines = ["决策树路径可视化:\n"]
    
    for i, (path, rule, match) in enumerate(paths, 1):
        text_lines.append(f"路径 {i}:")
        text_lines.append(f"  规则: {rule}")
        text_lines.append(f"  匹配: {match}")
        text_lines.append(f"  路径步骤:")
        for j, step in enumerate(path, 1):
            text_lines.append(f"    {j}. {step}")
        text_lines.append("")
    
    text = '\n'.join(text_lines)
    
    # 如果提供了输出文件路径，则保存文本
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"决策路径文本已保存到 {output_file}")
    
    return text

def main():
    # 确保输出目录存在
    output_dir = 'tree_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：构建决策树模型
    print("正在构建决策树模型...")
    model, feature_names, encoder = build_decision_tree('data.csv', target_column='flag', output_dir=output_dir)
    
    # 步骤2：可视化决策树为文本
    print("\n正在可视化决策树为文本...")
    df = pd.read_csv('data.csv')
    X = df.drop(columns=['flag'])
    visualize_decision_tree_to_text(model, 
                                   feature_names=feature_names, 
                                   encoder=encoder,
                                   X=X,
                                   output_file=os.path.join(output_dir, 'decision_tree_enhanced.txt'))
    
    # 步骤3：可视化决策树为图像
    print("\n正在可视化决策树为图像...")
    visualize_decision_tree_to_image(model, 
                                    feature_names=feature_names, 
                                    class_names=[str(c) for c in encoder.target_encoder.classes_],
                                    output_file=os.path.join(output_dir, 'decision_tree.png'))
    
    # 步骤4：定义匹配条件函数
    def match_condition(input_data):
        """匹配条件：当数据集中行数小于2时返回匹配"""
        if input_data.shape[0] < 2:
            return True, f"数据集行数 < 2 (当前行数: {input_data.shape[0]})"
        return False, "不匹配"
    
    # 步骤5：获取匹配的路径
    print("\n正在获取匹配的路径...")
    paths = traverse_decision_tree(model, df, feature_names, encoder, match_condition)
    
    # 步骤6：可视化路径为文本
    print("\n正在可视化路径为文本...")
    visualize_paths_to_text(paths, os.path.join(output_dir, 'decision_paths.txt'))
    
    # 打印结果
    print("\n可视化完成，请查看输出目录中的文件:")
    print(f"1. 决策树增强文本: {os.path.join(output_dir, 'decision_tree_enhanced.txt')}")
    print(f"2. 决策树图像: {os.path.join(output_dir, 'decision_tree.png')}")
    print(f"3. 决策路径文本: {os.path.join(output_dir, 'decision_paths.txt')}")
    print(f"4. 原始决策树文本: {os.path.join(output_dir, 'tree_closest.txt')}")

if __name__ == "__main__":
    main()