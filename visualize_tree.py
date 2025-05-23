import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn import tree
import graphviz
from IPython.display import display

# 导入自定义模块
from demo import build_decision_tree, traverse_decision_tree

def visualize_decision_tree(model, feature_names, class_names, output_file=None):
    """
    使用graphviz可视化决策树
    
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        class_names: 类别名称列表
        output_file: 输出文件路径（不包含扩展名）
    """
    dot_data = tree.export_graphviz(model, 
                                    out_file=None, 
                                    feature_names=feature_names,
                                    class_names=class_names,
                                    filled=True, 
                                    rounded=True,
                                    special_characters=True)
    
    # 创建Graphviz图形
    graph = graphviz.Source(dot_data)
    
    # 如果提供了输出文件路径，则保存图形
    if output_file:
        graph.render(output_file, format="png")
        print(f"决策树图形已保存到 {output_file}.png")
    
    return graph

def visualize_decision_paths(model, feature_names, encoder, data, match_condition=None, output_file=None):
    """
    可视化决策树中的路径
    
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        encoder: OneHot编码器
        data: 原始数据集
        match_condition: 匹配条件函数
        output_file: 输出文件路径（不包含扩展名）
    """
    # 获取匹配的路径
    paths = traverse_decision_tree(model, data, feature_names, encoder, match_condition)
    
    # 如果没有匹配的路径，返回None
    if not paths:
        print("没有找到匹配的路径")
        return None
    
    # 创建决策树的DOT表示
    dot_data = tree.export_graphviz(model, 
                                   out_file=None, 
                                   feature_names=feature_names,
                                   class_names=[str(c) for c in encoder.target_encoder.classes_],
                                   filled=True, 
                                   rounded=True,
                                   special_characters=True)
    
    # 为每个匹配的路径分配一个颜色
    colors = list(mcolors.TABLEAU_COLORS.values())
    
    # 创建一个新的DOT图形，用于高亮路径
    modified_dot_data = dot_data
    
    # 创建图例信息
    legend_info = []
    
    # 为每个路径添加高亮
    for i, (path, rule, match) in enumerate(paths):
        color = colors[i % len(colors)]
        color_hex = mcolors.to_hex(color)
        
        # 提取路径中的节点ID
        node_ids = []
        current_node = 0  # 根节点
        
        for step in path[:-1]:  # 不包括最后一个预测步骤
            if "<=" in step:
                feature_name = step.split(" <= ")[0]
                # 在树中找到对应的节点
                if current_node == 0 or current_node in node_ids:
                    # 左子节点
                    node_ids.append(current_node)
                    # 找到下一个节点
                    for j, feat in enumerate(feature_names):
                        if feat == feature_name:
                            current_node = model.tree_.children_left[current_node]
                            break
            elif ">" in step:
                feature_name = step.split(" > ")[0]
                # 在树中找到对应的节点
                if current_node == 0 or current_node in node_ids:
                    # 右子节点
                    node_ids.append(current_node)
                    # 找到下一个节点
                    for j, feat in enumerate(feature_names):
                        if feat == feature_name:
                            current_node = model.tree_.children_right[current_node]
                            break
        
        # 添加最后一个节点
        if current_node not in node_ids:
            node_ids.append(current_node)
        
        # 为路径中的每个节点添加颜色
        for node_id in node_ids:
            # 修改节点的颜色
            node_str = f'\n{node_id} ['
            if node_str in modified_dot_data:
                # 找到节点的开始位置
                start_idx = modified_dot_data.find(node_str)
                # 找到节点的结束位置
                end_idx = modified_dot_data.find(']', start_idx)
                # 提取节点的属性
                node_attrs = modified_dot_data[start_idx + len(node_str):end_idx]
                # 添加边框颜色和宽度
                new_attrs = node_attrs + f', color="{color_hex}", penwidth=3.0'
                # 替换原来的属性
                modified_dot_data = modified_dot_data[:start_idx + len(node_str)] + new_attrs + modified_dot_data[end_idx:]
        
        # 为连接节点的边添加颜色
        for j in range(len(node_ids) - 1):
            from_node = node_ids[j]
            to_node = node_ids[j + 1]
            edge_str = f'\n{from_node} -> {to_node}'
            if edge_str in modified_dot_data:
                # 找到边的开始位置
                start_idx = modified_dot_data.find(edge_str)
                # 找到边的结束位置（如果有属性）
                end_idx = modified_dot_data.find('\n', start_idx + 1)
                if end_idx == -1:
                    # 如果没有找到下一行，使用字符串结束位置
                    end_idx = len(modified_dot_data)
                # 检查是否已经有边的属性
                if '[' in modified_dot_data[start_idx:end_idx]:
                    # 找到属性的开始和结束位置
                    attr_start = modified_dot_data.find('[', start_idx)
                    attr_end = modified_dot_data.find(']', attr_start)
                    # 添加颜色属性
                    new_attrs = modified_dot_data[attr_start + 1:attr_end] + f', color="{color_hex}", penwidth=2.0'
                    # 替换原来的属性
                    modified_dot_data = modified_dot_data[:attr_start + 1] + new_attrs + modified_dot_data[attr_end:]
                else:
                    # 如果没有属性，添加属性
                    new_edge = f'{edge_str} [color="{color_hex}", penwidth=2.0]'
                    modified_dot_data = modified_dot_data[:start_idx] + new_edge + modified_dot_data[end_idx:]
        
        # 添加图例信息
        path_desc = f"路径 {i+1}: {rule}"
        legend_info.append((color_hex, path_desc))
    
    # 创建Graphviz图形
    graph = graphviz.Source(modified_dot_data)
    
    # 如果提供了输出文件路径，则保存图形
    if output_file:
        graph.render(output_file, format="png")
        print(f"决策树路径图形已保存到 {output_file}.png")
        
        # 创建带有图例的图形
        plt.figure(figsize=(10, 2))
        ax = plt.gca()
        ax.axis('off')
        
        # 添加图例
        patches = []
        for color, desc in legend_info:
            patch = mpatches.Patch(color=color, label=desc)
            patches.append(patch)
        
        ax.legend(handles=patches, loc='center', fontsize=12)
        
        # 保存图例
        legend_file = f"{output_file}_legend.png"
        plt.savefig(legend_file, bbox_inches='tight', dpi=300)
        print(f"图例已保存到 {legend_file}")
    
    return graph

def main():
    # 确保输出目录存在
    output_dir = 'tree_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：构建决策树模型
    print("正在构建决策树模型...")
    model, feature_names, encoder = build_decision_tree('data.csv', target_column='flag', output_dir=output_dir)
    
    # 步骤2：可视化决策树
    print("\n正在可视化决策树...")
    visualize_decision_tree(model, 
                           feature_names=feature_names, 
                           class_names=[str(c) for c in encoder.target_encoder.classes_],
                           output_file=os.path.join(output_dir, 'decision_tree'))
    
    # 步骤3：读取原始数据
    df = pd.read_csv('data.csv')
    print("\n原始数据集:")
    print(df)
    
    # 步骤4：定义匹配条件函数
    def match_condition(input_data):
        """匹配条件：当数据集中行数小于2时返回匹配"""
        if input_data.shape[0] < 2:
            return True, f"数据集行数 < 2 (当前行数: {input_data.shape[0]})"
        return False, "不匹配"
    
    # 步骤5：可视化决策路径
    print("\n正在可视化决策路径...")
    visualize_decision_paths(model, 
                            feature_names=feature_names, 
                            encoder=encoder, 
                            data=df, 
                            match_condition=match_condition,
                            output_file=os.path.join(output_dir, 'decision_paths'))
    
    print("\n可视化完成，请查看输出目录中的图像文件。")

if __name__ == "__main__":
    main()