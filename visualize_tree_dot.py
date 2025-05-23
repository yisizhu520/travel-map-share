import os
import pandas as pd
from demo import build_decision_tree, traverse_decision_tree

def main():
    # 确保输出目录存在
    output_dir = 'tree_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：构建决策树模型
    print("正在构建决策树模型...")
    model, feature_names, encoder = build_decision_tree('data.csv', target_column='flag', output_dir=output_dir)
    print(f"决策树已生成，请查看 {output_dir}/tree_closest.txt 以获取包含最接近离散值的决策树")
    print(f"决策树DOT文件已生成，请查看 {output_dir}/tree.dot")
    
    # 步骤2：读取原始数据
    df = pd.read_csv('data.csv')
    print("\n原始数据集:")
    print(df)
    
    # 步骤3：定义匹配条件函数
    def match_condition(input_data):
        """匹配条件：当数据集中行数小于2时返回匹配"""
        if input_data.shape[0] < 2:
            return True, f"数据集行数 < 2 (当前行数: {input_data.shape[0]})"
        return False, "不匹配"
    
    # 步骤4：遍历决策树获取路径
    print("\n正在获取决策树路径...")
    paths = traverse_decision_tree(model, df, feature_names, encoder, match_condition)
    
    # 步骤5：将路径保存到文本文件
    paths_file = os.path.join(output_dir, 'decision_paths.txt')
    with open(paths_file, 'w', encoding='utf-8') as f:
        f.write("决策树路径可视化:\n\n")
        if not paths:
            f.write("没有找到匹配的路径\n")
        else:
            for i, (path, rule, match) in enumerate(paths, 1):
                f.write(f"路径 {i}:\n")
                f.write(f"  规则: {rule}\n")
                f.write(f"  匹配: {match}\n")
                f.write(f"  路径步骤:\n")
                for j, step in enumerate(path, 1):
                    f.write(f"    {j}. {step}\n")
                f.write("\n")
    
    print(f"决策路径已保存到 {paths_file}")
    
    # 步骤6：生成使用说明
    instructions_file = os.path.join(output_dir, 'visualization_instructions.txt')
    with open(instructions_file, 'w', encoding='utf-8') as f:
        f.write("决策树可视化使用说明\n")
        f.write("=================\n\n")
        f.write("1. 文本格式的决策树:\n")
        f.write(f"   - 文件: {output_dir}/tree_closest.txt\n")
        f.write("   - 描述: 包含最接近离散值的决策树文本表示\n\n")
        
        f.write("2. DOT格式的决策树:\n")
        f.write(f"   - 文件: {output_dir}/tree.dot\n")
        f.write("   - 描述: 可以使用Graphviz工具可视化的DOT文件\n")
        f.write("   - 可视化方法:\n")
        f.write("     a) 在线可视化: 复制DOT文件内容到 https://dreampuf.github.io/GraphvizOnline/ 并点击刷新\n")
        f.write("     b) 本地可视化: 安装Graphviz后，使用命令 'dot -Tpng tree.dot -o tree.png'\n\n")
        
        f.write("3. 决策路径:\n")
        f.write(f"   - 文件: {output_dir}/decision_paths.txt\n")
        f.write("   - 描述: 包含匹配条件的决策路径文本表示\n\n")
    
    print(f"\n可视化使用说明已保存到 {instructions_file}")
    print("\n可视化完成，请查看输出目录中的文件:")
    print(f"1. 决策树文本: {os.path.join(output_dir, 'tree_closest.txt')}")
    print(f"2. 决策树DOT文件: {os.path.join(output_dir, 'tree.dot')}")
    print(f"3. 决策路径文本: {os.path.join(output_dir, 'decision_paths.txt')}")
    print(f"4. 可视化使用说明: {os.path.join(output_dir, 'visualization_instructions.txt')}")
    print("\n提示: 您可以使用在线工具如 https://dreampuf.github.io/GraphvizOnline/ 来可视化DOT文件")

if __name__ == "__main__":
    main()