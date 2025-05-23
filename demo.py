import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import os
import random

def build_decision_tree(csv_path, target_column='flag', output_dir=None):
    """
    从CSV文件读取数据，构建决策树模型
    
    参数:
        csv_path: CSV文件路径
        target_column: 目标分类列名
        output_dir: 输出目录，如果提供则保存决策树可视化
        
    返回:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        encoder: OneHot编码器
    """
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 将目标列移至特征列之后
    if target_column in df.columns and df.columns[-1] != target_column:
        cols = [col for col in df.columns if col != target_column] + [target_column]
        df = df[cols]
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 获取特征名称
    feature_names = X.columns.tolist()
    
    # 创建OneHotEncoder对所有特征列进行编码
    categorical_features = feature_names
    encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # 对特征进行编码
    X_encoded = encoder.fit_transform(X)
    
    # 获取OneHot编码后的特征名称
    onehot_feature_names = []
    for i, feature in enumerate(categorical_features):
        categories = encoder.transformers_[0][1].categories_[i]
        for category in categories:
            onehot_feature_names.append(f"{feature}_{category}")
    
    # 对目标变量进行编码
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    
    # 创建并训练决策树模型
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_encoded, y_encoded)
    
    # 将目标编码器添加到encoder对象中以便后续使用
    encoder.target_encoder = y_encoder
    
    # 如果提供了输出目录，则保存决策树可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存树的文本表示
        tree_text = export_text_with_closest_value(model, feature_names=onehot_feature_names, encoder=encoder, X=X)
        with open(os.path.join(output_dir, 'tree_closest.txt'), 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        # 保存DOT文件，可以用Graphviz可视化
        dot_file = os.path.join(output_dir, 'tree.dot')
        export_graphviz(model, out_file=dot_file, 
                        feature_names=onehot_feature_names,
                        class_names=[str(c) for c in y_encoder.classes_],
                        filled=True)
        
        print(f"决策树已保存到 {output_dir} 目录")
    
    return model, onehot_feature_names, encoder

def get_closest_value(feature, threshold, encoder, X, is_less_equal=True):
    """
    获取特定阈值最接近的原始离散值
    
    参数:
        feature: OneHot特征名称 (如 "feature_value")
        threshold: 数值阈值
        encoder: OneHot编码器
        X: 原始特征数据
        is_less_equal: 是否为小于等于条件
        
    返回:
        编码值最接近阈值的原始值
    """
    # 从OneHot特征名称中提取原始特征名和值
    if "_" not in feature:
        return ""
    
    parts = feature.split("_")
    original_feature = parts[0]
    feature_value = "_".join(parts[1:])
    
    # 获取该特征在OneHot编码中的索引
    try:
        categorical_features = X.columns.tolist()
        feature_idx = categorical_features.index(original_feature)
        
        # 获取该特征的所有类别
        categories = encoder.transformers_[0][1].categories_[feature_idx]
        
        # 找到对应的类别索引
        if feature_value in [str(c) for c in categories]:
            category_idx = [str(c) for c in categories].index(feature_value)
            
            # 判断是否满足条件
            if ((is_less_equal and threshold >= 0.5) or 
                (not is_less_equal and threshold < 0.5)):
                return feature_value
    except (ValueError, IndexError):
        pass
    
    return ""

def export_text_with_closest_value(model, feature_names, encoder, X):
    """
    导出决策树的文本表示，添加最接近的原始离散值信息
    
    参数:
        model: 训练好的决策树模型
        feature_names: OneHot编码后的特征名称列表
        encoder: OneHot编码器
        X: 原始特征数据
        
    返回:
        带有最接近原始值信息的决策树文本表示
    """
    # 获取原始的文本表示
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
                    closest_value = get_closest_value(feature, threshold, encoder, X, is_less_equal=True)
                    if closest_value != "":
                        line = f"{line} [最接近值: {closest_value}]"
                except ValueError:
                    pass
            elif feature in line and '>' in line:
                # 提取阈值
                parts = line.split('>')
                threshold_part = parts[1].strip()
                try:
                    threshold = float(threshold_part)
                    # 查找阈值对应的最接近原始值
                    closest_value = get_closest_value(feature, threshold, encoder, X, is_less_equal=False)
                    if closest_value != "":
                        line = f"{line} [最接近值: {closest_value}]"
                except ValueError:
                    pass
        
        enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)

def traverse_decision_tree(model, data, feature_names, encoder, match_condition=None):
    """
    遍历决策树，匹配数据集中符合条件的数据
    
    参数:
        model: 训练好的决策树模型
        data: 原始数据集 (DataFrame)
        feature_names: OneHot编码后的特征名称列表
        encoder: OneHot编码器
        match_condition: 匹配条件函数，接收过滤后的数据集，返回(match, rule)元组
        
    返回:
        匹配结果列表，每个元素为(路径, 规则, 匹配结果)的元组
    """
    # 存储匹配结果
    results = []
    
    # 如果没有提供匹配条件函数，使用默认函数（始终返回不匹配）
    if match_condition is None:
        def default_match_condition(input_data):
            return False, ""
        match_func = default_match_condition
    else:
        match_func = match_condition
    
    # 对输入数据进行OneHot编码
    data_encoded = encoder.transform(data)
    
    def traverse_node(node_id, current_encoded_data, current_original_data, path):
        """
        递归遍历决策树节点
        
        参数:
            node_id: 当前节点ID
            current_encoded_data: 当前节点的输入数据（经过编码和之前条件过滤）
            current_original_data: 当前节点的原始输入数据（未编码但经过之前条件过滤）
            path: 当前路径（条件序列）
        """
        if len(current_original_data) == 0:
            return
        
        tree = model.tree_
        
        if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
            class_idx = np.argmax(tree.value[node_id])
            # 使用target_encoder解码预测结果
            predicted_class_encoded = model.classes_[class_idx]
            try:
                prediction = encoder.target_encoder.inverse_transform([predicted_class_encoded])[0]
            except (ValueError, AttributeError):
                prediction = predicted_class_encoded

            leaf_path = path + [f"预测: {prediction}"]
            match, rule = match_func(current_original_data) # 使用原始数据进行匹配
            if match:
                results.append((leaf_path, rule, match))
            return
        
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        
        if feature_idx < len(feature_names):
            feature_name = feature_names[feature_idx]
            
            # 提取原始特征名
            if "_" in feature_name:
                parts = feature_name.split("_")
                original_feature = parts[0]
                feature_value = "_".join(parts[1:])
            else:
                original_feature = feature_name
                feature_value = ""
            
            # 左分支: feature <= threshold
            closest_value_left = ""
            if feature_value:
                closest_value_left = feature_value if threshold >= 0.5 else ""
            
            left_condition_text = f"{feature_name} <= {threshold:.1f}"
            if closest_value_left:
                left_condition_text += f" [最接近值: {closest_value_left}]"
            
            # 筛选满足左分支条件的数据
            left_mask = np.ones(len(current_original_data), dtype=bool)
            for i, row in enumerate(current_encoded_data):
                if row[feature_idx] > threshold:
                    left_mask[i] = False
            
            left_encoded_data = current_encoded_data[left_mask]
            left_original_data = current_original_data.iloc[left_mask].copy()
            
            match_left, rule_left = match_func(left_original_data)
            current_path_left = path + [left_condition_text]
            if match_left:
                results.append((current_path_left, rule_left, match_left))
            else:
                traverse_node(tree.children_left[node_id], left_encoded_data, left_original_data, current_path_left)
            
            # 右分支: feature > threshold
            closest_value_right = ""
            if feature_value:
                closest_value_right = feature_value if threshold < 0.5 else ""
            
            right_condition_text = f"{feature_name} > {threshold:.1f}"
            if closest_value_right:
                right_condition_text += f" [最接近值: {closest_value_right}]"
            
            # 筛选满足右分支条件的数据
            right_mask = np.ones(len(current_original_data), dtype=bool)
            for i, row in enumerate(current_encoded_data):
                if row[feature_idx] <= threshold:
                    right_mask[i] = False
            
            right_encoded_data = current_encoded_data[right_mask]
            right_original_data = current_original_data.iloc[right_mask].copy()
            
            match_right, rule_right = match_func(right_original_data)
            current_path_right = path + [right_condition_text]
            if match_right:
                results.append((current_path_right, rule_right, match_right))
            else:
                traverse_node(tree.children_right[node_id], right_encoded_data, right_original_data, current_path_right)
    
    # 从根节点开始遍历
    traverse_node(0, data_encoded, data.copy(), [])
    
    return results

# 使用示例
if __name__ == "__main__":
    # 构建决策树
    model, feature_names, encoder = build_decision_tree('test_data.csv', target_column='flag', output_dir='tree_output')
    
    print(f"决策树已生成，请查看 tree_output/tree_closest.txt 以获取包含最接近离散值的决策树")
    
    # 读取原始数据用于遍历示例
    df = pd.read_csv('data.csv')
    
    # 定义一个简单的匹配条件函数
    def example_match_condition(input_data):
        # 示例：当数据集中包含flag='a'的记录数大于2时返回匹配
        # Note: input_data here is the original, unencoded data subset
        # 如果 input_data 行数 < 2 ， 则返回 true
        if input_data.shape[0] < 2:
            return True, "input_data 行数 < 2"
        
        return False, "not match"
    
    # 遍历决策树
    results = traverse_decision_tree(model, df, feature_names, encoder, example_match_condition)
    
    # 打印结果
    print("\n遍历决策树匹配结果:")
    for path, rule, match in results:
        print(f"路径: {' -> '.join(path)}")
        print(f"规则: {rule}")
        print(f"匹配结果: {match}")
        print("---")