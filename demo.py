import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.preprocessing import LabelEncoder
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
        encoders: 各列的标签编码器字典
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
    
    # 对分类特征进行编码
    encoders = {}
    X_encoded = X.copy()
    
    for column in X.columns:
        if X[column].dtype == 'object' or X[column].nunique() < 10:
            encoder = LabelEncoder()
            # 处理缺失值
            mask = X[column].notna()
            if mask.any():
                # 编码非缺失值
                values = X.loc[mask, column].values
                encoded_values = encoder.fit_transform(values)
                X_encoded.loc[mask, column] = encoded_values
            # 将缺失值设为-1（使用更安全的方式避免链式赋值警告）
            X_encoded[column] = X_encoded[column].fillna(-1)
            encoders[column] = encoder
    
    # 对目标变量进行编码
    y_encoder = LabelEncoder()
    y_encoded = y_encoder.fit_transform(y)
    encoders['target'] = y_encoder
    
    # 创建并训练决策树模型
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_encoded, y_encoded)
    
    # 如果提供了输出目录，则保存决策树可视化
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存树的文本表示
        tree_text = export_text_with_closest_value(model, feature_names=feature_names, encoders=encoders, X=X)
        with open(os.path.join(output_dir, 'tree_closest.txt'), 'w', encoding='utf-8') as f:
            f.write(tree_text)
        
        # 保存DOT文件，可以用Graphviz可视化
        dot_file = os.path.join(output_dir, 'tree.dot')
        export_graphviz(model, out_file=dot_file, 
                        feature_names=feature_names,
                        class_names=[str(c) for c in y_encoder.classes_],
                        filled=True)
        
        print(f"决策树已保存到 {output_dir} 目录")
    
    return model, feature_names, encoders

def get_closest_value(feature, threshold, encoders, X, is_less_equal=True):
    """
    获取特定阈值最接近的原始离散值
    
    参数:
        feature: 特征名称
        threshold: 数值阈值
        encoders: 编码器字典
        X: 原始特征数据
        is_less_equal: 是否为小于等于条件
        
    返回:
        编码值最接近阈值的原始值
    """
    if feature not in encoders:
        return ""
    
    encoder = encoders[feature]
    
    # 获取特征的唯一值
    unique_values = X[feature].dropna().unique()
    
    closest_value = None
    min_distance = float('inf')
    
    for value in unique_values:
        try:
            # 编码该值
            encoded_value = encoder.transform([value])[0]
            
            # 检查是否满足条件
            if (is_less_equal and encoded_value <= threshold) or (not is_less_equal and encoded_value > threshold):
                # 计算与阈值的距离
                distance = abs(encoded_value - threshold)
                
                # 更新最接近的值
                if distance < min_distance:
                    min_distance = distance
                    closest_value = value
        except:
            continue
    
    return closest_value if closest_value is not None else ""

def export_text_with_closest_value(model, feature_names, encoders, X):
    """
    导出决策树的文本表示，添加最接近的原始离散值信息
    
    参数:
        model: 训练好的决策树模型
        feature_names: 特征名称列表
        encoders: 标签编码器字典
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
                    closest_value = get_closest_value(feature, threshold, encoders, X, is_less_equal=True)
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
                    closest_value = get_closest_value(feature, threshold, encoders, X, is_less_equal=False)
                    if closest_value != "":
                        line = f"{line} [最接近值: {closest_value}]"
                except ValueError:
                    pass
        
        enhanced_lines.append(line)
    
    return '\n'.join(enhanced_lines)

def traverse_decision_tree(model, data, feature_names, encoders, match_condition=None):
    """
    遍历决策树，匹配数据集中符合条件的数据
    
    参数:
        model: 训练好的决策树模型
        data: 原始数据集 (DataFrame)
        feature_names: 特征名称列表
        encoders: 特征编码器字典，用于将原始数据转换为模型训练时的格式
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
    
    # 对输入数据进行编码，与决策树模型训练时保持一致
    data_encoded = data.copy()
    for column in feature_names: # Iterate over feature_names used in the model
        if column in encoders and column in data_encoded.columns:
            encoder = encoders[column]
            mask = data_encoded[column].notna()
            if mask.any():
                values_to_encode = data_encoded.loc[mask, column].values
                try:
                    # Use transform, as encoder is already fit during build_decision_tree
                    encoded_values = encoder.transform(values_to_encode)
                    data_encoded.loc[mask, column] = encoded_values
                except ValueError: # Handle cases where new, unseen values are present
                    # Option 1: Fill with -1 (or other placeholder for unknowns)
                    # data_encoded.loc[mask & ~data_encoded[column].isin(encoder.classes_), column] = -1 
                    # Option 2: Or simply pass, keeping them as original, which might cause issues later if not numeric
                    # For now, let's convert to string and then try to convert to numeric, assigning -1 if fails
                    # This part needs careful consideration based on how unseen values should be handled.
                    # A robust way is to ensure all data passed to traverse is pre-processable by encoders.
                    # For simplicity in this fix, we'll assume values causing errors will be set to -1 after trying to encode.
                    current_col_encoded = []
                    for val in values_to_encode:
                        try:
                            current_col_encoded.append(encoder.transform([val])[0])
                        except ValueError:
                            current_col_encoded.append(-1) # Unknown value marker
                    data_encoded.loc[mask, column] = current_col_encoded

            # Ensure all values are numeric; fill NaNs that might have been introduced or were original
            # This needs to be done carefully if -1 is a valid encoded value from the encoder itself.
            # If encoders map to positive integers, -1 is a safe bet for missing/unknown.
            data_encoded[column] = pd.to_numeric(data_encoded[column], errors='coerce').fillna(-1).astype(int)

    def traverse_node(node_id, current_encoded_data, current_original_data, path):
        """
        递归遍历决策树节点
        
        参数:
            node_id: 当前节点ID
            current_encoded_data: 当前节点的输入数据（经过编码和之前条件过滤）
            current_original_data: 当前节点的原始输入数据（未编码但经过之前条件过滤）
            path: 当前路径（条件序列）
        """
        if current_encoded_data.empty:
            return
        
        tree = model.tree_
        
        if tree.children_left[node_id] == -1 and tree.children_right[node_id] == -1:
            class_idx = np.argmax(tree.value[node_id])
            # Handle cases where target encoder might not be present or classes_ is not transformed via standard LabelEncoder
            predicted_class_encoded = model.classes_[class_idx]
            prediction = predicted_class_encoded # Default to encoded if target encoder is missing
            if 'target' in encoders:
                try:
                    prediction = encoders['target'].inverse_transform([predicted_class_encoded])[0]
                except ValueError: # If inverse_transform fails for some reason
                    pass # Keep encoded prediction
            leaf_path = path + [f"预测: {prediction}"]
            match, rule = match_func(current_original_data) # Use original data for match_condition
            if match:
                results.append((leaf_path, rule, match))
            return
        
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = feature_names[feature_idx]

        if feature_name not in current_encoded_data.columns:
            return 

        # Left branch: feature <= threshold
        closest_value_left = get_closest_value(feature_name, threshold, encoders, data, is_less_equal=True)
        left_condition_text = f"{feature_name} <= {threshold:.1f}"
        if closest_value_left != "":
            left_condition_text += f" [最接近值: {closest_value_left}]"
        
        left_mask = current_encoded_data[feature_name] <= threshold
        left_encoded_data = current_encoded_data[left_mask].copy()
        left_original_data = current_original_data[left_mask].copy()
        
        match_left, rule_left = match_func(left_original_data)
        current_path_left = path + [left_condition_text]
        if match_left:
            results.append((current_path_left, rule_left, match_left))
        else:
            traverse_node(tree.children_left[node_id], left_encoded_data, left_original_data, current_path_left)
        
        # Right branch: feature > threshold
        closest_value_right = get_closest_value(feature_name, threshold, encoders, data, is_less_equal=False)
        right_condition_text = f"{feature_name} > {threshold:.1f}"
        if closest_value_right != "":
            right_condition_text += f" [最接近值: {closest_value_right}]"
            
        right_mask = current_encoded_data[feature_name] > threshold
        right_encoded_data = current_encoded_data[right_mask].copy()
        right_original_data = current_original_data[right_mask].copy()
        
        match_right, rule_right = match_func(right_original_data)
        current_path_right = path + [right_condition_text]
        if match_right:
            results.append((current_path_right, rule_right, match_right))
        else:
            traverse_node(tree.children_right[node_id], right_encoded_data, right_original_data, current_path_right)
    
    # Start traversal from the root node (0)
    traverse_node(0, data_encoded, data.copy(), []) # Pass encoded data for tree logic, original for match_func
    
    return results

# 使用示例
if __name__ == "__main__":
    # 构建决策树
    model, feature_names, encoders = build_decision_tree('data.csv', target_column='flag', output_dir='tree_output')
    
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
    results = traverse_decision_tree(model, df, feature_names, encoders, example_match_condition)
    
    # 打印结果
    print("\n遍历决策树匹配结果:")
    for path, rule, match in results:
        print(f"路径: {' -> '.join(path)}")
        print(f"规则: {rule}")
        print(f"匹配结果: {match}")
        print("---")