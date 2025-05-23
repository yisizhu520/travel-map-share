import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
import numpy as np

def discover_conditional_polynomial_rules(csv_file_path, decision_tree_max_depth=3, decision_tree_min_samples_leaf=50):
    """
    Discovers conditional polynomial rules from a CSV file.

    The process involves:
    1. Loading the data.
    2. Training a Decision Tree Regressor on a feature (e.g., 'x') to find split points.
    3. For each segment/leaf defined by the tree, fitting a Linear Regression model
       on other features (e.g., 'a', 'b', 'c') to find the polynomial rule.
    4. Printing the discovered conditional rules.

    Args:
        csv_file_path (str): The path to the CSV file.
        decision_tree_max_depth (int): Maximum depth of the decision tree.
        decision_tree_min_samples_leaf (int): Minimum number of samples required to be at a leaf node.
    """
    try:
        # 1. 数据加载
        data = pd.read_csv(csv_file_path)
        print(f"成功加载数据: {csv_file_path}")
        print("数据前5行:\n", data.head())

        # 定义用于多项式的特征列以及目标列
        polynomial_feature_cols = ['a', 'b', 'c'] 
        target_col = 'result'

        # 检查多项式特征列和目标列是否存在
        if not all(col in data.columns for col in polynomial_feature_cols):
            print(f"错误: 一个或多个多项式特征列 {polynomial_feature_cols} 在CSV文件中未找到。可用的列: {data.columns.tolist()}")
            return
        if target_col not in data.columns:
            print(f"错误: 目标列 '{target_col}' 在CSV文件中未找到。可用的列: {data.columns.tolist()}")
            return

        y_target = data[target_col]
        
        # 确定潜在的切分特征列 (排除目标列和已用于多项式的列)
        potential_split_cols = [col for col in data.columns if col not in polynomial_feature_cols and col != target_col and pd.api.types.is_numeric_dtype(data[col])]

        if not potential_split_cols:
            print("错误: 没有找到可用于切分的数值型特征列。请检查数据。")
            return
        
        print(f"潜在的切分特征列: {potential_split_cols}")
        all_rules_by_split_col = {}

        for current_split_col in potential_split_cols:
            print(f"\n{'='*20} 尝试使用 '{current_split_col}' 作为切分特征 {'='*20}")
            
            X_split = data[[current_split_col]]
            if X_split[current_split_col].nunique() <= 1 and len(data) > 1:
                print(f"  警告: 切分特征列 '{current_split_col}' 只有一个唯一值或所有值都相同。跳过此列。")
                all_rules_by_split_col[current_split_col] = [{"condition_feature": current_split_col, "condition": "特征列只有一个唯一值", "rule": "无法切分"}]
                continue

            # 2. 使用决策树寻找切分点
            tree_model = DecisionTreeRegressor(max_depth=decision_tree_max_depth, min_samples_leaf=decision_tree_min_samples_leaf, random_state=42)
            try:
                tree_model.fit(X_split, y_target)
            except ValueError as ve:
                print(f"  错误: 训练决策树时发生错误 (特征 '{current_split_col}'): {ve}")
                all_rules_by_split_col[current_split_col] = [{"condition_feature": current_split_col, "condition": "决策树训练错误", "rule": str(ve)}]
                continue
            
            print(f"\n  决策树结构 (基于特征 '{current_split_col}' 来划分数据子集):")
            try:
                tree_rules_text = export_text(tree_model, feature_names=[current_split_col])
                print(tree_rules_text)
            except Exception as e_export:
                print(f"  无法导出决策树文本规则: {e_export}")

            leaf_ids = tree_model.apply(X_split)
            unique_leaf_ids = np.unique(leaf_ids)

            if len(unique_leaf_ids) <= 1 and tree_model.tree_.node_count > 1:
                 print(f"  警告: 对于切分列 '{current_split_col}'，决策树只产生了一个有效的叶子节点（或没有有效的分裂）。")
                 print(f"  可能意味着 '{current_split_col}' 不是一个好的切分特征，或者决策树参数需要调整。")

            print(f"\n  基于 '{current_split_col}' 发现 {len(unique_leaf_ids)} 个数据子集 (决策树叶子节点)。")
            
            rules_for_this_split_col = []

            # 3. 对每个子集进行多项式拟合
            for leaf_id in unique_leaf_ids:
                subset_indices = data.index[leaf_ids == leaf_id]
                subset_data = data.loc[subset_indices]

                if subset_data.empty:
                    continue

                # 获取该叶子节点的规则 (current_split_col 的范围)
                # 我们使用叶子节点中 current_split_col 的实际范围作为简化条件
                # 更精确的条件可以从 export_text(tree_rules_text) 解析得到，但较为复杂
                min_val_in_leaf = subset_data[current_split_col].min()
                max_val_in_leaf = subset_data[current_split_col].max()
                condition_str = f"当 {min_val_in_leaf} <= {current_split_col} <= {max_val_in_leaf}"
                
                if min_val_in_leaf == max_val_in_leaf:
                    condition_str = f"当 {current_split_col} = {min_val_in_leaf}"
                # 更精确的条件可以从树结构中提取，但较为复杂，这里用范围近似

                print(f"\n    处理子集 ({current_split_col}): {condition_str} (包含 {len(subset_data)} 条数据)")

                X_poly_subset = subset_data[polynomial_feature_cols]
                y_poly_subset_leaf = subset_data[target_col]

                if len(X_poly_subset) < len(polynomial_feature_cols) + 1 or len(X_poly_subset) < 2: # 不足以拟合或数据点太少
                    print("      子集数据过少，无法进行有效的线性回归。")
                    rules_for_this_split_col.append({"condition_feature": current_split_col, "condition": condition_str, "rule": "数据过少，无法拟合"})
                    continue
                
                lin_reg_model = LinearRegression()
                try:
                    lin_reg_model.fit(X_poly_subset, y_poly_subset_leaf)
                except Exception as e_linreg:
                    print(f"      在线性回归拟合过程中发生错误: {e_linreg}")
                    rules_for_this_split_col.append({"condition_feature": current_split_col, "condition": condition_str, "rule": f"线性回归错误: {e_linreg}"})
                    continue

                coefficients = lin_reg_model.coef_
                intercept = lin_reg_model.intercept_

                rule_parts = []
                for i, col_name in enumerate(polynomial_feature_cols):
                    coeff_val = coefficients[i]
                    # 尝试四舍五入到最近的整数或简单小数，如果差异很小
                    coeff_rounded_int = round(coeff_val)
                    coeff_rounded_dec1 = round(coeff_val, 1)
                    coeff_rounded_dec2 = round(coeff_val, 2)

                    if abs(coeff_val - coeff_rounded_int) < 0.001: # 接近整数
                        rule_parts.append(f"{coeff_rounded_int} * {col_name}")
                    elif abs(coeff_val - coeff_rounded_dec1) < 0.001: # 接近一位小数
                        rule_parts.append(f"{coeff_rounded_dec1:.1f} * {col_name}")
                    elif abs(coeff_val - coeff_rounded_dec2) < 0.001: # 接近两位小数
                        rule_parts.append(f"{coeff_rounded_dec2:.2f} * {col_name}")
                    else: # 保留更多小数位
                        rule_parts.append(f"{coeff_val:.4f} * {col_name}")
                
                intercept_val = intercept
                intercept_rounded_int = round(intercept_val)
                intercept_rounded_dec1 = round(intercept_val, 1)
                intercept_rounded_dec2 = round(intercept_val, 2)

                if abs(intercept_val - intercept_rounded_int) < 0.001:
                    final_intercept_str = f"{intercept_rounded_int}"
                elif abs(intercept_val - intercept_rounded_dec1) < 0.001:
                    final_intercept_str = f"{intercept_rounded_dec1:.1f}"
                elif abs(intercept_val - intercept_rounded_dec2) < 0.001:
                    final_intercept_str = f"{intercept_rounded_dec2:.2f}"
                else:
                    final_intercept_str = f"{intercept_val:.4f}"

                polynomial_rule = f"{target_col} = {' + '.join(rule_parts)} + {final_intercept_str}"
                print(f"      发现的多项式规则: {polynomial_rule}")
                rules_for_this_split_col.append({"condition_feature": current_split_col, "condition": condition_str, "rule": polynomial_rule})

            all_rules_by_split_col[current_split_col] = rules_for_this_split_col

        # 4. 规则呈现
        print(f"\n\n{'='*30} 总结所有发现的条件规则和多项式 {'='*30}")
        if not all_rules_by_split_col:
            print("未能基于任何尝试的切分特征发现规则。请检查数据、决策树参数或特征选择。")
        else:
            # 先以文本形式输出详细规则
            for split_col, rules in all_rules_by_split_col.items():
                print(f"\n--- 基于切分特征 '{split_col}' 的规则 ---")
                if not rules or all(r['rule'] == "数据过少，无法拟合" or "错误" in r['rule'] for r in rules):
                    print("  未能为此切分特征发现有效的多项式规则或所有子集数据均不足/出错。")
                else:
                    for r in rules:
                        print(f"  条件 ({r['condition_feature']}): {r['condition']}")
                        print(f"    规则: {r['rule']}")
                        print("  ---")
            
            # 然后以表格形式总结所有有效规则
            print(f"\n\n{'='*30} 规则总结表格 {'='*30}")
            valid_rules_exist = False
            
            # 检查是否有有效规则
            for split_col, rules in all_rules_by_split_col.items():
                valid_rules = [r for r in rules if r['rule'] != "数据过少，无法拟合" and "错误" not in r['rule'] and "无法切分" not in r['rule']]
                if valid_rules:
                    valid_rules_exist = True
                    break
            
            if not valid_rules_exist:
                print("没有发现有效的规则，无法生成表格总结。")
            else:
                # 打印表头
                print("| 切分特征 | 条件 | 多项式规则 |")
                print("|---------|------|----------|")
                
                # 打印表内容
                for split_col, rules in all_rules_by_split_col.items():
                    valid_rules = [r for r in rules if r['rule'] != "数据过少，无法拟合" and "错误" not in r['rule'] and "无法切分" not in r['rule']]
                    if not valid_rules:
                        continue
                        
                    for r in valid_rules:
                        condition = r['condition']
                        rule = r['rule']
                        print(f"| {split_col} | {condition} | {rule} |")
                
                print("\n注: 表格中只显示有效的规则，排除了数据不足或出错的情况。")

    except FileNotFoundError:
        print(f"错误: 文件未找到 {csv_file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 确保 if_duoxiangshi.csv 文件与此脚本在同一目录下，或者提供完整路径
    # 例如: csv_file = "path/to/your/if_duoxiangshi.csv"
    csv_file = 'c:\\workspace\\private\\travel-map-share\\duoxiangshi\\if_duoxiangshi.csv'
    
    # 您可以调整决策树的参数来探索不同的切分效果
    # max_depth 控制树的最大深度，较小的值产生较粗略的划分，较大的值产生较细致的划分
    # min_samples_leaf 控制叶子节点最少需要的样本数，较大的值可以防止过拟合，产生更通用的规则
    dt_max_depth = 3    # 例如，尝试 2, 3, 4
    dt_min_samples = 50 # 例如，尝试数据量的1%, 5%, 或固定值如 20, 50, 100
                        # 这个值需要根据您的数据集大小和特性来调整

    print(f"开始规则发现，CSV文件: {csv_file}")
    print(f"决策树参数: max_depth={dt_max_depth}, min_samples_leaf={dt_min_samples}")
    print("注意: 如果CSV文件较大或特征较多，处理过程可能需要一些时间。")
    print("提示: 分析结果中的 '条件' 部分是基于决策树对切分特征的划分。")
    print("      '规则' 部分是对应条件下，其他特征 ('a', 'b', 'c') 与目标 ('result') 之间的多项式关系。")
    print("      如果某个切分特征下的规则不理想，可以尝试调整决策树参数或检查数据。")
    discover_conditional_polynomial_rules(csv_file, decision_tree_max_depth=dt_max_depth, decision_tree_min_samples_leaf=dt_min_samples)