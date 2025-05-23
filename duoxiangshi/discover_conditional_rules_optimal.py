import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import warnings
import argparse
from itertools import combinations
import time
warnings.filterwarnings('ignore')

class OptimalConditionalRuleDiscoverer:
    """
    优化版的条件多项式规则发现器
    
    主要改进：
    1. 智能特征组合穷举
    2. 交叉验证评估最优组合
    3. 支持离散型分段特征
    4. 动态特征分配优化
    """
    
    def __init__(self, max_depth=3, min_samples_leaf=50, cv_folds=3, 
                 max_combinations=100, enable_exhaustive_search=True):
        """
        初始化参数
        
        Args:
            max_depth: 决策树最大深度
            min_samples_leaf: 叶子节点最小样本数
            cv_folds: 交叉验证折数
            max_combinations: 最大尝试的特征组合数（防止计算量过大）
            enable_exhaustive_search: 是否启用穷举搜索
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.cv_folds = cv_folds
        self.max_combinations = max_combinations
        self.enable_exhaustive_search = enable_exhaustive_search
        self.discovered_rules = []
        self.best_configuration = None
        self.label_encoders = {}  # 存储分类特征的编码器
        self.categorical_features = []  # 存储分类特征列表
        
    def _identify_feature_types(self, data, target_col):
        """
        识别数值型和分类型特征
        
        Returns:
            numeric_features: 数值型特征列表
            categorical_features: 分类型特征列表
            all_split_candidates: 所有可用作分段的特征
        """
        # 排除目标列
        feature_cols = [col for col in data.columns if col != target_col]
        
        numeric_features = []
        categorical_features = []
        
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_features.append(col)
            else:
                # 检查是否为分类特征（字符串、对象类型，或数值但唯一值较少）
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    categorical_features.append(col)
                elif pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() <= 10:
                    # 数值型但唯一值较少，可能是编码的分类特征
                    print(f"将数值特征 '{col}' 视为分类特征（唯一值数量: {data[col].nunique()}）")
                    categorical_features.append(col)
                else:
                    numeric_features.append(col)
        
        # 所有特征都可以作为分段候选（分类特征用于分段，数值特征既可分段也可做多项式）
        all_split_candidates = numeric_features + categorical_features
        
        print(f"特征类型识别:")
        print(f"  数值特征: {numeric_features}")
        print(f"  分类特征: {categorical_features}")
        print(f"  可分段特征: {all_split_candidates}")
        
        return numeric_features, categorical_features, all_split_candidates
        
    def _encode_categorical_features(self, data, categorical_features):
        """
        对分类特征进行编码
        
        Returns:
            encoded_data: 编码后的数据
        """
        encoded_data = data.copy()
        
        for col in categorical_features:
            if col not in self.label_encoders:
                # 创建并拟合标签编码器
                le = LabelEncoder()
                encoded_data[col] = le.fit_transform(data[col].astype(str))
                self.label_encoders[col] = le
                print(f"对分类特征 '{col}' 进行编码: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            else:
                # 使用已有的编码器
                encoded_data[col] = self.label_encoders[col].transform(data[col].astype(str))
        
        return encoded_data
    
    def _generate_feature_combinations(self, numeric_features, categorical_features, target_col):
        """
        生成所有可能的特征组合
        
        Args:
            numeric_features: 数值特征列表
            categorical_features: 分类特征列表
            target_col: 目标列名
            
        Returns:
            combinations_list: [(split_features, poly_features), ...]
        """
        # 分段特征候选：数值特征 + 分类特征
        split_candidates = numeric_features + categorical_features
        # 多项式特征候选：只能是数值特征
        poly_candidates = numeric_features
        
        if len(split_candidates) < 1 or len(poly_candidates) < 1:
            print("警告: 没有足够的特征进行组合优化")
            return [([], poly_candidates)]
        
        combinations_list = []
        
        # 生成所有可能的分段特征组合
        for split_size in range(1, min(len(split_candidates), 4) + 1):  # 限制分段特征最多4个
            for split_features in combinations(split_candidates, split_size):
                # 多项式特征：数值特征中去除已用作分段的数值特征
                available_poly_features = [f for f in poly_candidates if f not in split_features]
                
                # 确保多项式特征至少有1个
                if len(available_poly_features) >= 1:
                    combinations_list.append((list(split_features), available_poly_features))
        
        # 如果组合数量太多，进行智能筛选
        if len(combinations_list) > self.max_combinations:
            print(f"警告: 特征组合数量 ({len(combinations_list)}) 超过最大限制 ({self.max_combinations})")
            print("将使用启发式方法筛选最有希望的组合...")
            combinations_list = self._select_promising_combinations(combinations_list, split_candidates, poly_candidates)
        
        return combinations_list
    
    def _select_promising_combinations(self, all_combinations, split_candidates, poly_candidates):
        """
        使用启发式方法筛选最有希望的特征组合
        """
        # 策略1: 优先选择分段特征数量适中的组合 (1-2个分段特征)
        moderate_combinations = [combo for combo in all_combinations 
                               if 1 <= len(combo[0]) <= 2]
        
        # 策略2: 优先选择包含分类特征的组合（分类特征通常是好的分段特征）
        categorical_combinations = [combo for combo in moderate_combinations
                                  if any(f in self.categorical_features for f in combo[0])]
        
        # 策略3: 如果分类特征组合不够，补充数值特征组合
        if len(categorical_combinations) < self.max_combinations // 2:
            remaining_combinations = [combo for combo in moderate_combinations 
                                    if combo not in categorical_combinations]
            categorical_combinations.extend(remaining_combinations[:self.max_combinations // 2])
        
        # 策略4: 如果还是太多，随机采样
        if len(categorical_combinations) > self.max_combinations:
            np.random.seed(42)
            indices = np.random.choice(len(categorical_combinations), 
                                     self.max_combinations, replace=False)
            categorical_combinations = [categorical_combinations[i] for i in indices]
        
        # 策略5: 总是包含一些基准组合
        baseline_combinations = []
        if split_candidates and poly_candidates:
            baseline_combinations = [
                ([split_candidates[0]], poly_candidates),  # 第一个特征作为分段
            ]
            if len(split_candidates) > 1:
                baseline_combinations.append(([split_candidates[-1]], [f for f in poly_candidates if f != split_candidates[-1]]))
        
        # 合并并去重
        final_combinations = baseline_combinations + categorical_combinations
        seen = set()
        unique_combinations = []
        for combo in final_combinations:
            combo_key = (tuple(sorted(combo[0])), tuple(sorted(combo[1])))
            if combo_key not in seen:
                seen.add(combo_key)
                unique_combinations.append(combo)
        
        return unique_combinations[:self.max_combinations]
    
    def _evaluate_combination(self, data, split_features, poly_features, target_col):
        """
        评估单个特征组合的效果
        
        Returns:
            score: 该组合的综合评分
            rules: 发现的规则列表
        """
        try:
            X_split = data[split_features]
            y_target = data[target_col]
            
            # 训练决策树
            tree_model = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=42
            )
            
            tree_model.fit(X_split, y_target)
            
            # 提取条件和分段
            conditions_by_leaf = self._extract_tree_conditions(tree_model, split_features, data)
            leaf_ids = tree_model.apply(X_split)
            
            segment_scores = []
            segment_rules = []
            
            # 评估每个分段的多项式拟合效果
            for leaf_id, conditions in conditions_by_leaf.items():
                subset_mask = (leaf_ids == leaf_id)
                subset_data = data[subset_mask]
                
                if len(subset_data) < self.min_samples_leaf // 2:
                    continue
                
                X_poly = subset_data[poly_features]
                y_poly = subset_data[target_col]
                
                # 使用交叉验证评估多项式拟合效果
                if len(X_poly) >= self.cv_folds and len(poly_features) > 0:
                    try:
                        model = LinearRegression()
                        cv_scores = cross_val_score(model, X_poly, y_poly, 
                                                   cv=min(self.cv_folds, len(X_poly)), 
                                                   scoring='r2')
                        avg_score = np.mean(cv_scores)
                        
                        if avg_score > 0.1:  # 最小质量阈值
                            model.fit(X_poly, y_poly)
                            rule_str = self._format_polynomial_rule(model, poly_features, target_col)
                            condition_str = " 且 ".join(conditions)
                            
                            segment_scores.append(avg_score)
                            segment_rules.append({
                                'split_features': split_features,
                                'poly_features': poly_features,
                                'condition': condition_str,
                                'rule': rule_str,
                                'cv_r2_score': avg_score,
                                'sample_count': len(subset_data),
                                'model': model
                            })
                            
                    except Exception as e:
                        continue
            
            # 计算整体评分：平均R²分数 × 规则数量权重
            if segment_scores:
                avg_score = np.mean(segment_scores)
                rule_count_bonus = min(len(segment_scores) / 10, 0.1)  # 规则数量奖励，最大10%
                total_score = avg_score + rule_count_bonus
                return total_score, segment_rules
            else:
                return 0.0, []
                
        except Exception as e:
            return 0.0, []
    
    def _extract_tree_conditions(self, tree_model, feature_names, original_data):
        """
        从决策树中提取精确的条件路径，并将编码后的分类值转换回原始值
        """
        tree = tree_model.tree_
        conditions_by_leaf = {}
        
        def extract_path(node_id, conditions):
            if tree.children_left[node_id] == -1:  # 叶子节点
                conditions_by_leaf[node_id] = conditions.copy()
                return
                
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]
            
            # 判断是否为分类特征
            if feature_name in self.categorical_features:
                # 分类特征：将阈值转换回原始分类值
                le = self.label_encoders[feature_name]
                
                # 获取该特征的所有可能值
                unique_encoded_values = sorted(original_data[feature_name].unique())
                
                # 根据阈值确定分类条件
                left_values = [val for val in unique_encoded_values if val <= threshold]
                right_values = [val for val in unique_encoded_values if val > threshold]
                
                # 转换回原始分类值
                if left_values:
                    left_original = [le.inverse_transform([int(val)])[0] for val in left_values]
                    left_condition = f"{feature_name} ∈ {{{', '.join(map(str, left_original))}}}"
                else:
                    left_condition = f"{feature_name} ∈ {{}}"
                
                if right_values:
                    right_original = [le.inverse_transform([int(val)])[0] for val in right_values]
                    right_condition = f"{feature_name} ∈ {{{', '.join(map(str, right_original))}}}"
                else:
                    right_condition = f"{feature_name} ∈ {{}}"
                
                # 左分支
                left_conditions = conditions + [left_condition]
                extract_path(tree.children_left[node_id], left_conditions)
                
                # 右分支
                right_conditions = conditions + [right_condition]
                extract_path(tree.children_right[node_id], right_conditions)
            else:
                # 数值特征：保持原有逻辑
                left_conditions = conditions + [f"{feature_name} <= {threshold:.2f}"]
                extract_path(tree.children_left[node_id], left_conditions)
                
                right_conditions = conditions + [f"{feature_name} > {threshold:.2f}"]
                extract_path(tree.children_right[node_id], right_conditions)
        
        extract_path(0, [])
        return conditions_by_leaf
    
    def _format_polynomial_rule(self, model, poly_features, target_col):
        """格式化多项式规则"""
        coefficients = model.coef_
        intercept = model.intercept_
        
        rule_parts = []
        for i, col_name in enumerate(poly_features):
            coeff_val = coefficients[i]
            
            if abs(coeff_val) < 0.001:  # 系数接近0，忽略此项
                continue
                
            coeff_rounded = self._smart_round(coeff_val)
            
            if coeff_rounded == 1:
                rule_parts.append(col_name)
            elif coeff_rounded == -1:
                rule_parts.append(f"-{col_name}")
            else:
                rule_parts.append(f"{coeff_rounded} * {col_name}")
        
        intercept_rounded = self._smart_round(intercept)
        
        if not rule_parts:
            return f"{target_col} = {intercept_rounded}"
            
        rule_str = f"{target_col} = {' + '.join(rule_parts)}"
        
        if intercept_rounded != 0:
            if intercept_rounded > 0:
                rule_str += f" + {intercept_rounded}"
            else:
                rule_str += f" - {abs(intercept_rounded)}"
                
        return rule_str
    
    def _smart_round(self, value):
        """智能四舍五入"""
        rounded_int = round(value)
        rounded_dec1 = round(value, 1)
        rounded_dec2 = round(value, 2)
        
        if abs(value - rounded_int) < 0.001:
            return rounded_int
        elif abs(value - rounded_dec1) < 0.001:
            return rounded_dec1
        elif abs(value - rounded_dec2) < 0.001:
            return rounded_dec2
        else:
            return round(value, 4)
    
    def discover_optimal_rules(self, csv_file_path, target_col=None, 
                              manual_split_features=None, manual_poly_features=None):
        """
        发现最优的条件多项式规则
        
        Args:
            csv_file_path: CSV文件路径
            target_col: 目标列名，如果为None则使用最后一列
            manual_split_features: 手动指定的分段特征
            manual_poly_features: 手动指定的多项式特征
            
        Returns:
            best_rules: 最优规则列表
        """
        try:
            print("=== 优化版条件规则发现（支持分类特征）===")
            start_time = time.time()
            
            # 1. 数据加载
            data = pd.read_csv(csv_file_path)
            print(f"成功加载数据: {csv_file_path}")
            print(f"数据形状: {data.shape}")
            
            # 2. 确定目标列
            if target_col is None:
                target_col = data.columns[-1]
                print(f"自动选择最后一列 '{target_col}' 作为目标列")
            
            # 3. 识别特征类型
            numeric_features, categorical_features, all_split_candidates = self._identify_feature_types(data, target_col)
            self.categorical_features = categorical_features
            
            if len(all_split_candidates) < 1 or len(numeric_features) < 1:
                print("错误: 没有足够的特征进行分析")
                return []
            
            # 4. 对分类特征进行编码
            encoded_data = self._encode_categorical_features(data, categorical_features)
            
            # 5. 特征组合策略
            if manual_split_features is not None and manual_poly_features is not None:
                # 使用手动指定的特征组合
                combinations_to_try = [(manual_split_features, manual_poly_features)]
                print(f"使用手动指定的特征组合")
            elif self.enable_exhaustive_search:
                # 生成所有可能的特征组合
                combinations_to_try = self._generate_feature_combinations(numeric_features, categorical_features, target_col)
                print(f"生成 {len(combinations_to_try)} 个特征组合进行穷举搜索")
            else:
                # 使用启发式方法生成少量高质量组合
                all_combinations = self._generate_feature_combinations(numeric_features, categorical_features, target_col)
                combinations_to_try = self._select_promising_combinations(all_combinations, all_split_candidates, numeric_features)
                print(f"使用启发式方法选择 {len(combinations_to_try)} 个特征组合")
            
            # 6. 评估所有组合
            print("\n开始评估特征组合...")
            best_score = -1
            best_rules = []
            
            for i, (split_features, poly_features) in enumerate(combinations_to_try):
                if i % max(1, len(combinations_to_try) // 10) == 0:
                    progress = (i + 1) / len(combinations_to_try) * 100
                    print(f"进度: {progress:.1f}% ({i+1}/{len(combinations_to_try)})")
                
                score, rules = self._evaluate_combination(encoded_data, split_features, poly_features, target_col)
                
                if score > best_score:
                    best_score = score
                    best_rules = rules
                    self.best_configuration = {
                        'split_features': split_features,
                        'poly_features': poly_features,
                        'score': score,
                        'num_rules': len(rules)
                    }
                    
                    print(f"  新的最佳组合! 分段特征: {split_features}, 多项式特征: {poly_features}")
                    print(f"  评分: {score:.3f}, 规则数量: {len(rules)}")
            
            # 7. 输出结果
            elapsed_time = time.time() - start_time
            print(f"\n搜索完成! 耗时: {elapsed_time:.2f}秒")
            
            if best_rules:
                print(f"\n最优特征配置:")
                print(f"  分段特征: {self.best_configuration['split_features']}")
                print(f"  多项式特征: {self.best_configuration['poly_features']}")
                print(f"  综合评分: {self.best_configuration['score']:.3f}")
                print(f"  发现规则数: {self.best_configuration['num_rules']}")
                
                self._display_optimal_results(best_rules)
            else:
                print("未发现有效的条件规则")
            
            return best_rules
            
        except FileNotFoundError:
            print(f"错误: 文件未找到 {csv_file_path}")
            return []
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _display_optimal_results(self, rules):
        """显示最优结果"""
        print(f"\n{'='*30} 最优规则详情 {'='*30}")
        
        # 按交叉验证R²分数排序
        sorted_rules = sorted(rules, key=lambda x: x['cv_r2_score'], reverse=True)
        
        for i, rule in enumerate(sorted_rules, 1):
            print(f"\n规则 {i}:")
            print(f"  条件: {rule['condition']}")
            print(f"  规则: {rule['rule']}")
            print(f"  交叉验证R²: {rule['cv_r2_score']:.3f}")
            print(f"  样本数: {rule['sample_count']}")
        
        # 生成表格总结
        print(f"\n{'='*20} 最优规则表格 {'='*20}")
        print("| 排名 | 条件 | 规则 | CV-R² | 样本数 |")
        print("|------|------|------|-------|--------|")
        
        for i, rule in enumerate(sorted_rules, 1):
            condition = rule['condition'][:35] + "..." if len(rule['condition']) > 35 else rule['condition']
            rule_str = rule['rule'][:45] + "..." if len(rule['rule']) > 45 else rule['rule']
            
            print(f"| {i} | {condition} | {rule_str} | {rule['cv_r2_score']:.3f} | {rule['sample_count']} |")

def discover_optimal_conditional_rules(csv_file_path, target_col=None,
                                      manual_split_features=None, manual_poly_features=None,
                                      max_depth=3, min_samples_leaf=50, 
                                      enable_exhaustive_search=True, max_combinations=100):
    """
    优化版条件多项式规则发现函数
    
    Args:
        csv_file_path: CSV文件路径
        target_col: 目标列名，默认使用最后一列
        manual_split_features: 手动指定分段特征
        manual_poly_features: 手动指定多项式特征
        max_depth: 决策树最大深度
        min_samples_leaf: 叶子节点最小样本数
        enable_exhaustive_search: 是否启用穷举搜索
        max_combinations: 最大尝试的组合数
        
    Returns:
        最优规则列表
    """
    discoverer = OptimalConditionalRuleDiscoverer(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        enable_exhaustive_search=enable_exhaustive_search,
        max_combinations=max_combinations
    )
    
    return discoverer.discover_optimal_rules(
        csv_file_path=csv_file_path,
        target_col=target_col,
        manual_split_features=manual_split_features,
        manual_poly_features=manual_poly_features
    )

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='优化版条件多项式规则发现工具')
    parser.add_argument('csv_file', type=str, help='要分析的CSV文件路径')
    parser.add_argument('--target-col', type=str, help='目标列的名称，如果不指定则使用最后一列')
    parser.add_argument('--split-features', type=str, nargs='+', help='手动指定分段特征')
    parser.add_argument('--poly-features', type=str, nargs='+', help='手动指定多项式特征')
    parser.add_argument('--max-depth', type=int, default=3, help='决策树最大深度')
    parser.add_argument('--min-samples', type=int, default=50, help='叶子节点最小样本数')
    parser.add_argument('--max-combinations', type=int, default=100, help='最大尝试的特征组合数')
    parser.add_argument('--disable-exhaustive', action='store_true', help='禁用穷举搜索，使用启发式方法')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print("=== 优化版条件规则发现工具 ===")
    print("主要特性:")
    print("1. 智能特征组合穷举")
    print("2. 交叉验证评估最优组合")
    print("3. 启发式搜索优化")
    print("4. 动态特征分配")
    print()
    
    # 运行优化版规则发现
    rules = discover_optimal_conditional_rules(
        csv_file_path=args.csv_file,
        target_col=args.target_col,
        manual_split_features=args.split_features,
        manual_poly_features=args.poly_features,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples,
        enable_exhaustive_search=not args.disable_exhaustive,
        max_combinations=args.max_combinations
    )
    
    print(f"\n最终发现的最优规则数量: {len(rules)}") 