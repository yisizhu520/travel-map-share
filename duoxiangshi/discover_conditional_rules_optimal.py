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
                            
                            # 🔧 新增：简化条件字符串
                            condition_str = self._simplify_condition_string(condition_str)
                            
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
    
    def _simplify_condition_string(self, condition_str):
        """
        简化条件字符串中的冗余条件
        
        例如: "x <= 39.50 且 x <= 29.50 且 y ∈ {A}" 
        简化为: "x <= 29.50 且 y ∈ {A}"
        
        Args:
            condition_str: 原始条件字符串
            
        Returns:
            simplified_str: 简化后的条件字符串
        """
        if not condition_str or condition_str.strip() == "":
            return condition_str
            
        # 解析条件
        conditions = self._parse_condition(condition_str)
        
        # 重新格式化（这会自动简化冗余条件）
        simplified_str = self._format_merged_conditions(conditions)
        
        return simplified_str
    
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
            print("\n🔍 开始评估特征组合...")
            best_score = -1
            best_rules = []
            progress_interval = max(1, len(combinations_to_try) // 10)
            
            for i, (split_features, poly_features) in enumerate(combinations_to_try):
                # 只在关键进度点显示信息
                if i % progress_interval == 0 or i == len(combinations_to_try) - 1:
                    progress = (i + 1) / len(combinations_to_try) * 100
                    print(f"   进度: {progress:.1f}% ({i+1}/{len(combinations_to_try)}) - 当前最佳分数: {best_score:.3f}")
                
                score, rules = self._evaluate_combination(encoded_data, split_features, poly_features, target_col)
                
                if score > best_score:
                    previous_score = best_score
                    best_score = score
                    best_rules = rules
                    self.best_configuration = {
                        'split_features': split_features,
                        'poly_features': poly_features,
                        'score': score,
                        'num_rules': len(rules)
                    }
                    
                    # 只在找到明显更好的组合时才输出
                    if previous_score <= 0 or score > previous_score * 1.05:  # 提升超过5%才报告
                        print(f"   ✨ 发现更优组合! 分段特征: {split_features}")
                        print(f"      多项式特征: {poly_features}")
                        print(f"      评分提升: {score:.3f} (之前: {previous_score:.3f})")
            
            # 7. 输出结果
            elapsed_time = time.time() - start_time
            print(f"\n✅ 搜索完成! 耗时: {elapsed_time:.2f}秒")
            
            if best_rules:
                print(f"\n🏆 最优特征配置:")
                print(f"   🔧 分段特征: {self.best_configuration['split_features']}")
                print(f"   📊 多项式特征: {self.best_configuration['poly_features']}")
                print(f"   📈 综合评分: {self.best_configuration['score']:.3f}")
                print(f"   📋 发现规则数: {self.best_configuration['num_rules']}")
                
                self._display_optimal_results(best_rules)
            else:
                print("❌ 未发现有效的条件规则")
                print("💡 建议:")
                print("   • 尝试减小 --min-samples 参数")
                print("   • 尝试增大 --max-depth 参数")
                print("   • 检查数据是否包含足够的数值特征")
            
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
        if not rules:
            print("\n未发现任何有效规则")
            return
            
        print(f"\n{'='*50} 最优规则详情 {'='*50}")
        
        # 按交叉验证R²分数排序，去重
        unique_rules = []
        seen_rules = set()
        
        for rule in rules:
            rule_key = (rule['condition'], rule['rule'])
            if rule_key not in seen_rules:
                seen_rules.add(rule_key)
                unique_rules.append(rule)
        
        sorted_rules = sorted(unique_rules, key=lambda x: x['cv_r2_score'], reverse=True)
        
        # 显示详细规则信息
        for i, rule in enumerate(sorted_rules, 1):
            print(f"\n规则 {i}:")
            print(f"  条件: {rule['condition']}")
            print(f"  规则: {rule['rule']}")
            print(f"  交叉验证R²: {rule['cv_r2_score']:.3f}")
            print(f"  样本数: {rule['sample_count']}")
            if i < len(sorted_rules):  # 不是最后一个规则
                print("  " + "-" * 60)
        
        # 生成清晰的表格总结
        print(f"\n{'='*50} 最优规则汇总表 {'='*50}")
        
        # 动态调整列宽
        max_condition_len = min(50, max(len(rule['condition']) for rule in sorted_rules) + 2)
        max_rule_len = min(40, max(len(rule['rule']) for rule in sorted_rules) + 2)
        
        # 打印表头
        header = f"| {'排名':^4} | {'条件':^{max_condition_len}} | {'规则':^{max_rule_len}} | {'R²':^5} | {'样本数':^6} |"
        separator = "|" + "-" * 6 + "|" + "-" * (max_condition_len + 2) + "|" + "-" * (max_rule_len + 2) + "|" + "-" * 7 + "|" + "-" * 8 + "|"
        
        print(separator)
        print(header)
        print(separator)
        
        # 打印规则行
        for i, rule in enumerate(sorted_rules, 1):
            condition = rule['condition']
            if len(condition) > max_condition_len:
                condition = condition[:max_condition_len-3] + "..."
                
            rule_str = rule['rule']
            if len(rule_str) > max_rule_len:
                rule_str = rule_str[:max_rule_len-3] + "..."
            
            row = f"| {i:^4} | {condition:<{max_condition_len}} | {rule_str:<{max_rule_len}} | {rule['cv_r2_score']:^5.3f} | {rule['sample_count']:^6} |"
            print(row)
        
        print(separator)
        
        # 统计信息
        print(f"\n📊 原始规则统计:")
        print(f"   • 总规则数: {len(sorted_rules)}")
        print(f"   • 平均R²分数: {np.mean([r['cv_r2_score'] for r in sorted_rules]):.3f}")
        print(f"   • 覆盖样本总数: {sum(r['sample_count'] for r in sorted_rules)}")
        
        # 质量分级统计
        excellent_rules = [r for r in sorted_rules if r['cv_r2_score'] >= 0.9]
        good_rules = [r for r in sorted_rules if 0.7 <= r['cv_r2_score'] < 0.9]
        fair_rules = [r for r in sorted_rules if r['cv_r2_score'] < 0.7]
        
        print(f"   • 优秀规则(R²≥0.9): {len(excellent_rules)}条")
        print(f"   • 良好规则(0.7≤R²<0.9): {len(good_rules)}条") 
        print(f"   • 一般规则(R²<0.7): {len(fair_rules)}条")
        
        # 🔗 智能合并相同规则
        merged_rules = self._merge_similar_rules(sorted_rules)
        
        if len(merged_rules) < len(sorted_rules):
            # 显示合并后的结果
            print(f"\n{'='*50} 智能合并后规则 {'='*50}")
            
            # 重新计算合并后规则的表格宽度
            merged_max_condition_len = min(60, max(len(rule['condition']) for rule in merged_rules) + 2)
            merged_max_rule_len = min(40, max(len(rule['rule']) for rule in merged_rules) + 2)
            
            # 合并后表格
            merged_header = f"| {'排名':^4} | {'条件':^{merged_max_condition_len}} | {'规则':^{merged_max_rule_len}} | {'R²':^5} | {'样本数':^6} | {'合并数':^6} |"
            merged_separator = "|" + "-" * 6 + "|" + "-" * (merged_max_condition_len + 2) + "|" + "-" * (merged_max_rule_len + 2) + "|" + "-" * 7 + "|" + "-" * 8 + "|" + "-" * 8 + "|"
            
            print(merged_separator)
            print(merged_header)
            print(merged_separator)
            
            # 按R²重新排序合并后的规则
            merged_sorted = sorted(merged_rules, key=lambda x: x['cv_r2_score'], reverse=True)
            
            for i, rule in enumerate(merged_sorted, 1):
                condition = rule['condition']
                if len(condition) > merged_max_condition_len:
                    condition = condition[:merged_max_condition_len-3] + "..."
                    
                rule_str = rule['rule']
                if len(rule_str) > merged_max_rule_len:
                    rule_str = rule_str[:merged_max_rule_len-3] + "..."
                
                merge_info = f"{rule.get('merged_from', 1)}" if 'merged_from' in rule else "1"
                
                row = f"| {i:^4} | {condition:<{merged_max_condition_len}} | {rule_str:<{merged_max_rule_len}} | {rule['cv_r2_score']:^5.3f} | {rule['sample_count']:^6} | {merge_info:^6} |"
                print(row)
            
            print(merged_separator)
            
            # 合并后统计
            print(f"\n📊 合并后规则统计:")
            print(f"   • 合并后规则数: {len(merged_rules)} (减少了 {len(sorted_rules) - len(merged_rules)} 条)")
            print(f"   • 平均R²分数: {np.mean([r['cv_r2_score'] for r in merged_rules]):.3f}")
            print(f"   • 覆盖样本总数: {sum(r['sample_count'] for r in merged_rules)}")
            
            # 突出显示简化效果
            simplification_rate = (len(sorted_rules) - len(merged_rules)) / len(sorted_rules) * 100
            print(f"   🎯 规则简化率: {simplification_rate:.1f}%")
            
            # 💾 保存发现的规则供预测器使用
            self.discovered_rules = merged_rules
            print(f"\n✅ 规则已保存到发现器，可用于后续预测")
            
        else:
            print(f"\n💡 所有规则都是唯一的，无需合并")
            # 💾 保存发现的规则供预测器使用
            self.discovered_rules = sorted_rules
            print(f"\n✅ 规则已保存到发现器，可用于后续预测")
            
        return merged_rules if len(merged_rules) < len(sorted_rules) else sorted_rules
    
    def _merge_similar_rules(self, rules):
        """
        合并相同规则的条件
        
        Args:
            rules: 规则列表
            
        Returns:
            merged_rules: 合并后的规则列表
        """
        if not rules:
            return []
            
        print(f"\n🔗 开始智能合并相同规则...")
        
        # 按规则内容分组
        rule_groups = {}
        for rule in rules:
            rule_formula = rule['rule']
            if rule_formula not in rule_groups:
                rule_groups[rule_formula] = []
            rule_groups[rule_formula].append(rule)
        
        merged_rules = []
        merge_count = 0
        
        for rule_formula, group_rules in rule_groups.items():
            if len(group_rules) == 1:
                # 只有一个规则，直接保留
                merged_rules.append(group_rules[0])
            else:
                # 多个规则需要合并
                print(f"   合并规则: {rule_formula}")
                print(f"   原有 {len(group_rules)} 个条件，尝试合并...")
                
                merged_rule = self._merge_conditions(group_rules, rule_formula)
                merged_rules.append(merged_rule)
                merge_count += len(group_rules) - 1
                
                print(f"   ✅ 合并完成，条件: {merged_rule['condition']}")
        
        print(f"🎯 合并统计: 原有 {len(rules)} 条规则，合并后 {len(merged_rules)} 条规则，共合并了 {merge_count} 条")
        return merged_rules
    
    def _merge_conditions(self, group_rules, rule_formula):
        """
        合并一组相同规则的条件
        
        Args:
            group_rules: 相同规则的列表
            rule_formula: 规则公式
            
        Returns:
            merged_rule: 合并后的规则
        """
        # 解析所有条件
        all_conditions = []
        total_samples = 0
        total_r2_weighted = 0
        
        for rule in group_rules:
            conditions = self._parse_condition(rule['condition'])
            all_conditions.append(conditions)
            total_samples += rule['sample_count']
            total_r2_weighted += rule['cv_r2_score'] * rule['sample_count']
        
        # 计算加权平均R²
        avg_r2 = total_r2_weighted / total_samples if total_samples > 0 else 0
        
        # 合并条件
        merged_conditions = self._merge_condition_logic(all_conditions)
        merged_condition_str = self._format_merged_conditions(merged_conditions)
        
        return {
            'condition': merged_condition_str,
            'rule': rule_formula,
            'cv_r2_score': avg_r2,
            'sample_count': total_samples,
            'merged_from': len(group_rules)
        }
    
    def _parse_condition(self, condition_str):
        """
        解析条件字符串为结构化格式
        
        Args:
            condition_str: 条件字符串，如 "x <= 39.50 且 y ∈ {y1}"
            
        Returns:
            conditions: 解析后的条件字典
        """
        conditions = {}
        
        # 按 "且" 分割条件
        parts = condition_str.split(' 且 ')
        
        for part in parts:
            part = part.strip()
            
            if ' ∈ ' in part:
                # 分类条件：y ∈ {y1, y2}
                feature, values_str = part.split(' ∈ ')
                feature = feature.strip()
                # 提取集合中的值
                values_str = values_str.strip().replace('{', '').replace('}', '')
                values = [v.strip() for v in values_str.split(',')]
                
                if feature not in conditions:
                    conditions[feature] = {'type': 'categorical', 'values': set()}
                conditions[feature]['values'].update(values)
                
            elif '<=' in part:
                # 数值条件：x <= 39.50
                feature, value_str = part.split('<=')
                feature = feature.strip()
                value = float(value_str.strip())
                
                if feature not in conditions:
                    conditions[feature] = {'type': 'numeric', 'upper': None, 'lower': None}
                
                if conditions[feature]['upper'] is None or value < conditions[feature]['upper']:
                    conditions[feature]['upper'] = value
                    
            elif '>' in part:
                # 数值条件：x > 39.50
                feature, value_str = part.split('>')
                feature = feature.strip()
                value = float(value_str.strip())
                
                if feature not in conditions:
                    conditions[feature] = {'type': 'numeric', 'upper': None, 'lower': None}
                
                if conditions[feature]['lower'] is None or value > conditions[feature]['lower']:
                    conditions[feature]['lower'] = value
        
        return conditions
    
    def _merge_condition_logic(self, all_conditions):
        """
        合并多个条件的逻辑
        
        Args:
            all_conditions: 多个条件的列表
            
        Returns:
            merged_conditions: 合并后的条件
        """
        merged = {}
        
        # 收集所有特征
        all_features = set()
        for conditions in all_conditions:
            all_features.update(conditions.keys())
        
        for feature in all_features:
            feature_conditions = []
            
            # 收集该特征的所有条件
            for conditions in all_conditions:
                if feature in conditions:
                    feature_conditions.append(conditions[feature])
            
            if not feature_conditions:
                continue
                
            first_condition = feature_conditions[0]
            
            if first_condition['type'] == 'categorical':
                # 分类特征：取并集
                all_values = set()
                for cond in feature_conditions:
                    all_values.update(cond['values'])
                
                merged[feature] = {
                    'type': 'categorical',
                    'values': all_values
                }
                
            elif first_condition['type'] == 'numeric':
                # 数值特征：计算真正的并集范围
                ranges = []
                
                # 收集所有的数值范围
                for cond in feature_conditions:
                    upper = cond.get('upper')
                    lower = cond.get('lower')
                    
                    # 构建范围元组 (lower_bound, upper_bound, inclusive_lower, inclusive_upper)
                    if upper is not None and lower is not None:
                        # 形如 lower < x <= upper
                        ranges.append((lower, upper, False, True))
                    elif upper is not None:
                        # 形如 x <= upper，相当于 (-∞, upper]
                        ranges.append((float('-inf'), upper, True, True))
                    elif lower is not None:
                        # 形如 x > lower，相当于 (lower, +∞)
                        ranges.append((lower, float('inf'), False, True))
                
                # 合并重叠的范围
                if ranges:
                    merged_range = self._merge_numeric_ranges(ranges)
                    
                    # 将合并后的范围转换回条件格式
                    if len(merged_range) == 1:
                        lower, upper, incl_lower, incl_upper = merged_range[0]
                        
                        final_condition = {'type': 'numeric', 'upper': None, 'lower': None}
                        
                        if lower != float('-inf'):
                            final_condition['lower'] = lower
                        if upper != float('inf'):
                            final_condition['upper'] = upper
                            
                        merged[feature] = final_condition
                    else:
                        # 多个不连续的范围，暂时取第一个范围（可以进一步优化）
                        lower, upper, incl_lower, incl_upper = merged_range[0]
                        final_condition = {'type': 'numeric', 'upper': None, 'lower': None}
                        
                        if lower != float('-inf'):
                            final_condition['lower'] = lower
                        if upper != float('inf'):
                            final_condition['upper'] = upper
                            
                        merged[feature] = final_condition
        
        return merged
    
    def _merge_numeric_ranges(self, ranges):
        """
        合并数值范围
        
        Args:
            ranges: 范围列表，每个元素为 (lower, upper, incl_lower, incl_upper)
            
        Returns:
            merged_ranges: 合并后的范围列表
        """
        if not ranges:
            return []
        
        # 排序范围
        sorted_ranges = sorted(ranges, key=lambda x: (x[0], x[1]))
        
        merged = [sorted_ranges[0]]
        
        for current in sorted_ranges[1:]:
            last = merged[-1]
            
            # 检查是否重叠或相邻
            if current[0] <= last[1]:  # 有重叠
                # 合并范围
                new_lower = min(last[0], current[0])
                new_upper = max(last[1], current[1])
                merged[-1] = (new_lower, new_upper, True, True)
            else:
                # 没有重叠，添加新范围
                merged.append(current)
        
        return merged
    
    def _format_merged_conditions(self, merged_conditions):
        """
        格式化合并后的条件
        
        Args:
            merged_conditions: 合并后的条件字典
            
        Returns:
            condition_str: 格式化的条件字符串
        """
        condition_parts = []
        
        for feature, condition in merged_conditions.items():
            if condition['type'] == 'categorical':
                values_str = ', '.join(sorted(condition['values']))
                condition_parts.append(f"{feature} ∈ {{{values_str}}}")
                
            elif condition['type'] == 'numeric':
                if condition['lower'] is not None and condition['upper'] is not None:
                    # 检查是否为无意义的范围
                    if condition['lower'] >= condition['upper']:
                        # 这种情况下，保留更宽泛的条件
                        if abs(condition['lower']) > abs(condition['upper']):
                            condition_parts.append(f"{feature} > {condition['lower']:.2f}")
                        else:
                            condition_parts.append(f"{feature} <= {condition['upper']:.2f}")
                    else:
                        condition_parts.append(f"{condition['lower']:.2f} < {feature} <= {condition['upper']:.2f}")
                elif condition['upper'] is not None:
                    condition_parts.append(f"{feature} <= {condition['upper']:.2f}")
                elif condition['lower'] is not None:
                    condition_parts.append(f"{feature} > {condition['lower']:.2f}")
        
        return ' 且 '.join(condition_parts)

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
    
    print("🚀 === 优化版条件规则发现工具 === 🚀")
    print("主要特性:")
    print("  ✓ 智能特征组合穷举")
    print("  ✓ 交叉验证评估最优组合")
    print("  ✓ 启发式搜索优化")
    print("  ✓ 动态特征分配")
    print("  ✓ 支持分类和数值特征")
    print("-" * 60)
    
    # 运行优化版规则发现
    try:
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
        
        # 最终总结
        print(f"\n🎉 分析完成！")
        print(f"   📋 最终发现的最优规则数量: {len(rules)}")
        
        if rules:
            avg_quality = np.mean([r['cv_r2_score'] for r in rules])
            print(f"   📈 规则平均质量(R²): {avg_quality:.3f}")
            if avg_quality >= 0.9:
                print("   🌟 规则质量: 优秀")
            elif avg_quality >= 0.7:
                print("   ⭐ 规则质量: 良好") 
            else:
                print("   📊 规则质量: 一般")
        else:
            print("   ⚠️  建议: 尝试调整参数或检查数据质量")
            
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc() 
        
    print("\n" + "="*60)
    print("感谢使用条件规则发现工具! 🙏") 