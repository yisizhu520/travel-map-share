import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import warnings
import argparse

warnings.filterwarnings('ignore')

class ConditionalRuleDiscoverer:
    """
    改进版的条件多项式规则发现器
    
    主要改进：
    1. 支持多特征决策树分段
    2. 精确的条件提取
    3. 参数化的特征设定
    4. 模型质量评估
    5. 更好的异常处理
    """
    
    def __init__(self, polynomial_features=None, target_col=None, 
                 split_features=None, max_depth=3, min_samples_leaf=50):
        """
        初始化参数
        
        Args:
            polynomial_features: 用于多项式拟合的特征列表，如果为None则自动检测
            target_col: 目标列名，如果为None则自动检测
            split_features: 用于分段的特征列表，如果为None则自动检测
            max_depth: 决策树最大深度
            min_samples_leaf: 叶子节点最小样本数
        """
        self.polynomial_features = polynomial_features
        self.target_col = target_col
        self.split_features = split_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.discovered_rules = []
        
    def _auto_detect_features(self, data):
        """
        自动检测特征列
        假设第一行是 header，最后一列是目标列，其他列是特征列
        """
        numeric_cols = [col for col in data.columns if pd.api.types.is_numeric_dtype(data[col])]
        
        # 如果没有指定目标列，假设最后一列是目标列
        if self.target_col is None:
            self.target_col = data.columns[-1]
            print(f"自动选择最后一列 '{self.target_col}' 作为目标列")
                
        # 如果没有指定多项式特征，使用除目标列外的所有数值型特征作为多项式特征
        if self.polynomial_features is None:
            # 优先使用常见多项式特征名
            common_poly_features = ['a', 'b', 'c', 'x1', 'x2', 'x3']
            common_features_found = [col for col in common_poly_features if col in data.columns]
            
            if common_features_found:
                self.polynomial_features = common_features_found
                print(f"发现常见多项式特征名: {self.polynomial_features}")
            else:
                # 如果没有找到常见特征名，使用除目标列和分段特征外的所有数值列
                self.polynomial_features = [col for col in numeric_cols if col != self.target_col]
                # 如果多项式特征太多，可能会使模型复杂度过高，此时只取前几个特征
                if len(self.polynomial_features) > 5:
                    print(f"警告: 多项式特征数量较多 ({len(self.polynomial_features)}个)，只使用前5个特征")
                    self.polynomial_features = self.polynomial_features[:5]
                
        # 如果没有指定分段特征，尝试使用 'x', 'if_condition' 等典型分段特征名
        # 如果找不到这些特征，则随机选择一个数值特征作为分段特征
        if self.split_features is None:
            typical_split_features = ['x', 'if_condition', 'condition', 'split', 'group']
            split_features_found = [col for col in typical_split_features if col in data.columns]
            
            if split_features_found:
                self.split_features = split_features_found
                print(f"发现典型分段特征名: {self.split_features}")
            else:
                # 从多项式特征中选择一个作为分段特征
                potential_split_features = [col for col in numeric_cols 
                                          if col not in self.polynomial_features and col != self.target_col]
                
                if potential_split_features:
                    self.split_features = potential_split_features
                else:
                    # 如果没有额外特征，则从多项式特征中取一个作为分段特征
                    if self.polynomial_features:
                        self.split_features = [self.polynomial_features[0]]
                        # 从多项式特征列表中移除该特征
                        self.polynomial_features = self.polynomial_features[1:]
                        print(f"警告: 没有明确的分段特征，使用 '{self.split_features[0]}' 作为分段特征")
                    else:
                        print("错误: 无法确定分段特征，数据列太少")
    
    def _extract_tree_conditions(self, tree_model, feature_names):
        """从决策树中提取精确的条件路径"""
        tree = tree_model.tree_
        conditions_by_leaf = {}
        
        def extract_path(node_id, conditions):
            if tree.children_left[node_id] == -1:  # 叶子节点
                conditions_by_leaf[node_id] = conditions.copy()
                return
                
            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            feature_name = feature_names[feature_idx]
            
            # 左分支：<= threshold
            left_conditions = conditions + [f"{feature_name} <= {threshold:.2f}"]
            extract_path(tree.children_left[node_id], left_conditions)
            
            # 右分支：> threshold  
            right_conditions = conditions + [f"{feature_name} > {threshold:.2f}"]
            extract_path(tree.children_right[node_id], right_conditions)
        
        extract_path(0, [])
        return conditions_by_leaf
    
    def _fit_polynomial_with_evaluation(self, X_subset, y_subset):
        """拟合多项式并评估质量"""
        if len(X_subset) < len(self.polynomial_features) + 1:
            return None, None, 0.0
            
        try:
            model = LinearRegression()
            model.fit(X_subset, y_subset)
            
            # 计算R²分数
            y_pred = model.predict(X_subset)
            r2 = r2_score(y_subset, y_pred)
            
            return model, self._format_polynomial_rule(model), r2
            
        except Exception as e:
            print(f"      线性回归拟合错误: {e}")
            return None, None, 0.0
    
    def _format_polynomial_rule(self, model):
        """格式化多项式规则"""
        coefficients = model.coef_
        intercept = model.intercept_
        
        rule_parts = []
        for i, col_name in enumerate(self.polynomial_features):
            coeff_val = coefficients[i]
            
            # 智能四舍五入
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
            return f"{self.target_col} = {intercept_rounded}"
            
        rule_str = f"{self.target_col} = {' + '.join(rule_parts)}"
        
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
    
    def discover_rules(self, csv_file_path):
        """
        发现条件多项式规则
        
        Args:
            csv_file_path: CSV文件路径
            
        Returns:
            discovered_rules: 发现的规则列表
        """
        try:
            # 1. 数据加载
            data = pd.read_csv(csv_file_path)
            print(f"成功加载数据: {csv_file_path}")
            print(f"数据形状: {data.shape}")
            print("数据前5行:\n", data.head())
            
            # 2. 自动检测特征
            self._auto_detect_features(data)
            
            print(f"\n特征配置:")
            print(f"  多项式特征: {self.polynomial_features}")
            print(f"  目标列: {self.target_col}")
            print(f"  分段特征: {self.split_features}")
            
            # 3. 验证特征存在性
            missing_poly = [f for f in self.polynomial_features if f not in data.columns]
            if missing_poly:
                print(f"错误: 多项式特征 {missing_poly} 在数据中不存在")
                return []
                
            if self.target_col not in data.columns:
                print(f"错误: 目标列 '{self.target_col}' 在数据中不存在")
                return []
                
            missing_split = [f for f in self.split_features if f not in data.columns]
            if missing_split:
                print(f"警告: 分段特征 {missing_split} 在数据中不存在，将被忽略")
                self.split_features = [f for f in self.split_features if f in data.columns]
            
            if not self.split_features:
                print("错误: 没有可用的分段特征")
                return []
            
            y_target = data[self.target_col]
            
            # 4. 方法1：基于单特征分段（原方法的改进版）
            print(f"\n{'='*20} 方法1: 单特征分段 {'='*20}")
            self._discover_single_feature_rules(data, y_target)
            
            # 5. 方法2：基于多特征分段（新增）
            print(f"\n{'='*20} 方法2: 多特征分段 {'='*20}")
            self._discover_multi_feature_rules(data, y_target)
            
            # 6. 输出结果
            self._display_results()
            
            return self.discovered_rules
            
        except FileNotFoundError:
            print(f"错误: 文件未找到 {csv_file_path}")
            return []
        except Exception as e:
            print(f"处理过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _discover_single_feature_rules(self, data, y_target):
        """基于单特征的分段规则发现（改进原方法）"""
        for split_col in self.split_features:
            print(f"\n--- 使用 '{split_col}' 进行单特征分段 ---")
            
            X_split = data[[split_col]]
            
            # 训练决策树
            tree_model = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_leaf=self.min_samples_leaf, 
                random_state=42
            )
            
            try:
                tree_model.fit(X_split, y_target)
            except ValueError as e:
                print(f"  决策树训练失败: {e}")
                continue
            
            # 提取精确条件
            conditions_by_leaf = self._extract_tree_conditions(tree_model, [split_col])
            leaf_ids = tree_model.apply(X_split)
            
            print(f"  发现 {len(conditions_by_leaf)} 个分段")
            
            # 对每个分段拟合多项式
            for leaf_id, conditions in conditions_by_leaf.items():
                subset_mask = (leaf_ids == leaf_id)
                subset_data = data[subset_mask]
                
                if len(subset_data) < self.min_samples_leaf // 2:
                    continue
                
                X_poly = subset_data[self.polynomial_features]
                y_poly = subset_data[self.target_col]
                
                model, rule_str, r2 = self._fit_polynomial_with_evaluation(X_poly, y_poly)
                
                if model is not None and r2 > 0.1:  # 最小R²阈值
                    condition_str = " 且 ".join(conditions)
                    
                    rule_info = {
                        'method': 'single_feature',
                        'split_feature': split_col,
                        'condition': condition_str,
                        'rule': rule_str,
                        'r2_score': r2,
                        'sample_count': len(subset_data),
                        'model': model
                    }
                    
                    self.discovered_rules.append(rule_info)
                    print(f"    条件: {condition_str}")
                    print(f"    规则: {rule_str}")
                    print(f"    R²: {r2:.3f}, 样本数: {len(subset_data)}")
    
    def _discover_multi_feature_rules(self, data, y_target):
        """基于多特征的分段规则发现（新方法）"""
        if len(self.split_features) < 2:
            print("  分段特征少于2个，跳过多特征分段")
            return
            
        print(f"\n--- 使用多个特征进行分段: {self.split_features} ---")
        
        X_split = data[self.split_features]
        
        # 训练决策树
        tree_model = DecisionTreeRegressor(
            max_depth=self.max_depth, 
            min_samples_leaf=self.min_samples_leaf, 
            random_state=42
        )
        
        try:
            tree_model.fit(X_split, y_target)
        except ValueError as e:
            print(f"  多特征决策树训练失败: {e}")
            return
        
        # 提取精确条件
        conditions_by_leaf = self._extract_tree_conditions(tree_model, self.split_features)
        leaf_ids = tree_model.apply(X_split)
        
        print(f"  发现 {len(conditions_by_leaf)} 个多特征分段")
        
        # 对每个分段拟合多项式
        for leaf_id, conditions in conditions_by_leaf.items():
            subset_mask = (leaf_ids == leaf_id)
            subset_data = data[subset_mask]
            
            if len(subset_data) < self.min_samples_leaf // 2:
                continue
            
            X_poly = subset_data[self.polynomial_features]
            y_poly = subset_data[self.target_col]
            
            model, rule_str, r2 = self._fit_polynomial_with_evaluation(X_poly, y_poly)
            
            if model is not None and r2 > 0.1:
                condition_str = " 且 ".join(conditions)
                
                rule_info = {
                    'method': 'multi_feature',
                    'split_features': self.split_features,
                    'condition': condition_str,
                    'rule': rule_str,
                    'r2_score': r2,
                    'sample_count': len(subset_data),
                    'model': model
                }
                
                self.discovered_rules.append(rule_info)
                print(f"    条件: {condition_str}")
                print(f"    规则: {rule_str}")
                print(f"    R²: {r2:.3f}, 样本数: {len(subset_data)}")
    
    def _display_results(self):
        """显示发现的规则"""
        print(f"\n\n{'='*30} 规则发现总结 {'='*30}")
        
        if not self.discovered_rules:
            print("未发现有效的条件规则")
            return
        
        # 按R²分数排序
        sorted_rules = sorted(self.discovered_rules, key=lambda x: x['r2_score'], reverse=True)
        
        print(f"总共发现 {len(sorted_rules)} 条有效规则\n")
        
        # 显示详细结果
        for i, rule in enumerate(sorted_rules, 1):
            print(f"规则 {i} ({rule['method']}):")
            print(f"  条件: {rule['condition']}")
            print(f"  规则: {rule['rule']}")
            print(f"  质量: R² = {rule['r2_score']:.3f}")
            print(f"  样本数: {rule['sample_count']}")
            print("  ---")
        
        # 生成表格总结
        print(f"\n{'='*20} 最佳规则表格 {'='*20}")
        print("| 排名 | 方法 | 条件 | 规则 | R² | 样本数 |")
        print("|------|------|------|------|----|----|")
        
        for i, rule in enumerate(sorted_rules[:10], 1):  # 显示前10条
            method = "单特征" if rule['method'] == 'single_feature' else "多特征"
            condition = rule['condition'][:30] + "..." if len(rule['condition']) > 30 else rule['condition']
            rule_str = rule['rule'][:40] + "..." if len(rule['rule']) > 40 else rule['rule']
            
            print(f"| {i} | {method} | {condition} | {rule_str} | {rule['r2_score']:.3f} | {rule['sample_count']} |")

def discover_conditional_polynomial_rules_improved(csv_file_path, 
                                                 polynomial_features=None,
                                                 target_col=None,
                                                 split_features=None,
                                                 max_depth=3, 
                                                 min_samples_leaf=50):
    """
    改进版的条件多项式规则发现函数
    
    Args:
        csv_file_path: CSV文件路径
        polynomial_features: 多项式特征列表，默认['a', 'b', 'c']或自动检测
        target_col: 目标列，默认'result'或自动检测
        split_features: 分段特征列表，默认自动检测
        max_depth: 决策树最大深度
        min_samples_leaf: 叶子节点最小样本数
        
    Returns:
        发现的规则列表
    """
    discoverer = ConditionalRuleDiscoverer(
        polynomial_features=polynomial_features,
        target_col=target_col,
        split_features=split_features,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf
    )
    
    return discoverer.discover_rules(csv_file_path)

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='条件多项式规则发现工具')
    parser.add_argument('csv_file', type=str, help='要分析的CSV文件路径')
    parser.add_argument('--max-depth', type=int, default=3, help='决策树最大深度，默认为3')
    parser.add_argument('--min-samples', type=int, default=30, help='叶子节点最小样本数，默认为30')
    parser.add_argument('--target-col', type=str, help='目标列的名称，如果不指定则使用最后一列')
    parser.add_argument('--poly-features', type=str, nargs='+', help='多项式特征的列名，多个列名用空格分隔')
    parser.add_argument('--split-features', type=str, nargs='+', help='用于分段的特征列名，多个列名用空格分隔')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    print("=== 改进版条件规则发现 ===")
    print(f"CSV文件: {args.csv_file}")
    print("主要改进:")
    print("1. 支持多特征分段")
    print("2. 精确的条件提取")
    print("3. 自动特征检测")
    print("4. R²质量评估")
    print("5. 更好的结果排序和展示")
    print()
    
    # 使用改进版方法
    rules = discover_conditional_polynomial_rules_improved(
        csv_file_path=args.csv_file,
        polynomial_features=args.poly_features,
        target_col=args.target_col,
        split_features=args.split_features,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples
    )
    
    print(f"\n发现的规则数量: {len(rules)}") 