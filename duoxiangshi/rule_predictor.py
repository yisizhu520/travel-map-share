#!/usr/bin/env python3
"""
基于条件规则的智能预测器
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class RuleBasedPredictor:
    """
    基于条件规则的智能预测器
    
    功能：
    1. 存储已发现的规则
    2. 对新数据进行预测
    3. 提供详细的预测解释
    4. 处理复杂的条件匹配
    """
    
    def __init__(self, rules: List[Dict] = None, label_encoders: Dict = None):
        """
        初始化预测器
        
        Args:
            rules: 已发现的规则列表
            label_encoders: 分类特征的编码器
        """
        self.rules = rules or []
        self.label_encoders = label_encoders or {}
        self.target_column = None
        
    def add_rules(self, rules: List[Dict], label_encoders: Dict = None):
        """添加规则到预测器"""
        self.rules.extend(rules)
        if label_encoders:
            self.label_encoders.update(label_encoders)
    
    def _parse_rule_formula(self, rule_formula: str) -> Tuple[Dict, float]:
        """
        解析规则公式，提取系数和截距
        
        Args:
            rule_formula: 规则公式字符串，如 "result = 2 * x + 3 * y + 5"
            
        Returns:
            coefficients: 特征系数字典
            intercept: 截距
        """
        # 提取目标变量
        if '=' in rule_formula:
            left, right = rule_formula.split('=', 1)
            self.target_column = left.strip()
            formula = right.strip()
        else:
            formula = rule_formula.strip()
        
        coefficients = {}
        intercept = 0.0
        
        # 处理公式中的项
        # 分割项（处理+和-）
        terms = re.split(r'(?=[+-])', formula)
        terms = [term.strip() for term in terms if term.strip()]
        
        for term in terms:
            term = term.strip()
            if not term:
                continue
                
            # 处理符号
            if term.startswith('+'):
                term = term[1:].strip()
                sign = 1
            elif term.startswith('-'):
                term = term[1:].strip()
                sign = -1
            else:
                sign = 1
            
            # 检查是否包含变量
            if '*' in term:
                # 形如 "2 * x" 或 "x * 2"
                parts = term.split('*')
                if len(parts) == 2:
                    part1, part2 = [p.strip() for p in parts]
                    
                    # 判断哪部分是系数，哪部分是变量
                    try:
                        coeff = float(part1)
                        var = part2
                    except ValueError:
                        try:
                            coeff = float(part2)
                            var = part1
                        except ValueError:
                            # 都不是数字，可能是两个变量相乘，暂时跳过
                            continue
                    
                    coefficients[var] = sign * coeff
            else:
                # 没有乘号，可能是单独的变量或常数
                try:
                    # 尝试解析为常数
                    const_val = float(term)
                    intercept += sign * const_val
                except ValueError:
                    # 是变量名，系数为1
                    coefficients[term] = sign * 1.0
        
        return coefficients, intercept
    
    def _evaluate_condition(self, condition_str: str, input_data: Dict) -> bool:
        """
        评估条件是否满足
        
        Args:
            condition_str: 条件字符串，如 "x <= 30.00 且 y ∈ {A}" 或 "39.50 < x <= 59.50"
            input_data: 输入数据字典
            
        Returns:
            是否满足条件
        """
        if not condition_str or condition_str.strip() == "":
            return True
        
        # 按 "且" 分割条件
        conditions = condition_str.split(' 且 ')
        
        for condition in conditions:
            condition = condition.strip()
            
            if ' ∈ ' in condition:
                # 分类条件：y ∈ {A, B}
                feature, values_str = condition.split(' ∈ ')
                feature = feature.strip()
                
                # 提取集合中的值
                values_str = values_str.strip().replace('{', '').replace('}', '')
                allowed_values = [v.strip() for v in values_str.split(',')]
                
                if feature not in input_data:
                    return False
                
                input_value = str(input_data[feature])
                if input_value not in allowed_values:
                    return False
            
            elif re.match(r'\d+\.?\d*\s*<\s*\w+\s*<=\s*\d+\.?\d*', condition):
                # 范围条件：39.50 < x <= 59.50 (使用正则表达式精确匹配)
                range_pattern = r'(\d+\.?\d*)\s*<\s*(\w+)\s*<=\s*(\d+\.?\d*)'
                match = re.match(range_pattern, condition)
                
                if match:
                    lower_str, feature, upper_str = match.groups()
                    lower_threshold = float(lower_str)
                    upper_threshold = float(upper_str)
                    
                    if feature not in input_data:
                        return False
                    
                    try:
                        input_value = float(input_data[feature])
                        # 检查 lower < input_value <= upper
                        if input_value <= lower_threshold or input_value > upper_threshold:
                            return False
                    except (ValueError, TypeError):
                        return False
                else:
                    # 解析失败，返回False
                    return False
                    
            elif '<=' in condition:
                # 数值条件：x <= 30.00
                feature, threshold_str = condition.split('<=')
                feature = feature.strip()
                threshold = float(threshold_str.strip())
                
                if feature not in input_data:
                    return False
                
                try:
                    input_value = float(input_data[feature])
                    if input_value > threshold:
                        return False
                except (ValueError, TypeError):
                    return False
                    
            elif '>' in condition:
                # 数值条件：x > 20.00
                feature, threshold_str = condition.split('>')
                feature = feature.strip()
                threshold = float(threshold_str.strip())
                
                if feature not in input_data:
                    return False
                
                try:
                    input_value = float(input_data[feature])
                    if input_value <= threshold:
                        return False
                except (ValueError, TypeError):
                    return False
        
        return True
    
    def predict(self, input_data: Dict, explain: bool = True) -> Dict:
        """
        对输入数据进行预测
        
        Args:
            input_data: 输入数据字典，如 {'x': 25, 'y': 'A', 'z': 1}
            explain: 是否提供详细解释
            
        Returns:
            预测结果字典，包含预测值和解释
        """
        if not self.rules:
            return {
                'prediction': None,
                'confidence': 0.0,
                'explanation': '❌ 没有可用的规则进行预测',
                'matched_rules': [],
                'status': 'no_rules'
            }
        
        # 寻找匹配的规则
        matched_rules = []
        
        for i, rule in enumerate(self.rules):
            if self._evaluate_condition(rule['condition'], input_data):
                matched_rules.append((i, rule))
        
        if not matched_rules:
            return {
                'prediction': None,
                'confidence': 0.0,
                'explanation': '❌ 输入数据不符合任何已知规则的条件',
                'matched_rules': [],
                'status': 'no_match',
                'input_data': input_data,
                'available_conditions': [rule['condition'] for rule in self.rules]
            }
        
        # 如果有多个匹配的规则，选择质量最高的
        best_rule_idx, best_rule = max(matched_rules, key=lambda x: x[1]['cv_r2_score'])
        
        # 应用规则进行预测
        try:
            coefficients, intercept = self._parse_rule_formula(best_rule['rule'])
            
            # 计算预测值
            prediction = intercept
            calculation_steps = [f"基础值: {intercept}"]
            
            for feature, coeff in coefficients.items():
                if feature in input_data:
                    feature_value = float(input_data[feature])
                    contribution = coeff * feature_value
                    prediction += contribution
                    
                    if coeff == 1:
                        calculation_steps.append(f"{feature}({feature_value}) = {contribution}")
                    elif coeff == -1:
                        calculation_steps.append(f"-{feature}({feature_value}) = {contribution}")
                    else:
                        calculation_steps.append(f"{coeff} × {feature}({feature_value}) = {contribution}")
                else:
                    calculation_steps.append(f"⚠️ 缺少特征: {feature}")
            
            # 生成解释
            explanation_parts = []
            
            if explain:
                explanation_parts.extend([
                    "🎯 预测结果分析:",
                    f"   📊 预测值: {prediction:.3f}",
                    f"   🎲 置信度: {best_rule['cv_r2_score']:.1%}",
                    "",
                    "📋 应用的规则:",
                    f"   🔍 条件: {best_rule['condition']}",
                    f"   📐 公式: {best_rule['rule']}",
                    f"   📈 质量(R²): {best_rule['cv_r2_score']:.3f}",
                    "",
                    "🧮 计算过程:"
                ])
                
                for step in calculation_steps:
                    explanation_parts.append(f"   {step}")
                
                explanation_parts.extend([
                    "",
                    f"   📊 最终结果: {prediction:.3f}"
                ])
                
                if len(matched_rules) > 1:
                    explanation_parts.extend([
                        "",
                        f"💡 备注: 发现 {len(matched_rules)} 个匹配规则，已选择质量最高的规则"
                    ])
            
            return {
                'prediction': prediction,
                'confidence': best_rule['cv_r2_score'],
                'explanation': '\n'.join(explanation_parts) if explain else '',
                'matched_rules': [rule for _, rule in matched_rules],
                'selected_rule': best_rule,
                'calculation_steps': calculation_steps,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'prediction': None,
                'confidence': 0.0,
                'explanation': f'❌ 预测计算失败: {str(e)}',
                'matched_rules': [rule for _, rule in matched_rules],
                'status': 'calculation_error',
                'error': str(e)
            }
    
    def predict_batch(self, input_dataframe: pd.DataFrame, explain: bool = False) -> pd.DataFrame:
        """
        批量预测
        
        Args:
            input_dataframe: 输入数据框
            explain: 是否包含解释列
            
        Returns:
            包含预测结果的数据框
        """
        results = []
        
        for idx, row in input_dataframe.iterrows():
            input_dict = row.to_dict()
            result = self.predict(input_dict, explain=explain)
            
            result_row = {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'status': result['status']
            }
            
            if explain:
                result_row['explanation'] = result['explanation']
            
            results.append(result_row)
        
        result_df = pd.concat([input_dataframe.reset_index(drop=True), 
                              pd.DataFrame(results).reset_index(drop=True)], axis=1)
        
        return result_df
    
    def explain_prediction_details(self, input_data: Dict) -> str:
        """
        提供详细的预测解释
        """
        result = self.predict(input_data, explain=True)
        
        if result['status'] != 'success':
            return result['explanation']
        
        detailed_explanation = [
            "=" * 60,
            "🔮 详细预测分析报告",
            "=" * 60,
            "",
            "📥 输入数据:",
        ]
        
        for key, value in input_data.items():
            detailed_explanation.append(f"   {key}: {value}")
        
        detailed_explanation.extend([
            "",
            result['explanation'],
            "",
            "🔍 规则匹配详情:",
            f"   总可用规则数: {len(self.rules)}",
            f"   匹配规则数: {len(result['matched_rules'])}",
        ])
        
        if len(result['matched_rules']) > 1:
            detailed_explanation.extend([
                "",
                "🏆 所有匹配规则(按质量排序):"
            ])
            
            sorted_rules = sorted(result['matched_rules'], 
                                key=lambda x: x['cv_r2_score'], reverse=True)
            
            for i, rule in enumerate(sorted_rules, 1):
                marker = "👑" if rule == result['selected_rule'] else "  "
                detailed_explanation.append(
                    f"   {marker} {i}. {rule['condition']} → {rule['rule']} (R²={rule['cv_r2_score']:.3f})"
                )
        
        detailed_explanation.extend([
            "",
            "=" * 60,
            "预测完成 ✅",
            "=" * 60
        ])
        
        return '\n'.join(detailed_explanation)

def create_predictor_from_discoverer(discoverer, csv_file_path: str = None) -> RuleBasedPredictor:
    """
    从规则发现器创建预测器
    
    Args:
        discoverer: OptimalConditionalRuleDiscoverer实例
        csv_file_path: 可选，用于获取更多上下文信息
        
    Returns:
        配置好的预测器
    """
    if not hasattr(discoverer, 'discovered_rules') or not discoverer.discovered_rules:
        print("⚠️ 发现器中没有已发现的规则，请先运行规则发现")
        return RuleBasedPredictor()
    
    # 获取规则和编码器
    rules = discoverer.discovered_rules
    label_encoders = discoverer.label_encoders
    
    predictor = RuleBasedPredictor(rules, label_encoders)
    
    print(f"✅ 成功创建预测器!")
    print(f"   📋 加载规则数: {len(rules)}")
    print(f"   🏷️ 分类特征编码器: {list(label_encoders.keys())}")
    
    return predictor

if __name__ == "__main__":
    # 示例用法
    print("🔮 === 规则预测器演示 === 🔮")
    
    # 创建示例规则
    sample_rules = [
        {
            'condition': 'x <= 30.00 且 y ∈ {A}',
            'rule': 'result = 2 * x + 5',
            'cv_r2_score': 0.95,
            'sample_count': 100
        },
        {
            'condition': 'x > 30.00 且 y ∈ {A}',
            'rule': 'result = 1.5 * x + 10',
            'cv_r2_score': 0.92,
            'sample_count': 80
        },
        {
            'condition': 'y ∈ {B}',
            'rule': 'result = 3 * x + 2',
            'cv_r2_score': 0.98,
            'sample_count': 120
        }
    ]
    
    # 创建预测器
    predictor = RuleBasedPredictor(sample_rules)
    
    # 测试预测
    test_inputs = [
        {'x': 25, 'y': 'A'},
        {'x': 35, 'y': 'A'},
        {'x': 20, 'y': 'B'},
        {'x': 40, 'y': 'C'},  # 不匹配任何规则
    ]
    
    for i, input_data in enumerate(test_inputs, 1):
        print(f"\n{'='*50}")
        print(f"测试案例 {i}: {input_data}")
        print('='*50)
        
        result = predictor.predict(input_data)
        print(result['explanation']) 