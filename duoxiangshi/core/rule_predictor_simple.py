#!/usr/bin/env python3
"""
使用simpleeval库的简化规则预测器
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from simpleeval import simple_eval, SimpleEval, EvalWithCompoundTypes

class SimpleRulePredictor:
    """
    使用simpleeval库的简化规则预测器
    
    优势：
    1. 不需要复杂的条件解析逻辑
    2. 安全的表达式求值
    3. 支持复杂的数学表达式
    4. 代码量大幅减少
    """
    
    def __init__(self, rules: List[Dict] = None):
        """
        初始化预测器
        
        Args:
            rules: 规则列表，每个规则包含condition和formula
        """
        self.rules = rules or []
        # 使用支持复合类型的evaluator
        self.evaluator = EvalWithCompoundTypes()
        
        # 添加一些有用的函数
        self.evaluator.functions.update({
            'abs': abs,
            'max': max,
            'min': min,
            'round': round,
        })
        
    def add_rules(self, rules: List[Dict]):
        """添加规则"""
        self.rules.extend(rules)
    
    def _convert_condition_to_eval_format(self, condition: str, input_data: Dict) -> str:
        """
        将我们的条件格式转换为simpleeval可以理解的格式
        
        Args:
            condition: 原始条件字符串，如 "x <= 29.50 且 y ∈ {y1}"
            input_data: 输入数据
            
        Returns:
            转换后的条件字符串
        """
        if not condition or condition.strip() == "":
            return "True"
        
        # 将中文"且"替换为Python的"and"
        eval_condition = condition.replace(" 且 ", " and ")
        
        # 处理集合条件 "y ∈ {y1, y2}" -> "y in ['y1', 'y2']"
        import re
        pattern = r'(\w+)\s*∈\s*\{([^}]+)\}'
        
        def replace_set_condition(match):
            var_name = match.group(1)
            values_str = match.group(2)
            # 解析集合中的值
            values = [v.strip() for v in values_str.split(',')]
            # 转换为Python列表格式
            python_list = str(values)
            return f"{var_name} in {python_list}"
        
        eval_condition = re.sub(pattern, replace_set_condition, eval_condition)
        
        return eval_condition
    
    def _parse_formula(self, formula: str) -> str:
        """
        解析公式，提取右侧表达式
        
        Args:
            formula: 规则公式，如 "result = 2 * a + b + 1"
            
        Returns:
            表达式部分，如 "2 * a + b + 1"
        """
        if '=' in formula:
            return formula.split('=', 1)[1].strip()
        return formula.strip()
    
    def predict(self, input_data: Dict, explain: bool = True) -> Dict:
        """
        对输入数据进行预测
        
        Args:
            input_data: 输入数据字典
            explain: 是否提供详细解释
            
        Returns:
            预测结果字典
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
            try:
                # 转换条件格式
                eval_condition = self._convert_condition_to_eval_format(
                    rule['condition'], input_data
                )
                
                # 使用EvalWithCompoundTypes评估条件
                self.evaluator.names = input_data
                condition_result = self.evaluator.eval(eval_condition)
                
                if condition_result:
                    matched_rules.append((i, rule))
                    
            except Exception as e:
                # 如果条件评估失败，跳过这个规则
                if explain:
                    print(f"⚠️ 规则 {i+1} 条件评估失败: {e}")
                continue
        
        if not matched_rules:
            return {
                'prediction': None,
                'confidence': 0.0,
                'explanation': '❌ 输入数据不符合任何已知规则的条件',
                'matched_rules': [],
                'status': 'no_match',
                'input_data': input_data
            }
        
        # 选择质量最高的规则
        def get_rule_score(rule_tuple):
            """获取规则的评分，兼容不同的评分字段"""
            _, rule = rule_tuple
            # 优先使用cv_r2_score，如果没有则使用score
            return rule.get('cv_r2_score', rule.get('score', 0))
        
        best_rule_idx, best_rule = max(matched_rules, key=get_rule_score)
        best_score = get_rule_score((best_rule_idx, best_rule))
        
        # 应用规则进行预测
        try:
            # 🆕 区分分类规则和回归规则
            rule_type = best_rule.get('rule_type', 'regression')
            
            if rule_type == 'classification':
                # 分类规则：直接返回目标值，或从输入数据中获取对应列的值
                target_value = best_rule.get('target_value')
                
                if target_value is None:
                    # 如果没有target_value，尝试从规则字符串中解析
                    rule_str = best_rule['rule']
                    if '=' in rule_str:
                        target_value = rule_str.split('=', 1)[1].strip()
                
                # 🆕 检查target_value是否是输入数据中的列名
                if target_value in input_data:
                    # 如果是列名，返回对应的具体值
                    prediction = input_data[target_value]
                else:
                    # 否则直接返回target_value
                    prediction = target_value
                
                # 生成分类规则的解释
                explanation_parts = []
                if explain:
                    explanation_parts.extend([
                        "🎯 分类预测结果:",
                        f"   📊 预测类别: {prediction}",
                        f"   🎲 准确率: {best_score:.1%}",
                        "",
                        "📋 应用的分类规则:",
                        f"   🔍 条件: {best_rule['condition']}",
                        f"   📐 规则: {best_rule['rule']}",
                        f"   📈 准确率: {best_score:.3f}",
                        "",
                        f"🔧 规则类型: 分类规则"
                    ])
                    
                    if len(matched_rules) > 1:
                        explanation_parts.extend([
                            "",
                            f"💡 备注: 发现 {len(matched_rules)} 个匹配规则，已选择质量最高的规则"
                        ])
            else:
                # 回归规则：使用表达式计算
                expression = self._parse_formula(best_rule['rule'])
                
                # 使用EvalWithCompoundTypes计算预测值
                self.evaluator.names = input_data
                prediction = self.evaluator.eval(expression)
                
                # 生成回归规则的解释
                explanation_parts = []
                if explain:
                    explanation_parts.extend([
                        "🎯 回归预测结果:",
                        f"   📊 预测值: {prediction:.3f}",
                        f"   🎲 置信度: {best_score:.1%}",
                        "",
                        "📋 应用的回归规则:",
                        f"   🔍 条件: {best_rule['condition']}",
                        f"   📐 公式: {best_rule['rule']}",
                        f"   📈 质量(R²): {best_score:.3f}",
                        "",
                        "🧮 计算过程:",
                        f"   表达式: {expression}",
                        f"   变量值: {input_data}",
                        f"   📊 最终结果: {prediction:.3f}"
                    ])
                    
                    if len(matched_rules) > 1:
                        explanation_parts.extend([
                            "",
                            f"💡 备注: 发现 {len(matched_rules)} 个匹配规则，已选择质量最高的规则"
                        ])
            
            return {
                'prediction': prediction,
                'confidence': best_score,
                'explanation': '\n'.join(explanation_parts) if explain else '',
                'matched_rules': [rule for _, rule in matched_rules],
                'selected_rule': best_rule,
                'rule_type': rule_type,
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
        """批量预测"""
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

def create_simple_predictor_from_discoverer(discoverer) -> SimpleRulePredictor:
    """
    从规则发现器创建简化预测器
    """
    if not hasattr(discoverer, 'discovered_rules') or not discoverer.discovered_rules:
        print("⚠️ 发现器中没有已发现的规则")
        return SimpleRulePredictor()
    
    rules = discoverer.discovered_rules
    predictor = SimpleRulePredictor(rules)
    
    print(f"✅ 成功创建简化预测器!")
    print(f"   📋 加载规则数: {len(rules)}")
    
    return predictor

if __name__ == "__main__":
    # 示例用法
    print("🔮 === 简化规则预测器演示 === 🔮")
    
    # 创建示例规则
    sample_rules = [
        {
            'condition': 'x <= 30 and y in ["A"]',
            'rule': 'result = 2 * x + 5',
            'cv_r2_score': 0.95,
        },
        {
            'condition': 'x > 30 and y in ["A"]',
            'rule': 'result = 1.5 * x + 10',
            'cv_r2_score': 0.92,
        },
        {
            'condition': 'y in ["B"]',
            'rule': 'result = 3 * x + 2',
            'cv_r2_score': 0.98,
        }
    ]
    
    # 创建预测器
    predictor = SimpleRulePredictor(sample_rules)
    
    # 测试预测
    test_inputs = [
        {'x': 25, 'y': 'A'},
        {'x': 35, 'y': 'A'},
        {'x': 20, 'y': 'B'},
    ]
    
    for i, input_data in enumerate(test_inputs, 1):
        print(f"\n{'='*50}")
        print(f"测试案例 {i}: {input_data}")
        print('='*50)
        
        result = predictor.predict(input_data)
        print(result['explanation']) 