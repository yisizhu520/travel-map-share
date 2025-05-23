#!/usr/bin/env python3
"""
测试简化版本的条件转换逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_predictor_simple import SimpleRulePredictor
from simpleeval import EvalWithCompoundTypes

def test_condition_conversion():
    print("🔍 === 测试条件转换逻辑 === 🔍")
    print()
    
    predictor = SimpleRulePredictor()
    evaluator = EvalWithCompoundTypes()
    
    test_cases = [
        {
            'original': 'x <= 29.50 且 y ∈ {y1}',
            'input': {'x': 25, 'y': 'y1', 'a': 3, 'b': 4},
            'expected': True
        },
        {
            'original': '29.50 < x <= 39.50 且 y ∈ {y2}',
            'input': {'x': 35, 'y': 'y2', 'a': 4, 'b': 5},
            'expected': True
        },
        {
            'original': 'x > 39.50 且 y ∈ {y2}',
            'input': {'x': 45, 'y': 'y2', 'a': 6, 'b': 7},
            'expected': True
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"测试 {i}:")
        print(f"  原始条件: {test['original']}")
        print(f"  输入数据: {test['input']}")
        
        # 转换条件
        converted = predictor._convert_condition_to_eval_format(
            test['original'], test['input']
        )
        print(f"  转换后: {converted}")
        
        # 测试评估
        try:
            # 设置变量并评估
            evaluator.names = test['input']
            result = evaluator.eval(converted)
            print(f"  评估结果: {result}")
            print(f"  期望结果: {test['expected']}")
            print(f"  ✅ 成功" if result == test['expected'] else "❌ 失败")
        except Exception as e:
            print(f"  ❌ 评估失败: {e}")
        
        print()

if __name__ == "__main__":
    test_condition_conversion() 