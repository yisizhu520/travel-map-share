#!/usr/bin/env python3
"""
测试条件匹配逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_predictor import RuleBasedPredictor

def test_condition_matching():
    print("🔍 === 条件匹配测试 === 🔍")
    print()
    
    # 创建一个简单的预测器用于测试条件匹配
    predictor = RuleBasedPredictor()
    
    # 测试案例
    test_conditions = [
        "x <= 29.50 且 y ∈ {y1}",
        "x <= 29.50 且 y ∈ {y2}",
        "29.50 < x <= 39.50 且 y ∈ {y1}",
        "29.50 < x <= 39.50 且 y ∈ {y2}",
        "39.50 < x <= 55.50 且 y ∈ {y1}",
        "39.50 < x <= 59.50 且 y ∈ {y2}",
        "x > 55.50 且 y ∈ {y1}",
        "x > 59.50 且 y ∈ {y2}",
    ]
    
    test_inputs = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '描述': '应该匹配规则1'},
        {'x': 25, 'y': 'y2', 'a': 3, 'b': 4, 'c': 5, '描述': '应该匹配规则2'},
        {'x': 35, 'y': 'y1', 'a': 4, 'b': 5, 'c': 6, '描述': '应该匹配规则3'},
        {'x': 35, 'y': 'y2', 'a': 4, 'b': 5, 'c': 6, '描述': '应该匹配规则4'},
        {'x': 45, 'y': 'y1', 'a': 5, 'b': 6, 'c': 7, '描述': '应该匹配规则5'},
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8, '描述': '应该匹配规则7'},
        {'x': 60, 'y': 'y1', 'a': 7, 'b': 8, 'c': 9, '描述': '应该匹配规则6'},
        {'x': 65, 'y': 'y2', 'a': 8, 'b': 9, 'c': 10, '描述': '应该匹配规则8'},
    ]
    
    print("📋 条件匹配测试结果:")
    print("-" * 80)
    
    for i, test_input in enumerate(test_inputs):
        description = test_input.pop('描述')
        
        print(f"\n测试 {i+1}: {description}")
        print(f"输入: x={test_input['x']}, y={test_input['y']}")
        
        # 测试每个条件
        matched_conditions = []
        for j, condition in enumerate(test_conditions, 1):
            result = predictor._evaluate_condition(condition, test_input)
            if result:
                matched_conditions.append(f"规则{j}")
                print(f"✅ 匹配规则{j}: {condition}")
        
        if not matched_conditions:
            print("❌ 没有匹配的规则")
            # 逐个测试条件部分
            print("   详细分析:")
            for j, condition in enumerate(test_conditions, 1):
                parts = condition.split(' 且 ')
                for part in parts:
                    part_result = predictor._evaluate_condition(part, test_input)
                    print(f"   规则{j}部分 '{part}': {'✅' if part_result else '❌'}")
        else:
            print(f"✅ 总计匹配: {', '.join(matched_conditions)}")
        
        test_input['描述'] = description
    
    print("\n" + "=" * 80)
    print("🔍 单独测试边界值")
    print("=" * 80)
    
    # 测试边界值
    boundary_tests = [
        {'x': 29.5, 'y': 'y1', '描述': '边界值29.5'},
        {'x': 29.51, 'y': 'y1', '描述': '刚好大于29.5'},
        {'x': 39.5, 'y': 'y1', '描述': '边界值39.5'},
        {'x': 39.51, 'y': 'y1', '描述': '刚好大于39.5'},
        {'x': 55.5, 'y': 'y1', '描述': '边界值55.5'},
        {'x': 55.51, 'y': 'y1', '描述': '刚好大于55.5'},
        {'x': 59.5, 'y': 'y2', '描述': '边界值59.5'},
        {'x': 59.51, 'y': 'y2', '描述': '刚好大于59.5'},
    ]
    
    for test in boundary_tests:
        description = test.pop('描述')
        print(f"\n🔸 {description}: x={test['x']}, y={test['y']}")
        
        matched = 0
        for j, condition in enumerate(test_conditions, 1):
            if predictor._evaluate_condition(condition, test):
                print(f"   ✅ 匹配规则{j}: {condition}")
                matched += 1
        
        if matched == 0:
            print("   ❌ 没有匹配的规则")
        
        test['描述'] = description

if __name__ == "__main__":
    test_condition_matching() 