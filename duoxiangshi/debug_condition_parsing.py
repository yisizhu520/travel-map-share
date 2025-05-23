#!/usr/bin/env python3
"""
调试条件解析问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_predictor_simple import SimpleRulePredictor

def debug_condition_parsing():
    print("🐛 === 调试条件解析问题 === 🐛")
    print()
    
    # 创建预测器
    predictor = SimpleRulePredictor()
    
    # 测试有问题的条件
    problematic_conditions = [
        "y ∈ {y1} 且 x > 55.50 且 39.50 < x <= 55.50",
        "y ∈ {y2} 且 x > 59.50 且 39.50 < x <= 59.50"
    ]
    
    test_inputs = [
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8},
        {'x': 60, 'y': 'y1', 'a': 5, 'b': 6, 'c': 7}
    ]
    
    print("🔍 分析有问题的条件:")
    for i, condition in enumerate(problematic_conditions, 1):
        print(f"\n条件 {i}: {condition}")
        
        # 转换条件
        try:
            converted = predictor._convert_condition_to_eval_format(condition, test_inputs[0])
            print(f"转换后: {converted}")
            
            # 分析逻辑矛盾
            if "x > 55.50" in converted and "x <= 55.50" in converted:
                print("❌ 发现逻辑矛盾: x > 55.50 且 x <= 55.50")
            if "x > 59.50" in converted and "x <= 59.50" in converted:
                print("❌ 发现逻辑矛盾: x > 59.50 且 x <= 59.50")
                
        except Exception as e:
            print(f"❌ 转换失败: {e}")
    
    print("\n🔧 分析应该有的正确条件:")
    correct_conditions = [
        "39.50 < x <= 55.50 且 y ∈ {y1}",  # 应该覆盖x=45的y1情况
        "39.50 < x <= 59.50 且 y ∈ {y2}",  # 应该覆盖x=45的y2情况  
        "x > 55.50 且 y ∈ {y1}",           # 应该覆盖x=60的y1情况
        "x > 59.50 且 y ∈ {y2}"            # 应该覆盖x=60的y2情况
    ]
    
    for i, condition in enumerate(correct_conditions, 1):
        print(f"\n正确条件 {i}: {condition}")
        converted = predictor._convert_condition_to_eval_format(condition, test_inputs[0])
        print(f"转换后: {converted}")
        
        # 测试是否匹配我们的测试用例
        for j, test_input in enumerate(test_inputs):
            try:
                predictor.evaluator.names = test_input
                result = predictor.evaluator.eval(converted)
                print(f"测试输入{j+1} {test_input}: {'✅匹配' if result else '❌不匹配'}")
            except Exception as e:
                print(f"测试输入{j+1} {test_input}: ❌错误 {e}")

if __name__ == "__main__":
    debug_condition_parsing() 