#!/usr/bin/env python3
"""
调试条件评估逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_predictor import RuleBasedPredictor

def debug_step_by_step():
    print("🐛 === 逐步调试条件评估 === 🐛")
    
    predictor = RuleBasedPredictor()
    
    # 手动实现条件评估以便调试
    def debug_evaluate_condition(condition_str, input_data):
        print(f"    🔍 开始评估条件: '{condition_str}'")
        print(f"    📥 输入数据: {input_data}")
        
        if not condition_str or condition_str.strip() == "":
            print(f"    ✅ 空条件，返回 True")
            return True
        
        # 按 "且" 分割条件
        conditions = condition_str.split(' 且 ')
        print(f"    📋 分割后条件: {conditions}")
        
        for i, condition in enumerate(conditions):
            condition = condition.strip()
            print(f"    🔸 评估子条件 {i+1}: '{condition}'")
            
            if ' ∈ ' in condition:
                print(f"      🏷️ 分类条件")
                feature, values_str = condition.split(' ∈ ')
                feature = feature.strip()
                
                values_str = values_str.strip().replace('{', '').replace('}', '')
                allowed_values = [v.strip() for v in values_str.split(',')]
                
                print(f"      特征: {feature}")
                print(f"      允许值: {allowed_values}")
                
                if feature not in input_data:
                    print(f"      ❌ 特征 '{feature}' 不在输入数据中")
                    return False
                
                input_value = str(input_data[feature])
                print(f"      输入值: '{input_value}'")
                
                if input_value not in allowed_values:
                    print(f"      ❌ 输入值不在允许值中")
                    return False
                else:
                    print(f"      ✅ 分类条件满足")
                    
            elif '<=' in condition:
                print(f"      📊 数值条件 (<=)")
                feature, threshold_str = condition.split('<=')
                feature = feature.strip()
                threshold = float(threshold_str.strip())
                
                print(f"      特征: {feature}")
                print(f"      阈值: {threshold}")
                
                if feature not in input_data:
                    print(f"      ❌ 特征 '{feature}' 不在输入数据中")
                    return False
                
                try:
                    input_value = float(input_data[feature])
                    print(f"      输入值: {input_value}")
                    print(f"      比较: {input_value} <= {threshold} = {input_value <= threshold}")
                    
                    if input_value > threshold:
                        print(f"      ❌ 数值条件不满足: {input_value} > {threshold}")
                        return False
                    else:
                        print(f"      ✅ 数值条件满足: {input_value} <= {threshold}")
                except (ValueError, TypeError) as e:
                    print(f"      ❌ 类型转换错误: {e}")
                    return False
            
            else:
                print(f"      ⚠️ 未识别的条件类型")
        
        print(f"    ✅ 所有子条件都满足，返回 True")
        return True
    
    # 测试案例
    test_cases = [
        {'condition': 'x <= 29.50', 'input': {'x': 25}},
        {'condition': 'x <= 29.50 且 y ∈ {y1}', 'input': {'x': 25, 'y': 'y1'}},
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*50}")
        print(f"测试 {i}:")
        
        # 使用原始方法
        original_result = predictor._evaluate_condition(test['condition'], test['input'])
        print(f"原始方法结果: {original_result}")
        
        print(f"\n调试步骤:")
        # 使用调试方法
        debug_result = debug_evaluate_condition(test['condition'], test['input'])
        print(f"调试方法结果: {debug_result}")
        
        print(f"\n结果对比: {'✅ 一致' if original_result == debug_result else '❌ 不一致'}")

if __name__ == "__main__":
    debug_step_by_step() 