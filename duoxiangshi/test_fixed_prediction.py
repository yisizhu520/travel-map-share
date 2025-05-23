#!/usr/bin/env python3
"""
测试修复后的预测功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rule_predictor_simple import SimpleRulePredictor

def test_fixed_prediction():
    print("🔧 === 测试修复后的预测功能 === 🔧")
    print()
    
    # 创建正确的测试规则（修复后的版本应该能生成这样的规则）
    fixed_rules = [
        {
            'condition': 'x <= 29.50 且 y ∈ {y1}',
            'rule': 'result = a',
            'cv_r2_score': 1.0,
            'sample_count': 100
        },
        {
            'condition': 'x <= 29.50 且 y ∈ {y2}',
            'rule': 'result = 2 * a',
            'cv_r2_score': 1.0,
            'sample_count': 80
        },
        {
            'condition': '29.50 < x <= 39.50 且 y ∈ {y1}',
            'rule': 'result = a + b',
            'cv_r2_score': 1.0,
            'sample_count': 120
        },
        {
            'condition': '29.50 < x <= 39.50 且 y ∈ {y2}',
            'rule': 'result = 2 * a + b',
            'cv_r2_score': 1.0,
            'sample_count': 90
        },
        {
            'condition': 'x > 39.50 且 y ∈ {y1}',
            'rule': 'result = a + b + c',
            'cv_r2_score': 1.0,
            'sample_count': 75
        },
        {
            'condition': 'x > 39.50 且 y ∈ {y2}',
            'rule': 'result = 2 * a + b + c',
            'cv_r2_score': 1.0,
            'sample_count': 85
        }
    ]
    
    # 创建预测器
    predictor = SimpleRulePredictor(fixed_rules)
    
    print("📋 使用的修复规则:")
    for i, rule in enumerate(fixed_rules, 1):
        print(f"   {i}. {rule['condition']} → {rule['rule']}")
    print()
    
    # 测试案例
    test_cases = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '描述': '中等值+类别y1', '期望': 3},
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8, '描述': '大值+类别y2', '期望': 27},  # 2*6 + 7 + 8 = 27
        {'x': 15, 'y': 'y1', 'a': 1, 'b': 2, 'c': 3, '描述': '小值+类别y1', '期望': 1},
        {'x': 60, 'y': 'y1', 'a': 5, 'b': 6, 'c': 7, '描述': '超大值+类别y1', '期望': 18},  # 5 + 6 + 7 = 18
        {'x': 35, 'y': 'y2', 'a': 4, 'b': 5, 'c': 6, '描述': '中大值+类别y2', '期望': 13},  # 2*4 + 5 = 13
    ]
    
    print("🎯 预测测试结果:")
    print("-" * 60)
    
    success_count = 0
    for i, test_case in enumerate(test_cases, 1):
        desc = test_case.pop('描述')
        expected = test_case.pop('期望')
        
        print(f"\n🔍 测试 {i}: {desc}")
        print(f"   输入: {test_case}")
        print(f"   期望: {expected}")
        
        result = predictor.predict(test_case, explain=False)
        
        if result['status'] == 'success':
            prediction = result['prediction']
            print(f"   ✅ 预测值: {prediction}")
            print(f"   🎲 置信度: {result['confidence']:.1%}")
            print(f"   📋 使用规则: {result['selected_rule']['condition']}")
            
            if abs(prediction - expected) < 0.01:
                print(f"   🎉 预测正确！")
                success_count += 1
            else:
                print(f"   ❌ 预测错误！期望 {expected}，得到 {prediction}")
        else:
            print(f"   ❌ 预测失败: {result['explanation']}")
        
        # 恢复字段
        test_case['描述'] = desc
        test_case['期望'] = expected
    
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    print(f"   测试案例总数: {len(test_cases)}")
    print(f"   成功案例: {success_count}")
    print(f"   成功率: {success_count/len(test_cases)*100:.1f}%")
    
    if success_count == len(test_cases):
        print("   🎉 所有测试案例都通过了！")
        print("   ✅ SimpleRulePredictor 工作正常")
    else:
        print(f"   ⚠️ 还有 {len(test_cases) - success_count} 个案例需要修复")
    
    return success_count == len(test_cases)

if __name__ == "__main__":
    test_fixed_prediction() 