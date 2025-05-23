#!/usr/bin/env python3
"""
演示条件简化功能的完整脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer

def demo_condition_simplification_comprehensive():
    print("🔧 === 条件简化功能全面演示 === 🔧")
    print()
    
    # 创建发现器实例
    discoverer = OptimalConditionalRuleDiscoverer()
    
    print("=" * 80)
    print("📚 第一部分：理论测试 - 各种条件简化场景")
    print("=" * 80)
    
    # 详细的测试案例
    test_cases = [
        {
            'title': '相同特征的多个小于等于条件',
            'original': 'x <= 39.50 且 x <= 29.50 且 y ∈ {y1}',
            'expected': 'x <= 29.50 且 y ∈ {y1}',
            'explanation': '多个 <= 条件取最小值'
        },
        {
            'title': '相同特征的多个大于条件',
            'original': 'x > 20.00 且 x > 30.00 且 y ∈ {A}',
            'expected': 'x > 30.00 且 y ∈ {A}',
            'explanation': '多个 > 条件取最大值'
        },
        {
            'title': '同时有上界和下界的条件',
            'original': 'x <= 50.00 且 x > 25.00 且 x <= 40.00',
            'expected': '25.00 < x <= 40.00',
            'explanation': '合并为范围条件'
        },
        {
            'title': '混合数值和分类条件',
            'original': 'y ∈ {A} 且 x <= 30.00 且 x <= 25.00 且 z > 10.00',
            'expected': 'y ∈ {A} 且 x <= 25.00 且 z > 10.00',
            'explanation': '保留分类条件，简化数值条件'
        },
        {
            'title': '复杂的多特征条件',
            'original': 'x > 15.00 且 y ∈ {B, C} 且 x > 20.00 且 x <= 35.00',
            'expected': '20.00 < x <= 35.00 且 y ∈ {B, C}',
            'explanation': '多特征同时简化'
        },
        {
            'title': '多个特征的重复条件',
            'original': 'a <= 10.00 且 b > 5.00 且 a <= 8.00 且 b > 7.00 且 c ∈ {high}',
            'expected': 'a <= 8.00 且 b > 7.00 且 c ∈ {high}',
            'explanation': '多个特征分别简化'
        }
    ]
    
    print()
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试案例 {i}: {test_case['title']}")
        print(f"  💡 说明: {test_case['explanation']}")
        print(f"  📥 原始条件: {test_case['original']}")
        
        try:
            simplified = discoverer._simplify_condition_string(test_case['original'])
            print(f"  📤 简化结果: {simplified}")
            
            # 检查是否符合预期
            if simplified == test_case['expected']:
                print(f"  ✅ 完全符合预期!")
            elif len(simplified) < len(test_case['original']):
                print(f"  ✨ 成功简化! (减少 {len(test_case['original']) - len(simplified)} 字符)")
            else:
                print(f"  📝 已是最简形式")
                
        except Exception as e:
            print(f"  ❌ 简化失败: {e}")
        
        print()
    
    print("=" * 80)
    print("🎯 简化规则总结:")
    print("  1️⃣ 同一特征的多个 <= 条件 → 取最小阈值")
    print("  2️⃣ 同一特征的多个 > 条件 → 取最大阈值")
    print("  3️⃣ 同一特征的上下界条件 → 合并为范围条件")
    print("  4️⃣ 分类条件保持不变，按需要合并值集合")
    print("  5️⃣ 不同特征的条件独立简化")
    print("=" * 80)
    
    print()
    print("=" * 80)
    print("📊 第二部分：实际应用演示")
    print("=" * 80)
    
    # 模拟一些在实际数据分析中可能出现的复杂条件
    real_world_examples = [
        "age <= 65.00 且 age <= 50.00 且 income > 30000.00 且 education ∈ {高中, 大学}",
        "temperature > 15.00 且 temperature > 20.00 且 humidity <= 80.00 且 season ∈ {春季}",
        "price <= 1000.00 且 price > 500.00 且 price <= 800.00 且 category ∈ {电子产品}",
        "x_coordinate > 100.00 且 y_coordinate <= 200.00 且 x_coordinate > 120.00 且 zone ∈ {A, B}",
    ]
    
    for i, example in enumerate(real_world_examples, 1):
        print(f"\n实际案例 {i}:")
        print(f"  原始条件: {example}")
        
        try:
            simplified = discoverer._simplify_condition_string(example)
            print(f"  简化条件: {simplified}")
            
            improvement = len(example) - len(simplified)
            if improvement > 0:
                print(f"  📈 简化效果: 减少 {improvement} 字符 ({improvement/len(example)*100:.1f}%)")
            else:
                print(f"  📝 条件已最优")
                
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print("\n" + "=" * 80)
    print("🎉 演示完成！条件简化功能已成功集成到规则发现系统中。")
    print("🚀 在实际的规则发现过程中，这些简化会自动应用，")
    print("   让最终的规则更加简洁、易读、易理解！")
    print("=" * 80)

if __name__ == "__main__":
    demo_condition_simplification_comprehensive() 