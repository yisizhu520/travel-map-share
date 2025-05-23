#!/usr/bin/env python3
"""
测试条件简化功能的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer

def test_condition_simplification():
    print("🔧 === 条件简化功能测试 === 🔧")
    print()
    
    # 创建发现器实例
    discoverer = OptimalConditionalRuleDiscoverer()
    
    # 测试案例
    test_cases = [
        "x <= 39.50 且 x <= 29.50 且 y ∈ {y1}",
        "x > 20.00 且 x > 30.00 且 y ∈ {A}",
        "x <= 50.00 且 x > 25.00 且 x <= 40.00",
        "y ∈ {A} 且 x <= 30.00 且 x <= 25.00 且 z > 10.00",
        "x > 15.00 且 y ∈ {B, C} 且 x > 20.00 且 x <= 35.00",
        "a <= 10.00 且 b > 5.00 且 a <= 8.00 且 b > 7.00 且 c ∈ {high}",
    ]
    
    print("📋 测试条件简化:")
    print("=" * 80)
    
    for i, original_condition in enumerate(test_cases, 1):
        print(f"\n测试案例 {i}:")
        print(f"  原始条件: {original_condition}")
        
        try:
            simplified_condition = discoverer._simplify_condition_string(original_condition)
            print(f"  简化条件: {simplified_condition}")
            
            # 检查是否真的简化了
            if len(simplified_condition) < len(original_condition):
                print(f"  ✅ 成功简化! (减少了 {len(original_condition) - len(simplified_condition)} 个字符)")
            elif simplified_condition != original_condition:
                print(f"  ✨ 条件已优化!")
            else:
                print(f"  📝 条件已是最简形式")
                
        except Exception as e:
            print(f"  ❌ 简化失败: {e}")
    
    print("\n" + "=" * 80)
    print("🎯 简化规则说明:")
    print("  • 多个 <= 条件: 取最小值")
    print("  • 多个 > 条件: 取最大值") 
    print("  • 分类条件: 自动合并")
    print("  • 冲突条件: 智能处理")
    
def test_with_real_data():
    print("\n" + "="*60)
    print("🔍 === 真实数据测试 === 🔍")
    
    # 使用测试数据进行完整的规则发现
    discoverer = OptimalConditionalRuleDiscoverer()
    
    # 测试简单数据
    print("\n📂 测试文件: test_merge.csv")
    try:
        rules = discoverer.discover_optimal_rules("test_merge.csv")
        
        if rules:
            print(f"\n✅ 发现 {len(rules)} 条规则，条件已自动简化")
            for i, rule in enumerate(rules[:3], 1):  # 只显示前3条
                print(f"  {i}. {rule['condition']} → {rule['rule']}")
        else:
            print("❌ 未发现规则")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

if __name__ == "__main__":
    test_condition_simplification()
    test_with_real_data() 