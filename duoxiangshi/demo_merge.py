#!/usr/bin/env python3
"""
演示规则合并功能的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer

def demo_rule_merging():
    print("🔗 === 规则合并功能演示 === 🔗")
    print()
    
    # 创建发现器实例
    discoverer = OptimalConditionalRuleDiscoverer()
    
    # 模拟一些发现的规则，其中有些是相同的
    demo_rules = [
        {
            'condition': 'x <= 30.00 且 y ∈ {A}',
            'rule': 'result = a + b',
            'cv_r2_score': 0.95,
            'sample_count': 100
        },
        {
            'condition': 'x > 30.00 且 y ∈ {A}',
            'rule': 'result = a + b',  # 相同规则
            'cv_r2_score': 0.92,
            'sample_count': 80
        },
        {
            'condition': 'x <= 25.00 且 y ∈ {B}',
            'rule': 'result = 2 * a + b',
            'cv_r2_score': 0.98,
            'sample_count': 90
        },
        {
            'condition': 'x > 25.00 且 x <= 50.00 且 y ∈ {B}',
            'rule': 'result = 2 * a + b',  # 相同规则
            'cv_r2_score': 0.96,
            'sample_count': 110
        },
        {
            'condition': 'x > 50.00 且 y ∈ {B}',
            'rule': 'result = 2 * a + b',  # 相同规则
            'cv_r2_score': 0.94,
            'sample_count': 70
        },
        {
            'condition': 'y ∈ {C}',
            'rule': 'result = a + 2 * b',
            'cv_r2_score': 0.89,
            'sample_count': 120
        }
    ]
    
    print("📋 原始规则列表:")
    for i, rule in enumerate(demo_rules, 1):
        print(f"  {i}. {rule['condition']} → {rule['rule']} (R²={rule['cv_r2_score']:.3f}, 样本={rule['sample_count']})")
    
    print()
    print("=" * 80)
    
    # 执行合并
    merged_rules = discoverer._merge_similar_rules(demo_rules)
    
    print()
    print("📋 合并后规则列表:")
    for i, rule in enumerate(merged_rules, 1):
        merge_info = f" (合并自{rule.get('merged_from', 1)}条规则)" if rule.get('merged_from', 1) > 1 else ""
        print(f"  {i}. {rule['condition']} → {rule['rule']} (R²={rule['cv_r2_score']:.3f}, 样本={rule['sample_count']}){merge_info}")
    
    print()
    print("🎯 合并效果:")
    print(f"   • 原始规则数: {len(demo_rules)}")
    print(f"   • 合并后规则数: {len(merged_rules)}")
    print(f"   • 简化率: {(len(demo_rules) - len(merged_rules)) / len(demo_rules) * 100:.1f}%")

if __name__ == "__main__":
    demo_rule_merging() 