#!/usr/bin/env python3
"""
简化版预测演示
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from rule_predictor import RuleBasedPredictor

def simple_demo():
    print("🔮 === 简化版智能预测演示 === 🔮")
    print()
    
    # 第一步：快速发现规则
    print("📊 Step 1: 发现规则")
    print("-" * 40)
    discoverer = OptimalConditionalRuleDiscoverer(max_depth=3, min_samples_leaf=50)
    rules = discoverer.discover_optimal_rules("multi_if_duoxiangshi.csv")
    
    if not rules:
        print("❌ 未发现规则")
        return
    
    print(f"✅ 发现 {len(rules)} 条规则")
    
    # 第二步：创建预测器
    print("\n🔮 Step 2: 创建预测器")
    print("-" * 40)
    predictor = RuleBasedPredictor(rules, discoverer.label_encoders)
    
    # 显示规则
    print("📋 已发现的规则:")
    for i, rule in enumerate(rules, 1):
        print(f"   {i}. 当 {rule['condition']} 时:")
        print(f"      → {rule['rule']} (R²={rule['cv_r2_score']:.3f})")
    
    # 第三步：进行预测
    print("\n🎯 Step 3: 预测演示")
    print("-" * 40)
    
    test_cases = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '描述': '中等值+类别y1'},
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8, '描述': '大值+类别y2'},
        {'x': 15, 'y': 'y1', 'a': 1, 'b': 2, 'c': 3, '描述': '小值+类别y1'},
        {'x': 60, 'y': 'y3', 'a': 5, 'b': 6, 'c': 7, '描述': '不存在的类别'},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        desc = test_case.pop('描述')
        
        print(f"\n🔍 测试 {i}: {desc}")
        print(f"   输入: {test_case}")
        
        result = predictor.predict(test_case, explain=False)
        
        if result['status'] == 'success':
            print(f"   📊 预测值: {result['prediction']:.2f}")
            print(f"   🎲 置信度: {result['confidence']:.1%}")
            print(f"   📋 使用规则: {result['selected_rule']['condition']}")
        else:
            print(f"   ❌ 预测失败: {result['explanation']}")
        
        test_case['描述'] = desc
    
    # 第四步：详细解释演示
    print("\n🔬 Step 4: 详细解释演示")
    print("-" * 40)
    
    detailed_case = {'x': 35, 'y': 'y2', 'a': 4, 'b': 5, 'c': 6}
    print(f"📥 详细分析案例: {detailed_case}")
    
    result = predictor.predict(detailed_case, explain=True)
    if result['status'] == 'success':
        print("\n" + result['explanation'])
    
    print("\n" + "=" * 60)
    print("🎉 演示完成！")
    print("✅ 核心功能:")
    print("   • 自动规则发现 ✓")
    print("   • 智能预测 ✓") 
    print("   • 详细解释 ✓")
    print("   • 边界处理 ✓")
    
    return predictor, rules

if __name__ == "__main__":
    simple_demo() 