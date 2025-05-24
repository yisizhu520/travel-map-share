#!/usr/bin/env python3
"""
lisan.csv 数据集规则发现演示

这个演示展示了如何在 lisan.csv 数据集上发现条件多项式规则。
数据集具有以下特征：
- x: 分类变量 (x1, x2, x3)
- a: 分类变量 (a1, a2, a3)  
- b: 分类变量 (b1, b2, b3)
- c: 数值变量 (1-9)
- result: 结果变量

预期规则：
1. 当 x = x1 时，result = a
2. 当 x = x2 时，result = b  
3. 当 x = x3 时，result = c
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from core.rule_predictor_simple import SimpleRulePredictor
import pandas as pd
import json

def demo_lisan_rules():
    print("🎯 === lisan.csv 数据集规则发现演示 === 🎯")
    
    # 数据文件路径
    data_file = '../data/lisan.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    # 加载数据
    df = pd.read_csv(data_file)
    print(f"\n📊 数据概览:")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   列名: {list(df.columns)}")
    
    print(f"\n📋 数据示例 (前5行):")
    print(df.head().to_string(index=False))
    
    print(f"\n🔍 === 开始规则发现 === 🔍")
    
    # 创建规则发现器
    discoverer = OptimalConditionalRuleDiscoverer()
    
    # 发现规则
    try:
        rules = discoverer.discover_optimal_rules(data_file)
        
        print(f"\n🎉 规则发现完成！")
        print(f"   发现的规则数量: {len(rules)}")
        
        # 显示发现的规则
        print(f"\n📜 发现的规则:")
        for i, rule in enumerate(rules, 1):
            condition = rule['condition']
            polynomial = rule['polynomial']
            coverage = rule.get('coverage', 0)
            accuracy = rule.get('accuracy', 0)
            
            print(f"   规则{i}: {condition} → {polynomial}")
            print(f"          覆盖率: {coverage:.2f}%, 准确率: {accuracy:.2f}%")
        
        # 分析规则是否符合预期
        print(f"\n🔎 === 规则分析 === 🔎")
        
        expected_rules = [
            ("x ∈ {x1}", "a"),
            ("x ∈ {x2}", "b"), 
            ("x ∈ {x3}", "c")
        ]
        
        print(f"📍 预期规则:")
        for i, (cond, poly) in enumerate(expected_rules, 1):
            print(f"   预期{i}: {cond} → result = {poly}")
        
        # 检查是否发现了预期规则
        found_expected = 0
        for rule in rules:
            condition = rule['condition']
            polynomial = rule['polynomial']
            
            for exp_cond, exp_poly in expected_rules:
                if exp_cond in condition and exp_poly in polynomial:
                    found_expected += 1
                    print(f"✅ 找到预期规则: {condition} → {polynomial}")
                    break
        
        print(f"\n📈 规则匹配结果:")
        print(f"   预期规则数: {len(expected_rules)}")
        print(f"   发现匹配数: {found_expected}")
        print(f"   匹配率: {found_expected/len(expected_rules)*100:.1f}%")
        
        # 测试规则预测
        print(f"\n🚀 === 规则预测测试 === 🚀")
        
        predictor = SimpleRulePredictor(rules)
        
        # 创建测试案例
        test_cases = [
            {'x': 'x1', 'a': 'a2', 'b': 'b3', 'c': 5},  # 应预测 a2
            {'x': 'x2', 'a': 'a1', 'b': 'b1', 'c': 7},  # 应预测 b1
            {'x': 'x3', 'a': 'a3', 'b': 'b2', 'c': 4},  # 应预测 4
        ]
        
        print(f"📝 测试案例:")
        for i, test_case in enumerate(test_cases, 1):
            prediction = predictor.predict(test_case)
            
            # 计算期望结果
            x_val = test_case['x']
            if x_val == 'x1':
                expected = test_case['a']
            elif x_val == 'x2':
                expected = test_case['b']
            elif x_val == 'x3':
                expected = test_case['c']
            else:
                expected = None
            
            status = "✅" if prediction == expected else "❌"
            
            print(f"   测试{i}: {test_case} → 预测={prediction}, 期望={expected} {status}")
        
        # 保存规则到文件
        rules_file = '../data/lisan_rules.json'
        with open(rules_file, 'w', encoding='utf-8') as f:
            json.dump(rules, f, ensure_ascii=False, indent=2)
        print(f"\n💾 规则已保存到: {rules_file}")
        
    except Exception as e:
        print(f"❌ 规则发现过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_lisan_rules() 