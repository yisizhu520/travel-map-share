#!/usr/bin/env python3
"""
使用修复后的规则发现结果进行预测测试
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from rule_predictor_simple import SimpleRulePredictor

def test_prediction_with_fixed_rules():
    print("🧪 === 测试修复后的完整工作流：规则发现 + 预测 === 🧪")
    print()
    
    # 1. 创建测试数据
    print("📊 创建测试数据...")
    np.random.seed(42)
    data = []
    
    for i in range(1000):
        x = np.random.uniform(10, 70)
        y = np.random.choice(['y1', 'y2'])
        a = np.random.randint(1, 10)
        b = np.random.randint(1, 10)
        c = np.random.randint(1, 10)
        
        # 根据已知规则生成result
        if x <= 29.5:
            if y == 'y1':
                result = a
            else:  # y2
                result = 2 * a
        elif x <= 39.5:
            if y == 'y1':
                result = a + b
            else:  # y2
                result = 2 * a + b
        else:  # x > 39.5
            if y == 'y1':
                result = a + b + c
            else:  # y2
                result = 2 * a + b + c
        
        data.append({'x': x, 'y': y, 'a': a, 'b': b, 'c': c, 'result': result})
    
    df = pd.DataFrame(data)
    csv_file = "test_prediction_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"✅ 创建了测试文件: {csv_file}, 样本数: {len(df)}")
    
    # 2. 发现规则
    print("\n🔍 发现规则...")
    discoverer = OptimalConditionalRuleDiscoverer(
        max_depth=3,
        min_samples_leaf=30,
        enable_exhaustive_search=True,
        max_combinations=50
    )
    
    discovered_rules = discoverer.discover_optimal_rules(
        csv_file_path=csv_file,
        target_col="result",
        manual_split_features=['x', 'y'],
        manual_poly_features=['a', 'b', 'c']
    )
    
    print(f"\n📋 发现了 {len(discovered_rules)} 条规则")
    
    if not discovered_rules:
        print("❌ 没有发现规则，测试失败")
        return False
    
    # 3. 创建预测器
    print("\n🤖 创建预测器...")
    predictor = SimpleRulePredictor(discovered_rules)
    
    # 4. 测试预测
    print("\n🎯 测试预测...")
    test_cases = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '期望': 3, '描述': '低范围+y1'},
        {'x': 35, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '期望': 7, '描述': '中范围+y1'},
        {'x': 45, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '期望': 12, '描述': '高范围+y1'},
        {'x': 25, 'y': 'y2', 'a': 3, 'b': 4, 'c': 5, '期望': 6, '描述': '低范围+y2'},
        {'x': 35, 'y': 'y2', 'a': 3, 'b': 4, 'c': 5, '期望': 10, '描述': '中范围+y2'},
        {'x': 45, 'y': 'y2', 'a': 3, 'b': 4, 'c': 5, '期望': 15, '描述': '高范围+y2'},
    ]
    
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
            print(f"   📋 匹配规则: {result['selected_rule']['condition']}")
            
            if abs(prediction - expected) < 0.1:  # 允许小的浮点误差
                print(f"   🎉 预测正确！")
                success_count += 1
            else:
                print(f"   ❌ 预测错误！期望 {expected}，得到 {prediction}")
        else:
            print(f"   ❌ 预测失败: {result['explanation']}")
        
        # 恢复字段用于下次迭代
        test_case['描述'] = desc
        test_case['期望'] = expected
    
    print("\n" + "=" * 60)
    print("📊 完整工作流测试总结:")
    print(f"   规则发现: {'✅ 成功' if discovered_rules else '❌ 失败'}")
    print(f"   测试案例总数: {len(test_cases)}")
    print(f"   预测成功: {success_count}")
    print(f"   成功率: {success_count/len(test_cases)*100:.1f}%")
    
    if success_count == len(test_cases):
        print("   🎉 完整工作流测试全部通过！")
        print("   ✅ 修复彻底成功：规则发现 → 预测都正常工作")
        return True
    else:
        print(f"   ⚠️ 还有 {len(test_cases) - success_count} 个案例需要处理")
        return False

if __name__ == "__main__":
    success = test_prediction_with_fixed_rules()
    if success:
        print("\n🎊 完整修复验证成功！问题已彻底解决！")
    else:
        print("\n❌ 还存在一些问题需要进一步修复") 