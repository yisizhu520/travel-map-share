#!/usr/bin/env python3
"""
覆盖率改进效果验证测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from core.rule_predictor_simple import SimpleRulePredictor
import pandas as pd

def test_coverage_improvement():
    print("🧪 === 覆盖率改进效果验证 === 🧪")
    print()
    
    # 加载测试数据
    data_file = 'data/lisan.csv'
    df = pd.read_csv(data_file)
    total_samples = len(df)
    
    print(f"📊 测试数据: {data_file}")
    print(f"   总样本数: {total_samples}")
    print(f"   数据列: {list(df.columns)}")
    print()
    
    # 创建规则发现器
    discoverer = OptimalConditionalRuleDiscoverer(
        max_depth=4, 
        min_samples_leaf=30,
        enable_exhaustive_search=True
    )
    
    print("🔍 开始规则发现...")
    rules = discoverer.discover_optimal_rules(data_file)
    
    if rules:
        # 计算覆盖率
        total_coverage = sum(rule['sample_count'] for rule in rules)
        coverage_rate = total_coverage / total_samples
        
        print(f"\n📈 === 测试结果 === 📈")
        print(f"✅ 发现规则数: {len(rules)}")
        print(f"✅ 覆盖样本数: {total_coverage}/{total_samples}")
        print(f"✅ 覆盖率: {coverage_rate:.1%}")
        
        # 检查规则质量
        perfect_rules = [r for r in rules if r['score'] >= 0.99]
        avg_score = sum(r['score'] for r in rules) / len(rules)
        
        print(f"✅ 平均准确率: {avg_score:.1%}")
        print(f"✅ 完美规则数(≥99%): {len(perfect_rules)}/{len(rules)}")
        
        # 验证预期模式
        expected_patterns = ['x1', 'x2', 'x3']
        found_patterns = set()
        
        for rule in rules:
            condition = rule['condition']
            for pattern in expected_patterns:
                if pattern in condition:
                    found_patterns.add(pattern)
        
        print(f"✅ 发现模式: {sorted(found_patterns)}")
        print(f"✅ 模式完整性: {len(found_patterns)}/{len(expected_patterns)}")
        
        # 预测测试
        print(f"\n🎯 === 预测验证 === 🎯")
        predictor = SimpleRulePredictor(rules)
        
        test_cases = [
            {'x': 'x1', 'a': 'a1', 'b': 'b1', 'c': 1, 'expected': 'a1'},
            {'x': 'x2', 'a': 'a2', 'b': 'b2', 'c': 2, 'expected': 'b2'},
            {'x': 'x3', 'a': 'a3', 'b': 'b3', 'c': 3, 'expected': 3},
        ]
        
        correct_predictions = 0
        for i, case in enumerate(test_cases, 1):
            expected = case.pop('expected')
            result = predictor.predict(case)
            
            if result['status'] == 'success':
                prediction = result['prediction']
                is_correct = str(prediction) == str(expected)
                correct_predictions += is_correct
                
                status = "✅" if is_correct else "❌"
                print(f"   测试{i}: 预测={prediction}, 期望={expected} {status}")
            else:
                print(f"   测试{i}: 预测失败 ❌")
            
            case['expected'] = expected  # 恢复数据
        
        prediction_accuracy = correct_predictions / len(test_cases)
        print(f"\n📊 预测准确率: {prediction_accuracy:.1%}")
        
        # 总体评价
        print(f"\n🏆 === 改进效果评价 === 🏆")
        if coverage_rate >= 0.9 and prediction_accuracy >= 0.8:
            print("🌟 改进效果: 优秀")
            print("   - 覆盖率 ≥ 90%")
            print("   - 预测准确率 ≥ 80%")
        elif coverage_rate >= 0.7 and prediction_accuracy >= 0.6:
            print("⭐ 改进效果: 良好")
            print("   - 覆盖率 ≥ 70%")
            print("   - 预测准确率 ≥ 60%")
        else:
            print("📊 改进效果: 需要进一步优化")
            
        return {
            'coverage_rate': coverage_rate,
            'prediction_accuracy': prediction_accuracy,
            'num_rules': len(rules),
            'avg_score': avg_score
        }
    else:
        print("❌ 未发现任何规则")
        return None

if __name__ == "__main__":
    test_coverage_improvement() 