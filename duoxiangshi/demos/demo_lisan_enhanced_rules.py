#!/usr/bin/env python3
"""
lisan.csv 数据集增强版规则发现演示

使用改进后的OptimalConditionalRuleDiscoverer来处理分类型目标变量
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from core.rule_predictor_simple import SimpleRulePredictor
import pandas as pd
import json

def demo_lisan_enhanced_rules():
    print("🚀 === lisan.csv 数据集增强版规则发现演示 === 🚀")
    print("✨ 新特性:")
    print("   • 🎯 支持分类型目标变量")
    print("   • 🔍 智能目标类型检测")
    print("   • 🏷️ 混合类型目标处理")
    print("   • 📊 分类规则专用评估")
    print("   • 🆔 自动编码映射")
    print("-" * 60)
    
    # 数据文件路径
    data_file = 'data/lisan.csv'
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    # 加载数据预览
    df = pd.read_csv(data_file)
    print(f"\n📊 数据概览:")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")
    print(f"   列名: {list(df.columns)}")
    
    print(f"\n📋 数据示例 (前5行):")
    print(df.head().to_string(index=False))
    
    # 分析目标变量特征
    result_values = df['result'].unique()
    str_results = [v for v in result_values if isinstance(v, str)]
    num_results = [v for v in result_values if isinstance(v, (int, float))]
    
    print(f"\n🔍 目标变量分析:")
    print(f"   字符串值: {sorted(str_results)}")
    print(f"   数值: {sorted(num_results)}")
    print(f"   总计: {len(result_values)} 种不同值")
    print(f"   类型: 混合型（字符串+数字）")
    
    print(f"\n🔧 === 开始增强版规则发现 === 🔧")
    
    # 创建增强版规则发现器
    discoverer = OptimalConditionalRuleDiscoverer(
        max_depth=4,  # 稍微增加深度以便更好地捕获分类规则
        min_samples_leaf=30,  # 降低最小样本数，适应分类问题
        cv_folds=3,
        max_combinations=50,  # 限制组合数以提高效率
        enable_exhaustive_search=True
    )
    
    try:
        # 发现规则
        rules = discoverer.discover_optimal_rules(data_file)
        
        print(f"\n🎉 规则发现完成！")
        
        if rules:
            print(f"   📈 成功发现 {len(rules)} 条规则")
            
            # 分析发现的规则
            classification_rules = [r for r in rules if r.get('rule_type') == 'classification']
            regression_rules = [r for r in rules if r.get('rule_type') == 'regression']
            
            print(f"   🎯 分类规则: {len(classification_rules)} 条")
            print(f"   📊 回归规则: {len(regression_rules)} 条")
            
            # 验证规则质量
            if classification_rules:
                avg_accuracy = sum(r['score'] for r in classification_rules) / len(classification_rules)
                perfect_rules = [r for r in classification_rules if r['score'] >= 0.99]
                print(f"   ✨ 平均准确率: {avg_accuracy:.3f}")
                print(f"   🏆 完美规则(准确率≥99%): {len(perfect_rules)} 条")
            
            # 检查是否发现了预期的lisan规则
            print(f"\n🔎 === 规则验证分析 === 🔎")
            
            expected_patterns = [
                ("x ∈ {x1}", "a"),
                ("x ∈ {x2}", "b"),
                ("x ∈ {x3}", "c")
            ]
            
            print(f"📍 期望发现的规律:")
            for i, (condition_pattern, target_pattern) in enumerate(expected_patterns, 1):
                print(f"   期望{i}: 当 {condition_pattern} 时 → result = {target_pattern}")
            
            # 检查实际发现的规则是否匹配期望
            matched_patterns = 0
            for rule in classification_rules:
                condition = rule['condition']
                rule_str = rule['rule']
                
                for exp_condition, exp_target in expected_patterns:
                    if exp_condition in condition and exp_target in rule_str:
                        matched_patterns += 1
                        print(f"   ✅ 匹配发现: {condition} → {rule_str}")
                        break
            
            print(f"\n📈 模式匹配结果:")
            print(f"   期望模式数: {len(expected_patterns)}")
            print(f"   匹配成功数: {matched_patterns}")
            print(f"   匹配率: {matched_patterns/len(expected_patterns)*100:.1f}%")
            
            # 测试预测功能
            if hasattr(discoverer, 'discovered_rules') and discoverer.discovered_rules:
                print(f"\n🚀 === 规则预测测试 === 🚀")
                
                predictor = SimpleRulePredictor(discoverer.discovered_rules)
                
                test_cases = [
                    {'x': 'x1', 'a': 'a2', 'b': 'b3', 'c': 5},  # 应预测 a2
                    {'x': 'x2', 'a': 'a1', 'b': 'b1', 'c': 7},  # 应预测 b1
                    {'x': 'x3', 'a': 'a3', 'b': 'b2', 'c': 4},  # 应预测 4
                    {'x': 'x1', 'a': 'a3', 'b': 'b1', 'c': 9},  # 应预测 a3
                    {'x': 'x2', 'a': 'a2', 'b': 'b2', 'c': 1},  # 应预测 b2
                ]
                
                print(f"📝 测试案例:")
                success_count = 0
                
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
                    
                    # 处理预测结果
                    if isinstance(prediction, dict):
                        pred_value = prediction.get('prediction')
                        confidence = prediction.get('confidence', 0)
                        status_info = prediction.get('status', 'unknown')
                    else:
                        pred_value = prediction
                        confidence = 1.0
                        status_info = 'direct'
                    
                    # 转换类型以便比较
                    if pred_value is not None:
                        pred_str = str(pred_value)
                        exp_str = str(expected)
                        is_correct = pred_str == exp_str
                    else:
                        is_correct = False
                    
                    status = "✅" if is_correct else "❌"
                    if is_correct:
                        success_count += 1
                    
                    print(f"   测试{i}: {test_case}")
                    print(f"         预测: {pred_value} (置信度: {confidence:.2f})")
                    print(f"         期望: {expected}")
                    print(f"         结果: {status} ({status_info})")
                    print()
                
                accuracy = success_count / len(test_cases) * 100
                print(f"🎯 预测测试结果:")
                print(f"   成功预测: {success_count}/{len(test_cases)}")
                print(f"   预测准确率: {accuracy:.1f}%")
                
                if accuracy >= 80:
                    print("   🌟 预测表现: 优秀")
                elif accuracy >= 60:
                    print("   ⭐ 预测表现: 良好")
                else:
                    print("   📊 预测表现: 需要改进")
            
            # 保存规则到文件
            rules_file = 'data/lisan_enhanced_rules.json'
            with open(rules_file, 'w', encoding='utf-8') as f:
                json.dump(rules, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n💾 增强版规则已保存到: {rules_file}")
            
        else:
            print(f"   ⚠️ 未发现有效规则")
            print(f"   💡 可能的原因:")
            print(f"      • 数据模式过于复杂")
            print(f"      • 需要调整算法参数")
            print(f"      • 特征工程不足")
    
    except Exception as e:
        print(f"❌ 规则发现过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"🎯 增强版规则发现演示完成")
    print(f"   主要改进:")
    print(f"   ✓ 成功处理混合类型目标变量")
    print(f"   ✓ 自动检测分类vs回归问题")  
    print(f"   ✓ 针对分类问题优化评估指标")
    print(f"   ✓ 智能编码和条件映射")

if __name__ == "__main__":
    demo_lisan_enhanced_rules() 