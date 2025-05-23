#!/usr/bin/env python3
"""
完整工作流程演示：从规则发现到智能预测
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from rule_predictor import RuleBasedPredictor
import pandas as pd

def complete_workflow_demo():
    print("🚀 === 完整工作流程演示：规则发现 → 智能预测 === 🚀")
    print()
    
    # =======================================
    # 第一步：规则发现
    # =======================================
    print("📊 第一步：规则发现")
    print("=" * 60)
    
    # 创建规则发现器
    discoverer = OptimalConditionalRuleDiscoverer(
        max_depth=2,
        min_samples_leaf=10,
        enable_exhaustive_search=True
    )
    
    # 使用测试数据进行规则发现
    print("📂 使用数据文件: test_merge.csv")
    rules = discoverer.discover_optimal_rules("test_merge.csv")
    
    if not rules:
        print("❌ 规则发现失败，无法继续演示")
        return
    
    print(f"\n✅ 规则发现完成！发现 {len(rules)} 条高质量规则")
    
    # =======================================
    # 第二步：创建预测器
    # =======================================
    print("\n" + "=" * 60)
    print("🔮 第二步：创建智能预测器")
    print("=" * 60)
    
    # 从发现器创建预测器
    predictor = RuleBasedPredictor(rules, discoverer.label_encoders)
    
    print(f"✅ 预测器创建成功！")
    print(f"   📋 加载规则数: {len(predictor.rules)}")
    print(f"   🏷️ 分类特征: {list(predictor.label_encoders.keys())}")
    
    # =======================================
    # 第三步：单次预测演示
    # =======================================
    print("\n" + "=" * 60)
    print("🎯 第三步：单次预测演示")
    print("=" * 60)
    
    # 准备测试数据
    test_cases = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5, '说明': '中等x值，类别y1'},
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8, '说明': '较大x值，类别y2'},
        {'x': 15, 'y': 'y1', 'a': 1, 'b': 2, 'c': 3, '说明': '较小x值，类别y1'},
        {'x': 55, 'y': 'y2', 'a': 8, 'b': 9, 'c': 10, '说明': '大x值，类别y2'},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        description = test_case.pop('说明')
        
        print(f"\n🔍 预测案例 {i}: {description}")
        print(f"   输入数据: {test_case}")
        print("-" * 50)
        
        # 进行预测
        result = predictor.predict(test_case, explain=True)
        
        if result['status'] == 'success':
            print(result['explanation'])
            print(f"\n   📊 预测结果: {result['prediction']:.2f}")
            print(f"   🎲 置信度: {result['confidence']:.1%}")
        else:
            print(f"   ❌ 预测失败: {result['explanation']}")
        
        # 重新添加说明（如果需要后续使用）
        test_case['说明'] = description
    
    # =======================================
    # 第四步：批量预测演示
    # =======================================
    print("\n" + "=" * 60)
    print("📊 第四步：批量预测演示")
    print("=" * 60)
    
    # 创建批量测试数据
    batch_data = pd.DataFrame([
        {'x': 20, 'y': 'y1', 'a': 2, 'b': 3, 'c': 4},
        {'x': 30, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5},
        {'x': 40, 'y': 'y2', 'a': 4, 'b': 5, 'c': 6},
        {'x': 50, 'y': 'y2', 'a': 5, 'b': 6, 'c': 7},
        {'x': 60, 'y': 'y1', 'a': 6, 'b': 7, 'c': 8},
    ])
    
    print("📋 批量预测数据:")
    print(batch_data.to_string(index=False))
    
    print("\n🔮 执行批量预测...")
    batch_results = predictor.predict_batch(batch_data, explain=False)
    
    print("\n📊 批量预测结果:")
    # 只显示关键列
    display_columns = ['x', 'y', 'a', 'b', 'c', 'prediction', 'confidence', 'status']
    print(batch_results[display_columns].to_string(index=False))
    
    # =======================================
    # 第五步：详细分析演示
    # =======================================
    print("\n" + "=" * 60)
    print("🔬 第五步：详细预测分析")
    print("=" * 60)
    
    # 选择一个案例进行详细分析
    detailed_case = {'x': 35, 'y': 'y2', 'a': 4, 'b': 5, 'c': 6}
    
    print("🔍 选择案例进行详细分析:")
    print(f"   输入: {detailed_case}")
    
    # 获取详细解释
    detailed_explanation = predictor.explain_prediction_details(detailed_case)
    print(detailed_explanation)
    
    # =======================================
    # 第六步：边界情况测试
    # =======================================
    print("\n" + "=" * 60)
    print("⚠️ 第六步：边界情况测试")
    print("=" * 60)
    
    # 测试不匹配的情况
    edge_cases = [
        {'x': 100, 'y': 'y3', 'a': 1, 'b': 2, 'c': 3, '描述': '不存在的类别'},
        {'x': -10, 'y': 'y1', 'a': 1, 'b': 2, 'c': 3, '描述': '超出训练数据范围'},
        {'x': 25, 'z': 'unknown', '描述': '缺少必要特征'},
    ]
    
    for i, edge_case in enumerate(edge_cases, 1):
        description = edge_case.pop('描述')
        
        print(f"\n⚠️ 边界测试 {i}: {description}")
        print(f"   输入: {edge_case}")
        
        result = predictor.predict(edge_case, explain=False)
        print(f"   状态: {result['status']}")
        
        if result['status'] == 'success':
            print(f"   结果: {result['prediction']:.2f}")
        else:
            print(f"   说明: {result['explanation']}")
        
        edge_case['描述'] = description
    
    # =======================================
    # 总结
    # =======================================
    print("\n" + "=" * 60)
    print("🎉 演示完成总结")
    print("=" * 60)
    
    print("✅ 演示内容回顾:")
    print("   1️⃣ 自动规则发现：从数据中发现条件规则")
    print("   2️⃣ 智能预测器：基于规则进行预测")
    print("   3️⃣ 单次预测：详细的预测过程和解释")
    print("   4️⃣ 批量预测：高效处理多个样本")
    print("   5️⃣ 详细分析：完整的预测分析报告")
    print("   6️⃣ 边界处理：处理异常和边界情况")
    
    print("\n🚀 核心优势:")
    print("   ✓ 规则透明：每个预测都有清晰的解释")
    print("   ✓ 质量保证：基于交叉验证的置信度")
    print("   ✓ 易于理解：用户友好的解释界面")
    print("   ✓ 鲁棒性强：优雅处理各种边界情况")
    
    print("\n💡 应用场景:")
    print("   • 业务规则挖掘和解释")
    print("   • 自动化决策支持系统")
    print("   • 可解释的机器学习")
    print("   • 专家系统和知识发现")
    
    print("\n🎯 从规则发现到智能预测的完整工作流程演示成功！")

def interactive_prediction_demo():
    """交互式预测演示"""
    print("\n" + "=" * 60)
    print("🎮 互动演示：自定义预测")
    print("=" * 60)
    
    # 创建简单的规则发现器
    discoverer = OptimalConditionalRuleDiscoverer(max_depth=2, min_samples_leaf=5)
    
    # 使用简单数据
    print("📂 加载规则...")
    rules = discoverer.discover_optimal_rules("test_merge.csv")
    
    if not rules:
        print("❌ 无法加载规则")
        return
    
    predictor = RuleBasedPredictor(rules, discoverer.label_encoders)
    print(f"✅ 规则加载完成，共 {len(rules)} 条规则")
    
    print("\n📋 可用特征:")
    print("   • x (数值): 例如 10, 20, 30...")
    print("   • y (分类): y1 或 y2")
    print("   • a (数值): 例如 1, 2, 3...")
    print("   • b (数值): 例如 2, 3, 4...")
    print("   • c (数值): 例如 3, 4, 5...")
    
    print("\n🎯 预设案例快速测试:")
    quick_tests = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5},
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8},
    ]
    
    for i, test_case in enumerate(quick_tests, 1):
        print(f"\n🔍 快速测试 {i}: {test_case}")
        result = predictor.predict(test_case)
        if result['status'] == 'success':
            print(f"   📊 预测结果: {result['prediction']:.2f}")
            print(f"   🎲 置信度: {result['confidence']:.1%}")
        else:
            print(f"   ❌ {result['explanation']}")

if __name__ == "__main__":
    complete_workflow_demo()
    interactive_prediction_demo() 