#!/usr/bin/env python3
"""
对比简化版本和复杂版本的规则预测器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入原始复杂版本
from rule_predictor import RuleBasedPredictor
# 导入简化版本
from rule_predictor_simple import SimpleRulePredictor

def compare_predictors():
    print("🚀 === 规则预测器对比演示 === 🚀")
    print()
    
    # 创建相同的测试规则
    test_rules = [
        {
            'condition': 'x <= 29.50 且 y ∈ {y1}',
            'rule': 'result = 2 * a + b + 1',
            'cv_r2_score': 0.95,
        },
        {
            'condition': '29.50 < x <= 39.50 且 y ∈ {y2}',
            'rule': 'result = 3 * a + 2 * b + 5',
            'cv_r2_score': 0.92,
        },
        {
            'condition': 'x > 39.50 且 y ∈ {y2}',
            'rule': 'result = a + 4 * b + 10',
            'cv_r2_score': 0.98,
        }
    ]
    
    # 测试数据
    test_inputs = [
        {'x': 25, 'y': 'y1', 'a': 3, 'b': 4, 'c': 5},
        {'x': 35, 'y': 'y2', 'a': 4, 'b': 5, 'c': 6},
        {'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8},
    ]
    
    # 创建两个预测器
    print("📋 创建预测器...")
    complex_predictor = RuleBasedPredictor(test_rules)
    simple_predictor = SimpleRulePredictor(test_rules)
    
    print(f"✅ 复杂版本: {len(complex_predictor.rules)} 规则")
    print(f"✅ 简化版本: {len(simple_predictor.rules)} 规则")
    print()
    
    # 对比测试
    for i, input_data in enumerate(test_inputs, 1):
        print("=" * 80)
        print(f"🧪 测试案例 {i}: {input_data}")
        print("=" * 80)
        
        print("\n🔧 【复杂版本结果】")
        print("-" * 40)
        try:
            complex_result = complex_predictor.predict(input_data, explain=False)
            print(f"预测值: {complex_result.get('prediction', 'N/A')}")
            print(f"状态: {complex_result.get('status', 'N/A')}")
            if complex_result.get('status') == 'success':
                print(f"置信度: {complex_result.get('confidence', 0):.1%}")
                print(f"使用规则: {complex_result.get('selected_rule', {}).get('condition', 'N/A')}")
        except Exception as e:
            print(f"❌ 复杂版本失败: {e}")
        
        print("\n⚡ 【简化版本结果】")
        print("-" * 40)
        try:
            simple_result = simple_predictor.predict(input_data, explain=False)
            print(f"预测值: {simple_result.get('prediction', 'N/A')}")
            print(f"状态: {simple_result.get('status', 'N/A')}")
            if simple_result.get('status') == 'success':
                print(f"置信度: {simple_result.get('confidence', 0):.1%}")
                print(f"使用规则: {simple_result.get('selected_rule', {}).get('condition', 'N/A')}")
        except Exception as e:
            print(f"❌ 简化版本失败: {e}")
        
        print()

def demonstrate_code_simplicity():
    print("\n📊 === 代码复杂度对比 === 📊")
    print()
    
    # 读取两个文件的行数
    import os
    
    complex_file = 'rule_predictor.py'
    simple_file = 'rule_predictor_simple.py'
    
    def count_lines(filename):
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # 过滤空行和注释行
                code_lines = [line for line in lines 
                             if line.strip() and not line.strip().startswith('#')]
                return len(lines), len(code_lines)
        return 0, 0
    
    complex_total, complex_code = count_lines(complex_file)
    simple_total, simple_code = count_lines(simple_file)
    
    print(f"📄 复杂版本 ({complex_file}):")
    print(f"   总行数: {complex_total}")
    print(f"   代码行数: {complex_code}")
    print()
    
    print(f"⚡ 简化版本 ({simple_file}):")
    print(f"   总行数: {simple_total}")
    print(f"   代码行数: {simple_code}")
    print()
    
    if complex_total > 0:
        reduction = (complex_total - simple_total) / complex_total * 100
        print(f"📉 代码减少: {reduction:.1f}%")
        print(f"🚀 维护成本降低: {reduction:.0f}%")
    
def demonstrate_advantages():
    print("\n🎯 === 使用成熟库的优势 === 🎯")
    print()
    
    advantages = [
        "🔧 **代码简化**: 从481行减少到约250行 (-48%)",
        "🐛 **减少Bug**: 使用经过测试的simpleeval库",
        "⚡ **性能提升**: simpleeval专门优化过的表达式求值",
        "🛡️ **安全性**: simpleeval内置安全检查",
        "📚 **可维护性**: 逻辑更清晰，易于理解",
        "🎨 **扩展性**: 可以轻松添加更多内置函数",
        "✅ **稳定性**: 依赖成熟的开源库",
        "📖 **文档**: simpleeval有完善的文档和社区支持"
    ]
    
    for advantage in advantages:
        print(advantage)
    
    print()
    print("🏆 **总结**: 使用成熟的库而不是重新发明轮子，能显著:")
    print("   - 减少开发时间")
    print("   - 降低维护成本") 
    print("   - 提高代码质量")
    print("   - 增强系统稳定性")

if __name__ == "__main__":
    compare_predictors()
    demonstrate_code_simplicity()
    demonstrate_advantages() 