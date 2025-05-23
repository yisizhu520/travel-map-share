#!/usr/bin/env python3
"""
命令行交互式预测工具
"""

import sys
import os
import argparse
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
from rule_predictor import RuleBasedPredictor
import pandas as pd

def load_rules_from_csv(csv_file: str, target_col=None, **kwargs):
    """从CSV文件训练并获取规则"""
    print(f"📂 正在分析数据文件: {csv_file}")
    print("🔍 发现规则中...")
    
    discoverer = OptimalConditionalRuleDiscoverer(**kwargs)
    rules = discoverer.discover_optimal_rules(csv_file, target_col=target_col)
    
    if not rules:
        print("❌ 未能发现有效规则")
        return None, None
    
    print(f"✅ 成功发现 {len(rules)} 条规则")
    return RuleBasedPredictor(rules, discoverer.label_encoders), rules

def interactive_predict(predictor, rules):
    """交互式预测"""
    print("\n" + "=" * 60)
    print("🎯 交互式预测模式")
    print("=" * 60)
    
    # 显示规则摘要
    print("📋 可用规则:")
    for i, rule in enumerate(rules, 1):
        print(f"   {i}. 当 {rule['condition']} 时:")
        print(f"      → {rule['rule']} (准确度: {rule['cv_r2_score']:.1%})")
    
    print("\n💡 使用提示:")
    print("   • 输入特征值，格式: 特征名=值")
    print("   • 多个特征用空格分隔")
    print("   • 例如: x=25 y=A a=3 b=4")
    print("   • 输入 'quit' 或 'exit' 退出")
    print("   • 输入 'help' 查看帮助")
    print("   • 输入 'rules' 查看详细规则")
    
    while True:
        print("\n" + "-" * 40)
        user_input = input("🔮 请输入预测数据: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 谢谢使用！")
            break
        
        if user_input.lower() == 'help':
            show_help()
            continue
        
        if user_input.lower() == 'rules':
            show_detailed_rules(rules)
            continue
            
        if not user_input:
            print("⚠️ 请输入数据")
            continue
        
        # 解析输入
        try:
            input_data = parse_input(user_input)
            if not input_data:
                print("❌ 输入格式错误，请重试")
                continue
            
            print(f"📥 解析的输入: {input_data}")
            
            # 进行预测
            result = predictor.predict(input_data, explain=True)
            
            if result['status'] == 'success':
                print("\n✅ 预测成功！")
                print(result['explanation'])
            else:
                print(f"\n❌ 预测失败: {result['explanation']}")
                
                if result['status'] == 'no_match':
                    print("\n💡 建议:")
                    print("   检查输入值是否在以下条件范围内:")
                    for condition in result.get('available_conditions', []):
                        print(f"   • {condition}")
                        
        except Exception as e:
            print(f"❌ 处理错误: {e}")

def parse_input(user_input: str) -> dict:
    """解析用户输入"""
    input_data = {}
    
    try:
        # 支持多种格式
        if '=' in user_input:
            # 格式: x=25 y=A
            pairs = user_input.split()
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # 尝试转换为数值
                    try:
                        if '.' in value:
                            input_data[key] = float(value)
                        else:
                            input_data[key] = int(value)
                    except ValueError:
                        input_data[key] = value
        
        elif user_input.startswith('{') and user_input.endswith('}'):
            # JSON格式: {"x": 25, "y": "A"}
            input_data = json.loads(user_input)
        
        else:
            print("❌ 不支持的输入格式")
            return {}
            
    except Exception as e:
        print(f"❌ 解析错误: {e}")
        return {}
    
    return input_data

def show_help():
    """显示帮助信息"""
    print("\n📖 帮助信息:")
    print("=" * 50)
    print("支持的输入格式:")
    print("  1. 键值对格式: x=25 y=A a=3")
    print("  2. JSON格式: {\"x\": 25, \"y\": \"A\", \"a\": 3}")
    print()
    print("特殊命令:")
    print("  • help - 显示此帮助")
    print("  • rules - 显示详细规则")
    print("  • quit/exit/q - 退出程序")
    print()
    print("数据类型:")
    print("  • 数值: 直接输入数字 (如 25, 3.14)")
    print("  • 文本: 直接输入文本 (如 A, B, y1)")

def show_detailed_rules(rules):
    """显示详细规则"""
    print("\n📋 详细规则列表:")
    print("=" * 60)
    
    for i, rule in enumerate(rules, 1):
        print(f"\n规则 {i}:")
        print(f"  🔍 条件: {rule['condition']}")
        print(f"  📐 公式: {rule['rule']}")
        print(f"  📈 准确度: {rule['cv_r2_score']:.3f}")
        print(f"  📊 样本数: {rule['sample_count']}")

def batch_predict_from_file(predictor, input_file: str, output_file: str = None):
    """从文件批量预测"""
    try:
        # 读取输入文件
        input_data = pd.read_csv(input_file)
        print(f"📂 读取输入文件: {input_file}")
        print(f"   数据形状: {input_data.shape}")
        
        # 执行批量预测
        print("🔮 执行批量预测...")
        results = predictor.predict_batch(input_data, explain=False)
        
        # 保存结果
        if output_file:
            results.to_csv(output_file, index=False)
            print(f"💾 结果已保存到: {output_file}")
        else:
            output_file = input_file.replace('.csv', '_predictions.csv')
            results.to_csv(output_file, index=False)
            print(f"💾 结果已保存到: {output_file}")
        
        # 显示摘要
        success_count = (results['status'] == 'success').sum()
        print(f"\n📊 预测摘要:")
        print(f"   总样本数: {len(results)}")
        print(f"   成功预测: {success_count}")
        print(f"   失败预测: {len(results) - success_count}")
        
        if success_count > 0:
            avg_confidence = results[results['status'] == 'success']['confidence'].mean()
            print(f"   平均置信度: {avg_confidence:.1%}")
        
        return results
        
    except Exception as e:
        print(f"❌ 批量预测失败: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='基于规则的交互式预测工具')
    parser.add_argument('csv_file', help='训练数据的CSV文件路径')
    parser.add_argument('--target-col', type=str, help='目标列名称')
    parser.add_argument('--max-depth', type=int, default=3, help='决策树最大深度')
    parser.add_argument('--min-samples', type=int, default=50, help='叶子节点最小样本数')
    parser.add_argument('--mode', choices=['interactive', 'batch'], default='interactive',
                        help='运行模式: interactive(交互) 或 batch(批量)')
    parser.add_argument('--input-file', type=str, help='批量预测的输入文件')
    parser.add_argument('--output-file', type=str, help='批量预测的输出文件')
    
    args = parser.parse_args()
    
    print("🔮 === 基于规则的智能预测工具 === 🔮")
    print(f"数据文件: {args.csv_file}")
    print(f"运行模式: {args.mode}")
    
    # 加载规则
    predictor, rules = load_rules_from_csv(
        args.csv_file,
        target_col=args.target_col,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples
    )
    
    if not predictor:
        print("❌ 无法加载规则，程序退出")
        return
    
    if args.mode == 'interactive':
        interactive_predict(predictor, rules)
    elif args.mode == 'batch':
        if not args.input_file:
            print("❌ 批量模式需要指定 --input-file")
            return
        batch_predict_from_file(predictor, args.input_file, args.output_file)

if __name__ == "__main__":
    main() 