#!/usr/bin/env python3
"""
lisan.csv 数据集简单规则发现演示

使用原版规则发现器来处理分类规则
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.discover_conditional_rules import discover_conditional_polynomial_rules
import pandas as pd

def demo_lisan_simple_rules():
    print("🎯 === lisan.csv 数据集简单规则发现演示 === 🎯")
    
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
    
    try:
        # 发现规则 - 这个函数直接处理文件并打印结果
        discover_conditional_polynomial_rules(data_file, decision_tree_max_depth=5, decision_tree_min_samples_leaf=20)
        
        print(f"\n✅ 规则发现过程完成")
        print(f"\n💡 注意: 原版规则发现器主要用于发现数值多项式规则，")
        print(f"   对于像lisan.csv这样的分类规则可能效果有限。")
        print(f"   建议手动分析数据规律或使用专门的分类规则发现方法。")
            
    except Exception as e:
        print(f"❌ 规则发现过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demo_lisan_simple_rules() 