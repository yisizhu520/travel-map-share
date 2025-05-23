#!/usr/bin/env python3
"""
测试修复后的规则发现功能
"""

import sys
import os
import pandas as pd
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer

def test_fixed_rule_discovery():
    print("🔧 === 测试修复后的规则发现功能 === 🔧")
    print()
    
    # 使用测试数据
    csv_file = "../test_data.csv"  # 上级目录的测试文件
    
    if not os.path.exists(csv_file):
        print("❌ 找不到测试数据文件，创建临时测试文件...")
        
        # 创建简单的测试数据
        np.random.seed(42)
        data = []
        
        for i in range(1000):
            x = np.random.uniform(10, 70)
            y = np.random.choice(['y1', 'y2'])
            a = np.random.randint(1, 10)
            b = np.random.randint(1, 10)
            c = np.random.randint(1, 10)
            
            # 根据规则生成result
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
        csv_file = "temp_test_data.csv"
        df.to_csv(csv_file, index=False)
        print(f"✅ 创建了临时测试文件: {csv_file}")
        print(f"   数据样本数: {len(df)}")
        print(f"   特征: {list(df.columns)}")
    else:
        # 先检查现有数据文件的结构
        df = pd.read_csv(csv_file)
        print(f"✅ 找到测试数据文件: {csv_file}")
        print(f"   数据样本数: {len(df)}")
        print(f"   特征: {list(df.columns)}")
        
        # 检查是否是合适的测试数据（有x, y, a, b, c等特征）
        required_features = {'x', 'y', 'a', 'b', 'c'}
        available_features = set(df.columns)
        
        if not required_features.issubset(available_features):
            print(f"⚠️  现有数据缺少必要特征，需要创建新的测试数据")
            print(f"   需要: {required_features}")
            print(f"   现有: {available_features}")
            
            # 重新创建合适的测试数据
            csv_file = "temp_test_data.csv"
            # 创建数据的代码保持不变...
            data = []
            for i in range(1000):
                x = np.random.uniform(10, 70)
                y = np.random.choice(['y1', 'y2'])
                a = np.random.randint(1, 10)
                b = np.random.randint(1, 10)
                c = np.random.randint(1, 10)
                
                # 根据规则生成result
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
            df.to_csv(csv_file, index=False)
            print(f"✅ 创建了新的测试文件: {csv_file}")
    
    print(f"\n📂 使用数据文件: {csv_file}")
    
    # 创建规则发现器
    discoverer = OptimalConditionalRuleDiscoverer(
        max_depth=3,
        min_samples_leaf=30,  # 减少最小样本数以获得更多规则
        enable_exhaustive_search=True,
        max_combinations=50
    )
    
    print("\n🔍 开始规则发现...")
    
    # 手动指定分段特征和多项式特征，确保测试的确定性
    rules = discoverer.discover_optimal_rules(
        csv_file_path=csv_file,
        target_col="result",
        manual_split_features=['x', 'y'],  # 手动指定分段特征
        manual_poly_features=['a', 'b', 'c']  # 手动指定多项式特征
    )
    
    print(f"\n📋 发现的规则数量: {len(rules)}")
    
    if rules:
        print("\n🔍 详细检查发现的规则:")
        
        logical_errors = 0
        missing_features = 0
        
        for i, rule in enumerate(rules, 1):
            condition = rule['condition']
            print(f"\n规则 {i}:")
            print(f"  条件: {condition}")
            print(f"  规则: {rule['rule']}")
            print(f"  R²: {rule['cv_r2_score']:.3f}")
            
            # 检查逻辑错误
            has_x = 'x' in condition
            has_y = 'y' in condition
            
            if not has_x:
                print(f"  ⚠️  警告: 条件中缺少特征 'x'")
                missing_features += 1
            
            if not has_y:
                print(f"  ⚠️  警告: 条件中缺少特征 'y'")
                missing_features += 1
            
            # 检查是否有矛盾条件
            if 'x >' in condition and 'x <=' in condition:
                # 分析是否矛盾
                import re
                x_greater = re.findall(r'x > ([\d.]+)', condition)
                x_less_equal = re.findall(r'x <= ([\d.]+)', condition)
                
                if x_greater and x_less_equal:
                    for gt_val in x_greater:
                        for le_val in x_less_equal:
                            if float(gt_val) >= float(le_val):
                                print(f"  ❌ 逻辑错误: x > {gt_val} 且 x <= {le_val}")
                                logical_errors += 1
        
        print(f"\n📊 质量检查结果:")
        print(f"   总规则数: {len(rules)}")
        print(f"   逻辑错误: {logical_errors}")
        print(f"   特征缺失: {missing_features}")
        
        if logical_errors == 0 and missing_features == 0:
            print("   🎉 所有规则都通过了质量检查！")
            return True
        else:
            print("   ⚠️  仍存在一些问题需要进一步修复")
            return False
    else:
        print("❌ 没有发现任何规则")
        return False

if __name__ == "__main__":
    success = test_fixed_rule_discovery()
    if success:
        print("\n✅ 修复验证成功！")
    else:
        print("\n❌ 修复验证失败，需要进一步调试") 