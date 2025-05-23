import os
import argparse

# 导入评估模块
from evaluate_tree_accuracy import evaluate_decision_tree, evaluate_full_dataset
from evaluate_tree_cross_validation import evaluate_with_cross_validation

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='评估决策树模型的预测准确率')
    parser.add_argument('--csv_path', type=str, default='data.csv', help='CSV文件路径')
    parser.add_argument('--target_column', type=str, default='flag', help='目标分类列名')
    parser.add_argument('--output_dir', type=str, default='tree_evaluation', help='输出目录')
    parser.add_argument('--method', type=str, choices=['split', 'full', 'cv', 'all'], default='all',
                        help='评估方法: split-训练测试集划分, full-完整数据集, cv-交叉验证, all-全部方法')
    parser.add_argument('--test_size', type=float, default=0.3, help='测试集比例 (仅用于split方法)')
    parser.add_argument('--n_folds', type=int, default=5, help='交叉验证折数 (仅用于cv方法)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 根据选择的方法执行评估
    if args.method in ['split', 'all']:
        print("\n正在使用训练集和测试集评估决策树模型...")
        accuracy, report, conf_matrix = evaluate_decision_tree(
            args.csv_path, target_column=args.target_column, 
            test_size=args.test_size, output_dir=args.output_dir
        )
        
        print(f"\n模型在测试集上的准确率: {accuracy:.4f}")
        print(f"\n分类报告:\n{report}")
        print(f"\n混淆矩阵:\n{conf_matrix}")
    
    if args.method in ['full', 'all']:
        print("\n正在使用完整数据集评估决策树模型...")
        full_accuracy, full_report, full_conf_matrix = evaluate_full_dataset(
            args.csv_path, target_column=args.target_column, 
            output_dir=args.output_dir
        )
        
        print(f"\n模型在完整数据集上的准确率: {full_accuracy:.4f}")
        print(f"\n分类报告:\n{full_report}")
        print(f"\n混淆矩阵:\n{full_conf_matrix}")
    
    if args.method in ['cv', 'all']:
        print("\n正在使用交叉验证评估决策树模型...")
        cv_scores, mean_accuracy, std_accuracy, cv_report, cv_conf_matrix = evaluate_with_cross_validation(
            args.csv_path, target_column=args.target_column, 
            n_folds=args.n_folds, output_dir=args.output_dir
        )
        
        print(f"\n交叉验证各折准确率: {cv_scores}")
        print(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
        print(f"\n分类报告:\n{cv_report}")
        print(f"\n混淆矩阵:\n{cv_conf_matrix}")
    
    print(f"\n评估结果已保存到 {args.output_dir} 目录")

if __name__ == "__main__":
    main()