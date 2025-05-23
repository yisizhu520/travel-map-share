import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict

# 导入自定义模块
from demo import build_decision_tree

def evaluate_decision_tree_simple(csv_path, target_column='flag', test_size=0.3, output_dir=None):
    """
    评估决策树模型的预测准确率（简化版，不使用matplotlib）
    
    参数:
        csv_path: CSV文件路径
        target_column: 目标分类列名
        test_size: 测试集比例
        output_dir: 输出目录，如果提供则保存评估结果
        
    返回:
        accuracy: 模型准确率
        report: 分类报告
        conf_matrix: 混淆矩阵
    """
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # 构建决策树模型（仅使用训练集）
    train_df = pd.concat([X_train, y_train], axis=1)
    train_csv = os.path.join(output_dir, 'train_data.csv') if output_dir else 'train_data.csv'
    train_df.to_csv(train_csv, index=False)
    
    model, feature_names, encoder = build_decision_tree(train_csv, target_column=target_column, output_dir=output_dir)
    
    # 对测试集进行编码
    X_test_encoded = encoder.transform(X_test)
    
    # 对测试集进行预测
    y_test_encoded = encoder.target_encoder.transform(y_test)
    y_pred_encoded = model.predict(X_test_encoded)
    
    # 将编码后的预测结果转换回原始标签
    y_pred = encoder.target_encoder.inverse_transform(y_pred_encoded)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    
    # 生成分类报告
    report = classification_report(y_test, y_pred)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # 如果提供了输出目录，则保存评估结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存评估结果
        with open(os.path.join(output_dir, 'evaluation_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"决策树模型评估报告\n")
            f.write(f"===================\n\n")
            f.write(f"数据集: {csv_path}\n")
            f.write(f"目标列: {target_column}\n")
            f.write(f"测试集比例: {test_size}\n\n")
            f.write(f"准确率: {accuracy:.4f}\n\n")
            f.write(f"分类报告:\n{report}\n\n")
            f.write(f"混淆矩阵:\n{conf_matrix}\n")
    
    return accuracy, report, conf_matrix

def evaluate_full_dataset_simple(csv_path, target_column='flag', output_dir=None):
    """
    使用完整数据集评估决策树模型的预测准确率（简化版，不使用matplotlib）
    
    参数:
        csv_path: CSV文件路径
        target_column: 目标分类列名
        output_dir: 输出目录，如果提供则保存评估结果
        
    返回:
        accuracy: 模型准确率
        report: 分类报告
        conf_matrix: 混淆矩阵
    """
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 构建决策树模型（使用全部数据）
    model, feature_names, encoder = build_decision_tree(csv_path, target_column=target_column, output_dir=output_dir)
    
    # 对全部数据进行编码
    X_encoded = encoder.transform(X)
    
    # 对全部数据进行预测
    y_encoded = encoder.target_encoder.transform(y)
    y_pred_encoded = model.predict(X_encoded)
    
    # 将编码后的预测结果转换回原始标签
    y_pred = encoder.target_encoder.inverse_transform(y_pred_encoded)
    
    # 计算准确率
    accuracy = accuracy_score(y, y_pred)
    
    # 生成分类报告
    report = classification_report(y, y_pred)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y, y_pred)
    
    # 如果提供了输出目录，则保存评估结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存评估结果
        with open(os.path.join(output_dir, 'full_dataset_evaluation.txt'), 'w', encoding='utf-8') as f:
            f.write(f"决策树模型完整数据集评估报告\n")
            f.write(f"==========================\n\n")
            f.write(f"数据集: {csv_path}\n")
            f.write(f"目标列: {target_column}\n\n")
            f.write(f"准确率: {accuracy:.4f}\n\n")
            f.write(f"分类报告:\n{report}\n\n")
            f.write(f"混淆矩阵:\n{conf_matrix}\n")
    
    return accuracy, report, conf_matrix

def evaluate_with_cross_validation_simple(csv_path, target_column='flag', n_folds=5, output_dir=None):
    """
    使用交叉验证评估决策树模型的预测准确率（简化版，不使用matplotlib）
    
    参数:
        csv_path: CSV文件路径
        target_column: 目标分类列名
        n_folds: 交叉验证折数
        output_dir: 输出目录，如果提供则保存评估结果
        
    返回:
        cv_scores: 交叉验证得分
        mean_accuracy: 平均准确率
        std_accuracy: 准确率标准差
        report: 分类报告
        conf_matrix: 混淆矩阵
    """
    # 读取CSV数据
    df = pd.read_csv(csv_path)
    
    # 分离特征和目标变量
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 构建决策树模型（使用全部数据）
    model, feature_names, encoder = build_decision_tree(csv_path, target_column=target_column, output_dir=output_dir)
    
    # 对全部数据进行编码
    X_encoded = encoder.transform(X)
    y_encoded = encoder.target_encoder.transform(y)
    
    # 设置交叉验证
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # 执行交叉验证
    cv_scores = cross_val_score(model, X_encoded, y_encoded, cv=kf)
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)
    
    # 使用交叉验证进行预测，用于生成混淆矩阵和分类报告
    y_pred_encoded = cross_val_predict(model, X_encoded, y_encoded, cv=kf)
    y_pred = encoder.target_encoder.inverse_transform(y_pred_encoded)
    
    # 生成分类报告
    report = classification_report(y, y_pred)
    
    # 计算混淆矩阵
    conf_matrix = confusion_matrix(y, y_pred)
    
    # 如果提供了输出目录，则保存评估结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存评估结果
        with open(os.path.join(output_dir, 'cross_validation_report.txt'), 'w', encoding='utf-8') as f:
            f.write(f"决策树模型交叉验证评估报告\n")
            f.write(f"==========================\n\n")
            f.write(f"数据集: {csv_path}\n")
            f.write(f"目标列: {target_column}\n")
            f.write(f"交叉验证折数: {n_folds}\n\n")
            f.write(f"各折准确率: {cv_scores}\n")
            f.write(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}\n\n")
            f.write(f"分类报告:\n{report}\n\n")
            f.write(f"混淆矩阵:\n{conf_matrix}\n")
    
    return cv_scores, mean_accuracy, std_accuracy, report, conf_matrix

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='评估决策树模型的预测准确率（简化版，不使用matplotlib）')
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
        accuracy, report, conf_matrix = evaluate_decision_tree_simple(
            args.csv_path, target_column=args.target_column, 
            test_size=args.test_size, output_dir=args.output_dir
        )
        
        print(f"\n模型在测试集上的准确率: {accuracy:.4f}")
        print(f"\n分类报告:\n{report}")
        print(f"\n混淆矩阵:\n{conf_matrix}")
    
    if args.method in ['full', 'all']:
        print("\n正在使用完整数据集评估决策树模型...")
        full_accuracy, full_report, full_conf_matrix = evaluate_full_dataset_simple(
            args.csv_path, target_column=args.target_column, 
            output_dir=args.output_dir
        )
        
        print(f"\n模型在完整数据集上的准确率: {full_accuracy:.4f}")
        print(f"\n分类报告:\n{full_report}")
        print(f"\n混淆矩阵:\n{full_conf_matrix}")
    
    if args.method in ['cv', 'all']:
        print("\n正在使用交叉验证评估决策树模型...")
        cv_scores, mean_accuracy, std_accuracy, cv_report, cv_conf_matrix = evaluate_with_cross_validation_simple(
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