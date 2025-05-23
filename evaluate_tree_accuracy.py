import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# 导入自定义模块
from demo import build_decision_tree

def evaluate_decision_tree(csv_path, target_column='flag', test_size=0.3, output_dir=None):
    """
    评估决策树模型的预测准确率
    
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
        
        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        
        # 设置坐标轴
        classes = encoder.target_encoder.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 添加数值标签
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"评估结果已保存到 {output_dir} 目录")
    
    return accuracy, report, conf_matrix

def evaluate_full_dataset(csv_path, target_column='flag', output_dir=None):
    """
    在完整数据集上评估决策树模型的预测准确率
    
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
            f.write(f"决策树模型在完整数据集上的评估报告\n")
            f.write(f"===============================\n\n")
            f.write(f"数据集: {csv_path}\n")
            f.write(f"目标列: {target_column}\n\n")
            f.write(f"准确率: {accuracy:.4f}\n\n")
            f.write(f"分类报告:\n{report}\n\n")
            f.write(f"混淆矩阵:\n{conf_matrix}\n")
        
        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('完整数据集上的混淆矩阵')
        plt.colorbar()
        
        # 设置坐标轴
        classes = encoder.target_encoder.classes_
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # 添加数值标签
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.savefig(os.path.join(output_dir, 'full_dataset_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"完整数据集评估结果已保存到 {output_dir} 目录")
    
    return accuracy, report, conf_matrix

def main():
    # 确保输出目录存在
    output_dir = 'tree_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 步骤1：使用训练集和测试集评估模型
    print("\n正在使用训练集和测试集评估决策树模型...")
    accuracy, report, conf_matrix = evaluate_decision_tree('data.csv', target_column='flag', test_size=0.3, output_dir=output_dir)
    
    print(f"\n模型在测试集上的准确率: {accuracy:.4f}")
    print(f"\n分类报告:\n{report}")
    print(f"\n混淆矩阵:\n{conf_matrix}")
    
    # 步骤2：在完整数据集上评估模型
    print("\n正在使用完整数据集评估决策树模型...")
    full_accuracy, full_report, full_conf_matrix = evaluate_full_dataset('data.csv', target_column='flag', output_dir=output_dir)
    
    print(f"\n模型在完整数据集上的准确率: {full_accuracy:.4f}")
    print(f"\n分类报告:\n{full_report}")
    print(f"\n混淆矩阵:\n{full_conf_matrix}")
    
    print(f"\n评估结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()