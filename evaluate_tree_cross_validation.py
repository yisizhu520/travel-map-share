import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict

# 导入自定义模块
from demo import build_decision_tree

def evaluate_with_cross_validation(csv_path, target_column='flag', n_folds=5, output_dir=None):
    """
    使用交叉验证评估决策树模型的预测准确率
    
    参数:
        csv_path: CSV文件路径
        target_column: 目标分类列名
        n_folds: 交叉验证折数
        output_dir: 输出目录，如果提供则保存评估结果
        
    返回:
        cv_scores: 交叉验证得分
        mean_accuracy: 平均准确率
        std_accuracy: 准确率标准差
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
        
        # 可视化交叉验证得分
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_folds + 1), cv_scores, color='skyblue')
        plt.axhline(y=mean_accuracy, color='r', linestyle='-', label=f'平均准确率: {mean_accuracy:.4f}')
        plt.axhline(y=mean_accuracy + std_accuracy, color='g', linestyle='--', label=f'+1 标准差: {mean_accuracy + std_accuracy:.4f}')
        plt.axhline(y=mean_accuracy - std_accuracy, color='g', linestyle='--', label=f'-1 标准差: {mean_accuracy - std_accuracy:.4f}')
        plt.xlabel('交叉验证折数')
        plt.ylabel('准确率')
        plt.title('决策树模型交叉验证准确率')
        plt.xticks(range(1, n_folds + 1))
        plt.ylim([0, 1])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, 'cross_validation_scores.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 可视化混淆矩阵
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('交叉验证混淆矩阵')
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
        plt.savefig(os.path.join(output_dir, 'cross_validation_confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"交叉验证评估结果已保存到 {output_dir} 目录")
    
    return cv_scores, mean_accuracy, std_accuracy, report, conf_matrix

def main():
    # 确保输出目录存在
    output_dir = 'tree_evaluation'
    os.makedirs(output_dir, exist_ok=True)
    
    # 使用交叉验证评估模型
    print("\n正在使用交叉验证评估决策树模型...")
    cv_scores, mean_accuracy, std_accuracy, report, conf_matrix = evaluate_with_cross_validation(
        'data.csv', target_column='flag', n_folds=5, output_dir=output_dir
    )
    
    print(f"\n交叉验证各折准确率: {cv_scores}")
    print(f"平均准确率: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"\n分类报告:\n{report}")
    print(f"\n混淆矩阵:\n{conf_matrix}")
    
    print(f"\n评估结果已保存到 {output_dir} 目录")

if __name__ == "__main__":
    main()