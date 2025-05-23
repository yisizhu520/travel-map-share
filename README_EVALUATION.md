# 决策树模型评估工具

本项目提供了一系列工具，用于评估决策树模型的预测准确率。这些工具可以帮助您了解模型在不同数据集上的表现，以及模型的分类性能。

## 文件说明

- `evaluate_tree_accuracy.py`: 提供基本的模型评估功能，包括训练测试集划分评估和完整数据集评估
- `evaluate_tree_cross_validation.py`: 使用交叉验证方法评估模型性能
- `evaluate_tree_main.py`: 综合评估脚本，提供命令行接口，可选择不同的评估方式

## 使用方法

### 基本评估

运行以下命令，使用训练测试集划分和完整数据集两种方式评估模型：

```bash
python evaluate_tree_accuracy.py
```

### 交叉验证评估

运行以下命令，使用交叉验证方法评估模型：

```bash
python evaluate_tree_cross_validation.py
```

### 综合评估

运行以下命令，使用所有评估方法：

```bash
python evaluate_tree_main.py
```

您也可以指定特定的评估方法：

```bash
# 仅使用训练测试集划分评估
python evaluate_tree_main.py --method split

# 仅使用完整数据集评估
python evaluate_tree_main.py --method full

# 仅使用交叉验证评估
python evaluate_tree_main.py --method cv
```

### 其他参数

您可以通过命令行参数自定义评估过程：

```bash
python evaluate_tree_main.py --csv_path 您的数据文件.csv --target_column 目标列名 --output_dir 输出目录 --test_size 0.2 --n_folds 10
```

参数说明：
- `--csv_path`: CSV文件路径，默认为'data.csv'
- `--target_column`: 目标分类列名，默认为'flag'
- `--output_dir`: 输出目录，默认为'tree_evaluation'
- `--method`: 评估方法，可选'split'、'full'、'cv'或'all'，默认为'all'
- `--test_size`: 测试集比例，默认为0.3（仅用于split方法）
- `--n_folds`: 交叉验证折数，默认为5（仅用于cv方法）

## 输出结果

评估结果将保存在指定的输出目录中（默认为'tree_evaluation'），包括：

1. 文本报告：包含准确率、分类报告和混淆矩阵
2. 可视化图表：包含混淆矩阵可视化和交叉验证得分图表

## 示例输出

运行评估脚本后，您将在控制台看到类似以下的输出：

```
正在使用训练集和测试集评估决策树模型...

模型在测试集上的准确率: 0.8500

分类报告:
              precision    recall  f1-score   support

           a       0.86      0.86      0.86        14
           b       0.83      0.83      0.83         6

    accuracy                           0.85        20
   macro avg       0.85      0.85      0.85        20
weighted avg       0.85      0.85      0.85        20


混淆矩阵:
[[12  2]
 [ 1  5]]
```

同时，在输出目录中，您将找到详细的评估报告和可视化图表。

## 注意事项

- 确保您的数据文件格式正确，且目标列存在
- 对于分类问题，建议使用交叉验证方法获得更稳定的评估结果
- 如果数据集较小，完整数据集评估可能会过于乐观，请结合其他评估方法一起使用