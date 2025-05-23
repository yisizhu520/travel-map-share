# 决策树模型评估工具 - 简化版

## 问题背景

在运行原始评估脚本时，您可能会遇到以下NumPy兼容性错误：

```
A module that was compiled using NumPy 1.x cannot be run in
NumPy 2.0.2 as it may crash. To support both 1.x and 2.x
versions of NumPy, modules must be compiled with NumPy 2.0.
```

这是因为您的环境中安装了NumPy 2.0.2版本，而某些依赖模块（如matplotlib）是使用NumPy 1.x编译的，导致不兼容。

## 解决方案

我们提供了两种解决方案：

1. **使用简化版评估脚本**：我们创建了一个不依赖matplotlib的简化版评估脚本`evaluate_tree_simple.py`，可以避免NumPy兼容性问题
2. **环境配置调整**：您也可以通过调整Python环境来解决兼容性问题，详见`环境配置指南.md`

## 简化版评估脚本使用方法

### 基本评估

运行以下命令，使用所有评估方法：

```bash
python evaluate_tree_simple.py
```

### 指定评估方法

```bash
# 仅使用训练测试集划分评估
python evaluate_tree_simple.py --method split

# 仅使用完整数据集评估
python evaluate_tree_simple.py --method full

# 仅使用交叉验证评估
python evaluate_tree_simple.py --method cv
```

### 其他参数

```bash
python evaluate_tree_simple.py --csv_path 您的数据文件.csv --target_column 目标列名 --output_dir 输出目录 --test_size 0.2 --n_folds 10
```

参数说明：
- `--csv_path`: CSV文件路径，默认为'data.csv'
- `--target_column`: 目标分类列名，默认为'flag'
- `--output_dir`: 输出目录，默认为'tree_evaluation'
- `--method`: 评估方法，可选'split'、'full'、'cv'或'all'，默认为'all'
- `--test_size`: 测试集比例，默认为0.3（仅用于split方法）
- `--n_folds`: 交叉验证折数，默认为5（仅用于cv方法）

## 功能说明

简化版评估脚本提供与原始脚本相同的核心功能：

1. **训练测试集划分评估**：将数据集分为训练集和测试集，评估模型在测试集上的表现
2. **完整数据集评估**：使用完整数据集评估模型性能
3. **交叉验证评估**：使用交叉验证方法评估模型性能

唯一的区别是简化版不生成可视化图表，只输出文本报告和保存评估结果到文本文件。

## 输出结果

评估结果将保存在指定的输出目录中（默认为'tree_evaluation'），包括：

1. 文本报告：包含准确率、分类报告和混淆矩阵

## 环境配置

如果您希望使用原始脚本（包含可视化功能），请参考`环境配置指南.md`文件，了解如何解决NumPy兼容性问题。

## 注意事项

- 简化版评估脚本不生成可视化图表，只输出文本报告
- 如果您需要可视化结果，请参考环境配置指南解决NumPy兼容性问题
- 确保您的数据文件格式正确，且目标列存在