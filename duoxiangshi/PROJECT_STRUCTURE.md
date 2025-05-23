# 项目结构说明

## 📁 目录结构

```
duoxiangshi/
├── core/                           # 🔧 核心功能代码
├── tests/                          # 🧪 测试代码
├── demos/                          # 🎯 演示代码
├── docs/                           # 📚 文档
├── data/                           # 📊 数据文件
├── utils/                          # 🛠️ 工具代码
└── __pycache__/                    # Python缓存文件
```

## 📂 各目录详细说明

### 🔧 `core/` - 核心功能代码
包含项目的主要功能模块：

- **`discover_conditional_rules_optimal.py`** - 🏆 **主要模块**：优化版条件多项式规则发现器
  - 支持分类和数值特征
  - 智能特征组合穷举
  - 交叉验证评估
  - 已修复条件矛盾问题

- **`rule_predictor_simple.py`** - 🚀 **推荐使用**：简化版规则预测器
  - 基于成熟的simpleeval库
  - 代码简洁，性能优秀
  - 安全的表达式评估

- **`rule_predictor.py`** - 📜 原版规则预测器（复杂实现）

- **`predict_cli.py`** - 💻 命令行预测工具

- **`discover_conditional_rules_improved.py`** - 📈 改进版规则发现器

- **`discover_conditional_rules.py`** - 📋 原版规则发现器

### 🧪 `tests/` - 测试代码
包含各种测试和调试脚本：

- **`test_condition_matching.py`** - 条件匹配测试
- **`test_condition_simplification.py`** - 条件简化测试
- **`test_simple_conversion.py`** - 简单转换测试
- **`debug_condition.py`** - 条件调试工具
- **`test_data_generate.py`** - 测试数据生成

### 🎯 `demos/` - 演示代码
包含各种功能演示和对比：

- **`demo_final_comparison.py`** - 🏆 **终极对比演示**：自制解析器 vs 成熟库
- **`simple_prediction_demo.py`** - 🚀 简单预测演示
- **`demo_complete_workflow.py`** - 完整工作流演示
- **`demo_simple_vs_complex.py`** - 简单版 vs 复杂版对比
- **`demo_simple_complete.py`** - 简单版完整演示
- **`demo_condition_simplification.py`** - 条件简化演示
- **`demo_merge.py`** - 规则合并演示

### 📚 `docs/` - 文档
包含项目文档和说明：

- **`README.md`** - 📖 **主要文档**：项目完整说明
- **`README_REFACTORING_JOURNEY.md`** - 🔄 重构历程文档
- **`BUGFIX_RULE_DISCOVERY.md`** - 🐛 Bug修复报告
- **`flowchart TD.mmd`** - 📊 流程图

### 📊 `data/` - 数据文件
包含各种测试和示例数据：

#### 主要数据集：
- **`duoxiangshi.csv`** - 主要数据集 (133KB, 10K条记录)
- **`if_duoxiangshi.csv`** - IF条件数据集 (159KB, 10K条记录)
- **`multi_if_duoxiangshi.csv`** - 多重IF数据集 (189KB, 10K条记录)
- **`test_data_2.csv`** - 测试数据集2 (240KB, 10K条记录)

#### 测试数据：
- **`test_sample.csv`** - 测试样本
- **`test_merge.csv`** - 合并测试数据
- **`test_complex_merge.csv`** - 复杂合并测试
- **`test_duplicate_rules.csv`** - 重复规则测试
- **`test_redundant_conditions.csv`** - 冗余条件测试
- **`test_complex_conditions.csv`** - 复杂条件测试
- **`test_predict_input.csv`** - 预测输入测试

### 🛠️ `utils/` - 工具代码
包含辅助工具和数据生成脚本：

- **`generate_multi_if_data.py`** - 多重IF数据生成器
- **`generate_if_data.py`** - IF数据生成器
- **`discover_polynomial_rule.py`** - 多项式规则发现工具

## 🚀 快速开始

### 1. 规则发现
```bash
cd core
python discover_conditional_rules_optimal.py ../data/duoxiangshi.csv
```

### 2. 规则预测
```bash
cd core
python predict_cli.py --rules-file rules.json --input ../data/test_predict_input.csv
```

### 3. 查看演示
```bash
cd demos
python demo_final_comparison.py
```

## 📋 推荐使用

### 🏆 **主要功能**：
1. **规则发现**：`core/discover_conditional_rules_optimal.py`
2. **规则预测**：`core/rule_predictor_simple.py`
3. **完整演示**：`demos/demo_final_comparison.py`

### 📖 **文档阅读顺序**：
1. `docs/README.md` - 了解项目概况
2. `docs/README_REFACTORING_JOURNEY.md` - 了解重构历程
3. `docs/BUGFIX_RULE_DISCOVERY.md` - 了解Bug修复过程

## 🔧 开发说明

- **核心代码**在 `core/` 目录，修改时请谨慎
- **测试代码**在 `tests/` 目录，可以自由添加新测试
- **演示代码**在 `demos/` 目录，可以参考学习
- **数据文件**在 `data/` 目录，注意大文件的版本控制

## 📝 版本历史

- **v1.0** - 初始版本，基础功能实现
- **v2.0** - 重构使用成熟库，性能提升41%
- **v2.1** - 修复条件矛盾Bug，质量检查100%通过
- **v2.2** - 项目结构重组，代码组织更清晰 