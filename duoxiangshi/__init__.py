"""
条件多项式规则发现与预测系统

这是一个用于发现和应用条件多项式规则的Python包。

主要功能：
1. 从数据中发现条件多项式规则
2. 使用发现的规则进行预测
3. 支持分类和数值特征
4. 提供多种演示和测试

快速开始：
    from duoxiangshi.core.discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
    from duoxiangshi.core.rule_predictor_simple import SimpleRulePredictor
    
    # 发现规则
    discoverer = OptimalConditionalRuleDiscoverer()
    rules = discoverer.discover_optimal_rules('data/your_data.csv')
    
    # 使用规则预测
    predictor = SimpleRulePredictor(rules)
    result = predictor.predict({'x': 45, 'y': 'y2', 'a': 6, 'b': 7, 'c': 8})
"""

__version__ = "2.2.0"
__author__ = "duoxiangshi Team"
__description__ = "条件多项式规则发现与预测系统"

# 导入主要类，方便用户使用
try:
    from .core.discover_conditional_rules_optimal import OptimalConditionalRuleDiscoverer
    from .core.rule_predictor_simple import SimpleRulePredictor
    from .core.rule_predictor import RulePredictor
    
    __all__ = [
        'OptimalConditionalRuleDiscoverer',
        'SimpleRulePredictor', 
        'RulePredictor'
    ]
except ImportError:
    # 如果导入失败，提供友好的错误信息
    import warnings
    warnings.warn(
        "无法导入核心模块。请确保所有依赖已安装：pip install -r requirements.txt",
        ImportWarning
    )
    __all__ = [] 