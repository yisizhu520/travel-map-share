import pandas as pd
from sklearn.linear_model import LinearRegression

def discover_polynomial_rule(csv_file_path):
    """
    Loads data from a CSV file, performs linear regression, and prints the discovered rule.

    Args:
        csv_file_path (str): The path to the CSV file.
    """
    try:
        # 1. 数据加载与预处理
        data = pd.read_csv(csv_file_path)
        print(f"成功加载数据: {csv_file_path}")
        print("数据前5行:\n", data.head())

        # 假设特征列是 'a', 'b', 'c'，目标列是 'result'
        # 如果您的列名不同，请在此处修改
        feature_cols = ['a', 'b', 'c']
        target_col = 'result'

        if not all(col in data.columns for col in feature_cols):
            print(f"错误: 特征列 {feature_cols} 在CSV文件中未全部找到。")
            print(f"可用的列: {data.columns.tolist()}")
            return

        if target_col not in data.columns:
            print(f"错误: 目标列 '{target_col}' 在CSV文件中未找到。")
            print(f"可用的列: {data.columns.tolist()}")
            return

        X = data[feature_cols]
        y = data[target_col]

        # 2. 模型选择与训练
        model = LinearRegression()
        model.fit(X, y)
        print("线性回归模型训练完成。")

        # 3. 规则提取与验证
        coefficients = model.coef_
        intercept = model.intercept_

        rule_parts = []
        for i, col_name in enumerate(feature_cols):
            rule_parts.append(f"{coefficients[i]:.2f} * {col_name}")
        
        rule = f"{target_col} = {' + '.join(rule_parts)} + {intercept:.2f}"
        
        print("\n发现的规则是:")
        print(rule)

        # (可选) 验证模型在训练数据上的表现
        predictions = model.predict(X)
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        print(f"\n模型在训练数据上的均方误差 (MSE): {mse:.2f}")
        print(f"模型在训练数据上的R^2分数: {r2:.2f}")

        if r2 > 0.95: # 如果R^2分数很高，说明模型拟合得很好
            print("模型拟合效果良好，规则很可能准确。")
        else:
            print("模型拟合效果一般，规则可能不是精确的一次方多项式，或者数据中存在噪声/其他复杂关系。")

    except FileNotFoundError:
        print(f"错误: 文件未找到 {csv_file_path}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")

if __name__ == "__main__":
    # 确保 duoxiangshi.csv 文件与此脚本在同一目录下，或者提供完整路径
    # 例如: discover_polynomial_rule("c:\\path\\to\\your\\duoxiangshi.csv")
    csv_path = "duoxiangshi.csv" # 假设CSV文件在脚本的同级目录
    discover_polynomial_rule(csv_path)
    print("\n请将 'duoxiangshi.csv' 文件放到与此脚本相同的目录下，然后运行脚本。")
    print(f"或者，修改脚本中的 csv_path = \"{csv_path}\" 为文件的绝对路径。")