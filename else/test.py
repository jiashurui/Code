import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 创建示例 DataFrame
data = {
    'Feature1': [1.0, 2.0, 3.0, 4.0],
    'Feature2': [5.0, 6.0, 7.0, 8.0],
    'Feature3': [9.0, 10.0, 11.0, 12.0],
}
df = pd.DataFrame(data)

# 初始化 MinMaxScaler
scaler = StandardScaler()

# 对整个 DataFrame 进行归一化
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

# 打印归一化后的 DataFrame
print("Min-Max Normalized DataFrame:")
print(df_normalized)
