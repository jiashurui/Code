from statsmodels.tsa.arima.model import ARIMA
import pandas as pd

# 载入数据
df = pd.read_csv('data.csv')
# 指定时间序列数据
ts = df['timeseries_column']

# 拟合模型
model = ARIMA(ts, order=(p, d, q))
model_fit = model.fit()

# 预测
forecast = model_fit.forecast(steps=5)
print(forecast)
