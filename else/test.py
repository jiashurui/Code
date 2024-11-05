import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def calculate_ar_coefficients(data, lags=1):
    """
    计算自回归系数
    data: numpy数组，表示时间序列数据
    lags: 滞后阶数，默认设置为1阶自回归
    返回: 自回归系数
    """
    model = AutoReg(data, lags=lags, old_names=False)
    model_fit = model.fit()
    return model_fit.params[1:]  # 返回自回归系数，去掉常数项

# 示例
data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
coefficients = calculate_ar_coefficients(data, 2)
print("自回归系数:", coefficients)
