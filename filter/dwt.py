import numpy as np
import pywt
import matplotlib.pyplot as plt
from PIL import Image

# 加载图像并转换为灰度
image = Image.open('/Users/jiashurui/Downloads/123.jpeg').convert('L')
image = np.array(image)

# 选择小波并执行DWT
coeffs = pywt.dwt2(image, 'haar')
cA, (cH, cV, cD) = coeffs

# 显示结果
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(cA, cmap='gray')
axs[0, 0].set_title('Approximation Coefficients', fontsize=12)
axs[0, 1].imshow(cH, cmap='gray')
axs[0, 1].set_title('Horizontal Detail Coefficients', fontsize=12)
axs[1, 0].imshow(cV, cmap='gray')
axs[1, 0].set_title('Vertical Detail Coefficients', fontsize=12)
axs[1, 1].imshow(cD, cmap='gray')
axs[1, 1].set_title('Diagonal Detail Coefficients', fontsize=12)

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
