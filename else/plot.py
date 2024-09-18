import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-8,8,100)
y=2**x
x2=np.linspace(0.001,8,100)
y2=np.log(x2) /np.log(2)

plt.figure(figsize=(5,5))
plt.plot(x,y,"black",linewidth=3)
plt.plot(x2,y2,"cornflowerblue",linewidth=3)
plt.plot(x,x,"black",linestyle="--",linewidth=1)
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.grid()
plt.show()