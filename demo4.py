import matplotlib.pyplot as plt
import numpy as np

b = -5
a = np.linspace(3, -3, 10)
x = np.arange(-5, 5, 0.1)
for a1 in a:
    y = a1 * x + b
    plt.plot(x, y, label=f'{a1:.1f}x+{b}')
    plt.legend(loc=2)
plt.axhline(0, color='black')  # horizontal 水平線
plt.axvline(0, color='black')  # vertical 垂直線
plt.show()
