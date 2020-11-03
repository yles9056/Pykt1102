import matplotlib.pyplot as plt
import numpy as np

b = np.linspace(5, -5, 10)
print(b)
a = 3
x = np.arange(-5, 5, 0.1)
for b1 in b:
    y = a * x + b1
    plt.plot(x, y, label=f"y={a}x+{b1:.1f}")  # {:3f}  小數點下3位數
    plt.legend(loc=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.show()
