import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')

x = np.array([1, 2, 4, 5, 9, 2, 2, 2])
y = np.array([1, 2, 4, 5, 9, 2, 2, 2])
z = np.array([1, 2, 4, 5, 9, 2, 2, 2])
bins = np.linspace(1, 10, 10)

plt.hist([x1, x2, x3, x4, x5], bins, label=['k = 1', 'k = 2', 'k = 3', 'k = 4', 'k = 5'])
plt.legend(loc='upper right')
plt.show()
