import numpy as np
from matplotlib import pyplot as plt
x = np.linspace(-4, 7.5, 10000)
y = .15*(x**4) -x ** 3 + x + 2
fig, ax = plt.subplots()
ax.grid()
ax.plot(x, y)
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_yticklabels([])
ax.set_xticklabels([])
plt.show()