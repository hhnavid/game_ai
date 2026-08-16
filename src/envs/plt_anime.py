import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)

point, = ax.plot([], [], 'ro')

def update(frame):
    x = np.sin(frame / 10) * 5 + 5
    y = np.cos(frame / 10) * 5 + 5
    point.set_data([x], [y])
    return point,

ani = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=20)

plt.show()
