import numpy as np
import matplotlib.pyplot as plot


rng = np.random.default_rng(0)

plot.figure(figsize=(6.4, 6.4))
plot.xlim = 1
plot.ylim = 1

rs = rng.uniform(0, 1, 10)
ts = rng.uniform(0, np.pi / 2, 10)

work_xs = rs * np.cos(ts)
work_ys = rs * np.sin(ts)

plot.scatter(work_xs, work_ys)
plot.plot(np.cos(np.linspace(0, np.pi / 2, 100)), np.sin(np.linspace(0, np.pi / 2, 100)))

plot.show()
