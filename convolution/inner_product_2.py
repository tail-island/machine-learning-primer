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

eval_t = np.pi / 2 * 0.8
eval_vector = np.array((np.cos(eval_t), np.sin(eval_t)))

plot.quiver(0, 0, *eval_vector, color='red', scale_units='xy', scale=1)

# ベクトルの内積を計算して、評価ベクトルでのスコアを計算します
scores = map(lambda work_vector: np.inner(eval_vector, work_vector), zip(work_xs, work_ys))

for work_x, work_y, score in zip(work_xs, work_ys, scores):
    plot.quiver(work_x, work_y, score * np.cos(eval_t) - work_x, score * np.sin(eval_t) - work_y, width=0.005, scale_units='xy', scale=1)

plot.show()
