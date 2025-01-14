"""understand why forward process looks the way it does"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = int(1e4)
num_T = 1000
beta = 1e-2

# %%
# sample x from a bi-model uniform distribution
x1 = np.random.uniform(0, 1, n) + 1.5
x2 = np.random.uniform(0, 1, n // 4) - 3.0

x0 = np.concatenate([x1, x2])
np.random.shuffle(x0)

# %%
# convert complex distribution to a simple normal one using a slow diffusion process

x = x0.copy()

for _ in range(num_T):
    x = np.sqrt(1 - beta) * x + np.sqrt(beta) * np.random.randn(x.size)

# %%
# plot the results

bins = np.linspace(-5, 5, 41)

fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True, tight_layout=True)
ax[0].hist(x0, bins=bins, density=True)
ax[1].hist(x, bins=bins, density=True)
fig.show()
