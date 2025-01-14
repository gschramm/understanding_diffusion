"""understand why forward process looks the way it does"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
n = int(1e4)
num_T = 1000
beta1 = 1e-4
betaT = 2e-2

# %%
# sample x from a bi-model uniform distribution
x1 = np.random.uniform(0, 1, n) + 1.5
x2 = np.random.uniform(0, 1, n // 4) - 3.0

x0 = np.concatenate([x1, x2])
np.random.shuffle(x0)

# %%
# convert complex distribution to a simple normal one using a slow diffusion process

x = x0.copy()
betas = np.linspace(beta1, betaT, num_T)
var = np.zeros(num_T)

for i, beta in enumerate(betas):
    x = np.sqrt(1 - beta) * x + np.sqrt(beta) * np.random.randn(x.size)
    var[i] = np.var(x)

# %%
# plot the results

bins = np.linspace(-5, 5, 41)

fig, ax = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
ax[0, 0].hist(x0, bins=bins, density=True)
ax[0, 0].set_title("distribution of x0")
ax[0, 1].hist(x, bins=bins, density=True)
ax[0, 1].set_title(f"distribution of x_{num_T}")
ax[1, 0].plot(betas)
ax[1, 0].set_title("beta")
ax[1, 1].plot(var)
ax[1, 1].set_title("variance of x_t")
fig.show()
