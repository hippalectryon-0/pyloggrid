"""see README.md: empirical determination of thresholds between numerical differences and higher-order Taylor terms in ETD4RK"""

import numpy as np
from matplotlib import pyplot as plt

# res = (e - 1) / (hc/h)
f = lambda x: (np.exp(x / 2) - 1) / x
g = lambda x: np.ones_like(x) * 1 / 2

x = np.logspace(-10, 0, int(1e4))
plt.loglog(x, f(x), label="numerical difference")
plt.loglog(x, g(x), "--", label="higher-order taylor")
plt.loglog(x, f(x) - g(x), ":", label="diff")
plt.axvline(x=1e-6, linestyle="--", label="threshold", color="black")
plt.xlabel("$x$")
plt.title("$(e^{x/2}-1)/x$")
plt.legend()
plt.show()

# res = (-4 - hc + e * (4 - 3 * hc + hc ** 2)) / ((hc)**3/h)
f = lambda x: (-4 - x + np.exp(x) * (4 - 3 * x + x**2)) / x**3
g = lambda x: np.ones_like(x) * 1 / 6

plt.loglog(x, f(x), label="numerical difference")
plt.loglog(x, g(x), "--", label="higher-order taylor")
plt.loglog(x, np.abs(f(x) - g(x)), ":", label="diff")
plt.axvline(x=1e-3, linestyle="--", label="threshold", color="black")
plt.xlabel("$x$")
plt.title("$(-4 - x + e^x (4 - 3x + x^2)) / x^3$")
plt.legend()
plt.show()

# res = (2 + hc + e * (-2 + hc)) / ((hc)**3/h)
f = lambda x: (2 + x + np.exp(x) * (-2 + x)) / x**3
g = lambda x: np.ones_like(x) * 1 / 6

plt.loglog(x, f(x), label="numerical difference")
plt.loglog(x, g(x), "--", label="higher-order taylor")
plt.loglog(x, np.abs(f(x) - g(x)), ":", label="diff")
plt.axvline(x=1e-3, linestyle="--", label="threshold", color="black")
plt.xlabel("$x$")
plt.title("$(2 + x + e^x (-2+x)) / x^3$")
plt.legend()
plt.show()

# res = (-4 - 3 * hc - hc ** 2 + e * (4 - hc)) / ((hc)**3/h)
f = lambda x: (-4 - 3 * x - x**2 + np.exp(x) * (4 - x)) / x**3
g = lambda x: np.ones_like(x) * 1 / 6

plt.loglog(x, f(x), label="numerical difference")
plt.loglog(x, g(x), "--", label="higher-order taylor")
plt.loglog(x, np.abs(f(x) - g(x)), ":", label="diff")
plt.axvline(x=6e-3, linestyle="--", label="threshold", color="black")
plt.xlabel("$x$")
plt.title("$(-4-3 x - x^2 + e^x (4 - x)) / x^3$")
plt.legend()
plt.show()
