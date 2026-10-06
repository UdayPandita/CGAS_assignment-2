"""Q2: size-controlled random cuisines (a: replica per recipe x10 sets, b: inverse transform x10 size)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from common import load_recipes

rng = np.random.default_rng(42)
recipes = load_recipes()
ings = np.array(sorted({i for r in recipes.values() for i in r}))
sizes = np.array([len(r) for r in recipes.values()])
N, M = len(sizes), len(ings)
maxs = sizes.max()
def dist(s):
    c = np.bincount(s, minlength=maxs + 1)[1:]
    return np.arange(1, maxs + 1), c / c.sum()

# (a) replica: each random recipe has the same size as its original, ingredients drawn uniformly w/o replacement
sets_a = [[rng.choice(M, size=k, replace=False) for k in sizes] for _ in range(10)]
x, p0 = dist(sizes)
plt.figure(figsize=(8, 5))
for i, s in enumerate(sets_a):
    plt.plot(*dist(np.array([len(r) for r in s])), color='tab:blue', alpha=.4, lw=1, label='random sets (10)' if i == 0 else None)
plt.plot(x, p0, 'k--', lw=2, label='original cuisine')
plt.xlabel('recipe size'); plt.ylabel('fraction of recipes'); plt.title('Q2a: replica-based size-controlled random cuisine')
plt.legend(); plt.xlim(0, 40); plt.tight_layout(); plt.savefig('outputs/Q2a_size_distribution.png', dpi=150); plt.close()

# (b) inverse transform sampling: draw u~U(0,1), size = smallest s with CDF(s) >= u. 10x recipes.
pmf = np.bincount(sizes, minlength=maxs + 1) / N
cdf = np.cumsum(pmf)
u = rng.random(10 * N)
new_sizes = np.searchsorted(cdf, u, side='left')
new_sizes = np.clip(new_sizes, 1, maxs)
rand_b = [rng.choice(M, size=k, replace=False) for k in new_sizes]
plt.figure(figsize=(8, 5))
plt.plot(x, p0, 'k--', lw=2, label=f'original ({N} recipes)')
plt.plot(*dist(new_sizes), color='tab:red', lw=1.5, label=f'inverse transform ({10*N} recipes)')
plt.xlabel('recipe size'); plt.ylabel('fraction of recipes'); plt.title('Q2b: inverse-transform size-controlled random cuisine')
plt.legend(); plt.xlim(0, 40); plt.tight_layout(); plt.savefig('outputs/Q2b_size_distribution.png', dpi=150); plt.close()
print('max abs diff pmf (b):', np.abs(dist(new_sizes)[1] - p0).max())
