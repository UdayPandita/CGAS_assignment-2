"""Q3: size- and ingredient-frequency-controlled random cuisines (10 sets)."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from common import load_recipes

rng = np.random.default_rng(7)
recipes = load_recipes()
ings = sorted({i for r in recipes.values() for i in r})
idx = {i: n for n, i in enumerate(ings)}
M = len(ings)
sizes = np.array([len(r) for r in recipes.values()])
N = len(sizes)
freq = np.zeros(M)
for r in recipes.values():
    for i in r: freq[idx[i]] += 1
prob = freq / freq.sum()

def rand_cuisine():
    """Each recipe keeps its original size; ingredients sampled w/o replacement with p proportional to
    original ingredient frequency (so expected usage of each ingredient matches the original)."""
    return [rng.choice(M, size=k, replace=False, p=prob) for k in sizes]

cuisines = [rand_cuisine() for _ in range(10)]
maxs = sizes.max()
def dist(s):
    c = np.bincount(s, minlength=maxs + 1)[1:]
    return np.arange(1, maxs + 1), c / c.sum()

# (a) size distribution
plt.figure(figsize=(8, 5))
for i, c in enumerate(cuisines):
    plt.plot(*dist(np.array([len(r) for r in c])), color='tab:green', alpha=.4, label='random cuisines (10)' if i == 0 else None)
plt.plot(*dist(sizes), 'k--', lw=2, label='original cuisine')
plt.xlabel('recipe size'); plt.ylabel('fraction of recipes'); plt.xlim(0, 40)
plt.title('Q3a: recipe size distribution'); plt.legend(); plt.tight_layout()
plt.savefig('outputs/Q3a_size_distribution.png', dpi=150); plt.close()

# (b) frequency-rank distribution
def freq_rank(counts):
    f = np.sort(counts[counts > 0])[::-1]
    return np.arange(1, len(f) + 1), f / N
plt.figure(figsize=(8, 5))
for i, c in enumerate(cuisines):
    cnt = np.bincount(np.concatenate(c), minlength=M)
    plt.loglog(*freq_rank(cnt), color='tab:green', alpha=.4, lw=1, label='random cuisines (10)' if i == 0 else None)
plt.loglog(*freq_rank(freq), 'k--', lw=2, label='original cuisine')
plt.xlabel('ingredient rank'); plt.ylabel('frequency (fraction of recipes)')
plt.title('Q3b: ingredient frequency-rank distribution'); plt.legend(); plt.tight_layout()
plt.savefig('outputs/Q3b_frequency_rank.png', dpi=150); plt.close()
