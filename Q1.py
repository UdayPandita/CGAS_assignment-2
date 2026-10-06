"""Q1: Frequent item set mining with Apriori (size 1, 2, 3)."""
from itertools import combinations
from collections import Counter
import pandas as pd
from common import load_recipes

MIN_COUNT = 2  # an itemset must occur in >= 2 recipes to be listed
recipes = load_recipes()
N = len(recipes)
tx = [frozenset(v) for v in recipes.values()]
print('recipes:', N)

def apriori_fast(tx, min_count, max_k=3):
    levels = {}
    c1 = Counter(i for t in tx for i in t)
    f1 = {i for i, c in c1.items() if c >= min_count}
    levels[1] = {(i,): c for i, c in c1.items()}  # size 1 list is complete (count>=1)
    ftx = [sorted(i for i in t if i in f1) for t in tx]
    c2 = Counter(p for t in ftx for p in combinations(t, 2))
    levels[2] = {k: c for k, c in c2.items() if c >= min_count}
    f2 = set(levels[2])
    c3 = Counter()
    for t in ftx:
        for tri in combinations(t, 3):
            # Apriori pruning: all 2-subsets must be frequent
            if (tri[0], tri[1]) in f2 and (tri[0], tri[2]) in f2 and (tri[1], tri[2]) in f2:
                c3[tri] += 1
    levels[3] = {k: c for k, c in c3.items() if c >= min_count}
    return levels

levels = apriori_fast(tx, MIN_COUNT)
rows = []
for k, d in levels.items():
    df = pd.DataFrame([{'itemset': ' | '.join(s), 'size': k, 'count': c, 'support': c / N}
                       for s, c in d.items()])
    df = df.sort_values(['count', 'itemset'], ascending=[False, True]).reset_index(drop=True)
    df.to_csv(f'outputs/Q1a_itemsets_size{k}.csv', index=False)
    df.head(20).to_csv(f'outputs/Q1b_top20_size{k}.csv', index=False)
    print(f'\nsize {k}: {len(df)} frequent itemsets (count>={MIN_COUNT}); top 20:')
    print(df.head(20).to_string())

# confidence examples for top size-2 itemsets: conf(A->B) = supp(AB)/supp(A)
single = {s[0]: c for s, c in levels[1].items()}
rules = []
for (a, b), c in sorted(levels[2].items(), key=lambda x: -x[1])[:20]:
    rules.append({'rule': f'{a} -> {b}', 'support': c / N, 'confidence': c / single[a]})
    rules.append({'rule': f'{b} -> {a}', 'support': c / N, 'confidence': c / single[b]})
pd.DataFrame(rules).to_csv('outputs/Q1c_example_rules.csv', index=False)

open('outputs/Q1c_definitions.txt', 'w').write("""Support: the fraction (or count) of transactions (recipes) that contain an itemset.
  support(X) = |{recipes containing all items of X}| / |all recipes|.
  Itemsets with support >= a user-chosen minimum threshold are 'frequent'. Support is anti-monotone:
  a superset can never have higher support than its subsets (basis of Apriori pruning).

Confidence: for an association rule X -> Y, the conditional probability that a recipe containing X
  also contains Y.  confidence(X -> Y) = support(X u Y) / support(X).
  It measures how reliable the rule is (e.g. conf(onion -> garlic)=0.4: 40% of onion recipes also have garlic).
""")
