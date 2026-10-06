import pandas as pd
def load_recipes(path='recipe_id_ingredient_name.csv'):
    d = pd.read_csv(path).dropna().drop_duplicates()
    d['ingredient_name'] = d['ingredient_name'].str.strip().str.lower()
    d = d.drop_duplicates()
    return {rid: sorted(set(g)) for rid, g in d.groupby('recipe_id')['ingredient_name']}
