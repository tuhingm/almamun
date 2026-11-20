"""Builds bayesian-guide/examples/hprice2_poly.csv: the eight regressors of the house-price data (z-scored), their squares and all
28 pairwise products, 44 regressors in all, for the Bayesian shrinkage guide."""
import itertools, pathlib
import pandas as pd
root = pathlib.Path(__file__).resolve().parents[2]
d = pd.read_csv(root / "ml-guide/examples/hprice2.csv"); x = [c for c in d.columns if c != "lprice"]
Z = (d[x] - d[x].mean()) / d[x].std()
F = pd.DataFrame({"lprice": d["lprice"]})
for c in x: F[c] = Z[c]
for c in x: F["sq_" + c] = Z[c] ** 2
for a, b in itertools.combinations(x, 2): F[f"{a}_x_{b}"] = Z[a] * Z[b]
F.round(6).to_csv(root / "bayesian-guide/examples/hprice2_poly.csv", index=False)
print(F.shape)
