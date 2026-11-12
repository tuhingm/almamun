#!/usr/bin/env python3
"""Simulate the null distribution of Pesaran's (2007) CIPS statistic (intercept, one lag) on an (N, T) grid.

The CADF regression for unit i is
    dy_it = a_i + b_i y_i,t-1 + c_i ybar_t-1 + d_i0 dybar_t + d_i1 dybar_t-1 + e_i1 dy_i,t-1 + u_it
and CIPS is the cross-sectional mean of the t-statistics on b_i. Under the null every y_i is a driftless random walk.
Writes cips_quantiles.json, which panel_analysis.py embeds to turn a CIPS statistic into a p-value.
"""
import json, pathlib
import numpy as np

P = 1
NS = [5, 10, 15, 20, 30, 50, 100, 200]
TS = [10, 15, 20, 25, 30, 40, 50, 75, 100]
PROBS = [0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def cadf_t_balanced(Y, p=P):
    """Y: array (R, T, N) of levels. Returns CADF t-statistics, shape (R, N)."""
    R, T, N = Y.shape
    yb = Y.mean(axis=2, keepdims=True)
    dY, dyb = np.diff(Y, axis=1), np.diff(yb, axis=1)          # (R, T-1, N)
    dep = dY[:, p:, :]                                          # dy_t for t = p+1 .. T-1 (index in diff array)
    L = dep.shape[1]
    ylag = Y[:, p:p + L, :]                                    # y_t-1 aligned with dy_t
    yblag = np.broadcast_to(yb[:, p:p + L, :], ylag.shape)
    cols = [np.ones_like(dep), ylag, yblag, np.broadcast_to(dyb[:, p:p + L, :], dep.shape)]
    for j in range(1, p + 1):
        cols.append(np.broadcast_to(dyb[:, p - j:p - j + L, :], dep.shape))
        cols.append(dY[:, p - j:p - j + L, :])
    X = np.stack(cols, axis=-1)                                # (R, L, N, k)
    X = np.moveaxis(X, 2, 1)                                   # (R, N, L, k)
    yv = np.moveaxis(dep, 2, 1)[..., None]                     # (R, N, L, 1)
    XtX = X.swapaxes(-1, -2) @ X
    Xty = X.swapaxes(-1, -2) @ yv
    b = np.linalg.solve(XtX, Xty)
    e = yv - X @ b
    k = X.shape[-1]
    s2 = (e[..., 0] ** 2).sum(-1) / (L - k)
    inv = np.linalg.inv(XtX)
    return b[..., 1, 0] / np.sqrt(s2 * inv[..., 1, 1])


def main(reps=4000, seed=7):
    rng = np.random.default_rng(seed)
    out = {"p": P, "N": NS, "T": TS, "probs": PROBS, "q": []}
    for N in NS:
        row = []
        for T in TS:
            stats = []
            per = max(50, int(4e6 // (N * T * 10)))
            done = 0
            while done < reps:
                r = min(per, reps - done)
                Y = np.cumsum(rng.standard_normal((r, T, N)), axis=1)
                stats.append(cadf_t_balanced(Y).mean(axis=1))
                done += r
            stats = np.concatenate(stats)
            row.append([round(float(v), 4) for v in np.quantile(stats, PROBS)])
            print(N, T, row[-1][4], row[-1][5])
        out["q"].append(row)
    pathlib.Path(__file__).with_name("cips_quantiles.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
