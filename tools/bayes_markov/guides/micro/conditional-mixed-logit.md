title: Conditional and Mixed Logit (Discrete Choice)
short: Conditional and mixed logit
when: People choose one option among several (travel mode, brand, product, site, school), and the options have attributes such as price and time
data: Greene's travel-mode data: 210 travellers choosing among air, train, bus and car for intercity trips
tools: Python · scipy · numpy · OES engine
oes: choice
lead: Use this guide when each decision-maker chooses one alternative from a set and you want to know how the attributes of the alternatives (cost, time) and the characteristics of the person (income) drive the choice. The conditional logit of McFadden is the workhorse: it gives choice probabilities, willingness-to-pay ratios and elasticities. Its main limitation is that it forces the same substitution pattern on all alternatives (independence of irrelevant alternatives), and it gives every traveller the same taste. The mixed logit relaxes both by letting coefficients vary across people. This guide takes the travel-mode data from the conditional logit to the mixed logit, computes the quantities a transport economist reports, and tests the independence assumption, with sample text for each part of the paper.
---

## The question and the model

**How do cost and travel time affect the choice of travel mode?** Traveller \(i\) chooses mode \(j\) with the highest utility

\[
U_{ij} = x_{ij}'\beta + \alpha_j + w_i'\delta_j + \varepsilon_{ij},
\]

where \(x_{ij}\) are attributes of the mode as the traveller faces them (cost, times), \(\alpha_j\) are mode-specific constants, \(w_i\) are traveller characteristics with mode-specific effects \(\delta_j\) (one mode is the base), and \(\varepsilon_{ij}\) is Gumbel. This gives the closed-form choice probability

\[
P_{ij} = \frac{\exp(x_{ij}'\beta + \alpha_j + w_i'\delta_j)}{\sum_k \exp(x_{ik}'\beta + \alpha_k + w_i'\delta_k)} .
\]

In the mixed logit some coefficients are random, \(\beta_i \sim N(\mu, \sigma^2)\), and the probability is the average of the logit formula over the distribution of \(\beta_i\), computed by simulation.

| Variable | Definition |
|---|---|
| `id`, `alt` | traveller and mode (air, train, bus, car); four rows per traveller |
| `choice` | 1 for the mode the traveller chose |
| `ttme` | terminal (waiting) time in minutes; 0 for car |
| `invc`, `invt` | in-vehicle cost in dollars and in-vehicle time in minutes |
| `gc` | generalized cost (cost plus time valued at the traveller's wage) |
| `hinc`, `psize` | household income (thousand dollars) and party size (do not vary across the modes) |

<div class="read" markdown="1">
**Before you estimate anything**

1. **Is the data in the right shape?** One row per traveller and alternative, an identifier for the traveller, one for the alternative and a 0/1 indicator that sums to 1 within each traveller.
2. **Which variables vary across alternatives?** Cost and time do, and enter with a single coefficient. Income does not, so it can only enter through coefficients that differ by mode (relative to a base mode).
3. **Is the substitution pattern plausible?** The conditional logit says that adding an alternative takes probability from all others in proportion to their shares. Test this.
</div>

<div class="write" markdown="1">
**How to write it: model section**

We model each traveller's choice among air, train, bus and car as a conditional logit in which the utility of a mode depends on the terminal time, in-vehicle cost, in-vehicle time and generalized cost of that mode, plus mode-specific constants (car is the base). We compute willingness-to-pay ratios and elasticities with the delta method, test the independence of irrelevant alternatives with a Hausman-McFadden test, and estimate a mixed logit in which the coefficient on terminal time is normally distributed across travellers.
</div>

## Load the data and look at the choices

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, optimize
from statsmodels.tools.numdiff import approx_hess

d = pd.read_csv("travel_mode.csv")
alts = ["air", "bus", "car", "train"]                                   # sorted; car is the base alternative
N = d.id.nunique()
print(f"{N} travellers, {d.alt.nunique()} modes")
ch = d[d.choice == 1].alt.value_counts(normalize=True).reindex(alts)
print("\nShare choosing each mode:", ch.round(3).to_dict())
print("\nAverage attributes of each mode")
print(d.groupby("alt")[["ttme", "invc", "invt", "gc"]].mean().reindex(alts).round(1).to_string())
```

<div class="read" markdown="1">
**How to read it**

The shares are balanced: train 30%, air 28%, car 28%, bus 14%. Air is the fastest in the vehicle (about 134 minutes) but the most expensive ($85) and needs the longest terminal time (about 61 minutes); car has no terminal time and is the cheapest ($21); train and bus are slow (about 10 hours in the vehicle) and cheaper than air. This variation in attributes across modes is what identifies the coefficients.
</div>

## Step 1: Conditional logit

```python
# arrays: travellers x modes x attributes, in the order of `alts`
d["_a"] = pd.Categorical(d.alt, alts); d = d.sort_values(["id", "_a"])
XA = d[["ttme", "invc", "invt", "gc"]].to_numpy(float).reshape(N, 4, 4)
ch_idx = d.choice.to_numpy().reshape(N, 4).argmax(1); ar = np.arange(N)
base = alts.index("car"); others = [j for j in range(4) if j != base]

def util(t, XA=XA):
    U = XA @ t[:4]
    for m, j in enumerate(others): U[:, j] += t[4 + m]
    return U
def nll(t):
    U = util(t); m = U.max(1, keepdims=True)
    return -(U[ar, ch_idx] - (m[:, 0] + np.log(np.exp(U - m).sum(1)))).sum()
o = optimize.minimize(nll, np.zeros(7), method="BFGS", options={"gtol": 1e-9})
V = np.linalg.inv(approx_hess(o.x, nll)); se = np.sqrt(np.diag(V)); th = o.x
names = ["ttme", "invc", "invt", "gc"] + [f"const: {alts[j]}" for j in others]
print(pd.DataFrame({"coef": th, "s.e.": se, "z": th / se}, index=names).round(4).to_string())
ll = -o.fun; ll0 = -N * np.log(4)
ll_c = sum((ch_idx == j).sum() * np.log((ch_idx == j).mean()) for j in range(4))
print(f"\nlog-likelihood {ll:.2f}; equal shares {ll0:.2f}; market shares {ll_c:.2f}; McFadden R2 = {1 - ll / ll0:.3f} (against market shares {1 - ll / ll_c:.3f})")
```

<div class="read" markdown="1">
**How to read it**

Longer terminal time, higher cost and longer in-vehicle time all lower the utility of a mode (z statistics between −4.4 and −9.5), and the signs are as expected; the coefficient on generalized cost is positive because it is highly collinear with the cost and time variables it is built from, so read it jointly with the others. The log-likelihood is −184.5. The constants say that, with everything else equal, air, bus and train are all preferred to car in this sample. The model has a McFadden R-squared of 0.37 against equal shares and 0.35 against market shares.
</div>

## Step 2: Willingness to pay and elasticities

```python
# Ratios of coefficients: dollars per minute of time. Delta method for the standard error.
def ratio(i, k):
    g = np.zeros(len(th)); g[i] = 1 / th[k]; g[k] = -th[i] / th[k] ** 2
    return th[i] / th[k], np.sqrt(g @ V @ g)
for nm, i in [("in-vehicle time", 2), ("terminal time", 0)]:
    r_, s_ = ratio(i, 1)
    print(f"{nm}: {r_:.3f} dollars per minute  (s.e. {s_:.3f})  = {60 * r_:.1f} dollars per hour")

def probs(t):
    U = util(t); e = np.exp(U - U.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
P = probs(th)
print("\nAverage own-price elasticity (cost) by mode:", {a: round(float((th[1] * XA[:, j, 1] * (1 - P[:, j])).mean()), 2) for j, a in enumerate(alts)})
print("Average cross-price elasticity (effect on every other mode when the cost of one mode rises):", {a: round(float((-th[1] * XA[:, j, 1] * P[:, j]).mean()), 2) for j, a in enumerate(alts)})
print("Predicted shares:", dict(zip(alts, P.mean(0).round(3))), " observed:", dict(zip(alts, np.bincount(ch_idx, minlength=4) / N)))
```

<div class="read" markdown="1">
**How to read it**

The cost coefficient converts the time coefficients into dollars. A minute of in-vehicle time is worth 16 cents, or $9.4 an hour, and a minute of terminal time is worth $1.22, or $73 an hour, about eight times as much: waiting at a terminal is much more costly than being on the move. The own-cost elasticities are −5.0 for air and −1.4 for car, so air demand is very sensitive to price and car demand is not. The cross elasticities are identical for every other mode (a feature of the conditional logit that the next step tests), and the predicted shares equal the observed shares by construction, since the constants fit them exactly.
</div>

<div class="write" markdown="1">
**How to write it: main result**

Travellers value a minute of in-vehicle time at 16 cents (standard error 1.6 cents), or $9.4 per hour, and a minute of terminal time at $1.22, or $73 per hour. Demand for air travel is elastic with respect to its cost (elasticity −5.0), while demand for car travel is not (−1.4). The model reproduces the observed mode shares and has a McFadden R-squared of 0.37.
</div>

## Step 3: Is the independence of irrelevant alternatives plausible?

```python
# Hausman-McFadden: drop one mode (and the travellers who chose it), re-estimate the model on the remaining modes, and compare the
# coefficients of the alternative-specific variables with the full-model ones.
def refit(drop):
    keep = [j for j in range(4) if j != drop]; mask = ch_idx != drop
    XA2, ch2 = XA[mask][:, keep, :], np.array([keep.index(c) for c in ch_idx[mask]]); b2 = keep.index(base) if base in keep else 0
    oth2 = [j for j in range(len(keep)) if j != b2]
    def u2(t):
        U = XA2 @ t[:4]
        for m, j in enumerate(oth2): U[:, j] += t[4 + m]
        return U
    def f2(t):
        U = u2(t); m = U.max(1, keepdims=True); return -(U[np.arange(len(ch2)), ch2] - (m[:, 0] + np.log(np.exp(U - m).sum(1)))).sum()
    r = optimize.minimize(f2, np.zeros(4 + len(oth2)), method="BFGS", options={"gtol": 1e-9})
    return r.x, np.linalg.inv(approx_hess(r.x, f2))
rows = []
for drop in [0, 1, 3]:
    b_r, V_r = refit(drop)
    diff = b_r[:4] - th[:4]; Dv = V_r[:4, :4] - V[:4, :4]
    ev, evec = np.linalg.eigh((Dv + Dv.T) / 2); pos = ev > 1e-10
    stat_ = ((evec[:, pos].T @ diff) ** 2 / ev[pos]).sum()
    rows.append([alts[drop], stat_, int(pos.sum()), stats.chi2.sf(stat_, pos.sum())])
print(pd.DataFrame(rows, columns=["mode dropped", "chi2", "df", "p-value"]).round(4).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

Removing air (chi-squared 34), bus (46) or train (30) changes the coefficients of the attributes far more than sampling error allows, and the test rejects the independence of irrelevant alternatives in each case. The traveller who is choosing between train and bus is not choosing in the same way when air is available as when it is not: there is a common unobserved taste for public transport. This rejects the strict substitution pattern of the conditional logit, and the results should be read as an approximation. Two remedies are a nested logit (public transport in one nest) and a mixed logit.
</div>

## Step 4: Mixed logit with a random coefficient on terminal time

```python
# Simulated maximum likelihood with R Halton draws per traveller; only the terminal-time coefficient is random (normal).
def halton(n, b):
    out = np.zeros(n)
    for i in range(n):
        f, r, k = 1.0, 0.0, i + 1
        while k > 0:
            f /= b; r += f * (k % b); k //= b
        out[i] = r
    return out
R = 100
eta = stats.norm.ppf(halton(N * R, 2).reshape(N, R))
XA3 = XA[:, :, :3]
from scipy.special import logsumexp
def util3(t):
    U = (XA3 @ t[:3])[:, :, None] + XA3[:, :, [0]] * (t[6] * eta[:, None, :])
    for m, j in enumerate(others): U[:, j, :] += t[3 + m]
    return U
def nll_mixed(t):
    U = util3(t); lp = U - logsumexp(U, axis=1, keepdims=True)
    return -(logsumexp(lp[ar, ch_idx, :], axis=1) - np.log(R)).sum()
# conditional logit on the same three attributes, for comparison
def nll3(t):
    U = XA3 @ t[:3]
    for m, j in enumerate(others): U[:, j] += t[3 + m]
    m_ = U.max(1, keepdims=True); return -(U[ar, ch_idx] - (m_[:, 0] + np.log(np.exp(U - m_).sum(1)))).sum()
c3 = optimize.minimize(nll3, np.zeros(6), method="BFGS", options={"gtol": 1e-9})
mx = optimize.minimize(nll_mixed, np.r_[c3.x, 0.3], method="BFGS", options={"gtol": 1e-7})
Vm = np.linalg.inv(approx_hess(mx.x, nll_mixed)); sem = np.sqrt(np.diag(Vm))
nm = ["ttme (mean)", "invc", "invt"] + [f"const: {alts[j]}" for j in others] + ["sd(ttme)"]
print(pd.DataFrame({"conditional logit": np.r_[c3.x, np.nan], "mixed logit": mx.x, "s.e.": sem}, index=nm).round(4).to_string())
lr = 2 * (-mx.fun + c3.fun)
print(f"\nlog-likelihood: conditional logit {-c3.fun:.2f}, mixed logit {-mx.fun:.2f};  LR test of no heterogeneity: {lr:.1f} (p < 0.001)")
print(f"Share of travellers with a positive (wrong-sign) terminal-time coefficient: {stats.norm.sf(-mx.x[0] / abs(mx.x[6])):.3f}")
```

<div class="read" markdown="1">
**How to read it**

Allowing the terminal-time coefficient to vary across travellers improves the log-likelihood from −193 to −178 (comparing models with the same three attributes), which rejects a common coefficient at any usual level (chi-squared 29.5). The mean coefficient is −0.18 and its standard deviation 0.11, so about 4% of travellers would have a positive coefficient. Terminal time is zero for car, so its coefficient also reflects how strongly each traveller dislikes the public modes relative to car, and the random coefficient captures part of the unobserved taste for car that the independence of irrelevant alternatives test detected. Mixed logit needs a choice of distribution and draws; check that the results are stable when you increase the number of draws.
</div>

## What to report in the paper

Report the number of travellers and alternatives; the coefficients, the base alternative and the constants; the log-likelihood and McFadden R-squared; willingness-to-pay ratios and elasticities; the test of independence of irrelevant alternatives; and, if you use a mixed logit, the distributions assumed, the number of draws and a likelihood-ratio test against the conditional logit.

<div class="write" markdown="1">
**How to write it: results and robustness**

Table 5 reports the conditional logit. A Hausman-McFadden test rejects the independence of irrelevant alternatives (chi-squared 30 to 46, p < 0.001 when dropping any of air, bus or train), so we also estimate a mixed logit with a random coefficient on terminal time (100 Halton draws). The mixed logit fits significantly better (likelihood-ratio statistic 29.5) with a mean coefficient of −0.18 and a standard deviation of 0.11. The cost and in-vehicle-time coefficients are similar in the two models (cost −0.021 against −0.014, in-vehicle time −0.0066 against −0.0040), so the willingness-to-pay ratios in the conditional logit should be treated as approximate.
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Wrong data shape.** The conditional logit needs one row per person and alternative, not one row per person.
2. **Putting a case-specific variable in with a single coefficient.** Income does not vary across modes, so it cancels out; it must have a mode-specific coefficient.
3. **Reading constants as preferences without the base.** Constants are relative to the base alternative.
4. **Ignoring the independence of irrelevant alternatives.** Test it, and use a nested or mixed logit if it fails.
5. **Including a generalized cost variable together with its components.** It is collinear with them and the signs become uninterpretable.
6. **Few draws in a mixed logit.** Use at least 100 Halton draws and check stability.
7. **Interpreting a normal random coefficient with a large spread literally.** The normal puts weight on the wrong sign; consider a lognormal for cost.
8. **Computing elasticities without averaging.** Own and cross elasticities differ across people; report the average over the sample.
</div>
