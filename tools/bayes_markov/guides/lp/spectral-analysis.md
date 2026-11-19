title: Spectral Analysis: Cycles and Coherence
short: Spectral analysis
when: You ask at which cycle lengths a series moves, or whether two series move together at business-cycle frequencies
data: U.S. quarterly GDP growth and unemployment, 1959–2009
tools: Python · scipy
lead: Use this guide when your question is about cycles: how long they are, how much of a series' variance they carry, and whether two series are linked at particular frequencies. The spectrum decomposes a series' variance by cycle length, and coherence measures how strongly two series are related at each cycle length. The guide estimates a smoothed spectrum of GDP growth, finds how much variance lies in the business-cycle band, and measures the coherence between GDP growth and unemployment.
---

## The question and the model

**At which cycle lengths does GDP growth vary, and how closely do output and unemployment move together at those cycle lengths?** Any stationary series can be written as a sum of cycles of different lengths. The spectral density \(f(\omega)\) is the contribution of the cycle at frequency \(\omega\) (cycles per period) to the variance:

\[
\mathrm{Var}(y_t) = \int_{-\pi}^{\pi} f(\omega)\, d\omega .
\]

The raw periodogram is very noisy, so we use the Welch method: split the series in overlapping segments, taper each with a window, and average the periodograms. For two series, the squared coherence

\[
\mathrm{coh}^2(\omega) = \frac{|f_{xy}(\omega)|^2}{f_{xx}(\omega)\, f_{yy}(\omega)} \in [0,1]
\]

is the R-squared of the relation between them at frequency \(\omega\), and the phase of \(f_{xy}\) shows whether they move in the same direction (phase near 0) or in opposite directions (phase near \(\pm\pi\)). A frequency \(\omega\) corresponds to a period of \(2\pi/\omega\) observations, which is \(1/(\text{frequency}\times\text{periods per year})\) years.

<div class="read" markdown="1">
**Before you estimate anything**

1. **Use stationary data.** A trend dominates the lowest frequencies; remove it, or use growth rates.
2. **Choose the segment length.** Long segments give sharp peaks but a noisy estimate; short ones are smoother and blur the peaks. With 200 observations, segments of one third of the sample are a reasonable compromise.
3. **Think of the business-cycle band.** For quarterly data this is usually 6 to 32 quarters (1.5 to 8 years).
</div>

<div class="write" markdown="1">
**How to write it: methods sentence**

We estimate spectral densities with the Welch method (67-quarter Hann segments with 50% overlap) after removing a linear trend, and measure the association between GDP growth and unemployment with the squared coherence and phase, which we summarise over the business-cycle band of 1.5 to 8 years.
</div>

## Load the data

```python
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

d = pd.read_csv("us_macro.csv"); n = len(d); ppy = 4
detr = lambda v: v - np.polyval(np.polyfit(np.arange(len(v)), v, 1), np.arange(len(v)))
g, u = detr(d["gdp_growth"].to_numpy(float)), detr(d["unemp"].to_numpy(float))
print(f"{n} quarters; GDP growth: mean {d.gdp_growth.mean():.2f}, sd {d.gdp_growth.std():.2f}; unemployment: mean {d.unemp.mean():.2f}, sd {d.unemp.std():.2f}")
print("correlation of GDP growth and unemployment:", round(np.corrcoef(d.gdp_growth, d.unemp)[0, 1], 3))
```

<div class="read" markdown="1">
**How to read it**

GDP growth and the unemployment rate are negatively correlated, as Okun's law says, but the correlation at the same date is weak, since the link is between growth and the change in unemployment, which happens over several quarters. Spectral analysis shows at which cycle lengths the two series are related.
</div>

## Step 1: The spectrum of GDP growth

```python
seg = 67
fr, P = signal.welch(g, fs=1.0, nperseg=seg, window="hann", noverlap=seg // 2, detrend=False); fr, P = fr[1:], P[1:]; per = 1 / fr / ppy
top = np.argsort(P)[::-1][:5]
print(pd.DataFrame({"period (years)": per[top].round(2), "cycles per year": (fr[top] * ppy).round(3), "density": P[top].round(1), "% of power": (100 * P[top] / P.sum()).round(1)}).sort_values("density", ascending=False).to_string(index=False))
for lab, lo, hi in (("short (< 1.5 years)", 0, 1.5), ("business cycle (1.5-8 years)", 1.5, 8), ("long (> 8 years)", 8, 1e9)):
    m = (per >= lo) & (per < hi); print(f"{lab}: {100 * P[m].sum() / P.sum():.1f}% of the power")
fig, ax = plt.subplots(figsize=(8, 3.2)); ax.plot(fr * ppy, P); ax.set_xlabel("cycles per year"); ax.set_title("Smoothed spectrum of GDP growth"); plt.tight_layout(); plt.show()
```

<div class="read" markdown="1">
**How to read it**

The spectrum has no single dominant peak: the largest, at a period of 5.6 years, carries only 10% of the power, and the next ones, at 8.4, 2.8, 4.2 and 2.4 years, carry 7% each. The variance is spread across many cycle lengths, as is typical of growth rates that are close to white noise plus a business-cycle component. 43% of the power is in the business-cycle band of 1.5 to 8 years, and 46% is at shorter periods, so a large part of quarterly GDP growth is high-frequency noise. Peaks of this size in a short sample should not be read as evidence of a regular cycle.
</div>

## Step 2: How much does the spectrum depend on the segment length?

```python
rows = []
for sg in (40, 67, 100):
    f_, P_ = signal.welch(g, fs=1.0, nperseg=sg, window="hann", noverlap=sg // 2, detrend=False); f_, P_ = f_[1:], P_[1:]; p_ = 1 / f_ / ppy
    m = (p_ >= 1.5) & (p_ < 8); rows.append([sg, round(float(p_[np.argmax(P_)]), 2), round(100 * P_[m].sum() / P_.sum(), 1)])
print(pd.DataFrame(rows, columns=["segment", "period of the largest peak (years)", "% power in business-cycle band"]).to_string(index=False))
```

<div class="read" markdown="1">
**How to read it**

With segments of 40, 67 and 100 quarters, the largest peak is at 5.0, 5.6 and 5.0 years and the share of power in the business-cycle band is 38.8%, 42.7% and 42.3%. The main statement, that about 40% of the variance is in the business-cycle band, is not sensitive to the choice. Report the segment length and show this check.
</div>

## Step 3: Coherence between GDP growth and unemployment

```python
fc, C = signal.coherence(g, u, fs=1.0, nperseg=seg, window="hann", noverlap=seg // 2)
_, Pxy = signal.csd(g, u, fs=1.0, nperseg=seg, window="hann", noverlap=seg // 2)
fc, C, ph = fc[1:], C[1:], np.angle(Pxy[1:]) / np.pi; pc = 1 / fc / ppy
K = (n - seg) // (seg - seg // 2) + 1; thr = 1 - 0.05 ** (1 / (K - 1))
rows = []
for lab, lo, hi in (("short (< 1.5 years)", 0, 1.5), ("business cycle (1.5-8 years)", 1.5, 8), ("long (> 8 years)", 8, 1e9)):
    m = (pc >= lo) & (pc < hi); rows.append([lab, round(float(C[m].mean()), 3), round(float(np.median(ph[m])), 2)])
print(pd.DataFrame(rows, columns=["band", "mean squared coherence", "median phase (units of pi)"]).to_string(index=False))
print(f"\n95% significance threshold for coherence with {K} overlapping segments: {thr:.2f}")
fig, ax = plt.subplots(figsize=(8, 3.2)); ax.plot(fc * ppy, C); ax.axhline(thr, color="k", ls="--", lw=.8); ax.set_xlabel("cycles per year"); ax.set_title("Squared coherence: GDP growth and unemployment"); plt.tight_layout(); plt.show()
```

<div class="read" markdown="1">
**How to read it**

The coherence is 0.70 in the business-cycle band, 0.79 at long periods and 0.38 at short periods. The threshold for a significant coherence at 5% with this number of segments is 0.63, so the link is significant at business-cycle and longer cycles and not at short ones. The median phase is about 0.45 in the business-cycle band (in units of π), which is close to a quarter of a cycle: the two series are not in step and not opposite, but a quarter-cycle apart, which is what you expect when one is a growth rate and the other a level that responds to accumulated growth. It also explains why their correlation at the same date is only −0.08. The relation is strong for the slow movements and weak for the quarter-to-quarter noise; it is Okun's law seen by frequency. Do not read the sign of the phase as a lead or a lag without checking the convention of the software.
</div>

## What to report in the paper

Report the detrending, the window and the segment length; the share of variance by band; the coherence and phase in the band of interest with the significance threshold; and a robustness check on the segment length. Do not interpret small peaks as cycles.

<div class="write" markdown="1">
**How to write it: results**

The spectrum of GDP growth has no dominant peak: 43% of its variance is at business-cycle periods of 1.5 to 8 years and 46% at shorter periods. GDP growth and unemployment are strongly coherent at business-cycle frequencies (squared coherence 0.70, above the 5% threshold of 0.63), with a phase of about a quarter of a cycle, but the coherence is below the threshold at periods shorter than 1.5 years (0.38).
</div>

## Common mistakes

<div class="warn" markdown="1">
**Mistakes to avoid**

1. **Using the raw periodogram.** It is very noisy; smooth it.
2. **Not detrending.** The trend puts all the power at the lowest frequency.
3. **Reading every peak as a cycle.** In a short sample, noise produces peaks.
4. **Confusing frequency and period.** A period is the inverse of the frequency.
5. **Coherence without a significance threshold.** With few segments, high coherence arises by chance.
6. **A segment length that is too short or too long.** The first loses resolution, the second has few segments.
7. **Applying spectral analysis to a non-stationary series.** Take growth rates or differences first.
8. **Reading the phase where the coherence is low.** The phase is meaningless without association.
</div>
