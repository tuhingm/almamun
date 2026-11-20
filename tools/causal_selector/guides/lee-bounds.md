title: Lee Bounds for Sample Selection and Attrition
short: Lee bounds
when: An outcome is observed only for some units, treatment changes who is observed, and you want an effect without a selection model
data: National Supported Work demonstration (LaLonde 1986): 445 workers randomly assigned to job training; the outcome, 1978 earnings among those who worked, exists only for workers with a job
tools: Python · OES engine · NumPy
oes: leebounds
lead: Use this guide when treatment is randomised but the outcome is observed only for a selected subset and treatment changes who is selected, as with wages (seen only for the employed), test scores (seen only for those who stay in school) or follow-up surveys with attrition. Comparing the observed outcomes mixes the treatment effect with a change in who is observed. Lee (2009) bounds the effect for the units that would be observed under either arm by trimming the arm with the higher selection rate until the two selection rates match. The guide builds the bounds, compares them with the worst-case bounds that assume nothing, and runs a placebo on a pre-treatment outcome.
---

## The question and the assumptions

**What is the effect of treatment on an outcome that is seen only for some units?** Let \(S\) indicate that the outcome is observed. If treatment raises \(P(S=1)\), the treated and control units who are observed are not comparable, so the plain difference is biased even under random assignment. Lee's argument: if treatment can only move units into selection, never out (**monotonicity**), the observed treated group is a mix of units that would be selected anyway and units pulled in by treatment. The share of the second kind is

\[
p = \frac{P(S=1\mid T=1) - P(S=1\mid T=0)}{P(S=1\mid T=1)} .
\]

Trimming the top \(p\) of the treated outcomes gives the lowest possible treated mean for the always-selected, and trimming the bottom \(p\) gives the highest. Subtracting the control mean gives a lower and an upper bound on the effect for the always-selected. The bounds are wide when \(p\) is large.

<div class="read" markdown="1">
**Three checks before you report Lee bounds**

1. **Is treatment randomised?** The method needs random assignment, or ignorability, to compare the arms at all.
2. **Is monotonicity plausible?** Ask whether treatment could plausibly push some units out of selection while pulling others in. If so, the bounds are not valid.
3. **Is the effect for the always-selected the quantity you need?** It is not the effect for everyone, and the always-selected cannot be identified unit by unit.
</div>

<div class="write" markdown="1">
**How to write it: the model**

We ask whether job training raised the earnings of workers in the National Supported Work demonstration. Earnings are observed as a wage-like outcome only for workers with a job in 1978, and assignment to training raised employment, so the observed earnings of the two arms are not comparable. We estimate Lee (2009) trimming bounds for the effect on earnings of workers who would have a job whether or not they were trained, assuming that training can only raise the chance of employment. Confidence intervals take the 2.5th percentile of the lower bound and the 97.5th percentile of the upper bound over 500 bootstrap resamples (Imbens and Manski 2004).
</div>

## Load the data and look at selection

```python
import numpy as np
import pandas as pd
import oes_engine as oes            # pip install -e open-econometric-systems/oes-engine

data = open("nsw_attrition.csv").read()
df = pd.read_csv("nsw_attrition.csv")
print(len(df), "workers;", int(df.train.sum()), "treated")
print(df.groupby("train")[["employed78", "earn_if_employed", "re78"]].mean().round(3))
print("observed earnings missing for", int(df.earn_if_employed.isna().sum()), "workers without a job")
```

<div class="read" markdown="1">
**How to read it**

Of 445 workers, 185 were assigned to training. In 1978, 75.7% of the trained had a job against 64.6% of the controls, so training raised the chance of observing earnings by 11 percentage points, and earnings among the employed are 8.39 for the trained against 7.05 for the controls (thousand dollars). The difference of 1.34 is not the effect of training on earnings: the trained group includes 14.6% of workers who would not have had a job without training, whose earnings may be lower or higher than those of the rest.
</div>

## Estimate the bounds

```python
spec = dict(outcome="earn_if_employed", treat="train", selected="employed78", boot=500)
r = oes.run("leebounds", data, spec)
oes.summary(r)
oes.show(r, "Selection")
oes.show(r, "Effect of")
```

<div class="read" markdown="1">
**How to read it**

The share to trim is 14.6% of the treated workers with a job. After trimming the top 14.6% the lower bound on the effect is -1.14, and after trimming the bottom 14.6% the upper bound is 2.62, so the effect of training on the earnings of workers who would be employed either way lies between -1.14 and 2.62 thousand dollars. With sampling error the interval is -2.56 to 4.47. The naive difference of 1.34 sits inside the bounds, but so does zero: selection alone prevents us from saying that training raised the earnings of the employed. The result that training raised total earnings (a gain of 1.79 per worker in this sample) therefore comes from getting people into jobs and not from higher pay once employed, which the data cannot establish.
</div>

## Compare with bounds that assume nothing

```python
for lo, hi in [(None, None), (0, 25), (0, 15)]:
    s = dict(spec, boot=200)
    if lo is not None: s.update(ymin=lo, ymax=hi)
    rr = oes.run("leebounds", data, s)
    row = [x for x in rr["tables"][1]["rows"] if x[0].startswith("Manski")][0]
    print(row[0], "->", row[1])
```

<div class="read" markdown="1">
**How to read it**

The worst-case bounds fill the missing outcomes with the lowest and highest possible values and make no assumption about who is selected. With the observed range of earnings (0.04 to 60.31) they run from -19.5 to 16.4, which says nothing; restricting earnings to the range 0 to 25 narrows them to -7.1 to 7.9, and the tighter the assumed range the more informative they become. The Lee bounds, from -1.14 to 2.62, are far narrower because the monotonicity assumption does the work that the range assumption does here. The price is that the bounds apply only to the always-selected.
</div>

## Placebo on a pre-treatment outcome

```python
d2 = df.copy()
d2["e75"] = (d2.re75 > 0).astype(int)
d2["re75_if_positive"] = d2.re75.where(d2.e75 == 1)
r0 = oes.run("leebounds", d2.to_csv(index=False), dict(outcome="re75_if_positive", treat="train", selected="e75", boot=500))
oes.summary(r0)
oes.show(r0, "Effect of")
```

<div class="read" markdown="1">
**How to read it**

Earnings in 1975 were fixed before assignment, so the true effect of training on them is zero. Treating positive earnings in 1975 as the selection indicator, 40.0% of the treated and 31.5% of the controls are selected, a gap of 8.5 points that arose by chance. The Lee bounds are -1.77 to 0.70 and contain zero, as they should, and the interval of -3.18 to 2.65 is wide because 21% of the treated are trimmed. A placebo that excludes zero would show that the design, not the programme, is producing the result.
</div>

## Write the results

<div class="write" markdown="1">
**How to write it: results**

Assignment to training raised employment in 1978 by 11 percentage points (75.7% against 64.6%), so earnings among the employed cannot be compared directly. Lee bounds for the earnings effect on workers who would be employed with or without training are -1.14 to 2.62 thousand dollars, with a 95% bootstrap interval of -2.56 to 4.47, trimming 14.6% of the treated workers. The bounds include zero and cannot rule out either a small loss or a gain of several thousand dollars. They are far narrower than worst-case bounds (-7.1 to 7.9 for earnings between 0 and 25), but they rely on monotonicity, that training does not lead any worker to give up a job they would otherwise have had. A placebo on earnings in 1975 gives bounds of -1.77 to 0.70, which contain the true effect of zero. We conclude that the gain in total earnings comes mainly from higher employment, and that the data do not identify an effect on pay for those already employed.
</div>

| Report | What | Where |
|---|---|---|
| Design | Randomisation, selection and monotonicity | Step 1 |
| Data | Selection rates by arm | Step 2 |
| Bounds | Lee bounds, trimming share, bootstrap interval | Step 3 |
| Worst case | Manski bounds under assumed outcome ranges | Step 4 |
| Placebo | Bounds on a pre-treatment outcome | Step 5 |

**References.** Lee, D. (2009), "Training, Wages, and Sample Selection: Estimating Sharp Bounds on Treatment Effects," *Review of Economic Studies* 76. Manski, C. (1990), "Nonparametric Bounds on Treatment Effects," *American Economic Review* 80. Imbens, G. and Manski, C. (2004), "Confidence Intervals for Partially Identified Parameters," *Econometrica* 72. Horowitz, J. and Manski, C. (2000), "Nonparametric Analysis of Randomized Experiments with Missing Covariate and Outcome Data," *Journal of the American Statistical Association* 95. Behaghel, L., Crépon, B., Gurgand, M. and Le Barbanchon, T. (2015), "Please Call Again: Correcting Nonresponse Bias in Treatment Effect Models," *Review of Economics and Statistics* 97.
