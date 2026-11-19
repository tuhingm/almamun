import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np, pandas as pd
from oes_engine import stat_anova as A
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
E=ROOT+'/stat-guide/examples/'
def T(r,prefix): return next(t for t in r['tables'] if t['title'].startswith(prefix))
d=pd.read_csv(E+'anorexia.csv')
r=A.ancova(d,dict(y='Postwt',group='Treat',covariates=['Prewt']))
m=smf.ols('Postwt ~ C(Treat, Sum) + Prewt',d).fit(); a=sm.stats.anova_lm(m,typ=3); print(a.round(5).values.tolist()); print(T(r,'ANCOVA')['rows'])
m2=smf.ols('Postwt ~ C(Treat) + Prewt',d).fit()
print('adjusted', T(r,'Unadjusted')['rows'])
mm=m2.get_prediction(pd.DataFrame({'Treat':['CBT','Cont','FT'],'Prewt':d.Prewt.mean()})).summary_frame(); print(mm[['mean','mean_se']].round(5).values.tolist())
print([x for x in T(r,'Assumption')['rows']])
# MANOVA iris
d=pd.read_csv(E+'iris.csv'); ys=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
r=A.manova(d,dict(y_vars=ys,group='Species'))
dd=d.rename(columns=lambda c:c.replace('.','_'))
mv=MANOVA.from_formula('Sepal_Length + Sepal_Width + Petal_Length + Petal_Width ~ Species',dd).mv_test().results['Species']['stat']; print(mv.round(5))
print(T(r,'Multivariate tests')['rows'])
print(T(r,'Assumption')['rows'])
# MANCOVA rohwer
d=pd.read_csv(E+'rohwer.csv'); d=d[d.group.isin([1,2])] if d.group.nunique()>2 else d; print(d.group.unique(), d.shape)
d=pd.read_csv(E+'rohwer.csv')
r=A.mancova(d,dict(y_vars=['SAT','PPVT','Raven'],group='group',covariates=['n','s','ns']))
mv=MANOVA.from_formula('SAT + PPVT + Raven ~ C(group) + n + s + ns',d).mv_test().results
for k in ('C(group)','n'): print(k, mv[k]['stat'].round(5).values.tolist())
for t in r['tables'][:4]: print(t['title'], t['rows'][:4])
