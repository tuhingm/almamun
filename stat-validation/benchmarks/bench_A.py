import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys, json; sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np, pandas as pd
from scipy import stats
from oes_engine import stat_anova as A
E=ROOT+'/stat-guide/examples/'
def T(r,prefix): return next(t for t in r['tables'] if t['title'].startswith(prefix))
def S(r,l): return next(s['value'] for s in r['summary'] if s['label']==l)
def chk(name,a,b,tol=2e-4):
    print(('PASS' if abs(a-b)<=tol*max(1,abs(b)) else 'FAIL'),name,a,b)
# A1
d=pd.read_csv(E+'plantgrowth.csv'); r=A.anova1(d,dict(y='weight',group='group'))
g=[v.weight.values for _,v in d.groupby('group')]; ref=stats.f_oneway(*g)
chk('A1 F',S(r,'F'),ref.statistic); chk('A1 p',S(r,'p-value'),ref.pvalue)
import pingouin as pg
pa=pg.anova(data=d,dv='weight',between='group',detailed=True); chk('A1 eta2',S(r,'Eta squared'),pa['np2'][0])
w=pg.welch_anova(data=d,dv='weight',between='group'); chk('A1 welch p',S(r,'Welch p'),w['p_unc'][0])
th=pg.pairwise_tukey(data=d,dv='weight',between='group'); print(th.columns.tolist(), th.iloc[:,[0,1,-5,-4,-3]].round(4).values.tolist() if False else th.round(4).values.tolist()); print([ (x[0],x[1],x[4]) for x in T(r,'Pairwise')['rows']])
gh=pg.pairwise_gameshowell(data=d,dv='weight',between='group'); r2=A.anova1(d,dict(y='weight',group='group',posthoc='games_howell')); print(gh.columns.tolist(), gh.round(5).values.tolist()); print([(x[0],x[4]) for x in T(r2,'Pairwise')['rows']])
# A2
d=pd.read_csv(E+'toothgrowth.csv'); d['dose']=d['dose'].astype(str)
import statsmodels.api as sm, statsmodels.formula.api as smf
for kind in ('II','III'):
    r=A.anova2(d,dict(y='len',factor_a='supp',factor_b='dose',ss_type=kind))
    m=smf.ols('len ~ C(supp, Sum)*C(dose, Sum)',d).fit(); a=sm.stats.anova_lm(m,typ=int(len(kind)) if False else {'II':2,'III':3}[kind])
    print(kind,[(x[0],x[1],x[5]) for x in T(r,'Two-way')['rows'][:3]]); print(a[['sum_sq','PR(>F)']].round(5).values.tolist())
# unbalanced
du=d.drop(index=[0,1,2,30,31,45]); r=A.anova2(du,dict(y='len',factor_a='supp',factor_b='dose',ss_type='III'))
m=smf.ols('len ~ C(supp, Sum)*C(dose, Sum)',du).fit(); a=sm.stats.anova_lm(m,typ=3); print('unbal III',[x[1] for x in T(r,'Two-way')['rows'][:3]],a['sum_sq'].round(4).values.tolist())
r=A.anova2(du,dict(y='len',factor_a='supp',factor_b='dose',ss_type='II')); a=sm.stats.anova_lm(m,typ=2); print('unbal II',[x[1] for x in T(r,'Two-way')['rows'][:3]],a['sum_sq'].round(4).values.tolist())
# A3
d=pd.read_csv(E+'sleepstudy.csv'); d=d[d.Days<=4]
r=A.rm_anova(d,dict(subject='Subject',within='Days',y='Reaction'))
pr=pg.rm_anova(data=d,dv='Reaction',within='Days',subject='Subject',detailed=True,correction=True); print(pr.round(5).to_string())
print(T(r,'Repeated-measures')['rows'][0]); print(T(r,'Sphericity')['rows'])
sp=pg.sphericity(d.pivot(index='Subject',columns='Days',values='Reaction'),method='mauchly'); print(sp)
