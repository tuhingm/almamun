import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np, pandas as pd
from scipy import stats
from oes_engine import stat_nonparam as N
E=ROOT+'/stat-guide/examples/'
def T(r,prefix): return next(t for t in r['tables'] if t['title'].startswith(prefix))
def S(r,l): return next(s['value'] for s in r['summary'] if s['label']==l)
def chk(name,a,b,tol=2e-4): print(('PASS' if abs(a-b)<=tol*max(1,abs(b)) else 'FAIL'),name,a,b)
# B1 ties (toothgrowth) and no-ties exact
d=pd.read_csv(E+'toothgrowth.csv'); a=d[d.supp=='VC'].len.values; b=d[d.supp=='OJ'].len.values  # engine orders groups by first appearance (VC first)
for alt in ('two-sided','greater','less'):
    r=N.mann_whitney(d,dict(y='len',group='supp',alternative=alt)); ref=stats.mannwhitneyu(a,b,alternative=alt,method='asymptotic',use_continuity=True)
    chk(f'B1 asym p {alt}',S(r,'p-value'),ref.pvalue); chk(f'B1 U',S(r,'U'),ref.statistic)
rng=np.random.default_rng(1); x=rng.normal(size=14); y=rng.normal(0.5,size=11); dd=pd.DataFrame({'v':np.r_[x,y],'g':['a']*14+['b']*11})
for alt in ('two-sided','greater','less'):
    r=N.mann_whitney(dd,dict(y='v',group='g',alternative=alt)); ref=stats.mannwhitneyu(x,y,alternative=alt,method='exact'); chk(f'B1 exact p {alt}',S(r,'p-value'),ref.pvalue)
# B2
d=pd.read_csv(E+'sleep_paired.csv')
for zero in ('wilcox','pratt','zsplit'):
    for alt in ('two-sided','greater','less'):
        r=N.wilcoxon(d,dict(x1='drug1',x2='drug2',alternative=alt,zero_method=zero)); ref=stats.wilcoxon(d.drug1,d.drug2,alternative=alt,zero_method=zero,method='approx',correction=True)
        chk(f'B2 {zero} {alt} p',S(r,'p-value'),ref.pvalue,1e-3)
    chk(f'B2 {zero} V',S(r,'V'),ref.statistic if alt=='less' else stats.wilcoxon(d.drug1,d.drug2,zero_method=zero,method='approx').statistic if False else S(r,'V'))
x=rng.normal(size=15); y=x+rng.normal(0.4,1,size=15); dd=pd.DataFrame({'x':x,'y':y})
for alt in ('two-sided','greater','less'):
    r=N.wilcoxon(dd,dict(x1='x',x2='y',alternative=alt)); ref=stats.wilcoxon(x,y,alternative=alt,method='exact'); chk(f'B2 exact {alt}',S(r,'p-value'),ref.pvalue)
print(T(r,'Effect size')['rows'])
# B3
d=pd.read_csv(E+'airquality_ozone.csv'); r=N.kruskal(d,dict(y='Ozone',group='Month')); ref=stats.kruskal(*[v.Ozone.values for _,v in d.groupby('Month')]); chk('B3 H',S(r,'H'),ref.statistic); chk('B3 p',S(r,'p-value'),ref.pvalue)
import scikit_posthocs as sp
dn=sp.posthoc_dunn(d,val_col='Ozone',group_col='Month',p_adjust='holm'); print(dn.round(5).values[0]); print([x[4] for x in T(r,'Dunn')['rows']][:4])
# B4
d=pd.read_csv(E+'roundingtimes.csv'); r=N.friedman(d,dict(subject='player',treatment='method',y='time')); w=d.pivot(index='player',columns='method',values='time'); ref=stats.friedmanchisquare(*[w[c] for c in w.columns]); chk('B4 chi2',S(r,'Chi-square'),ref.statistic); chk('B4 p',S(r,'p-value'),ref.pvalue); print(S(r,'Chi-square'),S(r,'p-value'),S(r,"Kendall's W"))
ne=sp.posthoc_nemenyi_friedman(w.values); print(ne.round(5).values[0]); print([x[2] for x in T(r,'Pairwise')['rows']])
co=sp.posthoc_conover_friedman(w.values,p_adjust='holm'); print(co.round(5).values[0]); print([x[4] for x in T(r,'Pairwise')['rows']])
# B5
d=pd.read_csv(E+'mtcars.csv'); r=N.kendall(d,dict(x='wt',y='mpg')); ref=stats.kendalltau(d.wt,d.mpg); chk('B5 tau',S(r,'Tau-b'),ref.statistic); chk('B5 p',S(r,'p-value'),ref.pvalue,1e-3)
rg=np.random.default_rng(2); x=rg.normal(size=20); y=0.4*x+rg.normal(size=20); dd=pd.DataFrame({'x':x,'y':y}); r=N.kendall(dd,dict(x='x',y='y')); ref=stats.kendalltau(x,y,method='exact'); chk('B5 exact p',S(r,'p-value'),ref.pvalue); chk('B5 exact tau',S(r,'Tau-b'),ref.statistic)
for alt in ('greater','less'):
    r=N.kendall(dd,dict(x='x',y='y',alternative=alt)); ref=stats.kendalltau(x,y,method='exact',alternative=alt); chk('B5 exact '+alt,S(r,'p-value'),ref.pvalue)
