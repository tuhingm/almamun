import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from oes_engine import stat_multivariate as M
E=ROOT+'/stat-guide/examples/'
def T(r,prefix): return next(t for t in r['tables'] if t['title'].startswith(prefix))
def S(r,l): return next(s['value'] for s in r['summary'] if s['label']==l)
# C1 PCA vs sklearn
from sklearn.decomposition import PCA
d=pd.read_csv(E+'usarrests.csv'); v=['Murder','Assault','UrbanPop','Rape']; r=M.pca(d,dict(vars=v))
X=(d[v]-d[v].mean())/d[v].std(ddof=1); p=PCA().fit(X)
print('eig ours',[x[1] for x in T(r,'Eigenvalues')['rows']],'sk',p.explained_variance_.round(5).tolist())
print('loadings ours',T(r,'Loadings')['rows'][:2]); print('sk loadings',(p.components_.T*np.sqrt(p.explained_variance_))[:2].round(4).tolist())
print(T(r,'Suitability')['rows'])
# C2 EFA vs statsmodels Factor
from statsmodels.multivariate.factor import Factor
b=pd.read_csv(E+'bfi_items.csv')[['N1','N2','N3','N4','N5','E1','E2','E3','E4','E5']].dropna()
for rot in ('none','varimax','promax','oblimin'):
    r=M.efa(b,dict(vars=list(b.columns),n_factors=2,rotation=rot))
    fa=Factor(b.values,n_factor=2,method='ml').fit()
    if rot!='none': fa.rotate(rot)
    ours=np.array([[x for x in row[1:3]] for row in T(r,'Factor loadings')['rows']],float)
    ref=fa.loadings
    # align columns by max abs correlation
    import itertools
    best=None
    for perm in itertools.permutations(range(2)):
        for signs in itertools.product([1,-1],repeat=2):
            R_=ref[:,list(perm)]*np.array(signs); e=np.abs(ours-R_).max()
            if best is None or e<best[0]: best=(e,perm,signs)
    print(rot,'max abs diff loadings',round(best[0],4),'uniq diff',np.abs(np.array([row[4] for row in T(r,'Factor loadings')['rows']],float)-fa.uniqueness).max().round(5))
r=M.efa(b,dict(vars=list(b.columns),n_factors=2,rotation='varimax')); print(T(r,'Model fit')['rows'][:4])
# C3 CFA
h=pd.read_csv(E+'holzinger_swineford.csv')
model="visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9"
r=M.cfa(h,dict(model=model)); print([s for s in r['summary']]); 
for row in T(r,'Model fit')['rows'][:6]: print(row)
print(T(r,'Factor loadings')['rows'][:6])
import semopy
mod=semopy.Model(model); mod.fit(h); print(semopy.calc_stats(mod).T.loc[['chi2','DoF','CFI','TLI','RMSEA']]); print(mod.inspect().head(8))
