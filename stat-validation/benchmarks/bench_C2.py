import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys; sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from oes_engine import stat_multivariate as M
E=ROOT+'/stat-guide/examples/'
def T(r,prefix): return next(t for t in r['tables'] if t['title'].startswith(prefix))
def S(r,l): return next(s['value'] for s in r['summary'] if s['label']==l)
# LDA iris vs MASS::lda known values and sklearn
d=pd.read_csv(E+'iris.csv'); v=['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
r=M.lda(d,dict(vars=v,group='Species')); print([ (x['label'],x['value']) for x in r['summary']])
print(T(r,'Discriminant function coefficients')['rows']); print(T(r,'Discriminant functions')['rows']); print(T(r,'Wilks')['rows'])
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import LeaveOneOut, cross_val_score
m=LDA().fit(d[v],d.Species); print('sk acc',m.score(d[v],d.Species),'loo',cross_val_score(LDA(),d[v],d.Species,cv=LeaveOneOut()).mean(), 'ratio',m.explained_variance_ratio_)
# CCA linnerud vs sklearn
l=pd.read_csv(E+'linnerud.csv'); xv=['Chins','Situps','Jumps']; yv=['Weight','Waist','Pulse']
r=M.cca(l,dict(x_vars=xv,y_vars=yv,perms=499)); print(T(r,'Canonical correlations')['rows'])
from sklearn.cross_decomposition import CCA
c=CCA(n_components=3,scale=True,max_iter=5000,tol=1e-12).fit(l[xv],l[yv]); u,vv=c.transform(l[xv],l[yv]); print('sk',[round(np.corrcoef(u[:,i],vv[:,i])[0,1],5) for i in range(3)])
from statsmodels.multivariate.cancorr import CanCorr
cc=CanCorr(l[yv],l[xv]); print('sm',cc.cancorr.round(5)); print(cc.corr_test().summary()) if False else None
print(T(r,'Permutation')['rows'])
# CA caith vs MASS::corresp known: canonical correlations 0.4464, 0.1735; total inertia 0.1994
c=pd.read_csv(E+'caith.csv'); r=M.ca(c,dict(row_var='eye',counts=['fair','red','medium','dark','black'])); print(T(r,'Dimensions')['rows']); print(T(r,'Test')['rows'])
print(T(r,'Row points')['rows'][:2])
# long format equals wide
rows=[]
for i,rw in c.iterrows():
    for col in ['fair','red','medium','dark','black']: rows += [(rw['eye'],col)]*int(rw[col])
long=pd.DataFrame(rows,columns=['eye','hair']); r2=M.ca(long,dict(row_var='eye',col_var='hair')); print(T(r2,'Dimensions')['rows'][:2])
