import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings,time; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd
from scipy import integrate, optimize, special, stats
from oes_engine import stat_mixed as M
E=ROOT+'/stat-guide/examples/'
c=pd.read_csv(E+'cbpp.csv')
for nodes in (1,15,25):
    t0=time.time()
    res=M.glmm(c,dict(y='incidence',trials='size',group='herd',factors=['period'],family='binomial',nodes=nodes))
    fx=res['tables'][0]['rows'];sd=res['tables'][1]['rows'][0]
    print(nodes,round(time.time()-t0,2),'s',[(r[0],r[1],r[2]) for r in fx],'sd',sd[1:],'ll',res['summary'][2]['value'])
# independent check: brute-force quad likelihood with scipy.integrate.quad, optimise with Nelder-Mead
X=np.column_stack([np.ones(len(c))]+[(c.period==k).astype(float) for k in (2,3,4)])
y=c.incidence.values.astype(float);n=c['size'].values.astype(float);g=c.herd.values
const=np.sum(special.gammaln(n+1)-special.gammaln(y+1)-special.gammaln(n-y+1))
gs=np.unique(g)
def ll(th):
    b,s=th[:4],np.exp(th[4]); tot=const
    xb=X@b
    for gg in gs:
        m=g==gg
        fn=lambda u: np.exp(np.sum(y[m]*(xb[m]+s*u)-n[m]*np.logaddexp(0,xb[m]+s*u)))*stats.norm.pdf(u)
        v,_=integrate.quad(fn,-12,12,epsabs=1e-14,epsrel=1e-12,limit=200)
        tot+=np.log(v)
    return tot
t0=time.time()
r=optimize.minimize(lambda th:-ll(th),np.array([-1.4,-1,-1.1,-1.6,np.log(.65)]),method='Nelder-Mead',options=dict(xatol=1e-7,fatol=1e-10,maxiter=3000))
print('quad MLE',r.x[:4].round(4),'sd',np.exp(r.x[4]).round(4),'ll',-r.fun,round(time.time()-t0,1),'s')
res=M.glmm(c,dict(y='incidence',trials='size',group='herd',factors=['period'],family='binomial',nodes=25))
print('mine ll at quad optimum',ll(np.array([ -1.4 ]*0+list(r.x))) )
