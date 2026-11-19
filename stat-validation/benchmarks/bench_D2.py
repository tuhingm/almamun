import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd
from oes_engine import stat_survival as S
# 1. FG internals == Cox Breslow when nobody has a competing event
r=pd.read_csv(ROOT+'/stat-guide/examples/rossi.csv')
X=r[['fin','age','prio']].values.astype(float);t=r.week.values.astype(float);e=r.arrest.values
ut=np.unique(t[e==1]); W=S._fg_weights(t,e,1,ut)
beta=np.zeros(3)
from scipy import optimize
negll=lambda b:-S._fg_terms(X,e==1,t,W,ut,b)[0]
b=optimize.minimize(negll,beta,jac=lambda b:-S._fg_terms(X,e==1,t,W,ut,b)[1],method='BFGS',options=dict(gtol=1e-9)).x
cx=S._cox_fit(t,e,X,'breslow')['beta']
print('FG==Cox(breslow)',np.abs(b-cx).max())
# gradient check
from scipy.optimize import approx_fprime
b0=np.array([.1,-.02,.05])
print('grad check',np.abs(approx_fprime(b0,lambda x:S._fg_terms(X,e==1,t,W,ut,x)[0],1e-6)-S._fg_terms(X,e==1,t,W,ut,b0)[1]).max())
# 2. Simulation: subdistribution PH generating (Fine-Gray 1999): F1(t|x)=1-(1-p(1-e^-t))^{exp(x b)}
def sim(n,bt,rng,p=0.66,cens=0.0):
    x=rng.binomial(1,.5,n).astype(float); eb=np.exp(bt*x)
    u=rng.uniform(size=n)
    pc1=1-(1-p)**eb          # P(cause1 ever)
    c1=rng.uniform(size=n)<pc1
    # time for cause1: solve 1-(1-p(1-e^-t))^eb = u*pc1 ...  F1(t)/F1(inf)=v
    v=rng.uniform(size=n)
    target=v*pc1
    tt1=-np.log(1-(1-(1-target)**(1/eb))/p)
    tt2=rng.exponential(1/ (0.8),n)  # competing time
    T=np.where(c1,tt1,tt2); st=np.where(c1,1,2)
    C=rng.uniform(0,6,n) if cens else np.full(n,np.inf)
    st=np.where(C<T,0,st);T=np.minimum(T,C)
    return pd.DataFrame(dict(t=T,s=st,x=x))
rng=np.random.default_rng(5); est=[];ses=[];cov=[]
for k in range(300):
    d=sim(400,0.7,rng,cens=1)
    res=S.fine_gray(d,dict(time='t',event='s',covariates=['x']))
    row=res['tables'][0]['rows'][0]
    est.append(row[1]);ses.append(row[3]); cov.append(row[6]<=np.exp(.7)<=row[7])
print('FG sim true .7: mean est',np.mean(est),'sd',np.std(est),'mean se',np.mean(ses),'coverage',np.mean(cov))
