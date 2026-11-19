import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings,time; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd
from scipy import stats
from oes_engine import stat_resample as R
E=ROOT+'/stat-guide/examples/'
tg=pd.read_csv(E+'toothgrowth.csv'); ar=pd.read_csv(E+'airquality_ozone.csv')
# bootstrap vs scipy
for name,data,kw in [('mean',tg.len.values,{}),('median',tg.len.values,{}),('sd',tg.len.values,{})]:
    t0=time.time()
    r=R.bootstrap(tg,dict(statistic=name,y='len',n_boot=20000,seed=1))
    rows=r['tables'][1]['rows']
    fn={'mean':np.mean,'median':np.median,'sd':lambda x,axis=-1:np.std(x,ddof=1,axis=axis)}[name]
    sc={}
    for meth in ['percentile','basic','BCa']:
        b=stats.bootstrap((data,),fn,n_resamples=20000,method=meth,random_state=2,vectorized=True,axis=-1)
        sc[meth]=(b.confidence_interval.low,b.confidence_interval.high)
    print(name,round(time.time()-t0,1),'s mine',[(x[0][:6],x[1],x[2]) for x in rows[:3]],'\n   scipy',{k:(round(v[0],4),round(v[1],4)) for k,v in sc.items()},'se',r['tables'][0]['rows'][1][1],np.std(stats.bootstrap((data,),np.mean,n_resamples=5000,random_state=3).bootstrap_distribution,ddof=1) if name=='mean' else '')
# correlation with scipy (paired)
a=ar.dropna()
x=a.Ozone.values; yv=a.Wind.values if 'Wind' in a else None
print(ar.columns.tolist())
