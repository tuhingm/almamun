import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd, time
from oes_engine import stat_mixed as M
import statsmodels.formula.api as smf
E=ROOT+'/stat-guide/examples/'
s=pd.read_csv(E+'sleepstudy.csv')
for slope,corr in [(None,'yes'),('Days','yes'),('Days','no')]:
  for meth in ['reml','ml']:
    t0=time.time()
    res=M.lmm(s,dict(y='Reaction',group='Subject',covariates=['Days'],slope=slope,method=meth,correlated=corr))
    fx=res['tables'][0]['rows']; vc=res['tables'][1]['rows']
    print(slope,corr,meth,round(time.time()-t0,2),'s')
    print(' fx',[(r[0],r[1],r[2],r[3],r[5]) for r in fx]); print(' vc',vc)
    if corr=='yes' or slope is None:
        re_='~Days' if slope else None
        md=smf.mixedlm('Reaction~Days',s,groups=s['Subject'],re_formula=re_).fit(reml=(meth=='reml'),method='lbfgs')
        print(' sm fe',md.fe_params.round(4).values,md.bse_fe.round(4).values,'cov_re',md.cov_re.values.round(3).tolist(),'scale',round(md.scale,3),'ll',round(md.llf,4))
