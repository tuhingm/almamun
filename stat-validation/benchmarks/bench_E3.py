import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd
from oes_engine import stat_mixed as M
import statsmodels.api as sm, statsmodels.formula.api as smf
import pingouin as pg
E=ROOT+'/stat-guide/examples/'
o=pd.read_csv(E+'orthodont.csv'); o['bin']=(o.distance>24.5).astype(int)
c=pd.read_csv(E+'cbpp.csv')
cases=[('gaussian','distance',o,'Subject','age',['age','Sex01']),('binomial','bin',o,'Subject','age',['age','Sex01']),('poisson','incidence',c,'herd','period',['period'])]
for fam,y,d,gc,tc,cov in cases:
    fm={'gaussian':sm.families.Gaussian(),'binomial':sm.families.Binomial(),'poisson':sm.families.Poisson()}[fam]
    for corr,cs in [('independence',sm.cov_struct.Independence()),('exchangeable',sm.cov_struct.Exchangeable()),('ar1',sm.cov_struct.Autoregressive())]:
        try:
            r=M.gee(d,dict(y=y,group=gc,time=tc,covariates=cov,family=fam,corr=corr))
        except Exception as e:
            print(fam,corr,'ERR',e); continue
        rows=r['tables'][0]['rows']; fit=dict((a,b) for a,b in r['tables'][1]['rows'])
        dd=d.sort_values([gc,tc]).reset_index(drop=True)
        try:
          m=smf.gee(f"{y}~"+'+'.join(cov),groups=gc,data=dd,family=fm,cov_struct=cs,time=dd[tc].values if corr=='ar1' else None).fit()
        except Exception as e:
          print(fam,corr,'statsmodels fails:',str(e)[:50],'| mine coef',[x[1] for x in rows],'alpha',fit['Estimated correlation parameter']); continue
        b=np.array([x[1] for x in rows]);se=np.array([x[2] for x in rows])
        print(fam,corr,'coef maxdiff',np.abs(b-m.params.values).max().round(6),'robust se maxdiff',np.abs(se-m.bse.values).max().round(6),'alpha',fit['Estimated correlation parameter'],getattr(m.cov_struct,'dep_params',None),'scale',fit['Scale (dispersion)'],round(m.scale,5),'QIC',fit['QIC'],[round(v,3) for v in m.qic(m.scale)])
# ICC vs pingouin
s=pd.read_csv(E+'shrout_fleiss.csv')
r=M.icc(s,dict(raters=['judge1','judge2','judge3','judge4']))
for row in r['tables'][0]['rows']: print(row[0],row[2],row[7],row[8],row[6])
long=s.melt(id_vars='target',var_name='rater',value_name='score')
print(pg.intraclass_corr(long,'target','rater','score').iloc[:,:8])
print('alpha',r['tables'][2]['rows'],pg.cronbach_alpha(s[['judge1','judge2','judge3','judge4']]))
print('--- AR1 with alpha fixed at statsmodels value')
dd=o.sort_values(['Subject','age']).reset_index(drop=True)
m=smf.gee("bin~age+Sex01",groups='Subject',data=dd,family=sm.families.Binomial(),cov_struct=sm.cov_struct.Autoregressive(),time=dd.age.values).fit()
r=M.gee(o,dict(y='bin',group='Subject',time='age',covariates=['age','Sex01'],family='binomial',corr='ar1',fixed_alpha=float(m.cov_struct.dep_params)))
print(np.abs(np.array([x[1] for x in r['tables'][0]['rows']])-m.params.values).max(),np.abs(np.array([x[2] for x in r['tables'][0]['rows']])-m.bse.values).max(), dict((a,b) for a,b in r['tables'][1]['rows'])['QIC'], m.qic(1.0))
