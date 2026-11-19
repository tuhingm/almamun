import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np, pandas as pd
from oes_engine import stat_survival as S
from lifelines import KaplanMeierFitter, CoxPHFitter, WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from lifelines.utils import concordance_index
r=pd.read_csv(ROOT+'/stat-guide/examples/rossi.csv')
v=pd.read_csv(ROOT+'/stat-guide/examples/veteran.csv')
# KM
res=S.kaplan_meier(v,dict(time='time',event='status',group='trt'))
kmf=KaplanMeierFitter().fit(v.time,v.status)
km=S.km_curve(v.time.values,v.status.values,'loglog')
print('KM S max diff',np.abs(km['S']-kmf.survival_function_.iloc[1:,0].values[:len(km['S'])]).max() if len(kmf.survival_function_)>len(km['S']) else 'n/a')
ci=kmf.confidence_interval_
print('KM median',kmf.median_survival_time_, S._first_below(km['t'],km['S'],.5))
lo=ci.iloc[:,0]; 
ev=np.unique(v.time[v.status==1])
print('CI diff',np.abs(ci.loc[ev].iloc[:,0].values-km['lo']).max(),np.abs(ci.loc[ev].iloc[:,1].values-km['hi']).max())
for w,lw in [('logrank',None),('gehan','wilcoxon'),('peto','peto')]:
    chi,df_,p,*_=S.logrank_family(v.time.values,v.status.values,v.celltype.tolist(),w)
    lt=multivariate_logrank_test(v.time,v.celltype,v.status) if lw is None else None
    from lifelines.statistics import pairwise_logrank_test
    print(w,chi,p, lt.test_statistic if lt else '')
# two-sample weighted vs lifelines logrank_test
from lifelines.statistics import logrank_test
a=v[v.trt==1];b=v[v.trt==2]
for w,lw in [('logrank',None),('gehan','wilcoxon'),('peto','peto')]:
    chi,_,p,*_=S.logrank_family(v.time.values,v.status.values,v.trt.tolist(),w)
    lt=logrank_test(a.time,b.time,a.status,b.status,weightings=lw)
    print('2samp',w,round(chi,5),round(lt.test_statistic,5))
# Cox
for ties in ['efron','breslow']:
    X=r[['fin','age','race','wexp','mar','paro','prio']].values.astype(float)
    fit=S._cox_fit(r.week.values.astype(float),r.arrest.values,X,ties)
    cph=CoxPHFitter(baseline_estimation_method='breslow') if False else CoxPHFitter()
    cph.fit(r,'week','arrest',formula='fin+age+race+wexp+mar+paro+prio')
    if ties=='efron':
        print('cox beta diff',np.abs(fit['beta']-cph.params_.values).max(),'se',np.abs(np.sqrt(np.diag(fit['cov']))-cph.standard_errors_.values).max(),'ll',fit['ll']-cph.log_likelihood_)
        per,glob,R,tt=S.ph_test(r.week.values.astype(float),r.arrest.values,X,fit['beta'],fit['cov'],'efron')
        ph=proportional_hazard_test(cph,r,time_transform='rank')
        print('ph',per.round(4),ph.summary['test_statistic'].values.round(4),'glob',glob)
        C,_=S.harrell_c(r.week.values,r.arrest.values,X@fit['beta']); print('C',C,cph.concordance_index_)
    else:
        print('breslow ll',fit['ll'],fit['beta'][:3])
res=S.cox_ph(r,dict(time='week',event='arrest',covariates=['fin','age','race','wexp','mar','paro','prio']))
print([x for x in res['summary']])
# AFT
for dist,L in [('weibull',WeibullAFTFitter),('lognormal',LogNormalAFTFitter),('loglogistic',LogLogisticAFTFitter)]:
    X=np.column_stack([np.ones(len(r)),r[['fin','age','prio']].values.astype(float)])
    fit=S._aft_fit(r.week.values.astype(float),r.arrest.values,X,dist)
    m=L().fit(r[['week','arrest','fin','age','prio']],'week','arrest')
    print(dist,'ll',fit['ll'],m.log_likelihood_)
    print(' theta',fit['theta'].round(4), 'se',np.sqrt(np.diag(fit['cov'])).round(4))
    print(' lifel',m.params_.round(4).values, m.standard_errors_.round(4).values)
