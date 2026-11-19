import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd
from oes_engine import stat_meta as G
from statsmodels.stats.meta_analysis import combine_effects
d=pd.read_csv(ROOT+'/stat-guide/examples/bcg_vaccine.csv')
sp=dict(input='binary',study='author',ev1='tpos',n1='ttotal',ev2='cpos',n2='ctotal',measure='lrr')
fx=G.meta_fixed(d,sp); print('FE',fx['tables'][0]['rows'][0][1:7],'metafor FE -0.4303 se 0.0481')
for m in ['dl','reml','pm','ml','he']:
    r=G.meta_random(d,dict(sp,tau_method=m))
    print(m,r['tables'][0]['rows'][0][1:7],'tau2',[x for x in r['summary'] if x['label']=='tau-squared'][0]['value'],'I2',[x for x in r['summary'] if 'I-sq' in x['label']][0]['value'])
r=G.meta_random(d,dict(sp,tau_method='reml',ci_method='hksj'));print('REML HKSJ',r['tables'][0]['rows'][0][1:8],r['tables'][2]['rows'][:3])
print([x for x in r['tables'][3]['rows']] if r['tables'][3]['title'].startswith('Pred') else r['tables'][3]['title'])
# statsmodels
y,v,_=None,None,None
labels,y,v,meas,_=G.effect_sizes(d,sp)
for mre,key in [('dl','dl'),('pm','pm'),('iterated','ml?')]:
    c=combine_effects(y,v,method_re=mre,use_t=False)
    print('sm',mre,c.mean_effect_re,c.sd_eff_w_re if False else '', 'tau2',c.tau2)
print('metafor published REML: -0.7145 se .1798 tau2 .3132 I2 92.22 Q 152.2330')
print('Q',[x for x in fx['tables'][2]['rows'] if x[0]=="Cochran's Q"])
print([x for x in fx['tables'][-1]['rows']])
