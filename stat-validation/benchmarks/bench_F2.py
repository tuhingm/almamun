import os
ROOT = os.environ.get("OES_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import sys,warnings,time; warnings.filterwarnings('ignore')
sys.path.insert(0,ROOT+'/oes/engine')
import numpy as np,pandas as pd
from scipy import stats
from oes_engine import stat_resample as R
E=ROOT+'/stat-guide/examples/'
tg=pd.read_csv(E+'toothgrowth.csv')
d=tg.len.values
for name,fn in [('mean',np.mean),('sd',lambda x,axis=-1:np.std(x,ddof=1,axis=axis))]:
    r=R.bootstrap(tg,dict(statistic=name,y='len',n_boot=100000,seed=7)); rows=r['tables'][1]['rows']
    out=[]
    for meth in ['percentile','basic','BCa']:
        b=stats.bootstrap((d,),fn,n_resamples=100000,method=meth,random_state=9,vectorized=True,axis=-1)
        out.append((round(float(b.confidence_interval.low),3),round(float(b.confidence_interval.high),3)))
    print(name,'mine',[(x[1],x[2]) for x in rows[:3]],'\n scipy',out)
# permutation vs scipy
ar=pd.read_csv(E+'airquality_ozone.csv').dropna()
sub=tg[tg.dose==0.5][['len','supp']]            # 20 obs, 10/10 -> C(20,10)=184756 exact
r=R.permutation(sub,dict(test='mean_diff',y='len',group='supp',alternative='two-sided',exact_max=200000))
g=sub.groupby('supp').len.apply(np.array); a,b=g['OJ'],g['VC']
sp=stats.permutation_test((a,b),lambda x,y,axis:np.mean(x,axis=axis)-np.mean(y,axis=axis),permutation_type='independent',n_resamples=10**6,alternative='two-sided')
print('perm exact two-sample',r['summary'][2]['value'],sp.pvalue,'nperm',r['summary'][3]['value'])
# correlation exact N=7
rng=np.random.default_rng(0); x=rng.normal(size=7); y=x*.7+rng.normal(size=7); small=pd.DataFrame(dict(x=x,y=y))
r=R.permutation(small,dict(test='correlation',y='y',x='x'))
def cstat(a,b,axis=-1):
    a=a-a.mean(axis=axis,keepdims=True); b=b-b.mean(axis=axis,keepdims=True)
    return (a*b).sum(axis)/np.sqrt((a*a).sum(axis)*(b*b).sum(axis))
sp=stats.permutation_test((x,y),cstat,permutation_type='pairings',n_resamples=10**6,alternative='two-sided')
print('perm exact corr',r['summary'][2]['value'],sp.pvalue)
# paired exact n=12
p=pd.DataFrame(dict(a=rng.normal(size=12)+.5,b=rng.normal(size=12)))
r=R.permutation(p,dict(test='paired',y='a',x='b'))
sp=stats.permutation_test((p.a.values,p.b.values),lambda x,y,axis:np.mean(x-y,axis=axis),permutation_type='samples',n_resamples=10**6,alternative='two-sided')
print('perm exact paired',r['summary'][2]['value'],sp.pvalue,r['summary'][3]['value'])
# monte carlo
r=R.permutation(tg,dict(test='mean_diff',y='len',group='supp',n_perm=100000,exact_max=1000))
print('MC mean diff p',r['summary'][2]['value'],'welch',r['tables'][1]['rows'][0])
r=R.permutation(tg,dict(test='anova_f',y='len',group='dose',n_perm=20000)); print('ANOVA',r['summary'][2]['value'],r['tables'][1]['rows'])
# power
import statsmodels.stats.power as sp_
from statsmodels.stats.power import TTestIndPower,TTestPower,FTestAnovaPower,NormalIndPower,GofChisquarePower
def P(**k): return R.power_analysis(None,k)
r=P(test='t_two',solve_for='n',effect=0.5,alpha=0.05,power=0.8); print('t_two n',r['summary'][1]['value'],TTestIndPower().solve_power(0.5,power=.8,alpha=.05))
r=P(test='t_two',solve_for='power',effect=0.5,n=30,alpha=0.05); print('t_two power',r['summary'][1]['value'],TTestIndPower().power(0.5,30,.05))
r=P(test='t_two',solve_for='power',effect=0.5,n=30,alpha=0.05,alternative='greater'); print('t_two 1-sided',r['summary'][1]['value'],TTestIndPower().power(0.5,30,.05,alternative='larger'))
r=P(test='t_one',solve_for='n',effect=0.4,alpha=0.05,power=0.9); print('t_one',r['summary'][1]['value'],TTestPower().solve_power(0.4,power=.9,alpha=.05))
r=P(test='anova',solve_for='n',effect=0.25,k=4,alpha=0.05,power=0.8); print('anova n/group',r['summary'][1]['value'],FTestAnovaPower().solve_power(0.25,k_groups=4,power=.8,alpha=.05)/4)
r=P(test='corr',solve_for='n',effect=0.3,alpha=0.05,power=0.8); print('corr n',r['summary'][1]['value'],'(pwr: 84.07 -> 85)')
r=P(test='prop',solve_for='n',effect=0.3,alpha=0.05,power=0.8); print('prop n',r['summary'][1]['value'],NormalIndPower().solve_power(0.3,power=.8,alpha=.05),'(pwr 174.4)')
r=P(test='chisq',solve_for='n',effect=0.3,df=2,alpha=0.05,power=0.8); print('chisq N',r['summary'][1]['value'],GofChisquarePower().solve_power(0.3,n_bins=3,power=.8,alpha=.05))
r=P(test='t_two',solve_for='effect',n=64,alpha=.05,power=.8); print('detectable d',r['summary'][1]['value'],TTestIndPower().solve_power(None,nobs1=64,power=.8,alpha=.05))
