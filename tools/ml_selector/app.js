const $=s=>document.querySelector(s), el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches, D=reduce?0:1;
const fmt=(v,d=3)=>v==null?'–':Number(v).toFixed(d), pf=p=>p==null?'–':(p<0.001?'<0.001':p.toFixed(3));
const big=v=>v==null?'–':Math.abs(v)>=1000?Math.round(v).toLocaleString('en-US'):fmt(v,Math.abs(v)<1?4:3);
const pct=v=>v==null?'–':(v*100).toFixed(1)+'%';
const pyList=a=>'['+a.map(v=>JSON.stringify(v)).join(', ')+']';
let run=0, QA={}, GOAL='predict';

/* ---------- decision tree ---------- */
const N={start:[440,26,120,'Your question'],goal:[440,92,150,'Predict or explain?'],
 ts:[215,160,124,'Time series?'],nn:[215,226,126,'N ≥ '+C.SMALLN+'?'],nl:[300,292,150,'Forest ≥ '+C.GAIN+'% better?'],
 lin:[130,362,120,'Linear model'],bign:[340,362,104,'N ≥ '+C.BIGN.toLocaleString('en-US')+'?'],
 ENET:[48,440,76,'Elastic net',1],LASSO:[130,440,66,'LASSO',1],RIDGE:[206,440,62,'Ridge',1],RF:[296,440,92,'Random forest',1],XGB:[392,440,72,'XGBoost',1],
 conf:[650,160,170,'Confounders observed?'],CIG:[838,160,76,'Design',1],ctl:[650,226,160,'Many or nonlinear controls?'],OLS:[838,226,56,'OLS',1],
 typ:[650,292,124,'Treatment type'],PLR:[548,362,86,'DML-PLR',1],IRM:[650,362,84,'DML-IRM',1],PLIV:[766,362,72,'DML-IV',1],
 het:[600,440,124,'Who gains most?'],CF:[760,440,108,'Causal forest',1]};
const E=[['start','goal'],['goal','ts','predict'],['goal','conf','causal effect'],['ts','nn'],['nn','lin','no'],['nn','nl','yes'],['nl','lin','no'],['nl','bign','yes'],
 ['lin','ENET'],['lin','LASSO'],['lin','RIDGE'],['bign','RF','no'],['bign','XGB','yes'],
 ['conf','CIG','no'],['conf','ctl','yes'],['ctl','OLS','no'],['ctl','typ','yes'],['typ','PLR'],['typ','IRM'],['typ','PLIV'],['PLR','het'],['IRM','het'],['het','CF','N ≥ '+C.CFN.toLocaleString('en-US')]];
const TL={start:'Your question',goal:'Aim',ts:'Time series?',nn:'Sample size',nl:'Nonlinearity test',lin:'Linear models',bign:'Large sample?',conf:'Confounders',ctl:'Controls',typ:'Treatment type',het:'Heterogeneity'};
function pathTo(m,R){const P=['start','goal'],pa=P.concat(['ts','nn']);
 if(['ENET','LASSO','RIDGE'].includes(m)){const small=R?R.small:QA.small;return pa.concat(small?['lin',m]:['nl','lin',m])}
 if(m==='RF'||m==='XGB')return pa.concat(['nl','bign',m]);
 const pc=P.concat(['conf']);if(m==='CIG')return pc.concat(['CIG']);if(m==='OLS')return pc.concat(['ctl','OLS']);
 if(m==='CF'){const k=R?R.kind:QA.kind;return pc.concat(['ctl','typ',k||'IRM','het','CF'])}
 return pc.concat(['ctl','typ',m])}
function drawTree(){
 const H=16;let s='<svg viewBox="0 0 900 470" role="img" aria-label="Decision tree from research question to machine-learning method">';
 for(const [a,b,lab] of E){const A=N[a],B=N[b];let d,lx,ly;
  if(A[1]===B[1]){const r=B[0]>A[0];d=r?`M${A[0]+A[2]/2} ${A[1]}H${B[0]-B[2]/2}`:`M${A[0]-A[2]/2} ${A[1]}H${B[0]+B[2]/2}`;lx=r?(A[0]+A[2]/2+B[0]-B[2]/2)/2:(A[0]-A[2]/2+B[0]+B[2]/2)/2;ly=A[1]-6}
  else{const my=(A[1]+B[1])/2;d=`M${A[0]} ${A[1]+H}V${my}H${B[0]}V${B[1]-H}`;lx=A[0]===B[0]?A[0]+16:(A[0]+B[0])/2;ly=A[0]===B[0]?my+4:my-5}
  s+=`<path class="e" id="e-${a}-${b}" pathLength="1" d="${d}"/>`;if(lab)s+=`<text class="el" x="${lx}" y="${ly}">${lab}</text>`}
 for(const k in N){const [x,y,w,t,leaf]=N[k];s+=`<g class="n${leaf?' leaf':''}" id="n-${k}"><rect x="${x-w/2}" y="${y-H}" width="${w}" height="${2*H}" rx="${leaf?16:8}"/><text x="${x}" y="${y}">${t}</text></g>`}
 $('#tree').innerHTML=s+'</svg>';
}
let lit=new Set(),tEnd=0;
function resetTree(){lit=new Set();tEnd=0;document.querySelectorAll('.tree .on,.tree .hl,.tree .alt').forEach(e=>e.classList.remove('on','hl','alt'));$('#trail').innerHTML=''}
function light(path,labels={},final=true){
 const my=run,now=performance.now();let at=Math.max(now,tEnd);
 path.forEach((id,i)=>{if(lit.has(id))return;lit.add(id);const wait=at-now;at+=380*D;
  setTimeout(()=>{if(my!==run)return;const n=$('#n-'+id);n&&n.classList.add('on');if(i>0){const e=document.getElementById(`e-${path[i-1]}-${id}`);e&&e.classList.add('hl')}
   const t=$('#trail');if(t.children.length)t.appendChild(el('i','','→'));const isLeaf=!!MODELS[id];t.appendChild(el('span',isLeaf&&final?'final':'',esc(labels[id]||(MODELS[id]?MODELS[id].short:TL[id])||id)))},wait)});
 tEnd=at;return at-now;
}

/* ---------- small charts ---------- */
function bars(rows){const mx=Math.max(...rows.map(r=>r[1]||0)),mn=Math.min(...rows.filter(r=>r[1]!=null).map(r=>r[1]));
 return `<div class="bars">${rows.filter(r=>r[1]!=null).map(r=>`<div class="bar${r[1]===mn?' best':''}"><span>${r[0]}</span><i style="width:${(r[1]/mx*100).toFixed(1)}%"></i><b>${big(r[1])}</b></div>`).join('')}</div>`}
function hist(O){const t=O.hist_t,c=O.hist_c,mx=Math.max(...t,...c,1),w=400/20,h=100,y=v=>h-v/mx*(h-6);
 let s='<div class="hist"><svg viewBox="0 0 400 110" preserveAspectRatio="none">';
 c.forEach((v,i)=>s+=`<rect class="c" x="${i*w+1}" y="${y(v)}" width="${w-2}" height="${h-y(v)}"/>`);t.forEach((v,i)=>s+=`<rect class="t" x="${i*w+w/4}" y="${y(v)}" width="${w/2}" height="${h-y(v)}"/>`);
 s+=`<line x1="${C.OVLO*400}" x2="${C.OVLO*400}" y1="0" y2="${h}"/><line x1="${C.OVHI*400}" x2="${C.OVHI*400}" y1="0" y2="${h}"/></svg><div class="legend" style="font-size:10.5px;color:var(--muted)"><span style="color:#0033a0">■</span> treated · <span style="color:#c46000">■</span> untreated · dashed lines: ${C.OVLO} and ${C.OVHI} · propensity score from 0 to 1</div></div>`;return s}

/* ---------- steps ---------- */
const code=c=>`<details class="code"><summary>Python code for this step</summary><pre>${esc(c)}</pre></details>`;
function addStep(n,title,html,delay){const s=el('div','step',`<div class="label">Step ${n}</div><h3>${title}</h3>${html}`);$('#steps').appendChild(s);const my=run;setTimeout(()=>{if(my!==run)return;s.classList.add('in');if(n>1&&!reduce)s.scrollIntoView({behavior:'smooth',block:'nearest'})},delay);return s}
function sumTable(R){const rows=R.summary,many=rows.length>12;
 const tb=`<div class="tw"><table><tr><th>Variable</th><th>Mean</th><th>Std. dev.</th><th>Min</th><th>Max</th><th>Type</th></tr>${rows.map(s=>`<tr><td>${esc(s.var)}</td><td>${big(s.mean)}</td><td>${big(s.sd)}</td><td>${big(s.min)}</td><td>${big(s.max)}</td><td>${s.binary?'0/1':'continuous'}</td></tr>`).join('')}</table></div>`;
 return many?`<details class="explain"><summary>Summary statistics for all ${rows.length} variables</summary>${tb}</details>`:tb}
function show(R,label){
 const my=++run;$('#steps').innerHTML='';$('#result').innerHTML='';resetTree();let t=150,gap=900*D,k=1;
 light(['start','goal'],{goal:R.goal==='causal'?'Causal effect':'Prediction'});
 if(R.goal==='causal')return showCausal(R,label,my,t,gap,k);
 addStep(k++,'The data',`<p><b>${R.n.toLocaleString('en-US')}</b> observations of <b>${esc(R.dep)}</b> and <b>${R.p}</b> predictors${label?' · '+esc(label):''}.${R.time?` Time column: <code>${esc(R.time)}</code> (${esc(R.first||'')}–${esc(R.last||'')}).`:''}${R.skipped&&R.skipped.length?' Ignored non-numeric columns: '+R.skipped.map(esc).join(', ')+'.':''}</p>
  <div class="stats"><div class="stat"><b>${R.n.toLocaleString('en-US')}</b><span>Observations (N)</span></div><div class="stat"><b>${R.p}</b><span>Predictors (p)</span></div><div class="stat"><b>${R.ts?'Yes':'No'}</b><span>Time series</span></div><div class="stat"><b>${fmt(R.corr.max_r,2)}</b><span>Largest |r| between predictors</span></div></div>
  ${sumTable(R)}
  <details class="explain" open><summary>Why the aim comes first</summary><p>Prediction and causal inference ask different questions of the same data. <b>Prediction</b> asks how well Y can be forecast from X; any variable that helps is welcome, and the test is accuracy on data the model has not seen. <b>Causal inference</b> asks how Y changes when one variable D changes, holding the others fixed; accuracy is not the test, bias is. This dataset is a prediction problem, so the methods are compared by out-of-sample error.</p></details>`+code(`import pandas as pd\n\ndf = pd.read_csv("data.csv")\ny, X = df["${R.dep}"], df[${pyList(R.xs.slice(0,8))}${R.xs.length>8?' + ...':''}]\nprint(df.describe().T)\nprint(X.corr().abs().where(lambda c: c < 1).max().max())   # largest |r| between predictors`),t);t+=gap;
 if(R.ts){const ur=R.unitroot||[];
  addStep(k++,'Time series: unit roots before learning',`<details class="explain" open><summary>Why stationarity still matters for machine learning</summary><p>Tree-based learners cannot extrapolate: a forest trained on a trending series predicts the highest value it has seen, never a new high. Linear learners on I(1) series suffer from the same spurious-regression problem as OLS. So each series is tested first, and I(1) series are differenced, exactly as in the Time-Series Guide. Lags of one series share its order.</p><ul><li><b>ADF test.</b> Null hypothesis: a unit root. p &lt; 0.05 means the series is stationary.</li><li><b>KPSS test.</b> Null hypothesis: stationary. Reported as a cross-check.</li></ul></details>
   <div class="tw"><table><tr><th>Series</th><th>ADF level (p)</th><th>KPSS level (p)</th><th>ADF difference (p)</th><th>Order</th></tr>${ur.map(u=>u.skipped?`<tr><td>${esc(u.var)}</td><td colspan="3">binary or few values: not tested</td><td><span class="tag ok">I(0)</span></td></tr>`:`<tr><td>${esc(u.var)}</td><td>${fmt(u.level.adf,2)} (${pf(u.level.adf_p)})</td><td>${fmt(u.level.kpss,2)} (${pf(u.level.kpss_p)})</td><td>${u.diff?fmt(u.diff.adf,2)+' ('+pf(u.diff.adf_p)+')':'–'}</td><td>${u.order===0?'<span class="tag ok">I(0)</span>':u.order===1?'<span class="tag i1">I(1)</span>':'<span class="tag i2">I(2)</span>'}</td></tr>`).join('')}</table></div>
   <p><b>Result:</b> ${R.diffed.length?`differenced before learning: ${R.diffed.map(esc).join(', ')}.`:'every series is stationary; nothing is differenced.'} Cross-validation below uses <b>rolling-origin splits</b>: each fold trains on the past and predicts the next block, never the other way round.</p>`+code(`from statsmodels.tsa.stattools import adfuller, kpss\nfrom sklearn.model_selection import TimeSeriesSplit\n\nfor v in ${pyList(ur.map(u=>u.var))}:\n    print(v, "ADF p =", round(adfuller(df[v].dropna(), autolag="AIC")[1], 3),\n             "KPSS p =", round(kpss(df[v].dropna(), nlags="auto")[1], 3))\ncv = TimeSeriesSplit(n_splits=5)          # never shuffle time`),t);t+=gap;
  setTimeout(()=>{if(my!==run)return;light(['start','goal','ts'],{ts:'Time series: yes'})},t-gap+300*D)}
 else setTimeout(()=>{if(my!==run)return;light(['start','goal','ts'],{ts:'Cross-section'})},t-gap+300*D);
 const rm=R.rmse;
 addStep(k++,'Sample size',`<p>${R.n_used.toLocaleString('en-US')} observations are available for learning${R.subsampled?` (a random ${(4000).toLocaleString('en-US')} are used for the forest checks in the browser)`:''}.</p><p><b>Rule A2:</b> ${R.small?`<span class="no">N &lt; ${C.SMALLN}</span>: too few observations for tree-based methods, which would memorise the sample. Only regularised linear models are compared.`:`<span class="yes">N ≥ ${C.SMALLN}</span>: enough observations to test for nonlinearity with a random forest.`}</p>`,t);t+=gap;
 setTimeout(()=>{if(my!==run)return;light(['start','goal','ts','nn'],{nn:'N = '+R.n_used.toLocaleString('en-US')})},t-gap+300*D);
 if(!R.small){
  addStep(k++,'Is the relationship nonlinear? Random forest vs LASSO',`<details class="explain" open><summary>How the test works</summary><p>LASSO can only fit a linear relationship; a random forest can fit any shape and any interaction. Both are scored by five-fold cross-validated RMSE${R.ts?' with rolling-origin splits':''}: the error on observations each model did not see. If the forest's error is at least <b>${C.GAIN}%</b> lower, the data contain nonlinearities or interactions worth modelling; if not, the simpler linear model is preferred.</p></details>
   ${bars([['OLS',rm.ols],['LASSO',rm.lasso],['Random forest',rm.rf],['Boosting',rm.gb]])}
   <p style="font-size:11px;color:#8793ab">Cross-validated RMSE in units of ${esc(R.dep)}; shorter is better. Standard deviation of ${esc(R.dep)}: ${big(R.sd_y)}.</p>
   <p><b>Rule A3:</b> the forest's RMSE is ${R.gain>=0?`${pct(R.gain)} lower`:`${pct(-R.gain)} higher`} than LASSO's. ${R.nonlinear?`<span class="yes">At least ${C.GAIN}%: the relationship is nonlinear.</span>`:`<span class="no">Less than ${C.GAIN}%: no evidence that a nonlinear model is worth its extra variance.</span>`}</p>`+code(`from sklearn.model_selection import KFold, cross_val_score\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.linear_model import LassoCV\nfrom sklearn.ensemble import RandomForestRegressor\n\ncv = ${R.ts?'TimeSeriesSplit(n_splits=5)':'KFold(5, shuffle=True, random_state=0)'}\nrmse = lambda m: (-cross_val_score(m, X, y, cv=cv, scoring="neg_mean_squared_error").mean()) ** .5\nlasso = make_pipeline(StandardScaler(), LassoCV(cv=5))\nforest = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, max_features=0.5, random_state=0)\nprint("LASSO", rmse(lasso), " forest", rmse(forest), " gain", 1 - rmse(forest) / rmse(lasso))`),t);t+=gap;
  setTimeout(()=>{if(my!==run)return;light(['start','goal','ts','nn','nl'],{nl:R.nonlinear?'Nonlinear':'Linear'})},t-gap+300*D)}
 if(R.model==='RF'||R.model==='XGB'){
  addStep(k++,'Forest or boosting? Sample size again',`<p><b>Rule A5:</b> ${R.n_used>=C.BIGN?`<span class="yes">N ≥ ${C.BIGN.toLocaleString('en-US')}</span>: enough data to tune a gradient-boosted model, which usually predicts best on tabular data${rm.gb!=null?` (here, scikit-learn's histogram boosting reaches an RMSE of ${big(rm.gb)} against ${big(rm.rf)} for the forest)`:''}. XGBoost is the standard implementation in papers.`:`<span class="no">N &lt; ${C.BIGN.toLocaleString('en-US')}</span>: boosting has many tuning parameters and overfits small samples easily. A random forest needs little tuning and is the safer choice.`}</p>`,t);t+=gap;
  setTimeout(()=>{if(my!==run)return;light(['start','goal','ts','nn','nl','bign'])},t-gap+300*D);
 }else{const kept=new Set(R.lasso_kept||[]);
  addStep(k++,'Which linear model? Correlation and sparsity',`<details class="explain" open><summary>LASSO, ridge or elastic net?</summary><ul><li><b>LASSO</b> sets weak coefficients to exactly zero: it suits a <b>sparse</b> problem in which a few predictors matter.</li><li><b>Ridge</b> shrinks all coefficients but drops none: it suits a <b>dense</b> problem with many small effects.</li><li><b>Elastic net</b> mixes both penalties. When predictors are strongly correlated, LASSO keeps one of a group almost arbitrarily; the elastic net keeps or drops them together.</li></ul><p>Sparsity is judged at the <b>one-standard-error penalty</b>: the largest penalty whose cross-validated error is within one standard error of the minimum (glmnet's <code>lambda.1se</code>).</p></details>
   ${bars([['OLS',rm.ols],['LASSO',rm.lasso],['Ridge',rm.ridge],['Elastic net',rm.enet]])}
   <ol class="urv" style="padding-left:18px"><li><b>Correlation.</b> The largest correlation between two predictors is ${fmt(R.corr.max_r,2)}${R.corr.pair.length?` (${R.corr.pair.map(esc).join(' and ')})`:''}. ${R.corr.max_r>C.HIGHR?`<span class="no">Above ${C.HIGHR}: correlated predictors</span>.`:`<span class="yes">Below ${C.HIGHR}</span>.`}</li>
   <li><b>Sparsity.</b> At the one-standard-error penalty LASSO keeps <b>${(R.lasso_kept||[]).length} of ${R.p}</b> predictors (${pct(R.lasso_share)}). ${R.lasso_share<C.SPARSE?'<span class="yes">Fewer than half: a sparse problem.</span>':'<span class="no">Half or more: a dense problem.</span>'}${(R.lasso_kept||[]).length===0?' None of the predictors improves on the sample mean by more than noise.':''}</li></ol>
   <div class="chips">${R.xs.slice(0,40).map(x=>`<span class="${kept.has(x)?'':'out'}">${esc(x)}</span>`).join('')}${R.xs.length>40?`<span class="out">+${R.xs.length-40} more</span>`:''}</div>
   <p><b>Rule A4:</b> ${R.model==='ENET'?`correlation above ${C.HIGHR}, so the elastic net.`:R.model==='LASSO'?`correlations below ${C.HIGHR} and fewer than half of the predictors kept, so LASSO.`:`correlations below ${C.HIGHR} and half or more of the predictors kept, so ridge.`}</p>`+code(`import numpy as np\nfrom sklearn.linear_model import LassoCV, Lasso\nfrom sklearn.preprocessing import StandardScaler\n\nXs = StandardScaler().fit_transform(X)\ncv = LassoCV(cv=5, random_state=0).fit(Xs, y)\nm, se = cv.mse_path_.mean(1), cv.mse_path_.std(1) / np.sqrt(5)\nlam_1se = cv.alphas_[m <= m.min() + se[m.argmin()]].max()      # one-standard-error penalty\nkept = X.columns[Lasso(alpha=lam_1se).fit(Xs, y).coef_ != 0]\nprint(len(kept), "of", X.shape[1], "kept:", list(kept))`),t);t+=gap;
  setTimeout(()=>{if(my!==run)return;light(R.small?['start','goal','ts','nn','lin']:['start','goal','ts','nn','nl','lin'])},t-gap+300*D)}
 setTimeout(()=>{if(my!==run)return;finish(R.model,R)},t);
}
function showCausal(R,label,my,t,gap,k){
 addStep(k++,'The question',`<p>Effect of <b>${esc(R.treat)}</b> (${R.binary?'binary':'continuous'}) on <b>${esc(R.dep)}</b>, with <b>${R.p}</b> controls${R.instr?` and the instrument <b>${esc(R.instr)}</b>`:''}. <b>${R.n.toLocaleString('en-US')}</b> observations${label?' · '+esc(label):''}.</p>
  <div class="stats"><div class="stat"><b>${R.n.toLocaleString('en-US')}</b><span>Observations (N)</span></div><div class="stat"><b>${R.p}</b><span>Controls (p)</span></div><div class="stat"><b>${R.binary?'Binary':'Continuous'}</b><span>Treatment</span></div><div class="stat"><b>${R.instr?'Yes':'No'}</b><span>Instrument</span></div></div>
  ${sumTable(R)}
  <details class="explain" open><summary>What machine learning can and cannot do for a causal effect</summary><p>The effect of D on Y is identified if, after adjusting for the controls X, D is as good as random (<b>selection on observables</b>), or if an instrument shifts D. Machine learning helps with the <b>adjustment</b>: it can model how Y and D depend on many controls, or on controls that enter nonlinearly, without the researcher guessing the functional form. It cannot help if an important confounder is missing from X.</p><p><b>Rule B1:</b> ${R.instr?`an instrument is available, so the question is whether it is valid (see below).`:`this analysis assumes the controls contain the confounders${R.b1?': '+esc(R.b1):''}. If they do not, use the Causal Inference Guide.`}</p></details>`,t);t+=gap;
 setTimeout(()=>{if(my!==run)return;light(['start','goal','conf'],{conf:R.instr?'Instrument':'Observed'})},t-gap+300*D);
 const nl=R.nl;
 addStep(k++,'Is OLS enough? Number of controls and their functional form',`<details class="explain" open><summary>When does double machine learning add something?</summary><p>OLS adjusts for the controls linearly. That is enough when there are few controls and they enter linearly; then machine learning only adds variance. DML is needed when there are <b>many controls</b> (OLS overfits, and choosing among them by hand invites specification search) or when the outcome or the treatment depends on them <b>nonlinearly</b>. The nonlinearity test is the same as for prediction: a random forest against LASSO, by cross-validated RMSE.</p></details>
  <div class="tw"><table><tr><th>Prediction of</th><th>LASSO RMSE</th><th>Forest RMSE</th><th>Forest gain</th></tr><tr><td>${esc(R.dep)} (outcome) from X</td><td>${big(nl.y.lasso)}</td><td>${big(nl.y.rf)}</td><td>${pct(nl.y.gain)}</td></tr><tr><td>${esc(R.treat)} (treatment) from X</td><td>${big(nl.d.lasso)}</td><td>${big(nl.d.rf)}</td><td>${pct(nl.d.gain)}</td></tr></table></div>
  <p>OLS benchmark (robust standard errors): effect of ${esc(R.treat)} = <b>${big(R.ols.b)}</b> (s.e. ${big(R.ols.se)}, p ${pf(R.ols.p)}).</p>
  <p><b>Rule B2:</b> ${R.p}${R.p<C.FEWCTRL?` &lt; ${C.FEWCTRL}`:` ≥ ${C.FEWCTRL}`} controls, and the forest ${R.controls_nonlinear?`beats LASSO by at least ${C.GAIN}%`:`does not beat LASSO by ${C.GAIN}%`}. ${R.ols_enough?`<span class="no">OLS is enough</span>${R.instr?', but the treatment is endogenous, so the instrument is still needed.':': DML is reported below as a robustness check.'}`:`<span class="yes">Double machine learning is needed.</span>`}</p>`+code(`import statsmodels.api as sm\n\nols = sm.OLS(df["${R.dep}"], sm.add_constant(df[["${R.treat}"] + controls])).fit(cov_type="HC1")\nprint(ols.params["${R.treat}"], ols.bse["${R.treat}"])\n# nonlinearity of the controls: forest vs LASSO for the outcome and for the treatment\nfor target in ["${R.dep}", "${R.treat}"]:\n    print(target, rmse(lasso, df[controls], df[target]), rmse(forest, df[controls], df[target]))`),t);t+=gap;
 setTimeout(()=>{if(my!==run)return;light(['start','goal','conf','ctl'],{ctl:R.ols_enough?'Few, linear':'Many or nonlinear'})},t-gap+300*D);
 if(!(R.ols_enough&&!R.instr)){
  let body='';
  if(R.instr){const F=R.first_stage;body=`<p>The instrument <b>${esc(R.instr)}</b> must (1) shift the treatment, (2) affect the outcome only through the treatment, given the controls, and (3) be as good as random given the controls. Only (1) can be tested.</p><div class="tw"><table><tr><th>First stage</th><th>Coefficient</th><th>Robust s.e.</th><th>F statistic</th></tr><tr><td>${esc(R.treat)} on ${esc(R.instr)} and the controls</td><td>${big(F.b)}</td><td>${big(F.se)}</td><td>${fmt(F.F,1)}</td></tr></table></div><p><b>Rule B3:</b> ${R.weak?`<span class="no">F &lt; ${C.WEAKF}: a weak instrument.</span> DML-IV is still the method, but report Anderson–Rubin confidence intervals, which stay valid when the instrument is weak.`:`<span class="yes">F ≥ ${C.WEAKF}</span>: the instrument is strong. DML-IV.`}</p>`}
  else if(R.binary){const O=R.overlap;body=`<p>With a binary treatment, the interactive model compares treated and untreated people with the same controls. That requires <b>overlap</b>: everyone must have some chance of being in either group. The propensity score (the probability of treatment given the controls) is estimated by cross-validated random forest.</p>${hist(O)}<p>Propensity scores range from ${fmt(O.min,3)} to ${fmt(O.max,3)}; <b>${pct(O.share_out)}</b> lie outside [${C.OVLO}, ${C.OVHI}]. ${pct(R.treated_share)} of the sample is treated.</p><p><b>Rule B3:</b> a binary treatment, so the <b>interactive model</b> (doubly robust AIPW score). ${O.share_out>0?`<span class="no">Trim or clip the ${pct(O.share_out)} of observations outside [${C.OVLO}, ${C.OVHI}]</span>, and report the estimate with and without trimming.`:`<span class="yes">Overlap is good.</span>`}</p>`}
  else body=`<p>The treatment is continuous, so the <b>partially linear model</b>: Y = θD + g(X) + ε, where g is any function of the controls, learned by machine learning. θ is a single slope, as in OLS.</p><p><b>Rule B3:</b> continuous treatment → DML partially linear model.</p>`;
  addStep(k++,R.instr?'The instrument: first stage':R.binary?'Binary treatment: propensity scores and overlap':'Continuous treatment',body+code(R.instr?`fs = sm.OLS(df["${R.treat}"], sm.add_constant(df[["${R.instr}"] + controls])).fit(cov_type="HC1")\nprint("first-stage F =", (fs.params["${R.instr}"] / fs.bse["${R.instr}"]) ** 2)`:R.binary?`from sklearn.model_selection import cross_val_predict\nfrom sklearn.ensemble import RandomForestClassifier\n\nps = cross_val_predict(RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=0),\n                       df[controls], df["${R.treat}"], cv=5, method="predict_proba")[:, 1]\nprint(ps.min(), ps.max(), ((ps < ${C.OVLO}) | (ps > ${C.OVHI})).mean())`:`# continuous treatment: partially linear model Y = theta * D + g(X) + e`),t);t+=gap;
  setTimeout(()=>{if(my!==run)return;light(['start','goal','conf','ctl','typ'],{typ:R.instr?'Instrument':R.binary?'Binary':'Continuous'})},t-gap+300*D)}
 const de=R.dml,lab={PLR:'partially linear model',IRM:'interactive model (ATE)',PLIV:'partially linear IV model'}[R.kind];
 addStep(k++,'Double machine learning: LASSO and random-forest learners',`<details class="explain" open><summary>How DML works, in three lines</summary><ol><li>Predict the outcome and the treatment${R.instr?' and the instrument':''} from the controls with a machine-learning model, each prediction made on data the model was not trained on (<b>cross-fitting</b>: five folds).</li><li>Take the residuals: the part of each that the controls do not explain.</li><li>Regress the outcome residual on the treatment residual${R.instr?', using the instrument residual as the instrument':''}${R.kind==='IRM'?'; for a binary treatment, combine outcome models and propensity scores in the doubly robust score':''}. The result is θ with a valid standard error.</li></ol><p>The estimate should not depend much on which learner is used. <b>Rule B4:</b> if LASSO and the forest differ by more than one standard error, the relationship is nonlinear and the forest is reported.</p></details>
  <div class="tw"><table><tr><th>Estimator</th><th>θ</th><th>s.e.</th><th>95% CI</th></tr><tr><td>OLS${R.instr?' (not IV)':''}</td><td>${big(R.ols.b)}</td><td>${big(R.ols.se)}</td><td>–</td></tr><tr><td>DML ${lab}, LASSO</td><td>${big(de.lasso.theta)}</td><td>${big(de.lasso.se)}</td><td>[${big(de.lasso.lo)}, ${big(de.lasso.hi)}]</td></tr><tr><td>DML ${lab}, random forest</td><td>${big(de.rf.theta)}</td><td>${big(de.rf.se)}</td><td>[${big(de.rf.lo)}, ${big(de.rf.hi)}]</td></tr></table></div>
  <p style="font-size:11px;color:#8793ab">Five-fold cross-fitting, two repetitions (the median is reported); the guide uses DoubleML with more repetitions.${R.subsampled?' A random 4,000 observations are used in the browser.':''}</p>
  <p><b>Result:</b> the two learners differ by ${fmt(R.gap_se,2)} standard errors. ${R.gap_se>1?'<span class="no">More than one: report the random-forest learner and discuss the difference.</span>':'<span class="yes">Less than one: the estimate is robust to the learner.</span>'}</p>`+code(`import doubleml as dml\nfrom sklearn.linear_model import LassoCV${R.kind==='IRM'?', LogisticRegressionCV':''}\nfrom sklearn.ensemble import RandomForestRegressor${R.kind==='IRM'?', RandomForestClassifier':''}\n\ndata = dml.DoubleMLData(df, y_col="${R.dep}", d_cols="${R.treat}", x_cols=controls${R.instr?`, z_cols="${R.instr}"`:''})\nrf = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5)\n${R.kind==='IRM'?`model = dml.DoubleMLIRM(data, ml_g=rf, ml_m=RandomForestClassifier(n_estimators=500, min_samples_leaf=5), n_folds=5, n_rep=10)`:R.kind==='PLIV'?`model = dml.DoubleMLPLIV(data, ml_l=rf, ml_m=rf, ml_r=rf, n_folds=5, n_rep=10)`:`model = dml.DoubleMLPLR(data, ml_l=rf, ml_m=rf, n_folds=5, n_rep=10)`}\nprint(model.fit().summary)`),t);t+=gap;
 if(R.model==='OLS'){setTimeout(()=>{if(my!==run)return;finish('OLS',R)},t);return}
 setTimeout(()=>{if(my!==run)return;
  if(R.cf_possible&&R.kind!=='PLIV')askHet(R);else finish(R.model,R)},t);
}
function askHet(R){const s=el('div','step',`<div class="label">One question</div><h3>Do you need to know who gains most from ${esc(R.treat)}, not only the average effect?</h3><p>With ${R.n.toLocaleString('en-US')} observations (at least ${C.CFN.toLocaleString('en-US')}), there are enough data to estimate how the effect varies with the controls.</p><div class="opts"><button type="button" data-a="${R.model}">No, the average effect answers my question<small>${MODELS[R.model].name}</small></button><button type="button" data-a="CF">Yes, for example by income or age<small>Causal forest, with the DML average effect as the benchmark</small></button></div>`);
 $('#steps').appendChild(s);requestAnimationFrame(()=>s.classList.add('in'));if(!reduce)s.scrollIntoView({behavior:'smooth',block:'nearest'});
 s.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{s.querySelectorAll('button').forEach(x=>x.disabled=true);b.style.borderColor='var(--navy)';b.style.background='var(--tint)';finish(b.dataset.a,R,b.dataset.a==='CF'?R.model:null)}))}

/* ---------- recommendation ---------- */
function evidence(R,m){if(!R)return '';const rm=R.rmse||{};
 if(m==='LASSO')return `No two predictors are correlated above ${C.HIGHR} (largest ${fmt(R.corr.max_r,2)}), and LASSO keeps ${(R.lasso_kept||[]).length} of ${R.p} predictors at the one-standard-error penalty.${R.nonlinear===false?` The forest is ${pct(-R.gain)} ${R.gain<0?'worse':'better'} than LASSO, below the ${C.GAIN}% threshold.`:''}`;
 if(m==='RIDGE')return `No two predictors are correlated above ${C.HIGHR}, and LASSO keeps ${(R.lasso_kept||[]).length} of ${R.p} predictors (${pct(R.lasso_share)}): the effects are dense. Cross-validated RMSE: ridge ${big(rm.ridge)}, LASSO ${big(rm.lasso)}.`;
 if(m==='ENET')return `${R.corr.pair.map(esc).join(' and ')} are correlated at ${fmt(R.corr.max_r,2)}, above ${C.HIGHR}. Cross-validated RMSE: elastic net ${big(rm.enet)}, LASSO ${big(rm.lasso)}, ridge ${big(rm.ridge)}.`;
 if(m==='RF')return `The forest's cross-validated RMSE is ${pct(R.gain)} below LASSO's (${big(rm.rf)} against ${big(rm.lasso)}), and N = ${R.n_used.toLocaleString('en-US')} is below ${C.BIGN.toLocaleString('en-US')}.`;
 if(m==='XGB')return `The forest's cross-validated RMSE is ${pct(R.gain)} below LASSO's, and N = ${R.n_used.toLocaleString('en-US')} is above ${C.BIGN.toLocaleString('en-US')}.${rm.gb!=null?` Boosting: ${big(rm.gb)}; forest: ${big(rm.rf)}.`:''}`;
 if(m==='OLS')return `${R.p} controls, and the forest improves on LASSO by only ${pct(Math.max(R.nl.y.gain,R.nl.d.gain))} at best. OLS: ${big(R.ols.b)} (s.e. ${big(R.ols.se)}); DML with a forest: ${big(R.dml.rf.theta)} (s.e. ${big(R.dml.rf.se)}).`;
 const th=R.learner==='rf'?R.dml.rf:R.dml.lasso;
 if(m==='CF')return `N = ${R.n.toLocaleString('en-US')}, enough for a causal forest. The DML average effect, ${big(R.dml.rf.theta)} (s.e. ${big(R.dml.rf.se)}), is the benchmark the forest's average must match.`;
 return `${R.p} controls${R.controls_nonlinear?', entering nonlinearly':''}. DML estimate: ${big(th.theta)} (95% CI ${big(th.lo)} to ${big(th.hi)}), against ${big(R.ols.b)} by OLS.${R.kind==='PLIV'?` First-stage F = ${fmt(R.first_stage.F,1)}.`:''}`}
function caution(R,m){if(!R)return QA.weak&&m==='PLIV'?`<div class="caution"><b>Check before you write.</b> A first-stage F below ${C.WEAKF} means a weak instrument: report Anderson–Rubin confidence intervals.</div>`:'';
 const c=[];if(m==='IRM'||m==='CF'){if(R.overlap&&R.overlap.share_out>0)c.push(`${pct(R.overlap.share_out)} of propensity scores lie outside [${C.OVLO}, ${C.OVHI}]: report the estimate with those observations trimmed.`)}
 if(m==='PLIV'&&R.weak)c.push(`The first-stage F is ${fmt(R.first_stage.F,1)}, below ${C.WEAKF}: report Anderson–Rubin confidence intervals.`);
 if(R.goal==='predict'&&R.ts)c.push('Time series: keep rolling-origin validation and compare every model with an AR benchmark.');
 if(['PLR','IRM','PLIV'].includes(m)&&R.gap_se>1)c.push(`LASSO and the forest differ by ${fmt(R.gap_se,1)} standard errors: report both, and lead with the forest.`);
 if((R.lasso_kept||[1]).length===0&&m==='LASSO')c.push('LASSO keeps no predictor: the variables in this file do not predict the outcome better than its mean. Look for better predictors before fitting any model.');
 return c.length?`<div class="caution"><b>Check before you write.</b> ${c.join(' ')}</div>`:''}
function finish(m,R,alt){const my=run,M=MODELS[m];const d=light(pathTo(m,R));if(alt){const a=$('#n-'+alt);a&&a.classList.add('alt')}
 setTimeout(()=>{if(my!==run)return;const ev=evidence(R,m);
  $('#result').innerHTML=`<div class="result"><div class="label">Recommended method</div><h2>${M.name}</h2><p>${M.why}${ev?' <b>In this data:</b> '+ev:''}</p>${caution(R,m)}${M.steps.length?`<div class="label" style="margin-bottom:6px">What the paper needs</div><ul>${M.steps.map(s=>`<li>${s}</li>`).join('')}</ul>`:''}<div class="acts">${GUIDE[m]?`<a class="btn" href="${GUIDE[m]}.html">Go to the ${M.short} guide →</a>`:m==='CIG'?'<a class="btn" href="../causal-inference-guide/index.html">Open the Causal Inference Guide →</a>':''}${M.lec.map((l,i)=>`<a class="btn${GUIDE[m]||m==='CIG'||i?' ghost':''}" href="${l[0]}">${l[1]} →</a>`).join('')}${alt?`<a class="btn ghost" href="#g-${alt}" data-go="${alt}">Benchmark: ${MODELS[alt].short}</a>`:''}${R?'':'<button class="btn ghost" type="button" id="again">Start over</button>'}</div></div>`;
  $('#result').scrollIntoView({behavior:reduce?'auto':'smooth',block:'center'});const ag=$('#again');ag&&ag.addEventListener('click',()=>setMode('ask'))},d+250*D)}

/* ---------- examples ---------- */
function initExamples(){let grp=null;EXAMPLES.forEach(x=>{if(x.group!==grp){grp=x.group;$('#ex').appendChild(el('div','exgroup',esc(grp)));$('#ex').appendChild(el('div','ex'))}
 const b=el('button','',`<small>${esc(x.meta)}</small><b>${esc(x.title)}</b><span>${esc(x.desc)}</span>`);b.type='button';
 b.addEventListener('click',()=>{document.querySelectorAll('.ex button').forEach(e=>e.classList.remove('sel'));b.classList.add('sel');show(x.result,x.title)});$('#ex').lastChild.appendChild(b)})}

/* ---------- upload ---------- */
let pyReady=null,pending=null;
function status(t){$('#status').classList.toggle('on',!!t);$('#status-text').textContent=t||''}
function loadPy(){if(pyReady)return pyReady;pyReady=(async()=>{status('Loading Python. The first time takes about 20 seconds…');
  await new Promise((res,rej)=>{const s=document.createElement('script');s.src=PYO+'pyodide.js';s.onload=res;s.onerror=()=>rej(new Error('Could not load Python. Check your internet connection and try again.'));document.head.appendChild(s)});
  const py=await loadPyodide({indexURL:PYO});status('Loading pandas, statsmodels and scikit-learn…');await py.loadPackage(['pandas','scipy','statsmodels','scikit-learn']);py.runPython($('#py-src').textContent);return py})();
 pyReady.catch(()=>{pyReady=null});return pyReady}
async function analyse(text,name,treat,instr){$('#upload-err').innerHTML='';$('#steps').innerHTML='';$('#result').innerHTML='';++run;resetTree();
 try{const py=await loadPy();status(GOAL==='causal'?'Running the checks and double machine learning. This can take one to three minutes…':'Running the checks on your data. This can take up to a minute…');await new Promise(r=>setTimeout(r,40));
  const R=JSON.parse(py.globals.get('analyze')(text,GOAL,treat||null,instr||null));status('');if(R.error)throw new Error(R.error);show(R,name)}
 catch(e){status('');$('#upload-err').innerHTML=`<div class="err">${esc(e.message||e)}</div>`}}
async function handleFile(f){$('#upload-err').innerHTML='';$('#cfg').innerHTML='';
 if(f.size>8e6){$('#upload-err').innerHTML='<div class="err">The file is larger than 8 MB. Please upload a smaller CSV.</div>';return}
 const text=await f.text();
 if(GOAL!=='causal')return analyse(text,f.name);
 const head=text.split(/\r?\n/)[0].split(',').map(s=>s.trim().replace(/^"|"$/g,''));
 if(head.length<3){$('#upload-err').innerHTML='<div class="err">A causal analysis needs at least three columns: the outcome, the treatment and one control.</div>';return}
 const opt=(sel)=>head.slice(1).map(h=>`<option${h===sel?' selected':''}>${esc(h)}</option>`).join('');
 $('#cfg').innerHTML=`<div class="cfg"><label>Treatment<select id="cT">${opt(head[1])}</select></label><label>Instrument (optional)<select id="cZ"><option value="">None</option>${opt('')}</select></label><button class="btn" type="button" id="cGo">Run the checks</button></div><div class="q" style="margin-top:12px"><div class="label">Rule B1</div><p style="margin:4px 0 8px;font-size:13px">Are all the important confounders of <b>${esc(head[1])}</b> among your controls? If not, and you have no instrument, machine learning cannot remove the bias.</p><div class="opts"><button type="button" id="cNo">No, and I have no instrument<small>Go to the research designs</small></button></div></div>`;
 $('#cT').addEventListener('change',e=>{$('#cfg .q b').textContent=e.target.value});
 $('#cNo').addEventListener('click',()=>{++run;resetTree();$('#steps').innerHTML='';light(['start','goal','conf'],{goal:'Causal effect',conf:'Unobserved'});finish('CIG',null)});
 $('#cGo').addEventListener('click',()=>{const T=$('#cT').value,Z=$('#cZ').value;if(Z&&Z===T){$('#upload-err').innerHTML='<div class="err">The instrument must be a different column from the treatment.</div>';return}analyse(text,f.name,T,Z)})}
function initUpload(){const d=$('#drop');$('#file').addEventListener('change',e=>{e.target.files[0]&&handleFile(e.target.files[0]);e.target.value=''});
 ['dragenter','dragover'].forEach(ev=>d.addEventListener(ev,e=>{e.preventDefault();d.classList.add('over')}));['dragleave','drop'].forEach(ev=>d.addEventListener(ev,e=>{e.preventDefault();d.classList.remove('over')}));
 d.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];f&&handleFile(f)});
 document.querySelectorAll('.goal button').forEach(b=>b.addEventListener('click',()=>{GOAL=b.dataset.goal;document.querySelectorAll('.goal button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));$('#cfg').innerHTML='';$('#upload-err').innerHTML=''}))}

/* ---------- questions ---------- */
const QS={
 goal:{t:'What do you want to do?',o:[['Predict or forecast an outcome','Accuracy on new data is what matters','ts',['start','goal','ts']],['Estimate the effect of one variable on another','A causal effect, with a standard error','conf',['start','goal','conf'],'causal',1]]},
 ts:{t:'Is your data a time series?',o:[['Yes','Test every series for a unit root first, difference I(1) series, and validate with rolling-origin splits','nn',['ts','nn'],'ts',1],['No, a cross-section or pooled data','','nn',['ts','nn']]]},
 nn:{t:'How many observations do you have?',o:[['Fewer than '+C.SMALLN,'Tree-based methods would overfit','lin',['nn','lin'],'small',1],[C.SMALLN+' or more','','nl',['nn','nl']]]},
 nl:{t:'Is a random forest’s cross-validated RMSE at least '+C.GAIN+'% below LASSO’s?',o:[['Yes','The relationship is nonlinear','bign',['nl','bign']],['No','A linear model is enough','lin',['nl','lin']]]},
 bign:{t:'Do you have at least '+C.BIGN.toLocaleString('en-US')+' observations?',o:[['Yes','Enough to tune a boosted model','XGB'],['No','','RF']]},
 lin:{t:'Are any two predictors correlated above '+C.HIGHR+'?',o:[['Yes','','ENET'],['No','','sp']]},
 sp:{t:'At the one-standard-error penalty, does LASSO keep fewer than half of the predictors?',o:[['Yes','A few predictors carry the signal','LASSO'],['No','Many small effects','RIDGE']]},
 conf:{t:'Are all the important confounders among your control variables?',o:[['Yes','Selection on observables','ctl',['conf','ctl']],['No, but I have an instrument','A variable that shifts the treatment and affects the outcome only through it','ivF',['conf','ctl','typ']],['No, and I have no instrument','','CIG']]},
 ivF:{t:'Is the first-stage F statistic of the instrument at least '+C.WEAKF+'?',o:[['Yes','','PLIV'],['No','A weak instrument','PLIV',null,'weak',1]]},
 ctl:{t:'Do you have at least '+C.FEWCTRL+' controls, or does a random forest beat LASSO by '+C.GAIN+'% in predicting the outcome or the treatment from them?',o:[['Yes','Many or nonlinear controls','typ',['ctl','typ']],['No','Few controls that enter linearly','OLS']]},
 typ:{t:'Is the treatment binary or continuous?',o:[['Binary (0/1)','For example a programme, a policy or eligibility','het',['typ','IRM','het'],'kind','IRM'],['Continuous','For example a price, an amount or a rate','het',['typ','PLR','het'],'kind','PLR']]},
 het:{t:'Do you need to know who gains most, and do you have at least '+C.CFN.toLocaleString('en-US')+' observations?',o:[['No, the average effect is enough','','KIND'],['Yes','Effects by subgroup','CF']]}
};
function ask(id,n=1){const Q=QS[id];$('#q').innerHTML=`<div class="label">Question ${n}</div><h3>${Q.t}</h3><div class="opts">${Q.o.map((o,i)=>`<button type="button" data-i="${i}">${o[0]}${o[1]?`<small>${o[1]}</small>`:''}</button>`).join('')}</div>`;
 $('#q').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{const o=Q.o[+b.dataset.i];let to=o[2];if(o[4])QA[o[4]]=o[5];
  if(to==='KIND')to=QA.kind;
  if(QS[to]){o[3]&&light(o[3]);ask(to,n+1)}else{$('#q').innerHTML='<p class="sub" style="margin:0">Your answers lead to the method shown below.</p>';finish(to,null,to==='CF'?QA.kind:to==='XGB'?'RF':to==='RF'?'XGB':null)}}))}

/* ---------- guides, modes, init ---------- */
function initGuides(){Object.entries(MODELS).filter(([k])=>ALLG[k]).forEach(([k,M])=>{const g=GUIDE[k];$('#gcards').appendChild(el('article','gcard',`<span class="tag${g?' ok':''}">${g?'Step-by-step guide':'In preparation'}</span><h3>${M.short}</h3><p class="when">${M.when}</p><p>${M.steps.slice(0,5).join(' · ')} …</p>${g?`<a href="${g}.html" style="font-size:12.5px">Open the ${M.short} guide →</a>`:''}${M.lec.map(l=>`<a href="${l[0]}">${l[1]} →</a>`).join('')}`)).id='g-'+k})}
document.addEventListener('click',e=>{const a=e.target.closest('[data-go]');if(!a)return;const c=$('#g-'+a.dataset.go);if(c){c.classList.remove('flash');void c.offsetWidth;c.classList.add('flash')}});
function setMode(m){++run;QA={};document.querySelectorAll('.modes button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===m)));['example','upload','ask'].forEach(x=>$('#m-'+x).hidden=x!==m);
 $('#steps').innerHTML='';$('#result').innerHTML='';$('#upload-err').innerHTML='';$('#cfg').innerHTML='';status('');resetTree();document.querySelectorAll('.ex button').forEach(e=>e.classList.remove('sel'));if(m==='ask'){light(['start']);ask('goal')}}
document.querySelectorAll('.modes button').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
drawTree();initExamples();initUpload();initGuides();$('#year').textContent=new Date().getFullYear();
