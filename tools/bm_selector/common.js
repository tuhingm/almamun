const PYO='https://cdn.jsdelivr.net/pyodide/v0.27.2/full/';
const $=s=>document.querySelector(s), el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e};
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches, D=reduce?0:1;
const pf=p=>p==null?'–':(p<0.001?'<0.001':p.toFixed(3)), nf=(v,d=2)=>v==null?'–':Number(v).toFixed(d);
let run=0;
const MODELS=HUB.MODELS,N=HUB.N,E=HUB.E,TL=HUB.TL,GUIDE=HUB.GUIDE;

/* ---------- decision tree ---------- */
function treePath(m){const kids={};E.forEach(([a,b])=>(kids[a]=kids[a]||[]).push(b));
 const dfs=(n,p)=>{p=p.concat(n);if(n===m)return p;for(const c of kids[n]||[]){const r=dfs(c,p);if(r)return r}return null};return dfs('start',[])||['start']}
function drawTree(){
 const H=16;let s=`<svg viewBox="${HUB.vb}" role="img" aria-label="Decision tree from your data to a model">`;
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

/* ---------- small charts and tags ---------- */
function spark(v){const a=v.filter(x=>x!=null);if(a.length<2)return '';const mn=Math.min(...a),mx=Math.max(...a),r=mx-mn||1;let d='',pen=false;
 v.forEach((y,i)=>{if(y==null){pen=false;return}d+=(pen?'L':'M')+(i/(v.length-1)*200).toFixed(1)+' '+(40-(y-mn)/r*36).toFixed(1);pen=true});
 return `<svg viewBox="0 0 200 44" preserveAspectRatio="none"><path pathLength="1" d="${d}"/></svg>`}
const tg=o=>/I\(1\)/.test(o)?(o==='I(1)'?'<span class="tag i1">I(1)</span>':'<span class="tag i2">I(1) or I(2)</span>'):o==='I(0)'?'<span class="tag ok">I(0)</span>':'<span class="tag">n/a</span>';
const flag=(on,yes,no)=>on?`<span class="tag i2">${yes}</span>`:`<span class="tag ok">${no}</span>`;
const code=c=>`<details class="code"><summary>Python code for this step</summary><pre>${esc(c)}</pre></details>`;
const explain=(s,h)=>`<details class="explain"><summary>${s}</summary>${h}</details>`;

/* ---------- reading the facts ---------- */
function flags(F){
 const so=Object.values(F.stationarity||{}),nI1=so.filter(s=>/I\(1\)/.test(s.order)).length,nonstat=so.length>0&&nI1>=so.length/2;
 let volBest=null,tvpBest=null;
 for(const [v,r] of Object.entries(F.volatility||{})){const ps=[r.levene_p,r.arch_p].filter(x=>x!=null),pm=ps.length?Math.min(...ps):1;
  const on=(r.levene_p!=null&&r.levene_p<.01&&r.ratio!=null&&(r.ratio<.5||r.ratio>2))||(r.arch_p!=null&&r.arch_p<.01);if(on&&(!volBest||pm<volBest.p))volBest={v,p:pm,r}}
 for(const [v,r] of Object.entries(F.instability||{})){if(r.cusum_p!=null&&r.cusum_p<.05&&(!tvpBest||r.cusum_p<tvpBest.p))tvpBest={v,p:r.cusum_p,r}}
 const T=F.threshold&&F.threshold.best?F.threshold.by_variable[F.threshold.best]:null,thr=T&&T.p<.05?Object.assign({v:F.threshold.best},T):null;
 const ms=F.ms;return {nonstat,volBest,tvpBest,thr,ms,hidden:!!(ms&&ms.lr>14),msWeak:!!(ms&&ms.aic_ms<ms.aic_linear)}}

/* ---------- results ---------- */
function addStep(n,title,html,delay){const s=el('div','step',`<div class="label">Step ${n}</div><h3>${title}</h3>${html}`);$('#steps').appendChild(s);const my=run;setTimeout(()=>{if(my!==run)return;s.classList.add('in')},delay)}
const PYTXT={
 stat:`from statsmodels.tsa.stattools import adfuller, kpss
p_adf  = adfuller(x, regression="c", autolag="AIC")[1]       # H0: unit root
p_adf1 = adfuller(np.diff(x), regression="c", autolag="AIC")[1]
p_kpss = kpss(x, regression="c", nlags="auto")[1]            # H0: stationary
# I(0): ADF rejects and KPSS does not. I(1): ADF does not reject in levels but rejects after differencing.`,
 vol:`# AR(1) residuals e for each series; compare the two halves
h = len(e)//2
ratio  = e[h:].var() / e[:h].var()
levene = scipy.stats.levene(e[:h], e[h:])[1]
arch   = statsmodels.stats.diagnostic.het_arch(e, nlags=4)[1]`,
 thr:`# Sup-LR threshold test for a VAR(1): for each candidate threshold variable z,
# fit the VAR with one regime and with two regimes split at a grid of quantiles of z (15%-85%),
# take the largest likelihood ratio, and get its p-value from a wild bootstrap under the linear null.`,
 ms:`import statsmodels.api as sm
mod = sm.tsa.MarkovRegression(x, k_regimes=2, trend="c", switching_variance=True)
res = mod.fit(search_reps=15)
lr = 2*(res.llf - loglik_one_regime)      # compare with the Garcia critical value, about 14
res.aic, res.bic, res.smoothed_marginal_probabilities`};
function show(F,label){
 const my=++run;$('#steps').innerHTML='';$('#result').innerHTML='';resetTree();
 if(F.error){$('#steps').innerHTML=`<div class="err">${esc(F.error)}</div>`;return}
 F.n_series=F.n_series||(F.tested_vars||F.vars||[]).length;
 const f=flags(F),Dn=decide(F,window.OPTS||{}),path=treePath(Dn.m),gap=900*D;let t=150,k=1,done=0;
 const lightTo=n=>light(path.slice(0,Math.max(1,Math.min(n,path.length-1))),{},false);
 const struct=F.aim==='regression'?'One equation':F.unit?'Panel':F.mixed_freq?'Mixed frequency':'Time series';
 lightTo(1);
 const vars=F.aim==='regression'?[F.reg.y].concat(F.vars.filter(v=>v!==F.reg.y)):(F.tested_vars||F.vars);
 const ser=F.series||{};
 addStep(k++,'Your data',`<p><b>${F.rows}</b> rows${label?' · '+esc(label):''}. ${F.time?`Time column <code>${esc(F.time)}</code>. `:''}${F.unit?`Unit column <code>${esc(F.unit)}</code> (${F.unit_count} units). `:''}${F.aim==='regression'?`Dependent variable <code>${esc(F.reg.y)}</code> with ${F.reg.k} regressors.`:`Variables: ${vars.map(v=>'<code>'+esc(v)+'</code>').join(', ')}.`}${F.mixed_freq?` The variable${F.sparse_vars.length>1?'s':''} ${F.sparse_vars.map(v=>'<code>'+esc(v)+'</code>').join(', ')} ${F.sparse_vars.length>1?'are':'is'} missing in most rows, so the data mix frequencies.`:''}</p>
  <div class="stats"><div class="stat"><b>${struct}</b><span>Structure</span></div><div class="stat"><b>${F.aim==='regression'?F.reg.n:F.periods}</b><span>${F.aim==='regression'?'Observations':'Periods'}</span></div><div class="stat"><b>${F.unit?F.unit_count:(F.aim==='regression'?F.reg.k+1:(F.n_series||vars.length))}</b><span>${F.unit?'Units':(F.aim==='regression'?'Variables':'Series')}</span></div><div class="stat"><b>${F.mixed_freq?'Yes':'No'}</b><span>Mixed frequency</span></div></div>
  <div class="sparks">${Object.entries(ser).map(([v,s])=>`<div class="spark"><b>${esc(v)}${F.unit?' (unit average)':''}</b>${spark(s)}</div>`).join('')}</div>`,t);
 t+=gap;lightTo(Math.min(2,path.length));
 const st=F.stationarity||{},stK=Object.keys(st);
 if(stK.length){
  addStep(k++,'Are the series stationary?',`<p>Both models in this guide are built for stationary data (or, for a VAR in levels, for a prior that expects a unit root). The ADF test has a unit root as its null; KPSS has stationarity as its null. ${f.nonstat?'<b>At least half of the series look I(1) in levels.</b>':'The series look stationary, or close enough to continue.'}</p>
  <table class="etable"><tr><th>Variable</th><th>ADF p</th><th>KPSS p</th><th>Reading</th></tr>${stK.map(v=>`<tr><td>${esc(v)}</td><td>${pf(st[v].adf_p)}</td><td>${pf(st[v].kpss_p)}</td><td>${tg(st[v].order)}</td></tr>`).join('')}</table>
  ${F.unit?'<p class="hint">For a panel the test is run on the average across units after removing unit means.</p>':''}
  ${explain('How to read the unit-root tests','<p>An I(0) series fluctuates around a stable mean. An I(1) series wanders: its shocks are permanent and a regression in levels can look significant by accident. For a Bayesian VAR in levels this is fine, because the Minnesota prior is centred on a random walk. For a Markov-switching model it is a problem, because a trending series is cut into pieces that follow the trend.</p>')}${code(PYTXT.stat)}`,t);t+=gap}
 lightTo(Math.min(3,path.length));
 if(F.aim==='regression'){const r=F.reg;
  addStep(k++,'Sample size and collinearity',`<p>${r.n} observations, ${r.k} regressors: <b>${r.n_per_k.toFixed(0)}</b> observations per regressor. The largest variance-inflation factor is <b>${r.vif_max.toFixed(1)}</b> (above 10 is a warning).</p>
  <div class="stats"><div class="stat"><b>${r.n}</b><span>Observations</span></div><div class="stat"><b>${r.k}</b><span>Regressors</span></div><div class="stat"><b>${r.n_per_k.toFixed(0)}</b><span>Per regressor</span></div><div class="stat"><b>${r.vif_max.toFixed(1)}</b><span>Max VIF</span></div></div>
  ${explain('Why this matters','<p>A prior matters most when there are few observations per parameter or when regressors are strongly correlated, because the likelihood alone cannot pin the coefficients down. With a large sample and low collinearity, a Bayesian regression and OLS give nearly the same answer.</p>')}`,t);t+=gap}
 else if(F.volatility){
  const rows=Object.keys(F.volatility).map(v=>{const a=F.volatility[v],b=(F.instability||{})[v]||{};
   const vo=(a.levene_p!=null&&a.levene_p<.01&&a.ratio!=null&&(a.ratio<.5||a.ratio>2))||(a.arch_p!=null&&a.arch_p<.01),io=b.cusum_p!=null&&b.cusum_p<.05;
   return `<tr><td>${esc(v)}</td><td>${nf(a.ratio)}</td><td>${pf(a.levene_p)}</td><td>${pf(a.arch_p)}</td><td>${pf(b.cusum_p)}</td><td>${nf(b.supF,1)}</td><td>${flag(vo,'volatility','stable')} ${flag(io,'unstable','stable')}</td></tr>`}).join('');
  addStep(k++,'Does the volatility or the coefficient pattern change over time?',`<p>These tests run on a VAR(1) equation by equation. They tell you whether a constant-parameter VAR is adequate, or whether the size of shocks or the coefficients move.${f.nonstat?' <b>Because the series look I(1), read these with care: they are not used for the choice.</b>':''}</p>
  <table class="etable"><tr><th>Equation</th><th>Var. ratio 2nd/1st half</th><th>Levene p</th><th>ARCH p</th><th>CUSUM p</th><th>sup-F</th><th>Reading</th></tr>${rows}</table>
  ${explain('What each column means','<p><b>Variance ratio and Levene:</b> the residual variance in the second half of the sample divided by the first half, and a test of equal variances. <b>ARCH:</b> whether big residuals cluster together. Any of these points to a stochastic-volatility VAR. <b>CUSUM and sup-F:</b> whether the coefficients are stable (CUSUM) or break at some date (sup-F). Instability points to a time-varying-parameter VAR.</p>')}${code(PYTXT.vol)}`,t);t+=gap;
  const T=F.threshold,tr=Object.keys(T.by_variable).map(v=>{const r=T.by_variable[v];return `<tr><td>${esc(v)}${v===T.best?' ★':''}</td><td>${nf(r.supLR,1)}</td><td>${r.threshold==null?'–':nf(r.threshold)}</td><td>${pf(r.p)}</td><td>${flag(r.p<.05,'threshold','linear')}</td></tr>`}).join('');
  addStep(k++,'Do the dynamics differ above and below a threshold?',`<p>The sup-LR test asks, for each variable in turn, whether splitting the sample at one of its values gives a better VAR than a single regime. Wild-bootstrap p-values are used because the threshold is estimated. The strongest candidate is marked ★.</p>
  <table class="etable"><tr><th>Threshold variable</th><th>sup-LR</th><th>Threshold</th><th>Bootstrap p</th><th>Reading</th></tr>${tr}</table>
  ${explain('What a significant threshold means','<p>A significant threshold means that you can name the regime with a variable you observe (inflation above 4%, the interest rate near zero). That is a threshold VAR, covered in the Bayesian Guide. If the instability is real but no observed variable explains it, the regimes may be hidden, and a Markov-switching model is the next thing to try.</p>')}${code(PYTXT.thr)}`,t);t+=gap}
 else{addStep(k++,'Structure decides the model',`<p>${F.unit?'For a panel the tests of time variation need one long time series, so the structure itself decides: a panel of units is estimated as a panel VAR (Bayesian) or with a shared regime path (Markov).':F.mixed_freq?'With mixed frequencies the structure itself decides: only a mixed-frequency model uses all of the data as it arrives.':'The volatility, instability and threshold tests need a single complete time series.'}</p>`,t);t+=gap}
 lightTo(Math.min(path.length-1,Math.max(4,path.length-1)));
 if(F.ms&&!F.mixed_freq){const m=F.ms;
  addStep(k++,'Are there hidden regimes?',`<p>For <code>${esc(F.y||'')}</code>, a two-regime Markov-switching model (mean and variance switch) is compared with a single regime. The likelihood-ratio statistic does not have the usual chi-square distribution, so it is compared with the Garcia critical values (about 14 for the mean and variance switching). This is a screening fit with no autoregressive terms; the full test for your own specification is in the guide for the model.</p>
  <div class="stats"><div class="stat"><b>${nf(m.lr,1)}</b><span>LR (critical ≈ 14)</span></div><div class="stat"><b>${nf(m.aic_ms,0)} / ${nf(m.aic_linear,0)}</b><span>AIC switching / one regime</span></div><div class="stat"><b>${nf(m.bic_ms,0)} / ${nf(m.bic_linear,0)}</b><span>BIC switching / one regime</span></div><div class="stat"><b>${m.min_regime} (${(100*m.min_share).toFixed(0)}%)</b><span>Periods in the smaller regime</span></div></div>
  <p>${m.lr>14?'<b>The test rejects a single regime.</b>':'<b>The test does not reject a single regime.</b>'} ${m.aic_ms<m.aic_linear?'AIC prefers the switching model':'AIC prefers one regime'}, ${m.bic_ms<m.bic_linear?'and BIC does too.':'while BIC prefers one regime.'} ${m.min_share<.05?'<b>The smaller regime holds very few periods: it may be a collapsed regime fitting a few outliers.</b>':''}</p>
  ${explain('How to read this','<p>Under the null of one regime, the regime parameters are not identified, so the ordinary chi-square test is invalid. A statistic above the Garcia critical value is evidence of regimes; below it, the data do not clearly require them. Information criteria give a second opinion, and a regime that holds only a handful of periods is a warning sign that the model is fitting outliers.</p>')}${code(PYTXT.ms)}`,t);t+=gap}
 setTimeout(()=>{if(my!==run)return;finish(Dn,F)},t);
}
function finish(Dn,F){
 const my=run,M=MODELS[Dn.m],path=treePath(Dn.m),d=light(path);
 if(Dn.alt){const a=$('#n-'+Dn.alt);a&&a.classList.add('alt')}
 const href=M.href?M.href:(GUIDE[Dn.m]?GUIDE[Dn.m]+'.html':null),btn=href?`<a class="btn" href="${href}">${GUIDE[Dn.m]?'Go to the '+M.short+' guide':'Open the '+M.short+' guide'} →</a>`:'';
 setTimeout(()=>{if(my!==run)return;
  $('#result').innerHTML=`<div class="result"><div class="label">Recommended model</div><h2>${M.name}</h2><p>${M.why}</p>
  ${Dn.why.length?`<div class="label" style="margin:14px 0 6px">${F?'Why this model in your data':'Your answers'}</div><ul>${Dn.why.map(w=>`<li>${w}</li>`).join('')}</ul>`:''}
  ${Dn.cau.length?`<div class="label" style="margin:14px 0 6px">Check before you rely on it</div><ul>${Dn.cau.map(w=>`<li>${w}</li>`).join('')}</ul>`:''}
  <div class="label" style="margin:14px 0 6px">What the paper needs</div><ul>${M.steps.map(s=>`<li>${s}</li>`).join('')}</ul>
  <div class="acts">${btn}${Dn.alt&&GUIDE[Dn.alt]?`<a class="btn ghost" href="#g-${Dn.alt}" data-go="${Dn.alt}">Compare: ${MODELS[Dn.alt].short}</a>`:''}${F?'':'<button class="btn ghost" type="button" id="again">Start over</button>'}</div></div>`;
  $('#result').scrollIntoView({behavior:reduce?'auto':'smooth',block:'center'});const ag=$('#again');ag&&ag.addEventListener('click',()=>setMode('ask'))},d+250*D)}

/* ---------- mode: examples ---------- */
function initExamples(){EXAMPLES.forEach(x=>{const b=el('button','',`<small>${esc(x.meta)}</small><b>${esc(x.title)}</b><span>${esc(x.desc)}</span>`);b.type='button';
 b.addEventListener('click',()=>{document.querySelectorAll('.ex button').forEach(e=>e.classList.remove('sel'));b.classList.add('sel');window.OPTS={};show(x.result,x.title)});$('#ex').appendChild(b)})}

/* ---------- mode: upload (Python in the browser) ---------- */
let pyReady=null,CSV=null,CSVNAME='';
function status(t){$('#status').classList.toggle('on',!!t);$('#status-text').textContent=t||''}
function loadPy(){if(pyReady)return pyReady;pyReady=(async()=>{status('Loading Python. The first time takes about 15 seconds…');
  await new Promise((res,rej)=>{const s=document.createElement('script');s.src=PYO+'pyodide.js';s.onload=res;s.onerror=()=>rej(new Error('Could not load Python. Check your internet connection and try again.'));document.head.appendChild(s)});
  const py=await loadPyodide({indexURL:PYO});status('Loading pandas, SciPy and statsmodels…');await py.loadPackage(['pandas','scipy','statsmodels']);py.runPython($('#py-src').textContent);return py})();
 pyReady.catch(()=>{pyReady=null});return pyReady}
function splitRow(l){const o=[];let c='',q=false;for(let i=0;i<l.length;i++){const ch=l[i];if(ch==='"'){if(q&&l[i+1]==='"'){c+='"';i++}else q=!q}else if(ch===','&&!q){o.push(c);c=''}else c+=ch}o.push(c);return o}
async function handleFile(f){$('#upload-err').innerHTML='';$('#steps').innerHTML='';$('#result').innerHTML='';$('#opts').hidden=true;++run;resetTree();
 try{if(f.size>8e6)throw new Error('The file is larger than 8 MB. Please upload a smaller CSV.');
  CSV=await f.text();CSVNAME=f.name;const lines=CSV.split(/\r?\n/).filter(x=>x.trim());if(lines.length<12)throw new Error('The file needs a header row and at least 11 data rows.');
  const head=splitRow(lines[0]).map(h=>h.trim()),rows=lines.slice(1,201).map(splitRow),num=head.filter((h,j)=>{const v=rows.map(r=>(r[j]||'').trim()).filter(x=>x!==''&&x.toLowerCase()!=='na'&&x!=='.');return v.length>=3&&v.filter(x=>isFinite(+x)).length>=.8*v.length});
  if(num.length<1)throw new Error('No numeric columns found. Check that the first row holds the column names.');
  const opt=(a,sel)=>a.map(x=>`<option value="${esc(x)}">${esc(x)}</option>`).join('');
  $('#opts').innerHTML=`<h3 style="font:800 14.5px Manrope;margin:16px 0 4px">${esc(f.name)}: ${lines.length-1} rows, ${head.length} columns</h3><p class="hint" style="margin-top:0">The columns are detected automatically. Change them below if the detection is wrong, then run the tests.</p>
  <div class="ctrl"><label>What do you want to estimate?<select id="oAim"><option value="var">Several series, or a panel (a VAR)</option><option value="regression">One regression equation</option></select></label>
  <label>Date column<select id="oTime"><option value="">Detect automatically</option>${opt(head)}</select></label>
  <label>Unit column (panel)<select id="oUnit"><option value="">Detect automatically</option><option value="none">None: a single time series</option>${opt(head)}</select></label>
  <label>Dependent (or first) variable<select id="oY"><option value="">First numeric column</option>${opt(num)}</select></label></div>
  <div class="vlist"><span class="label">Variables to include</span>${num.map(h=>`<label><input type="checkbox" class="oV" value="${esc(h)}" checked> ${esc(h)}</label>`).join('')}</div>
  ${HUB.known?'<label class="known"><input type="checkbox" id="oKnown"> I know the long-run means of the variables from theory or policy (for example an inflation target)</label>':''}
  <button class="btn" type="button" id="oRun">Run the tests →</button>`;
  $('#opts').hidden=false;$('#oRun').addEventListener('click',runUpload)}
 catch(e){status('');$('#upload-err').innerHTML=`<div class="err">${esc(e.message||e)}</div>`}}
async function runUpload(){$('#upload-err').innerHTML='';$('#steps').innerHTML='';$('#result').innerHTML='';++run;resetTree();
 try{const py=await loadPy();status('Running the tests on your data. Nonlinearity tests use a bootstrap and can take a minute…');await new Promise(r=>setTimeout(r,30));
  const vs=[...document.querySelectorAll('.oV:checked')].map(x=>x.value).join('\x1f');window.OPTS={known:!!($('#oKnown')&&$('#oKnown').checked)};
  const F=JSON.parse(py.globals.get('analyse_json')(CSV,$('#oTime').value,$('#oUnit').value,$('#oY').value,$('#oAim').value,vs));status('');if(F.error)throw new Error(F.error);
  show(F,CSVNAME)}
 catch(e){status('');$('#upload-err').innerHTML=`<div class="err">${esc(e.message||e)}</div>`}}
function initUpload(){const d=$('#drop');$('#file').addEventListener('change',e=>{e.target.files[0]&&handleFile(e.target.files[0]);e.target.value=''});
 ['dragenter','dragover'].forEach(ev=>d.addEventListener(ev,e=>{e.preventDefault();d.classList.add('over')}));['dragleave','drop'].forEach(ev=>d.addEventListener(ev,e=>{e.preventDefault();d.classList.remove('over')}));
 d.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];f&&handleFile(f)})}

/* ---------- mode: questions ---------- */
let ANS=[];
function ask(id,n=1){const Q=HUB.QS[id];$('#q').innerHTML=`<div class="label">Question ${n}</div><h3>${Q.t}</h3><div class="opts">${Q.o.map((o,i)=>`<button type="button" data-i="${i}">${o[0]}${o[1]?`<small>${o[1]}</small>`:''}</button>`).join('')}</div>`;
 $('#q').querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{const o=Q.o[+b.dataset.i],to=o[2];ANS.push(o[0]);light(treePath(to));
  if(HUB.QS[to])ask(to,n+1);else{$('#q').innerHTML='<p class="sub" style="margin:0">Your answers lead to the model shown below.</p>';
   finish({m:to,why:ANS.map(a=>'You answered: '+a),cau:[],alt:null},null)}}))}

/* ---------- guides, modes, init ---------- */
function initGuides(){Object.entries(GUIDE).forEach(([k,slug])=>{const M=MODELS[k];$('#gcards').appendChild(el('article','gcard',`<span class="tag ok">Step-by-step guide</span><h3>${M.short}</h3><p class="when">${M.when}</p><p>${M.steps.slice(0,5).join(' · ')} …</p><a href="${slug}.html" style="font-size:12.5px">Open the ${M.short} guide →</a>`)).id='g-'+k})}
document.addEventListener('click',e=>{const a=e.target.closest('[data-go]');if(!a)return;const c=$('#g-'+a.dataset.go);if(c){c.classList.remove('flash');void c.offsetWidth;c.classList.add('flash')}});
function setMode(m){++run;ANS=[];window.OPTS={};document.querySelectorAll('.modes button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.mode===m)));['example','upload','ask'].forEach(x=>$('#m-'+x).hidden=x!==m);
 $('#steps').innerHTML='';$('#result').innerHTML='';$('#upload-err').innerHTML='';status('');resetTree();document.querySelectorAll('.ex button').forEach(e=>e.classList.remove('sel'));if(m==='ask'){light(['start']);ask(HUB.first)}}
document.querySelectorAll('.modes button').forEach(b=>b.addEventListener('click',()=>setMode(b.dataset.mode)));
drawTree();initExamples();initUpload();initGuides();$('#year').textContent=new Date().getFullYear();
