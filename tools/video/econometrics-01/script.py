# Lecture 1 video script. Each slide: kicker, title, html body (elements with data-s="k" appear at step k),
# and `say`: one narration paragraph per step (step 0 = initial state).
COURSE = "Econometrics"
LECTURE = "Lecture 1"
TITLE = "From Probability to the Linear Regression Model"

SLIDES = [
dict(kind="title", say=[
 "Welcome to Econometrics. This is lecture one: from probability to the linear regression model. "
 "Before we ask whether a regression coefficient is causal, we need to know what it actually is. "
 "So in this lecture we build regression from probability alone, and we add assumptions only when we need them."]),

dict(kicker="Roadmap", title="What you will be able to do", body="""
<ol class="big">
<li data-s="1">Define the <b>conditional expectation function</b> and prove it is the best predictor</li>
<li data-s="2">Derive the <b>best linear predictor</b> — the linear projection</li>
<li data-s="3">Derive <b>OLS</b> as its sample analogue, and read its <b>geometry</b></li>
<li data-s="4">Prove the <b>Frisch–Waugh–Lovell</b> theorem</li>
<li data-s="5">Prove <b>unbiasedness</b> and the <b>Gauss–Markov</b> theorem — and state exactly what they assume</li>
</ol>""", say=[
 "Here is the plan.",
 "First, the conditional expectation function, and why it is the best possible predictor.",
 "Second, the best linear predictor, also called the linear projection.",
 "Third, ordinary least squares as the sample version of that projection, and its geometry.",
 "Fourth, the Frisch Waugh Lovell theorem, which tells us what it means to hold other variables constant.",
 "And fifth, unbiasedness and the Gauss Markov theorem, with a careful list of what each result assumes."]),

dict(kicker="1 · The conditional expectation function", title="A decomposition, not a model", body="""
<div class="def" data-s="0">For any \\((Y, X)\\) with \\(E[Y^2]<\\infty\\), define
\\[ m(x)=E[Y\\mid X=x], \\qquad e = Y - m(X). \\]</div>
<div class="eq-big" data-s="1">\\[ Y = m(X) + e \\]</div>
<p class="note" data-s="2">This holds for <b>every</b> joint distribution. No linearity, no normality, no causal story.</p>
""", say=[
 "Start with any outcome Y and any vector of variables X. The conditional expectation function, m of x, is the expected value of Y given that X equals x. "
 "Define the error e as Y minus m of X.",
 "Then Y equals m of X plus e. Notice what this is.",
 "It is a decomposition, not a model. It holds for every joint distribution with a finite second moment. We have assumed nothing about linearity, normality, or causality."]),

dict(kicker="1 · The conditional expectation function", title="Properties of the CEF error", body="""
<div class="thm"><b>Theorem 1.1.</b> With \\(e=Y-m(X)\\):
<ol>
<li data-s="1">\\(E[e\\mid X]=0\\)</li>
<li data-s="2">\\(E[e]=0\\)</li>
<li data-s="3">\\(E[h(X)\\,e]=0\\) for <b>any</b> function \\(h\\)</li>
</ol></div>
<div class="proof" data-s="4"><i>Proof of 3.</i> By iterated expectations,
\\[ E[h(X)e] = E\\big\\{h(X)\\,E[e\\mid X]\\big\\} = 0. \\]</div>
""", say=[
 "The CEF error has three properties.",
 "First, its conditional mean given X is zero. That follows directly from the definition.",
 "Second, its unconditional mean is zero, by the law of iterated expectations.",
 "Third, and most important: e is uncorrelated with any function of X at all, not just with X itself.",
 "The proof is one line. Condition on X. Then h of X is known, so it comes outside the inner expectation, and the expectation of e given X is zero."]),

dict(kicker="1 · The conditional expectation function", title="The CEF is the best predictor", body="""
<div class="thm"><b>Theorem 1.2.</b> For every \\(g\\) with \\(E[g(X)^2]<\\infty\\):
\\[ E\\big[(Y-g(X))^2\\big] \\;\\ge\\; E\\big[(Y-m(X))^2\\big]. \\]</div>
<div class="proof" data-s="1">Write \\(Y-g = e + (m-g)\\) and expand:
\\[ E[(Y-g)^2] = E[e^2] + 2\\underbrace{E[e\\,(m-g)]}_{=\\,0} + \\underbrace{E[(m-g)^2]}_{\\ge 0}. \\]</div>
<p class="note" data-s="2">Law of total variance: \\(\\operatorname{Var}(Y)=\\operatorname{Var}(m(X)) + E[\\operatorname{Var}(Y\\mid X)]\\).</p>
""", say=[
 "Why do we care about the CEF? Because it is the best predictor of Y given X in mean square. No other function of X does better.",
 "Here is the proof. Write Y minus g as e plus m minus g, and expand the square. The cross term is zero by property three, because m minus g is a function of X. "
 "The last term is a square, so it is at least zero. So the mean squared error is minimized exactly when g equals m.",
 "A corollary is the law of total variance: total variation splits into explained variation plus average conditional variance."]),

dict(kicker="2 · The best linear predictor", title="The linear projection", body="""
<div class="def">\\[ \\beta = \\arg\\min_{b}\\; E\\big[(Y - X'b)^2\\big] \\]</div>
<div class="eq-big" data-s="1">\\[ \\beta = \\big(E[XX']\\big)^{-1} E[XY], \\qquad u = Y - X'\\beta, \\qquad E[Xu]=0 \\]</div>
<ul class="facts">
<li data-s="2">Exists under <b>moment conditions alone</b></li>
<li data-s="3">\\(E[Xu]=0\\) is <b>weaker</b> than \\(E[u\\mid X]=0\\) — they coincide only if the CEF is linear</li>
</ul>""", say=[
 "In practice we approximate the CEF with a linear function. The best linear predictor chooses b to minimize the mean squared error over linear functions, X prime b.",
 "Setting the derivative to zero gives beta equals the inverse of E of X X prime, times E of X Y. The projection error u is then orthogonal to X: the expectation of X times u is zero.",
 "This always exists, as long as the moments are finite and E of X X prime is invertible.",
 "But be careful. Orthogonality, E of X u equals zero, is weaker than mean independence, E of u given X equals zero. The two coincide only when the conditional expectation function is actually linear."]),

dict(kicker="2 · The best linear predictor", title="When the CEF is not linear", body="""
<div class="split"><img src="assets/cef_blp.svg" class="fig"><div class="side">
<p data-s="0">\\(X\\sim U[0,1]\\), \\(\\;Y = X^2 + \\varepsilon\\)</p>
<p data-s="1">\\(\\beta_1=\\dfrac{\\operatorname{Cov}(X,X^2)}{\\operatorname{Var}(X)}=\\dfrac{1/12}{1/12}=1\\)</p>
<p data-s="2">BLP: \\(\\;X - \\tfrac16\\)</p>
<p class="note" data-s="3">\\(u\\) is uncorrelated with \\(X\\), yet \\(E[u\\mid X]=X^2-X+\\tfrac16\\neq0\\).</p>
</div></div>""", say=[
 "Let's see this in a picture. X is uniform on zero one, and Y equals X squared plus noise. The true conditional mean, in blue, is a parabola.",
 "The slope of the best linear predictor is the covariance of X and X squared over the variance of X. Both are one twelfth, so the slope is one.",
 "The intercept works out to minus one sixth, giving the dashed line.",
 "The projection error is uncorrelated with X by construction. But its conditional mean is not zero: it is positive at the edges and negative in the middle. "
 "So regression still estimates something well defined, the best linear approximation to the CEF, even when the CEF is curved."]),

dict(kicker="3 · Models and assumptions", title="Which “regression model”?", body="""
<table class="tbl">
<tr><th>Model</th><th>Assumption</th><th>What it buys</th></tr>
<tr data-s="1"><td>Linear projection</td><td>\\(E[Xu]=0\\)</td><td>OLS consistent</td></tr>
<tr data-s="2"><td>Linear CEF</td><td>\\(E[u\\mid X]=0\\)</td><td>OLS unbiased</td></tr>
<tr data-s="3"><td>+ Homoskedastic</td><td>\\(E[u^2\\mid X]=\\sigma^2\\)</td><td>Gauss–Markov efficiency</td></tr>
<tr data-s="4"><td>+ Normal</td><td>\\(u\\mid X\\sim N(0,\\sigma^2)\\)</td><td>exact \\(t\\) and \\(F\\)</td></tr>
<tr data-s="5" class="warn"><td>Structural</td><td>\\(\\beta\\) = effect of an intervention</td><td>causal meaning — needs identification</td></tr>
</table>""", say=[
 "People say 'the linear regression model' as if there were only one. There are several, and they are nested.",
 "The weakest is the linear projection, where orthogonality holds by construction. That is enough for OLS to be consistent.",
 "If the CEF is linear, so the error is mean independent of X, OLS is also unbiased.",
 "Add homoskedasticity, and OLS is efficient among linear unbiased estimators.",
 "Add normality, and we get exact t and F distributions in finite samples.",
 "The last row is different in kind. A structural or causal interpretation says beta measures the effect of an intervention. "
 "No amount of statistical structure in the rows above delivers that. It requires an identification argument, which is the subject of later lectures."]),

dict(kicker="4 · Ordinary least squares", title="OLS is the sample projection", body="""
<div class="eq-big">\\[ \\widehat\\beta=\\Big(\\tfrac1n\\textstyle\\sum_i X_iX_i'\\Big)^{-1}\\tfrac1n\\textstyle\\sum_i X_iY_i=(X'X)^{-1}X'y \\]</div>
<div class="def" data-s="1">Normal equations: \\(\\;X'(y-X\\widehat\\beta)=0 \\iff X'\\widehat u=0\\)</div>
<ul class="facts">
<li data-s="2">Exists and is unique iff \\(\\operatorname{rank}(X)=k\\) — no perfect multicollinearity</li>
<li data-s="3">A <b>method-of-moments</b> estimator for \\(E[X(Y-X'\\beta)]=0\\) — the template for IV and GMM</li>
</ul>""", say=[
 "Ordinary least squares simply replaces population moments with sample averages. In matrix form, beta hat equals X prime X inverse, times X prime y.",
 "The first order conditions are the normal equations: X prime times the residual vector equals zero. That is the sample version of orthogonality.",
 "The estimator exists and is unique exactly when X has full column rank, so no regressor is a perfect linear combination of the others.",
 "And notice that OLS is a method of moments estimator. That view generalizes directly to instrumental variables and to GMM later in the course."]),

dict(kicker="5 · Geometry", title="Projection matrices", body="""
<div class="split"><img src="assets/proj.svg" class="fig"><div class="side">
<p data-s="0">\\(P_X = X(X'X)^{-1}X'\\), \\(\\;M_X = I - P_X\\)</p>
<p data-s="1">Symmetric and idempotent: \\(P_X^2=P_X\\)</p>
<p data-s="2">\\(P_XX=X,\\;\\; M_XX=0,\\;\\; P_XM_X=0\\)</p>
<p data-s="3">\\(\\operatorname{tr}(P_X)=k,\\;\\; \\operatorname{tr}(M_X)=n-k\\)</p>
</div></div>""", say=[
 "Now the geometry. P X projects any vector onto the column space of X, and M X, the residual maker, projects onto its orthogonal complement. "
 "Fitted values are P X times y, and residuals are M X times y, and the two are at right angles.",
 "Both matrices are symmetric and idempotent: projecting twice is the same as projecting once.",
 "P X leaves X unchanged, M X annihilates X, and the two projections are orthogonal to each other.",
 "Finally, the trace of P X is k and the trace of M X is n minus k. That trace fact is where the n minus k degrees of freedom correction comes from."]),

dict(kicker="6 · Partitioned regression", title="Frisch–Waugh–Lovell", body="""
<div class="thm"><b>Theorem 1.5.</b> In the regression of \\(y\\) on \\([X_1\\; X_2]\\),
\\[ \\widehat\\beta_2 = (X_2'M_1X_2)^{-1}X_2'M_1y . \\]</div>
<div class="proof" data-s="1">Premultiply \\(y = X_1\\widehat\\beta_1 + X_2\\widehat\\beta_2 + \\widehat u\\) by \\(M_1\\):
\\[ M_1y = M_1X_2\\,\\widehat\\beta_2 + \\widehat u . \\]</div>
<p class="note" data-s="2">“Holding \\(X_1\\) constant” = using only the part of \\(X_2\\) not linearly explained by \\(X_1\\).</p>""", say=[
 "The Frisch Waugh Lovell theorem says: the coefficient on X two in the long regression equals the coefficient from regressing the residualized y on the residualized X two, after partialling out X one.",
 "The proof: multiply the fitted regression by M one. M one kills X one, and leaves the residual unchanged because the residual is already orthogonal to X one. "
 "What's left is a regression of M one y on M one X two.",
 "This makes 'holding other variables constant' precise. A multiple regression coefficient uses only the variation in X two that X one cannot explain linearly. "
 "Demeaning, detrending, and the fixed effects estimator are all special cases."]),

dict(kicker="Worked example", title="Returns to schooling in two steps", body="""
<div class="eq-big">\\[ \\log w = \\beta_0 + \\beta_1\\,educ + \\beta_2\\,exper + u \\]</div>
<ol class="big">
<li data-s="1">Regress \\(educ\\) on \\((1,\\,exper)\\); keep the residual \\(\\tilde r\\)</li>
<li data-s="2">Slope of \\(\\log w\\) on \\(\\tilde r\\): \\(\\;\\widehat\\beta_1=\\sum_i\\tilde r_i\\log w_i\\,/\\,\\sum_i\\tilde r_i^2\\)</li>
</ol>
<p class="note" data-s="3">Identical to the long regression. Causal only if \\(E[u\\mid educ, exper]=0\\) is credible — ability is the classic worry.</p>""", say=[
 "Here is a worked example. Log wages on education and experience.",
 "Step one: regress education on a constant and experience, and keep the residual. That is the part of schooling not predicted by experience.",
 "Step two: regress log wages on that residual. The slope is exactly the long regression coefficient on education.",
 "In the lab you will verify that the two routes match to machine precision. But whether this number is the return to schooling depends on whether the error is mean independent of schooling. "
 "Unobserved ability is the classic reason to doubt it, and lecture three derives the size of that bias."]),

dict(kicker="7 · Finite-sample properties", title="Unbiasedness and Gauss–Markov", body="""
<div class="eq-big">\\[ \\widehat\\beta-\\beta=(X'X)^{-1}X'u \\]</div>
<div class="thm" data-s="1"><b>Theorem 1.6.</b> If \\(E[u\\mid X]=0\\): \\(E[\\widehat\\beta\\mid X]=\\beta\\). If also \\(E[uu'\\mid X]=\\sigma^2 I\\): \\(\\operatorname{Var}(\\widehat\\beta\\mid X)=\\sigma^2(X'X)^{-1}\\).</div>
<div class="thm" data-s="2"><b>Theorem 1.7 (Gauss–Markov).</b> Under these assumptions OLS is <b>BLUE</b>: for any linear unbiased \\(\\tilde\\beta=Ay\\), \\(\\operatorname{Var}(\\tilde\\beta\\mid X)-\\operatorname{Var}(\\widehat\\beta\\mid X)\\succeq 0\\).</div>
<p class="note" data-s="3">No normality needed. Homoskedasticity is needed. And it only compares <b>linear unbiased</b> estimators.</p>""", say=[
 "Every finite sample property of OLS comes from this one expression: beta hat minus beta equals X prime X inverse, X prime u.",
 "If the error is mean independent of X, take conditional expectations, and OLS is unbiased. If the errors are also homoskedastic and uncorrelated, the variance is sigma squared times X prime X inverse.",
 "The Gauss Markov theorem says that, under these assumptions, OLS has the smallest variance among all linear unbiased estimators.",
 "Three warnings. Gauss Markov does not need normality. It does need homoskedasticity. And it only compares linear unbiased estimators. "
 "A biased estimator, such as ridge regression, can have lower mean squared error."]),

dict(kicker="Summary", title="Key takeaways", body="""
<ul class="facts big">
<li data-s="0">Regression always estimates a well-defined object: the <b>linear projection</b></li>
<li data-s="1">Consistency needs \\(E[Xu]=0\\); unbiasedness needs \\(E[u\\mid X]=0\\)</li>
<li data-s="2">FWL: a coefficient uses only the <b>residual variation</b> in its regressor</li>
<li data-s="3">Causal interpretation requires <b>identification</b>, not more statistics</li>
</ul>
<div class="next" data-s="4">Next · Lecture 2 — OLS inference: asymptotics, tests and confidence intervals</div>""", say=[
 "Let's summarize. A regression always estimates something well defined: the best linear approximation to the conditional expectation function.",
 "Consistency needs only orthogonality. Unbiasedness needs mean independence.",
 "By Frisch Waugh Lovell, each coefficient uses only the residual variation in its own regressor.",
 "And a causal interpretation comes from an identification argument, not from adding more statistical assumptions.",
 "Next time, in lecture two, we turn to inference: asymptotic theory, hypothesis tests, and confidence intervals. "
 "The full notes, problem set, and lab for this lecture are on the course website. Thanks for watching."]),
]
OUTNAME = "econometrics-lecture-01"
