"""Registry of the Statistics Division: 31 methods in seven categories, with their input fields and the example each one is
tested on. __init__.py calls build() with the field helpers, so the models are registered like every other OES model.
Each entry carries the category (group), a code (A1 ... G2), the guide page and a documented public example dataset."""
from . import stat_anova, stat_meta, stat_mixed, stat_multivariate, stat_nonparam, stat_resample, stat_survival

SX = "stat-guide/examples/"
G_ANOVA = "Analysis of variance"
G_NONPAR = "Nonparametric statistics"
G_MULTI = "Multivariate analysis"
G_SURV = "Survival analysis"
G_MIXED = "Mixed-effects and longitudinal"
G_RESAMP = "Resampling and simulation"
G_META = "Meta-analysis"
STAT_GROUPS = [G_ANOVA, G_NONPAR, G_MULTI, G_SURV, G_MIXED, G_RESAMP, G_META]


def build(col, cols, num, sel):
    ALPHA = num("alpha", "Significance level", 0.05)
    ALT = sel("alternative", "Alternative hypothesis", [["two-sided", "Two-sided"], ["greater", "Greater"], ["less", "Less"]], "two-sided")
    EXACT = sel("method", "p-value method", [["auto", "Automatic (exact for small samples without ties)"], ["exact", "Exact"], ["asymptotic", "Normal approximation"]], "auto")
    CONT = sel("continuity", "Continuity correction", [["yes", "Yes"], ["no", "No"]], "yes")
    ADJ = sel("adjust", "Multiple-comparison adjustment", [["holm", "Holm"], ["bonferroni", "Bonferroni"], ["bh", "Benjamini-Hochberg"], ["none", "None"]], "holm")
    SS = sel("ss_type", "Sums of squares", [["III", "Type III (each term adjusted for all others)"], ["II", "Type II (adjusted for terms not containing it)"]], "III")
    M = []

    def add(**kw):
        kw.setdefault("division", "statistics")
        M.append(kw)

    add(id="stat_anova1", code="A1", group=G_ANOVA, name="One-way ANOVA", guide="stat-guide/anova-one-way.html", fn=stat_anova.anova1,
        fields=[col("y", "Outcome variable"), col("group", "Grouping variable"),
                sel("posthoc", "Pairwise comparisons", [["tukey", "Tukey HSD"], ["games_howell", "Games-Howell (unequal variances)"], ["none", "None"]], "tukey"), ALPHA],
        example=dict(file=SX + "plantgrowth.csv", spec=dict(y="weight", group="group")))
    add(id="stat_anova2", code="A2", group=G_ANOVA, name="Two-way ANOVA", guide="stat-guide/anova-two-way.html", fn=stat_anova.anova2,
        fields=[col("y", "Outcome variable"), col("factor_a", "First factor"), col("factor_b", "Second factor"), SS,
                sel("interaction", "Interaction term", [["yes", "Include"], ["no", "Omit"]], "yes"), ALPHA],
        example=dict(file=SX + "toothgrowth.csv", spec=dict(y="len", factor_a="supp", factor_b="dose", ss_type="III")))
    add(id="stat_rm_anova", code="A3", group=G_ANOVA, name="Repeated-measures ANOVA", guide="stat-guide/anova-repeated-measures.html", fn=stat_anova.rm_anova,
        fields=[col("subject", "Subject identifier"), col("within", "Within-subject factor (condition or time)"), col("y", "Outcome variable (one row per subject and condition)"),
                sel("correction", "Sphericity correction", [["auto", "Automatic (Mauchly test)"], ["gg", "Greenhouse-Geisser"], ["hf", "Huynh-Feldt"], ["none", "None"]], "auto"), ALPHA],
        example=dict(file=SX + "sleepstudy.csv", spec=dict(subject="Subject", within="Days", y="Reaction")))
    add(id="stat_ancova", code="A4", group=G_ANOVA, name="Analysis of covariance (ANCOVA)", guide="stat-guide/ancova.html", fn=stat_anova.ancova,
        fields=[col("y", "Outcome variable"), col("group", "Grouping variable"), cols("covariates", "Covariates"), SS, ALPHA],
        example=dict(file=SX + "anorexia.csv", spec=dict(y="Postwt", group="Treat", covariates=["Prewt"])))
    add(id="stat_manova", code="A5", group=G_ANOVA, name="Multivariate ANOVA (MANOVA)", guide="stat-guide/manova.html", fn=stat_anova.manova,
        fields=[cols("y_vars", "Response variables (two or more)"), col("group", "Grouping variable"), ALPHA],
        example=dict(file=SX + "iris.csv", spec=dict(y_vars=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], group="Species")))
    add(id="stat_mancova", code="A6", group=G_ANOVA, name="Multivariate ANCOVA (MANCOVA)", guide="stat-guide/mancova.html", fn=stat_anova.mancova,
        fields=[cols("y_vars", "Response variables (two or more)"), col("group", "Grouping variable"), cols("covariates", "Covariates"), ALPHA],
        example=dict(file=SX + "rohwer.csv", spec=dict(y_vars=["SAT", "PPVT", "Raven"], group="group", covariates=["n", "s", "ns"])))
    add(id="stat_mann_whitney", code="B1", group=G_NONPAR, name="Mann-Whitney U test", guide="stat-guide/mann-whitney.html", fn=stat_nonparam.mann_whitney,
        fields=[col("y", "Outcome variable"), col("group", "Grouping variable (exactly two groups)"), ALT, EXACT, CONT, ALPHA],
        example=dict(file=SX + "toothgrowth.csv", spec=dict(y="len", group="supp")))
    add(id="stat_wilcoxon", code="B2", group=G_NONPAR, name="Wilcoxon signed-rank test", guide="stat-guide/wilcoxon.html", fn=stat_nonparam.wilcoxon,
        fields=[col("x1", "First measurement (or the sample for a one-sample test)"), col("x2", "Second measurement (blank = one-sample test)", required=False), num("mu0", "Hypothesised median (one-sample test)", 0), ALT, EXACT, CONT,
                sel("zero_method", "Zero differences", [["wilcox", "Drop them (Wilcoxon)"], ["pratt", "Rank them, then drop (Pratt)"], ["zsplit", "Split their ranks between both signs"]], "wilcox"), ALPHA],
        example=dict(file=SX + "sleep_paired.csv", spec=dict(x1="drug1", x2="drug2")))
    add(id="stat_kruskal", code="B3", group=G_NONPAR, name="Kruskal-Wallis test", guide="stat-guide/kruskal-wallis.html", fn=stat_nonparam.kruskal,
        fields=[col("y", "Outcome variable"), col("group", "Grouping variable (three or more groups)"), ADJ, ALPHA],
        example=dict(file=SX + "airquality_ozone.csv", spec=dict(y="Ozone", group="Month")))
    add(id="stat_friedman", code="B4", group=G_NONPAR, name="Friedman test", guide="stat-guide/friedman.html", fn=stat_nonparam.friedman,
        fields=[col("subject", "Subject (block) identifier"), col("treatment", "Treatment or condition"), col("y", "Outcome variable (one row per subject and condition)"), ADJ, ALPHA],
        example=dict(file=SX + "roundingtimes.csv", spec=dict(subject="player", treatment="method", y="time")))
    add(id="stat_kendall", code="B5", group=G_NONPAR, name="Kendall's tau correlation", guide="stat-guide/kendall-tau.html", fn=stat_nonparam.kendall,
        fields=[col("x", "First variable"), col("y", "Second variable"), ALT, EXACT, ALPHA],
        example=dict(file=SX + "mtcars.csv", spec=dict(x="wt", y="mpg")))
    add(id="stat_pca", code="C1", group=G_MULTI, name="Principal component analysis (PCA)", guide="stat-guide/pca.html", fn=stat_multivariate.pca,
        fields=[cols("vars", "Variables"), sel("basis", "Analyse the", [["correlation", "Correlation matrix (standardised variables)"], ["covariance", "Covariance matrix"]], "correlation"),
                num("n_components", "Components to report (blank = parallel analysis)", ""), num("seed", "Random seed (parallel analysis)", 0)],
        example=dict(file=SX + "usarrests.csv", spec=dict(vars=["Murder", "Assault", "UrbanPop", "Rape"])))
    add(id="stat_efa", code="C2", group=G_MULTI, name="Exploratory factor analysis (EFA)", guide="stat-guide/efa.html", fn=stat_multivariate.efa,
        fields=[cols("vars", "Variables (at least three)"), num("n_factors", "Number of factors", 2),
                sel("rotation", "Rotation", [["varimax", "Varimax (orthogonal)"], ["promax", "Promax (oblique)"], ["oblimin", "Oblimin (oblique)"], ["none", "None"]], "varimax"), num("seed", "Random seed (parallel analysis)", 0)],
        example=dict(file=SX + "bfi_items.csv", spec=dict(vars=["N1", "N2", "N3", "N4", "N5", "E1", "E2", "E3", "E4", "E5"], n_factors=2, rotation="varimax")))
    add(id="stat_cfa", code="C3", group=G_MULTI, name="Confirmatory factor analysis (CFA)", guide="stat-guide/cfa.html", fn=stat_multivariate.cfa,
        fields=[dict(key="model", label="Measurement model (one factor per line: factor =~ item1 + item2 + item3)", type="text", required=True, default="visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9", help=""),
                sel("identification", "Scale of each factor", [["marker", "First loading fixed to 1"], ["std", "Factor variance fixed to 1"]], "marker"),
                sel("correlated", "Factor correlations", [["yes", "Free"], ["no", "Fixed at zero"]], "yes"), ALPHA],
        example=dict(file=SX + "holzinger_swineford.csv", spec=dict(model="visual =~ x1 + x2 + x3\ntextual =~ x4 + x5 + x6\nspeed =~ x7 + x8 + x9")))
    add(id="stat_lda", code="C4", group=G_MULTI, name="Linear discriminant analysis (LDA)", guide="stat-guide/lda.html", fn=stat_multivariate.lda,
        fields=[cols("vars", "Predictors"), col("group", "Class variable"), sel("priors", "Prior class probabilities", [["proportional", "Class proportions"], ["equal", "Equal"]], "proportional")],
        example=dict(file=SX + "iris.csv", spec=dict(vars=["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"], group="Species")))
    add(id="stat_cca", code="C5", group=G_MULTI, name="Canonical correlation analysis (CCA)", guide="stat-guide/cca.html", fn=stat_multivariate.cca,
        fields=[cols("x_vars", "First variable set"), cols("y_vars", "Second variable set"), num("perms", "Permutations for the permutation test (0 = skip)", 999), num("seed", "Random seed", 0)],
        example=dict(file=SX + "linnerud.csv", spec=dict(x_vars=["Chins", "Situps", "Jumps"], y_vars=["Weight", "Waist", "Pulse"], perms=499)))
    add(id="stat_ca", code="C6", group=G_MULTI, name="Correspondence analysis (CA)", guide="stat-guide/correspondence-analysis.html", fn=stat_multivariate.ca,
        fields=[col("row_var", "Row variable (a category, or the row label of a count table)"), col("col_var", "Column variable (one row per case; leave blank for a count table)", required=False), cols("counts", "Count columns (for a table that is already cross-tabulated)", required=False)],
        example=dict(file=SX + "caith.csv", spec=dict(row_var="eye", counts=["fair", "red", "medium", "dark", "black"])))
    TIME = col("time", "Follow-up time")
    EVENT = col("event", "Event indicator (1 = event, 0 = censored)")
    add(id="stat_km", code="D1", group=G_SURV, name="Kaplan-Meier estimator and log-rank test", guide="stat-guide/kaplan-meier.html", fn=stat_survival.kaplan_meier,
        fields=[TIME, EVENT, col("group", "Grouping variable (optional)", required=False),
                sel("test", "Test across groups", [["logrank", "Log-rank"], ["gehan", "Gehan-Breslow (Wilcoxon)"], ["peto", "Peto-Peto"]], "logrank"),
                sel("ci_type", "Confidence interval", [["loglog", "Log-log"], ["log", "Log"], ["plain", "Plain"]], "loglog"), ALPHA],
        example=dict(file=SX + "veteran.csv", spec=dict(time="time", event="status", group="trt")))
    add(id="stat_cox", code="D2", group=G_SURV, name="Cox proportional hazards model", guide="stat-guide/cox-ph.html", fn=stat_survival.cox_ph,
        fields=[TIME, EVENT, cols("covariates", "Numeric covariates", required=False), cols("factors", "Categorical covariates (dummy-coded)", required=False),
                sel("ties", "Handling of tied event times", [["efron", "Efron"], ["breslow", "Breslow"]], "efron"), ALPHA],
        example=dict(file=SX + "rossi.csv", spec=dict(time="week", event="arrest", covariates=["fin", "age", "race", "wexp", "mar", "paro", "prio"])))
    add(id="stat_aft", code="D3", group=G_SURV, name="Accelerated failure time model", guide="stat-guide/aft.html", fn=stat_survival.aft,
        fields=[TIME, EVENT, cols("covariates", "Numeric covariates", required=False), cols("factors", "Categorical covariates (dummy-coded)", required=False),
                sel("distribution", "Error distribution", [["weibull", "Weibull"], ["lognormal", "Log-normal"], ["loglogistic", "Log-logistic"], ["exponential", "Exponential"]], "weibull"), ALPHA],
        example=dict(file=SX + "rossi.csv", spec=dict(time="week", event="arrest", covariates=["fin", "age", "prio"], distribution="weibull")))
    add(id="stat_finegray", code="D4", group=G_SURV, name="Fine-Gray competing-risks regression", guide="stat-guide/fine-gray.html", fn=stat_survival.fine_gray,
        fields=[TIME, col("event", "Status (0 = censored, 1, 2, ... = event types)"), num("cause", "Event code of interest", 1), cols("covariates", "Numeric covariates", required=False), cols("factors", "Categorical covariates (dummy-coded)", required=False), ALPHA],
        example=dict(file=SX + "melanoma.csv", spec=dict(time="time", event="status", cause=1, covariates=["age", "thickness", "ulcer", "sex"])))
    add(id="stat_parametric", code="D5", group=G_SURV, name="Parametric survival (Weibull, exponential, log-normal)", guide="stat-guide/parametric-survival.html", fn=stat_survival.parametric,
        fields=[TIME, EVENT, cols("covariates", "Numeric covariates (optional)", required=False), cols("factors", "Categorical covariates (optional)", required=False), ALPHA],
        example=dict(file=SX + "veteran.csv", spec=dict(time="time", event="status", covariates=["karno"])))
    COVS = cols("covariates", "Numeric covariates (fixed effects)", required=False)
    FACS = cols("factors", "Categorical covariates (dummy-coded)", required=False)
    add(id="stat_lmm", code="E1", group=G_MIXED, name="Linear mixed-effects model (LMM)", guide="stat-guide/lmm.html", fn=stat_mixed.lmm,
        fields=[col("y", "Outcome variable"), col("group", "Grouping variable (cluster or subject)"), COVS, FACS,
                col("slope", "Random-slope variable (optional, numeric)", required=False),
                sel("correlated", "Intercept-slope correlation", [["yes", "Estimated"], ["no", "Fixed at zero"]], "yes"),
                sel("method", "Estimation", [["reml", "REML"], ["ml", "Maximum likelihood"]], "reml"), ALPHA],
        example=dict(file=SX + "sleepstudy.csv", spec=dict(y="Reaction", group="Subject", covariates=["Days"], slope="Days", method="reml")))
    add(id="stat_glmm", code="E2", group=G_MIXED, name="Generalised linear mixed model (GLMM)", guide="stat-guide/glmm.html", fn=stat_mixed.glmm,
        fields=[col("y", "Outcome (successes, 0/1, or a count)"), col("trials", "Number of trials (binomial counts only; blank for 0/1)", required=False), col("group", "Grouping variable (cluster)"), COVS, FACS,
                sel("family", "Family", [["binomial", "Binomial (logit link)"], ["poisson", "Poisson (log link)"]], "binomial"),
                num("nodes", "Quadrature points (1 = Laplace)", 15), ALPHA],
        example=dict(file=SX + "cbpp.csv", spec=dict(y="incidence", trials="size", group="herd", factors=["period"], family="binomial", nodes=15)))
    add(id="stat_gee", code="E3", group=G_MIXED, name="Generalised estimating equations (GEE)", guide="stat-guide/gee.html", fn=stat_mixed.gee,
        fields=[col("y", "Outcome variable"), col("group", "Cluster (subject) identifier"), col("time", "Time or visit order within cluster (optional)", required=False), COVS, FACS,
                sel("family", "Family", [["gaussian", "Gaussian (identity)"], ["binomial", "Binomial (logit, 0/1 outcome)"], ["poisson", "Poisson (log)"]], "gaussian"),
                sel("corr", "Working correlation", [["exchangeable", "Exchangeable"], ["ar1", "AR(1)"], ["independence", "Independence"]], "exchangeable"),
                num("fixed_alpha", "Fix the correlation parameter (blank = estimate)", ""), ALPHA],
        example=dict(file=SX + "orthodont.csv", spec=dict(y="distance", group="Subject", time="age", covariates=["age", "Sex01"], family="gaussian", corr="exchangeable")))
    add(id="stat_icc", code="E4", group=G_MIXED, name="Intraclass correlation (ICC) and reliability", guide="stat-guide/icc.html", fn=stat_mixed.icc,
        fields=[cols("raters", "Rater / measurement columns (one row per subject)"), ALPHA],
        example=dict(file=SX + "shrout_fleiss.csv", spec=dict(raters=["judge1", "judge2", "judge3", "judge4"])))
    SEED = num("seed", "Random seed", 12345)
    add(id="stat_bootstrap", code="F1", group=G_RESAMP, name="Bootstrap confidence intervals", guide="stat-guide/bootstrap.html", fn=stat_resample.bootstrap,
        fields=[col("y", "Variable (outcome)"), col("x", "Second variable (correlation, slope)", required=False), col("group", "Grouping variable with two levels (difference statistics)", required=False),
                sel("statistic", "Statistic", [[k, v] for k, v in stat_resample._BOOT_STATS.items()], "mean"), num("n_boot", "Bootstrap resamples", 2000), SEED, ALPHA],
        example=dict(file=SX + "toothgrowth.csv", spec=dict(statistic="median", y="len", n_boot=4000, seed=12345)))
    add(id="stat_permutation", code="F2", group=G_RESAMP, name="Permutation tests", guide="stat-guide/permutation.html", fn=stat_resample.permutation,
        fields=[col("y", "Outcome variable"), col("group", "Grouping variable (group tests)", required=False), col("x", "Second variable (paired difference, correlation)", required=False),
                sel("test", "Test", [[k, v] for k, v in stat_resample._PERM_TESTS.items()], "mean_diff"), ALT, num("n_perm", "Random permutations (when not exact)", 10000),
                num("exact_max", "Enumerate exactly up to this many rearrangements", 200000), SEED, ALPHA],
        example=dict(file=SX + "toothgrowth.csv", spec=dict(test="mean_diff", y="len", group="supp", n_perm=20000, seed=12345)))
    add(id="stat_power", code="F3", group=G_RESAMP, name="Power and sample-size analysis", guide="stat-guide/power.html", fn=stat_resample.power_analysis, nodata=True,
        fields=[sel("test", "Test", [[k, v[0]] for k, v in stat_resample._POWER_TESTS.items()], "t_two"),
                sel("solve_for", "Solve for", [["n", "Sample size"], ["power", "Power"], ["effect", "Smallest detectable effect size"]], "n"),
                num("effect", "Effect size (d, f, r, h or w, depending on the test)", 0.5), num("n", "Sample size (per group, or total for chi-square)", 30), num("power", "Target power", 0.8), ALPHA, ALT,
                num("k", "Number of groups (ANOVA)", 3), num("df", "Degrees of freedom (chi-square)", 2), num("ratio", "Allocation ratio, second group / first", 1),
                num("p1", "Proportion in group 1 (two proportions; replaces the effect size)", ""), num("p2", "Proportion in group 2", "")],
        example=dict(file=None, spec=dict(test="t_two", solve_for="n", effect=0.5, power=0.8, alpha=0.05)))
    META_IN = sel("input", "Type of study data", [["binary", "Event counts in two groups"], ["means", "Means and SDs in two groups"], ["generic", "Effect size with standard error or variance"]], "generic")
    META_FIELDS = [META_IN, col("study", "Study label", required=False),
                   col("yi", "Effect size (generic input)", required=False), col("sei", "Standard error (generic input)", required=False), col("vi", "Variance (generic input, if no standard error)", required=False),
                   col("ev1", "Events, treatment group (counts input)", required=False), col("n1", "Group size, treatment (counts or means input)", required=False),
                   col("ev2", "Events, control group (counts input)", required=False), col("n2", "Group size, control (counts or means input)", required=False),
                   sel("measure", "Effect measure for counts", [["lrr", "Log risk ratio"], ["lor", "Log odds ratio"], ["rd", "Risk difference"]], "lrr"),
                   col("m1", "Mean, treatment (means input)", required=False), col("sd1", "SD, treatment (means input)", required=False), col("m2", "Mean, control (means input)", required=False), col("sd2", "SD, control (means input)", required=False)]
    add(id="stat_meta_fixed", code="G1", group=G_META, name="Fixed-effect meta-analysis", guide="stat-guide/meta-fixed.html", fn=stat_meta.meta_fixed, fields=META_FIELDS + [ALPHA],
        example=dict(file=SX + "bcg_vaccine.csv", spec=dict(input="binary", study="author", ev1="tpos", n1="ttotal", ev2="cpos", n2="ctotal", measure="lrr")))
    add(id="stat_meta_random", code="G2", group=G_META, name="Random-effects meta-analysis", guide="stat-guide/meta-random.html", fn=stat_meta.meta_random,
        fields=META_FIELDS + [sel("tau_method", "Between-study variance estimator", [["reml", "REML"], ["dl", "DerSimonian-Laird"], ["pm", "Paule-Mandel"], ["ml", "Maximum likelihood"], ["he", "Hedges"]], "reml"),
                              sel("ci_method", "Confidence interval", [["wald", "Wald (z)"], ["hksj", "Hartung-Knapp-Sidik-Jonkman"], ["hksj_mod", "Hartung-Knapp, never narrower than Wald"]], "wald"), ALPHA],
        example=dict(file=SX + "bcg_vaccine.csv", spec=dict(input="binary", study="author", ev1="tpos", n1="ttotal", ev2="cpos", n2="ctotal", measure="lrr", tau_method="reml", ci_method="hksj")))
    return M
