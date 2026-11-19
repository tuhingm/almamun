"""Registry of the Statistics Division: 31 methods in seven categories, with their input fields and the example each one is
tested on. __init__.py calls build() with the field helpers, so the models are registered like every other OES model.
Each entry carries the category (group), a code (A1 ... G2), the guide page and a documented public example dataset."""
from . import stat_anova, stat_multivariate, stat_nonparam

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
    return M
