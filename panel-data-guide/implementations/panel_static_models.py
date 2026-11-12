"""
Panel-data lab: pooled OLS, random effects, fixed effects, first differences,
and the Mundlak correlated-random-effects test.

Data: Wooldridge wagepan (545 workers, 1980-1987) and crime4
(90 North Carolina counties, 1981-1987).

Install:
    python -m pip install wooldridge pandas statsmodels linearmodels

Run:
    python panel_static_models.py
"""
import pandas as pd
import wooldridge as wd
import statsmodels.formula.api as smf
from linearmodels.panel import PooledOLS, RandomEffects, PanelOLS, FirstDifferenceOLS


def fit_wagepan():
    raw = wd.data("wagepan").copy().sort_values(["nr", "year"])
    panel = raw.set_index(["nr", "year"])
    year_dummies = " + ".join(f"d8{i}" for i in range(1, 8))
    clustered = {"cov_type": "clustered", "cluster_entity": True}

    pooled = PooledOLS.from_formula(
        f"lwage ~ 1 + union + married + expersq + educ + black + hisp + {year_dummies}",
        data=panel,
    ).fit(**clustered)
    random = RandomEffects.from_formula(
        f"lwage ~ 1 + union + married + expersq + educ + black + hisp + {year_dummies}",
        data=panel,
    ).fit(**clustered)
    fixed = PanelOLS.from_formula(
        "lwage ~ union + married + expersq + TimeEffects + EntityEffects",
        data=panel,
    ).fit(**clustered)
    first_difference = FirstDifferenceOLS.from_formula(
        f"lwage ~ union + married + expersq + {year_dummies}", data=panel
    ).fit(**clustered)

    table = pd.DataFrame(
        {
            "Pooled OLS": [pooled.params["union"], pooled.std_errors["union"]],
            "Random effects": [random.params["union"], random.std_errors["union"]],
            "Fixed effects": [fixed.params["union"], fixed.std_errors["union"]],
            "First differences": [
                first_difference.params["union"],
                first_difference.std_errors["union"],
            ],
        },
        index=["Union coefficient", "Entity-clustered SE"],
    )
    print("\nWAGEPAN: UNION WAGE PREMIUM")
    print(table.round(4).to_string())

    for variable in ("union", "married", "expersq"):
        raw[variable + "_bar"] = raw.groupby("nr")[variable].transform("mean")
    mundlak = smf.ols(
        f"lwage ~ union + married + expersq + union_bar + married_bar + "
        f"expersq_bar + educ + black + hisp + {year_dummies}",
        data=raw,
    ).fit(cov_type="cluster", cov_kwds={"groups": raw["nr"]})
    test = mundlak.wald_test(
        "union_bar = 0, married_bar = 0, expersq_bar = 0", scalar=True
    )
    print("\nMUNDLAK CORRELATED RANDOM-EFFECTS TEST")
    print(f"Within-equivalent union coefficient: {mundlak.params['union']:.4f}")
    print(
        f"Joint test of unit-mean coefficients: chi2(3) = "
        f"{float(test.statistic):.2f}, p = {float(test.pvalue):.4g}"
    )


def fit_crime4():
    panel = wd.data("crime4").set_index(["county", "year"])
    regressors = "lprbarr + lprbconv + lprbpris + lavgsen + lpolpc"
    year_dummies = " + ".join(f"d8{i}" for i in range(2, 8))
    clustered = {"cov_type": "clustered", "cluster_entity": True}
    pooled = PooledOLS.from_formula(
        f"lcrmrte ~ 1 + {regressors} + {year_dummies}", data=panel
    ).fit(**clustered)
    fixed = PanelOLS.from_formula(
        f"lcrmrte ~ {regressors} + EntityEffects + TimeEffects", data=panel
    ).fit(**clustered)
    names = regressors.split(" + ")
    table = pd.DataFrame(
        {
            "Pooled OLS": pooled.params[names],
            "Pooled SE": pooled.std_errors[names],
            "Two-way FE": fixed.params[names],
            "FE SE": fixed.std_errors[names],
        }
    )
    print("\nCRIME4: POOLED OLS VERSUS COUNTY AND YEAR FIXED EFFECTS")
    print(table.round(3).to_string())
    print(f"\nWithin R-squared: {fixed.rsquared_within:.3f}")
    print(f"Number of counties: {panel.index.get_level_values(0).nunique()}")


if __name__ == "__main__":
    fit_wagepan()
    fit_crime4()
