"""
Dynamic panel lab: pooled OLS, fixed effects and Anderson-Hsiao IV.

Data: Wooldridge wagepan (545 workers, 1980-1987).

Install:
    python -m pip install wooldridge pandas numpy statsmodels linearmodels

Run:
    python dynamic_panel_models.py

The IV regression instruments Delta(y_{i,t-1}) with y_{i,t-2}.
This is Anderson-Hsiao IV, not Arellano-Bond or System GMM.
"""
import numpy as np
import pandas as pd
import wooldridge as wd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS
from linearmodels.panel import PanelOLS


def main():
    df = wd.data("wagepan").sort_values(["nr", "year"]).copy()
    group = df.groupby("nr")
    df["y_l1"] = group["lwage"].shift(1)
    df["y_l2"] = group["lwage"].shift(2)
    df["y_l3"] = group["lwage"].shift(3)
    df["dy"] = df["lwage"] - df["y_l1"]
    df["dyl1"] = df["y_l1"] - df["y_l2"]
    df["dunion"] = df["union"] - group["union"].shift(1)

    pooled_sample = df.dropna(subset=["y_l1"]).copy()
    pooled = smf.ols(
        "lwage ~ y_l1 + union + C(year)", data=pooled_sample
    ).fit(cov_type="cluster", cov_kwds={"groups": pooled_sample["nr"]})
    fixed = PanelOLS.from_formula(
        "lwage ~ y_l1 + union + TimeEffects + EntityEffects",
        data=pooled_sample.set_index(["nr", "year"]),
    ).fit(cov_type="clustered", cluster_entity=True)

    iv_sample = df.dropna(subset=["dy", "dyl1", "y_l2", "dunion"]).copy()
    ah = IV2SLS.from_formula(
        "dy ~ 1 + dunion + C(year) + [dyl1 ~ y_l2]", data=iv_sample
    ).fit(
        cov_type="clustered",
        clusters=iv_sample["nr"].astype("category").cat.codes,
    )
    first_stage = smf.ols(
        "dyl1 ~ y_l2 + dunion + C(year)", data=iv_sample
    ).fit(cov_type="cluster", cov_kwds={"groups": iv_sample["nr"]})

    print("DYNAMIC WAGE EQUATION")
    print(f"Pooled OLS lag coefficient: {pooled.params['y_l1']:.4f}")
    print(f"Fixed-effects lag coefficient: {fixed.params['y_l1']:.4f}")
    print(f"Anderson-Hsiao IV coefficient: {ah.params['dyl1']:.4f}")
    print(f"Anderson-Hsiao clustered SE: {ah.std_errors['dyl1']:.4f}")
    print(f"First-stage clustered t-squared: {first_stage.tvalues['y_l2'] ** 2:.2f}")

    residuals = pd.DataFrame(
        {
            "id": iv_sample["nr"].values,
            "year": iv_sample["year"].values,
            "resid": np.asarray(ah.resids).reshape(-1),
        }
    ).sort_values(["id", "year"])
    residuals["resid_l2"] = residuals.groupby("id")["resid"].shift(2)
    ar2_sample = residuals.dropna(subset=["resid_l2"])
    ar2 = smf.ols("resid ~ resid_l2", data=ar2_sample).fit(
        cov_type="cluster", cov_kwds={"groups": ar2_sample["id"]}
    )
    print(
        "Descriptive lag-2 residual correlation: "
        f"{ar2_sample['resid'].corr(ar2_sample['resid_l2']):.4f}"
    )
    print(f"Clustered t-statistic on lag-2 residual: {ar2.tvalues['resid_l2']:.3f}")
    print(
        "\nInterpretation: pooled OLS and within FE are biased in opposite "
        "directions in a short dynamic panel. Anderson-Hsiao uses a lagged "
        "level as an instrument for the differenced lagged outcome. For formal "
        "Arellano-Bond diagnostics and GMM estimation, use a dedicated "
        "implementation such as xtabond2 or pydynpd."
    )


if __name__ == "__main__":
    main()
