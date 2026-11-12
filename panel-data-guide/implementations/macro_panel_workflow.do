/**********************************************************************
 MACRO PANEL IMPLEMENTATION: CROSS-SECTIONAL DEPENDENCE, CIPS,
 WESTERLUND COINTEGRATION, PMG AND CS-ARDL / CCE

 Map the variable names in the local macros to your dataset. This file
 is a Stata workflow template and is executable after that mapping.

 User-written packages:
   ssc install xtdcce2
   ssc install xtcips
   ssc install xtwest
**********************************************************************/

clear all
set more off

* Load your dataset before running:
* use "my_macro_panel.dta", clear

* ---- Map the dataset variables here ----
local id      country_id
local time    year
local y       ln_gdp
local x1      ln_energy
local x2      ln_co2
local x3      ln_renewables

xtset \`id' \`time'
xtdescribe
xtsum \`y' \`x1' \`x2' \`x3'

* 1. Static pooled / FE / RE and correlated random effects
xtreg \`y' \`x1' \`x2' \`x3' i.\`time', fe vce(cluster \`id')
estimates store FE_tw
xtreg \`y' \`x1' \`x2' \`x3' i.\`time', re vce(cluster \`id')
estimates store RE_tw

foreach v in \`x1' \`x2' \`x3' {
    bysort \`id': egen mean_\`v' = mean(\`v')
}
xtreg \`y' \`x1' \`x2' \`x3' mean_\`x1' mean_\`x2' mean_\`x3' ///
    i.\`time', re vce(cluster \`id')
test mean_\`x1' mean_\`x2' mean_\`x3'
estimates store CRE

* 2. Cross-sectional dependence (xtcd2 is distributed with xtdcce2)
xtcd2 \`y'
xtcd2 \`x1'
xtcd2 \`x2'
xtcd2 \`x3'

* 3. Pesaran CIPS: requires balanced panel data
xtcips \`y',  maxlags(2) bglags(1)
xtcips \`x1', maxlags(2) bglags(1)
xtcips \`x2', maxlags(2) bglags(1)
xtcips \`x3', maxlags(2) bglags(1)
* Add trend only when the deterministic specification warrants it:
* xtcips \`y', maxlags(2) bglags(1) trend

* 4. Cointegration: run only after establishing suitable I(1) properties
xtcointtest pedroni \`y' \`x1' \`x2' \`x3'
xtcointtest westerlund \`y' \`x1' \`x2' \`x3', demean
* User-written Westerlund ECM test with bootstrap inference:
* xtwest \`y' \`x1' \`x2' \`x3', lags(1 2) leads(0 1) ///
*     constant bootstrap(400)

* 5. PMG-ARDL benchmark: common long-run coefficients
xtpmg D.\`y' D.\`x1' D.\`x2' D.\`x3', ///
    lr(L.\`y' \`x1' \`x2' \`x3') ec(ec_pmg) replace pmg

* 6. Mean Group and CCE / CS-ARDL using xtdcce2
xtdcce2 \`y' L.\`y' \`x1' \`x2' \`x3', nocrosssectional
xtdcce2 \`y' \`x1' \`x2' \`x3', ///
    crosssectional(\`y' \`x1' \`x2' \`x3') cr_lags(0)

* CS-ARDL long-run option: choose lag lengths in light of T and theory.
xtdcce2 D.\`y' \`x1' \`x2' \`x3' D.(\`x1' \`x2' \`x3'), ///
    crosssectional(D.\`y' D.\`x1' D.\`x2' D.\`x3') ///
    cr_lags(0 3 3 3) ///
    lr(L.\`y' \`x1' \`x2' \`x3', ardl)

display "Macro-panel workflow completed. Review assumptions and diagnostics."
