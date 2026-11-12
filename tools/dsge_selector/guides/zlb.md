title: The Zero Lower Bound with OccBin
short: ZLB
when: Nonlinearity: an occasionally binding constraint
data: Calibrated New Keynesian model, deep demand shock
lead: Use this guide when a constraint binds only sometimes: the zero lower bound on interest rates, a borrowing limit, a capacity or collateral constraint. A linear solution ignores the constraint and can let the interest rate fall to minus five percent. Dynare's OccBin solver (Guerrieri and Iacoviello 2015) switches between the regime where the constraint is slack and the regime where it binds, and shows how much worse a recession is when policy cannot respond.
---

## The question and the model

**How much deeper is a recession when the central bank cannot cut the interest rate below zero?** Between 2008 and 2015, and again in 2020, policy rates in many economies sat at or near zero, and a central question of macroeconomics became how the zero lower bound (ZLB) changes the effects of shocks.

The model is the three-equation New Keynesian model of the NK guide, with one change: the interest rate follows the Taylor rule only when the rule prescribes a rate above the bound.

\[
i_t = \max\left\{\, \underline{i},\ \ \phi_\pi \pi_t + \phi_x x_t \,\right\}
\]

Variables are deviations from the steady state, so the bound is \(\underline{i} = -i^*\), minus the steady-state nominal rate. With a 2% annual real rate and a 2% inflation target, \(i^* = 4\%\) a year, 1% a quarter.

<div class="read" markdown="1">
**How OccBin works**

OccBin treats the model as two linear models, one for each regime: the Taylor rule (constraint slack) and \(i_t = \underline{i}\) (constraint binding). For a given path of shocks it guesses in which periods the constraint binds, solves the model backwards through time with the right regime in each period, checks whether the guess is consistent (the constraint binds exactly when the shadow rate is below the bound), and iterates until it is. The solution is nonlinear globally but piecewise linear, and expectations take into account that the economy will leave the ZLB later.
</div>

## Write the model in Dynare

```dynare zlb.mod to="POLICY AND TRANSITION"
// New Keynesian model with the zero lower bound, solved with OccBin. Quarterly percent deviations.
@#ifndef ILB
  @#define ILB = -1.0
@#endif
var x pi i rn inot;
varexo e_rn;
parameters beta sigma kappa phi_pi phi_x rho_rn ilb;

beta = 0.995;  sigma = 1;  kappa = 0.1717;  phi_pi = 1.5;  phi_x = 0.125;  rho_rn = 0.8;
ilb = @{ILB};                                   // the bound: minus the steady-state nominal rate (1% a quarter)

model;
  x = x(+1) - (1/sigma) * (i - pi(+1) - rn);    // IS curve
  pi = beta * pi(+1) + kappa * x;               // Phillips curve
  inot = phi_pi * pi + phi_x * x;               // the rate the Taylor rule prescribes (the shadow rate)
  [name = 'policy', relax = 'zlb']
  i = inot;                                     // regime 1: the rule applies
  [name = 'policy', bind = 'zlb']
  i = ilb;                                      // regime 2: the rate is stuck at the bound
  rn = rho_rn * rn(-1) + e_rn;                  // natural-rate (demand) shock
end;

occbin_constraints;
  name 'zlb'; bind inot <= ilb; relax inot > ilb;
end;

steady;
check;

shocks(surprise);
  var e_rn; periods 1; values -1.25;            // a fall in the natural rate of 5 percentage points (annualised)
end;

occbin_setup;
occbin_solver(simul_periods = 40);
```

<div class="read" markdown="1">
**How to read it**

- **Two versions of the policy equation**, tagged with the same `name` and with `relax` and `bind`, tell Dynare which equation holds in which regime.
- **`occbin_constraints`** states when the constraint binds and when it is relaxed, in terms of model variables. Using the shadow rate `inot` makes the condition transparent.
- **`shocks(surprise)`** defines the shock path for the simulation: an unexpected fall in the natural rate in period 1. **`occbin_solver`** computes the piecewise-linear solution and, for comparison, the linear solution that ignores the bound.
- The `check` output confirms that the model is determinate in the regime where the Taylor rule applies.
</div>

## Compare the OccBin and the linear solution

```octave
P = oo_.occbin.simul.piecewise; L = oo_.occbin.simul.linear;
names = cellstr(M_.endo_names); id = @(v) find(strcmp(names, v));
T = 16; t = 1:T; ilb = M_.params(strcmp(cellstr(M_.param_names), 'ilb'));
f = figure('position', [0 0 900 300]);
vv = {'x', 'pi', 'i'}; tt = {'Output gap (%)', 'Inflation (annualised, pp)', 'Interest rate (annualised, pp)'}; sc = [1 4 4];
for j = 1:3
  subplot(1, 3, j); plot(t, sc(j)*P(t, id(vv{j}))); hold on; plot(t, sc(j)*L(t, id(vv{j})), '--');
  if j == 3, plot(t, sc(j)*ilb*ones(size(t)), 'k:'); end
  title(tt{j}); grid on; xlim([1 T]);
  if j == 1, legend('With the ZLB (OccBin)', 'Ignoring the ZLB (linear)', 'location', 'southeast'); end
end
print(f, '-dpng', '-r110', 'zlb.png');
nb = sum(P(:, id('i')) <= ilb + 1e-8);
fprintf('Quarters at the ZLB: %d\n', nb);
fprintf('Trough of the output gap: OccBin %.2f%%, linear %.2f%%  (amplification x%.1f)\n', min(P(:, id('x'))), min(L(:, id('x'))), min(P(:, id('x')))/min(L(:, id('x'))));
fprintf('Trough of annual inflation: OccBin %.2f pp, linear %.2f pp\n', 4*min(P(:, id('pi'))), 4*min(L(:, id('pi'))));
fprintf('Lowest interest rate (annualised): OccBin %.2f pp, linear %.2f pp\n', 4*min(P(:, id('i'))), 4*min(L(:, id('i'))));
```

<div class="read" markdown="1">
**How to read it**

- **The linear solution** lets the interest rate fall below the bound, by about 3.6 percentage points (annualised) beyond the steady-state rate of 4%, so it implies a negative nominal rate.
- **With the ZLB**, the rate stays at the bound for a few quarters. The real interest rate rises as inflation falls, the opposite of what the economy needs, so the output gap and inflation fall **much further** than in the linear solution.
- **Expectations matter.** Households and firms know the rate will be stuck for a while, which lowers expected inflation and raises the expected real rate even before the bound binds: the amplification comes partly from the future.
- This is why the ZLB is a nonlinearity a linear model cannot approximate: the size of a recession depends on whether the constraint binds.
</div>

## Policy question: does a higher inflation target help?

```octave
fprintf('%-28s %10s %14s %16s\n', 'Inflation target (annual)', 'ZLB qtrs', 'trough x (%)', 'trough pi (pp)');
for target = [1 2 3 4]
  b = -(2 + target) / 4;                          % bound = -(real rate + target), quarterly
  clear functions; evalc(sprintf('dynare zlb noclearall nograph nolog console -DILB=%g', b));
  P2 = oo_.occbin.simul.piecewise;
  fprintf('%-28s %10d %14.2f %16.2f\n', sprintf('%d%%', target), sum(P2(:, id('i')) <= b + 1e-8), min(P2(:, id('x'))), 4*min(P2(:, id('pi'))));
end
```

<div class="read" markdown="1">
**How to read it**

A higher inflation target raises the steady-state nominal rate and so gives the central bank more room to cut before reaching zero. The same shock then produces a shorter spell at the bound and a milder recession: with a 4% target instead of 2%, the trough of the output gap is about 45% shallower. This is the argument of Blanchard, Dell'Ariccia and Mauro (2010) for raising inflation targets. The model leaves out the costs of higher steady-state inflation (price dispersion, relative-price distortions), which a welfare analysis must add before drawing a policy conclusion.
</div>

```octave hide
clear functions; evalc('dynare zlb noclearall nograph nolog console -DILB=-1.0');
P = oo_.occbin.simul.piecewise; L = oo_.occbin.simul.linear;
ex = struct('x_occ', P(1:16, id('x'))', 'x_lin', L(1:16, id('x'))', 'pi_occ', 4*P(1:16, id('pi'))', 'pi_lin', 4*L(1:16, id('pi'))', ...
            'i_occ', 4*P(1:16, id('i'))', 'i_lin', 4*L(1:16, id('i'))', 'ilb', 4*ilb, 'nb', sum(P(:, id('i')) <= ilb + 1e-8));
fid = fopen(fullfile(export_dir, 'zlb.json'), 'w'); fputs(fid, jsonencode(ex)); fclose(fid);
```

## Write the paper

<div class="write" markdown="1">
**How to write it: results**

We solve the model with the occasionally binding zero lower bound using the OccBin algorithm of Guerrieri and Iacoviello (2015), as implemented in Dynare. Following a large adverse shock to the natural rate of interest, the policy rate reaches the bound and remains there for several quarters. Relative to a linear solution that ignores the bound, the fall in the output gap is more than twice as large and the fall in inflation substantially deeper, because the real interest rate rises as inflation falls. A higher inflation target shortens the spell at the bound and attenuates the recession.
</div>

<div class="warn" markdown="1">
**Common mistakes**

- Simulating a ZLB episode with a linear model and truncating the interest rate afterwards: expectations are then wrong.
- Writing the bound in levels when the model is in deviations (or the reverse). The bound in deviations is minus the steady-state rate.
- Forgetting that OccBin is a perfect-foresight solution conditional on the shock path: agents do not take into account the probability of future ZLB episodes. Global methods do, at a much higher computational cost.
- Estimating a model on ZLB-era data with a linear filter. Dynare's OccBin also provides a piecewise Kalman filter for estimation.
</div>
