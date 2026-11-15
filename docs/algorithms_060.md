# Algorithm definitions and verification

WIND 0.6.0 has 17 active implementations in `src/algorithms/optimizers.py`. The catalog
records a source and a specific computational variant; it does not claim that
fixed tracking settings satisfy every convergence assumption in that source.
The following table links each update to independent finite-step tests in
`tests/test_algorithms_060.py`. Those tests use hand expansions and closed-form
directional derivatives, not a second call to the implementation under test.

Let g be the observed gradient, S(a,b)=sign(a) max(|a|-b,0), and sym(A)=(A+Aᵀ)/2.
All vector products, signs, squares and maxima below are coordinatewise unless
indicated otherwise. Iteration t begins at 1 for moment corrections and schedules.

| Algorithm and primary source | Declared update and initialization | Independent checks |
|---|---|---|
| [SGD, Robbins and Monro](https://doi.org/10.1214/aoms/1177729586) | x⁺=x−ηg(x); constant-step gradient specialization of stochastic approximation; no momentum | `test_gradient_updates_two_steps` |
| [PolyakRuppertSGD, Polyak and Juditsky](https://doi.org/10.1137/0330046) | Raw z_t=z_(t−1)−η t^−0.75 g(z_(t−1)); recommendation is mean(z_1,…,z_t), no burn-in; average never feeds back | `test_pr_average_never_feeds_back_and_uses_updated_iterates` |
| [HeavyBall, Polyak](https://doi.org/10.1016/0041-5553(64)90137-5) | v⁺=βv−ηg(x), x⁺=x+v⁺; v=0 initially | `test_gradient_updates_two_steps` |
| [Nesterov, Sutskever et al., Eqs. 3–4](https://proceedings.mlr.press/v28/sutskever13.pdf) | Constant-momentum lookahead specialization: y=x+β(x−x_previous), x⁺=y−ηg(y); x_previous=x0 initially | `test_gradient_updates_two_steps`, including actual second query |
| [Adam, Kingma and Ba, Algorithm 1](https://arxiv.org/pdf/1412.6980) | m=β1 m+(1−β1)g, v=β2 v+(1−β2)g², x⁺=x−η[m/(1−β1^t)]/[sqrt(v/(1−β2^t))+ε]; m=v=0 | `test_adaptive_moments_against_hand_expansion`, `test_adaptive_nonzero_momentum_after_variance_decreases` |
| [AdamW, Loshchilov and Hutter, Algorithm 2](https://arxiv.org/abs/1711.05101) | Adam moments use g only; x⁺=(1−ηλ)x−η m_corrected/(sqrt(v_corrected)+ε). λ is decay per unit learning rate; schedule multiplier is constant | Same moment tests explicitly check decay and unaffected moments |
| [AMSGrad, Reddi et al., Algorithm 2](https://arxiv.org/pdf/1904.09237) | Raw m,v; v_max=max(v_max,v); x⁺=x−ηm/(sqrt(v_max)+ε). All moments start at zero; no bias correction; unconstrained constant-step variant with numerical ε | Same tests distinguish raw moments, decreasing v and retained maximum |
| [SignSGD, Bernstein et al., Algorithm 1](https://proceedings.mlr.press/v80/bernstein18a/bernstein18a.pdf) | Single-worker x⁺=x−η sign(g), no momentum or majority vote | `test_gradient_updates_two_steps` |
| [ProxSGD, forward-backward step in Beck and Teboulle](https://doi.org/10.1137/080716542) | Stochastic-gradient specialization with exact L1 prox: x⁺=S(x−ηg,ηλ); no acceleration | `test_gradient_updates_two_steps` checks exact threshold |
| [RDA, Xiao, Algorithm 1](https://www.jmlr.org/papers/volume11/xiao10a/xiao10a.pdf) | Quadratic prox, L1 regularizer: G_t=sum(g_s), β_t=sqrt(t)/η, x_t=S(−G_t/β_t,tλ/β_t); x0=0 | `test_rda_square_root_schedule_and_initial_state` |
| [SPSA, Spall](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF) | Mean of m fresh pairs [f(x+hu)−f(x−hu)]u/(2h), independent Rademacher u; constant η,h; no past-value reuse | `test_paired_estimators_and_exact_query_count` |
| [CoordinateFiniteDifferenceSGD, Kiefer and Wolfowitz](https://doi.org/10.1214/aoms/1177729392) | Multidimensional central-coordinate specialization: g_i=[f(x+he_i)−f(x−he_i)]/(2h); fixed η,h; exactly 2d calls per complete update | Same directional-derivative and call-count test |
| [GaussianZOSGD, Nesterov and Spokoiny](https://doi.org/10.1007/s10208-015-9296-2) | Antithetic symmetrization of the Gaussian smoothing estimator, averaged over m pairs; u~N(0,I), no normalization or factor d | Same test separately reconstructs Gaussian directions and scale |
| [SphericalZOSGD, Shamir, Algorithm 1 estimator](https://www.jmlr.org/papers/volume18/16-632/16-632.pdf) | Mean of m independent symmetric unit-sphere pairs, each multiplied by d; Euclidean SGD update, fixed η,h | Same test independently normalizes directions and checks d/m |
| [ExponentiatedGradient, Beck and Teboulle](https://www.tau.ac.il/~becka/3.pdf) | Entropic simplex step x_i⁺ proportional to x_i exp(−ηg_i), implemented in persistent log space. Initial zero coordinates retain zero support | Simplex formula, extreme-gradient and zero-support checks |
| [ProjectedGradient, Duchi et al. simplex projection](https://icml.cc/Conferences/2008/papers/361.pdf) | x⁺=Π_simplex(x−ηg), exact sorted-threshold Euclidean projection | Simplex hand solution and feasibility checks |
| [RiemannianSGD, Absil et al., chapters 3–4](https://sites.uclouvain.be/absil/amsbook/) | Π_X(G)=G−X sym(XᵀG); X⁺=polar(X−ηΠ_X(G)); embedded Euclidean metric, feasible initial frame | Tangency, polar retraction and independent feasible-frame checks |

Gaussian antithetic symmetrization averages the one-sided estimator at u and −u;
their common f(x) term cancels. The source also defines this symmetric oracle
explicitly in Section 3. This is an explicitly selected estimator variant,
not a claim that the source's complete optimizer or theorem is reproduced.
Likewise the spherical source supplies the estimator, while WIND uses the
declared Euclidean update and independent-pair batching.

The multidimensional central-coordinate formula is also written explicitly in
[Spall's overview, Eq. 2](https://www.jhuapl.edu/SPSA/PDF-SPSA/Spall_An_Overview.PDF).
That equation is the direct formula-level reference for the coordinate test;
the Kiefer--Wolfowitz citation identifies its historical origin.

## Defaults and numerical tolerances

SGD and HeavyBall use η=0.1; HeavyBall has β=0.9. PolyakRuppertSGD uses
η_t=0.1 t^−0.75. Nesterov uses η=0.05, β=0.9. Adam, AdamW and AMSGrad use
η=0.001, β1=0.9, β2=0.999, ε=1e−8; AdamW decay is 0.01. SignSGD uses
η=0.05. ProxSGD and RDA use η=0.1, λ=0.01. All three geometric methods
use η=0.1. SPSA uses η=0.005,h=0.1,m=1; Gaussian and spherical methods
use η=0.005,h=0.01,m=1; coordinate differences use η=0.02,h=1e−4.

Small float64 formula checks use atol=1e−12 and rtol=1e−10. RL float32
comparisons use atol=1e−6 and rtol=1e−5; geometric tests specify their
own residual tolerances. All active methods are reinitialized and checked
against fresh instances. Partial batches, zero remaining budget and distinct
recommended/query positions have separate contract tests.

Regularizers belong to the algorithms, not the benchmark loss. Their fixed
configurations can therefore be biased relative to the unregularized target.
The Riemannian update's embedded metric is distinct from the canonical
Stiefel geodesic distance implemented for evaluation.

Random frames use Gaussian QR with column signs corrected from diagonal R,
following [Mezzadri, section 5](https://arxiv.org/pdf/math-ph/0609050).
`test_haar_qr_column_signs_follow_positive_r_diagonal` verifies reconstruction
and positive diagonal signs; it is not a statistical proof of uniformity.
