const LANDSCAPES = [
  { id: "quadratic", name: "Quadratic", desc: "Conditioned convex bowl", a: ["Condition number", 5], b: ["Unused", 1], formula: String.raw`f_t(x)=\frac12(x-\theta_t)^\top A(x-\theta_t),\qquad \kappa(A)=\frac{\lambda_{\max}}{\lambda_{\min}}` },
  { id: "pnorm", name: "P-norm", desc: "Configurable ℓp geometry", a: ["p", 1.5], b: ["Condition number", 2], formula: String.raw`f_t(x)=\frac1p\left\|M_\kappa(x-\theta_t)\right\|_p^p` },
  { id: "rosenbrock", name: "Rosenbrock", desc: "Curved non-convex valley", a: ["Unused", 0], b: ["Unused", 0], formula: String.raw`f_t(x)=\sum_i\left[100(z_{i+1}-z_i^2)^2+(1-z_i)^2\right],\quad z=x-\theta_t+\mathbf1` },
  { id: "multiextremal", name: "Multi-extremal", desc: "Multiple local basins", a: ["Centers", 3], b: ["Width", 1], formula: String.raw`f_t(x)=\sum_i\left[(x_i-\theta_{t,i})^2+A\left(1-\cos(2\pi(x_i-\theta_{t,i}))\right)\right]` },
  { id: "robust", name: "Robust", desc: "Huber-like landscape", a: ["Delta", 0.1], b: ["Unused", 0], formula: String.raw`f_t(x)=\sum_i H_\delta(x_i-\theta_{t,i}),\quad H_\delta(u)=\begin{cases}\frac12u^2,&|u|\le\delta\\ \delta(|u|-\frac12\delta),&|u|>\delta\end{cases}` },
  { id: "simplex", name: "Simplex", desc: "Probability-constrained space", a: ["Unused", 0], b: ["Unused", 0], formula: String.raw`f_t(x)=\|x-\theta_t\|_2^2,\qquad x\in\Delta^{d-1}` },
  { id: "stiefel", name: "Stiefel", desc: "Orthonormal frame tracking", a: ["Ambient d", 5], b: ["Frame rank r", 2], formula: String.raw`f_t(X)=\|X-\Theta_t\|_F^2,\qquad X^\top X=I_r` },
  { id: "grassmann", name: "Grassmann", desc: "Basis-invariant subspaces", a: ["Ambient d", 5], b: ["Subspace rank r", 2], formula: String.raw`f_t(X)=\frac12\|XX^\top-\Theta_t\Theta_t^\top\|_F^2` },
];

const DRIFTS = [
  { id: "stationary", name: "Stationary", desc: "Fixed optimum", a: ["Unused", 0], b: ["Unused", 0], formula: String.raw`\theta_{t+1}=\theta_t` },
  { id: "linear", name: "Linear", desc: "Constant velocity", a: ["Velocity / coordinate", 0.02], b: ["Unused", 0], formula: String.raw`\theta_{t+1}=\theta_t+v` },
  { id: "random_walk", name: "Random walk", desc: "Stochastic displacement", a: ["Sigma", 0.02], b: ["Sparsity", 0], formula: String.raw`\theta_{t+1}=\theta_t+m_t\odot\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2I)` },
  { id: "cyclic", name: "Cyclic", desc: "Periodic movement", a: ["Amplitude", 1], b: ["Period", 100], c: ["Center / coordinate", 0], formula: String.raw`\theta_t=c+A\sin\!\left(\frac{2\pi t}{T}\right)\mathbf1` },
  { id: "jump", name: "Jump", desc: "Abrupt regime changes", a: ["Jump magnitude", 1], b: ["Interval", 50], formula: String.raw`\theta_{t+1}=\begin{cases}\theta_t+Ju_t,&(t+1)\bmod T=0\\ \theta_t,&\text{otherwise}\end{cases}` },
  { id: "adaptive", name: "Adaptive", desc: "Action-aware pursuit", a: ["Alpha", 0.1], b: ["Threshold", 10], mode: true, formula: String.raw`\theta_{t+1}=\theta_t\pm\alpha\,\operatorname{sign}(x_t-\theta_t),\qquad \|x_t-\theta_t\|_2\le\tau` },
  { id: "sparse", name: "Sparse", desc: "Coordinate-sparse motion", a: ["Sigma", 0.1], b: ["Active coordinates", 1], formula: String.raw`\theta_{t+1,i}=\theta_{t,i}+\sigma\varepsilon_{t,i}\ \text{for }i\in S_t,\qquad |S_t|=k` },
  { id: "stiefel", name: "Stiefel drift", desc: "Manifold-preserving motion", a: ["Sigma", 0.05], b: ["Unused", 0], formula: String.raw`\Theta_{t+1}=\operatorname{Cayley}\!\left(\sigma(A_t-A_t^\top)\right)\Theta_t` },
];

const ORACLES = [
  { id: "first-order", tag: "FO", name: "First-order", desc: "Gradient, optionally value" },
  { id: "zero-order", tag: "ZO", name: "Zero-order", desc: "Function values only" },
  { id: "hybrid", tag: "HY", name: "Hybrid", desc: "Value + gradient" },
  { id: "scheduled", tag: "SC", name: "Scheduled", desc: "Time-multiplexed FO / ZO" },
  { id: "offline", tag: "OF", name: "Offline replay", desc: "Recorded θ trajectory" },
];

const ORACLE_FORMULAS = {
  "first-order": String.raw`\mathcal O_t(x)=\nabla f_t(x)\quad\text{or}\quad\bigl(f_t(x),\nabla f_t(x)\bigr)`,
  "zero-order": String.raw`\mathcal O_t(x)=\bigl(f_t(x),\varnothing\bigr)`,
  hybrid: String.raw`\mathcal O_t(x)=\bigl(f_t(x)+\xi_t^v,\nabla f_t(x)+\xi_t^g\bigr),\qquad \xi^v\perp\xi^g`,
  scheduled: String.raw`\mathcal O_t(x)=\begin{cases}\mathcal O_t^{FO}(x),&t\in\mathcal T_{FO}\\ \mathcal O_t^{ZO}(x),&t\in\mathcal T_{ZO}\end{cases}`,
  offline: String.raw`\theta_t=\theta_t^{\mathrm{recorded}},\qquad \mathcal O_t(x)=\bigl(f(x,\theta_t),\nabla f(x,\theta_t)\bigr)`,
};

const NOISE_MODELS = {
  none: [],
  gaussian: [["sigma", 0.01]],
  heavy_tailed: [["alpha", 1.5], ["scale", 1]],
  correlated: [["sigma", 0.01], ["phi", 0.8]],
  quantized: [["delta", 0.1]],
  multiplicative: [["sigma_rel", 0.1]],
  sparse: [["sigma", 0.1], ["p", 0.1]],
};
const NOISES = Object.keys(NOISE_MODELS);
const NOISE_FORMULAS = {
  none: String.raw`\widetilde y_t=y_t`,
  gaussian: String.raw`\widetilde y_t=y_t+\varepsilon_t,\qquad \varepsilon_t\sim\mathcal N(0,\sigma^2)`,
  heavy_tailed: String.raw`\widetilde y_t=y_t+s_t\,\mathrm{scale}\left((1-U_t)^{-1/\alpha}-1\right)`,
  correlated: String.raw`\xi_t=\phi\xi_{t-1}+\sqrt{1-\phi^2}\,\varepsilon_t,\qquad \widetilde y_t=y_t+\xi_t`,
  quantized: String.raw`\widetilde y_t=\delta\,\operatorname{round}\!\left(\frac{y_t}{\delta}\right)`,
  multiplicative: String.raw`\widetilde y_t=y_t(1+\varepsilon_t),\qquad \varepsilon_t\sim\mathcal N(0,\sigma_{\mathrm{rel}}^2)`,
  sparse: String.raw`\widetilde y_t=y_t+B_t\varepsilon_t,\qquad B_t\sim\operatorname{Bernoulli}(p),\quad \varepsilon_t\sim\mathcal N(0,\sigma^2)`,
};

const OPTIMIZERS = [
  ["SGD", "first-order", { lr: 0.1, momentum: 0 }, "vₜ = μvₜ₋₁ − ηgₜ;  xₜ₊₁ = xₜ + vₜ"],
  ["SGD_Polyak", "first-order", { lr: 0.1 }, "zₜ = xₜ − ηgₜ;  x̄ₜ = (1 − 1/t)x̄ₜ₋₁ + zₜ/t"],
  ["HeavyBall", "first-order", { lr: 0.1, beta: 0.9 }, "vₜ = βvₜ₋₁ − ηgₜ;  xₜ₊₁ = xₜ + vₜ"],
  ["Nesterov", "first-order", { lr: 0.05, beta: 0.9 }, "vₜ = βvₜ₋₁ − η∇f(xₜ + βvₜ₋₁);  xₜ₊₁ = xₜ + vₜ"],
  ["Adam", "first-order", { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8 }, "mₜ = β₁mₜ₋₁ + (1−β₁)gₜ;  vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²;  xₜ₊₁ = xₜ − ηm̂ₜ/(√v̂ₜ+ε)"],
  ["AdamW", "first-order", { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 }, "g̃ₜ = gₜ + λxₜ;  xₜ₊₁ = xₜ − ηm̂(g̃ₜ)/(√v̂(g̃ₜ)+ε)"],
  ["AMSGrad", "first-order", { lr: 0.001, beta1: 0.9, beta2: 0.999, eps: 1e-8 }, "v̂ₜ = max(v̂ₜ₋₁, vₜ);  xₜ₊₁ = xₜ − ηm̂ₜ/(√v̂ₜ+ε)"],
  ["SMD", "first-order", { lr: 0.1 }, "xₜ₊₁,ᵢ = xₜ,ᵢ exp(−ηgₜ,ᵢ) / Σⱼ xₜ,ⱼ exp(−ηgₜ,ⱼ)"],
  ["RDA", "first-order", { lr: 0.1, lambda_reg: 0.01 }, "Gₜ = Σₛ₌₁ᵗ gₛ;  xₜ₊₁ = soft(−ηGₜ, ηλt)"],
  ["ProxSGD", "first-order", { lr: 0.1, lambda_reg: 0.01 }, "xₜ₊₁ = proxηλ‖·‖₁(xₜ − ηgₜ) = soft(xₜ − ηgₜ, ηλ)"],
  ["AdaptiveLR", "first-order", { lr0: 0.1 }, "ηₜ = η₀/(1+‖gₜ‖₂);  xₜ₊₁ = xₜ − ηₜgₜ"],
  ["SignSGD", "first-order", { lr: 0.05 }, "xₜ₊₁ = xₜ − η sign(gₜ)"],
  ["RandomSearch", "zero-order", { lr: 0.1, scale: 0.5 }, "xₜ₊₁ = xbest + σ εₜ,  εₜ ~ N(0,I)"],
  ["OnePointSPSA", "zero-order", { lr: 0.005, perturb: 0.1 }, "ĝₜ = [f(xₜ+cΔₜ)−f(xₜ)]Δₜ/c;  xₜ₊₁ = xₜ − ηĝₜ"],
  ["FiniteDiffCentral", "zero-order", { lr: 0.02, h: 0.0001 }, "ĝₜ,ᵢ = [f(xₜ+heᵢ)−f(xₜ−heᵢ)]/(2h);  xₜ₊₁ = xₜ − ηĝₜ"],
  ["FDSA", "zero-order", { lr: 0.02, h: 0.0001 }, "ĝₜ = [f(xₜ+δₜ)−f(xₜ)]δₜ/‖δₜ‖²;  xₜ₊₁ = xₜ − ηĝₜ"],
  ["SPSA", "zero-order", { lr: 0.005, perturb: 0.1 }, "ĝₜ = [f(xₜ+cΔₜ)−f(xₜ−cΔₜ)]Δₜ/(2c);  xₜ₊₁ = xₜ − ηĝₜ"],
  ["ZOSGD", "zero-order", { lr: 0.005, mu: 0.01 }, "ĝₜ = [f(xₜ+μuₜ)−f(xₜ)]uₜ/(μ‖uₜ‖²);  xₜ₊₁ = xₜ − ηĝₜ"],
  ["ZOSignSGD", "zero-order", { lr: 0.005, mu: 0.01 }, "xₜ₊₁ = xₜ − η sign([f(xₜ+μuₜ)−f(xₜ)]uₜ)"],
  ["QuadraticInterpolation", "zero-order", { lr: 0.1 }, "q(s)=as²+bs+f₀;  s* = clip(−b/(2a), −2, 2);  xₜ₊₁=xₜ+s*d"],
  ["KieferWolfowitz", "zero-order", { lr: 0.005, cn: 0.1 }, "cₙ=c/√n;  ĝₙ,ᵢ=[f(xₙ+cₙeᵢ)−f(xₙ−cₙeᵢ)]/(2cₙ);  xₙ₊₁=xₙ−ηĝₙ/√n"],
  ["NedicSubgradient", "zero-order", { lr: 0.005 }, "ĝₜ = [f(xₜ+δₜ)−f(xₜ)]δₜ/‖δₜ‖²;  xₜ₊₁=xₜ−ηĝₜ/√t"],
  ["AcceleratedSPSA", "zero-order", { lr: 0.005, perturb: 0.1, beta: 0.9 }, "mₜ=βmₜ₋₁+(1−β)ĝSPSAₜ;  xₜ₊₁=xₜ−ηmₜ"],
  ["CMAES", "zero-order", { sigma: 0.5, population_size: 0 }, "xᵢ ~ N(mₜ,Cₜ);  mₜ₊₁ = mean(elite(xᵢ));  Cₜ₊₁ = cov(elite(xᵢ))"],
  ["GPUCB", "zero-order", { beta: 2 }, "xₜ₊₁ = xₜ + clip(βuₜ,0,2)dₜ,  ‖dₜ‖₂=1"],
].map(([name, order, params, formula]) => ({ name, order, params, formula }));

const optimizerStages = (...stages) => String.raw`\begin{aligned}`
  + stages.map((stage, index) => String.raw`${index + 1}.\quad &${stage}`).join(String.raw`\\[5pt]`)
  + String.raw`\end{aligned}`;

const OPTIMIZER_LATEX = {
  SGD: optimizerStages(
    String.raw`v_t=\mu v_{t-1}-\eta g_t`,
    String.raw`x_{t+1}=x_t+v_t`,
  ),
  SGD_Polyak: optimizerStages(
    String.raw`z_t=x_t-\eta g_t`,
    String.raw`\bar x_t=\left(1-\frac1t\right)\bar x_{t-1}+\frac1t z_t`,
  ),
  HeavyBall: optimizerStages(
    String.raw`v_t=\beta v_{t-1}-\eta g_t`,
    String.raw`x_{t+1}=x_t+v_t`,
  ),
  Nesterov: optimizerStages(
    String.raw`y_t=x_t+\beta v_{t-1}`,
    String.raw`v_t=\beta v_{t-1}-\eta\nabla f(y_t)`,
    String.raw`x_{t+1}=x_t+v_t`,
  ),
  Adam: optimizerStages(
    String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)g_t`,
    String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`,
    String.raw`\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t}`,
    String.raw`x_{t+1}=x_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}`,
  ),
  AdamW: optimizerStages(
    String.raw`\widetilde g_t=g_t+\lambda x_t`,
    String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)\widetilde g_t`,
    String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)\widetilde g_t^2`,
    String.raw`x_{t+1}=x_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}`,
  ),
  AMSGrad: optimizerStages(
    String.raw`m_t=\beta_1m_{t-1}+(1-\beta_1)g_t`,
    String.raw`v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2`,
    String.raw`\widehat v_t=\max(\widehat v_{t-1},v_t)`,
    String.raw`x_{t+1}=x_t-\eta\frac{\widehat m_t}{\sqrt{\widehat v_t}+\varepsilon}`,
  ),
  SMD: optimizerStages(
    String.raw`w_{t+1,i}=x_{t,i}\exp(-\eta g_{t,i})`,
    String.raw`x_{t+1,i}=\frac{w_{t+1,i}}{\sum_jw_{t+1,j}}`,
  ),
  RDA: optimizerStages(
    String.raw`G_t=\sum_{s=1}^{t}g_s`,
    String.raw`x_{t+1}=\operatorname{soft}(-\eta G_t,\eta\lambda t)`,
  ),
  ProxSGD: optimizerStages(
    String.raw`z_t=x_t-\eta g_t`,
    String.raw`x_{t+1}=\operatorname{prox}_{\eta\lambda\|\cdot\|_1}(z_t)=\operatorname{soft}(z_t,\eta\lambda)`,
  ),
  AdaptiveLR: optimizerStages(
    String.raw`\eta_t=\frac{\eta_0}{1+\|g_t\|_2}`,
    String.raw`x_{t+1}=x_t-\eta_tg_t`,
  ),
  SignSGD: String.raw`x_{t+1}=x_t-\eta\,\operatorname{sign}(g_t)`,
  RandomSearch: optimizerStages(
    String.raw`\varepsilon_t\sim\mathcal N(0,I)`,
    String.raw`x_{t+1}=x_{\mathrm{best}}+\sigma\varepsilon_t`,
  ),
  OnePointSPSA: optimizerStages(
    String.raw`\Delta_t\sim\operatorname{Rad}(\{-1,+1\}^d)`,
    String.raw`\widehat g_t=\frac{f(x_t+c\Delta_t)-f(x_t)}{c}\Delta_t`,
    String.raw`x_{t+1}=x_t-\eta\widehat g_t`,
  ),
  FiniteDiffCentral: optimizerStages(
    String.raw`y_{t,i}^{\pm}=f(x_t\pm he_i)`,
    String.raw`\widehat g_{t,i}=\frac{y_{t,i}^{+}-y_{t,i}^{-}}{2h}`,
    String.raw`x_{t+1}=x_t-\eta\widehat g_t`,
  ),
  FDSA: optimizerStages(
    String.raw`\delta_t\sim\mathcal N(0,h^2I)`,
    String.raw`\widehat g_t=\frac{f(x_t+\delta_t)-f(x_t)}{\|\delta_t\|_2^2}\delta_t`,
    String.raw`x_{t+1}=x_t-\eta\widehat g_t`,
  ),
  SPSA: optimizerStages(
    String.raw`\Delta_t\sim\operatorname{Rad}(\{-1,+1\}^d)`,
    String.raw`y_t^{\pm}=f(x_t\pm c\Delta_t)`,
    String.raw`\widehat g_t=\frac{y_t^+-y_t^-}{2c}\Delta_t`,
    String.raw`x_{t+1}=x_t-\eta\widehat g_t`,
  ),
  ZOSGD: optimizerStages(
    String.raw`u_t\sim\mathcal N(0,I)`,
    String.raw`\widehat g_t=\frac{f(x_t+\mu u_t)-f(x_t)}{\mu\|u_t\|_2^2}u_t`,
    String.raw`x_{t+1}=x_t-\eta\widehat g_t`,
  ),
  ZOSignSGD: optimizerStages(
    String.raw`u_t\sim\mathcal N(0,I)`,
    String.raw`s_t=\operatorname{sign}\!\left([f(x_t+\mu u_t)-f(x_t)]u_t\right)`,
    String.raw`x_{t+1}=x_t-\eta s_t`,
  ),
  QuadraticInterpolation: optimizerStages(
    String.raw`q(s)=as^2+bs+f_0`,
    String.raw`s^*=\operatorname{clip}\!\left(-\frac{b}{2a},-2,2\right)`,
    String.raw`x_{t+1}=x_t+s^*d_t`,
  ),
  KieferWolfowitz: optimizerStages(
    String.raw`c_n=\frac{c}{\sqrt n}`,
    String.raw`\widehat g_{n,i}=\frac{f(x_n+c_ne_i)-f(x_n-c_ne_i)}{2c_n}`,
    String.raw`x_{n+1}=x_n-\frac{\eta}{\sqrt n}\widehat g_n`,
  ),
  NedicSubgradient: optimizerStages(
    String.raw`\delta_t\sim\mathcal N(0,\sigma^2I)`,
    String.raw`\widehat g_t=\frac{f(x_t+\delta_t)-f(x_t)}{\|\delta_t\|_2^2}\delta_t`,
    String.raw`x_{t+1}=x_t-\frac{\eta}{\sqrt t}\widehat g_t`,
  ),
  AcceleratedSPSA: optimizerStages(
    String.raw`\widehat g_t=\frac{f(x_t+c\Delta_t)-f(x_t-c\Delta_t)}{2c}\Delta_t`,
    String.raw`m_t=\beta m_{t-1}+(1-\beta)\widehat g_t`,
    String.raw`x_{t+1}=x_t-\eta m_t`,
  ),
  CMAES: optimizerStages(
    String.raw`x_i\sim\mathcal N(m_t,C_t)`,
    String.raw`E_t=\operatorname{elite}(x_1,\ldots,x_\lambda)`,
    String.raw`m_{t+1}=\operatorname{mean}(E_t)`,
    String.raw`C_{t+1}=\operatorname{cov}(E_t)`,
  ),
  GPUCB: optimizerStages(
    String.raw`\|d_t\|_2=1`,
    String.raw`x_{t+1}=x_t+\operatorname{clip}(\beta u_t,0,2)d_t`,
  ),
};
OPTIMIZERS.forEach((optimizer) => { optimizer.formula = OPTIMIZER_LATEX[optimizer.name]; });

const METRICS = [
  ["tracking_error", "Tracking error", "Geometry-aware distance", String.raw`e_t=\|x_t-\theta_t\|_p`],
  ["max_coordinate_error", "Max coordinate", "Worst coordinate deviation", String.raw`e_t^{\infty}=\max_i|x_{t,i}-\theta_{t,i}|`],
  ["instant_loss", "Instant loss", "fₜ(xₜ)", String.raw`\ell_t=f_t(x_t)`],
  ["dynamic_regret", "Dynamic regret", "Cumulative moving regret", String.raw`R_T=\sum_{t=1}^{T}\left[f_t(x_t)-f_t(\theta_t)\right]`],
  ["time_to_recovery", "Time to recovery", "Recovery after jumps", String.raw`\operatorname{TTR}(\tau)=\min\{k\ge0:\|x_{\tau+k}-\theta_{\tau+k}\|_2\le\varepsilon\}`],
  ["drift_adaptation", "Drift adaptation", "Motion alignment score", String.raw`A_t=\frac{\langle\Delta x_t,\Delta\theta_t\rangle}{\|\Delta x_t\|_2\,\|\Delta\theta_t\|_2}`],
  ["adaptivity", "Adaptivity", "Recovery relative to ideal", String.raw`\operatorname{Adaptivity}=\frac{\operatorname{TTR}_{\mathrm{oracle}}}{\operatorname{TTR}_{\mathrm{algo}}}`],
  ["query_efficiency", "Query efficiency", "Error per oracle query", String.raw`\operatorname{QE}=\frac{\operatorname{mean}_{\mathrm{tail}}\|x_t-\theta_t\|_2}{Q_T}`],
  ["lyapunov", "Lyapunov", "Stability function Vₙ", String.raw`V_t=\|x_t-\theta_t\|_{\rho+1}^{\rho+1}`],
  ["asymptotic_bound", "Asymptotic bound", "Tail lim-sup estimate", String.raw`L=\limsup_{t\to\infty}\mathbb E[V_t]`],
].map(([id, name, desc, formula]) => ({ id, name, desc, formula }));

const I18N = {
  ru: {
    navOverview:"Главная",navEnvironment:"Среда",navOracle:"Оракул",navOptimizers:"Примеры оптимизаторов",navRunner:"Запуск",navResults:"Результаты",navAnalysis:"Анализ",navGym:"RL: Gymnasium",documentation:"Документация ↗",workspaceReady:"Конфигурация готова",
    controlCenter:"ОБЗОР КОНФИГУРАЦИИ",overviewTitle:"WIND Benchmark",overviewDescription:"Настройка среды, оракула, алгоритма, запуска, метрик и выходных данных.",
    landscapes:"Ландшафтов",landscapesList:"Quadratic, P-norm, Rosenbrock, Multi-extremal, Robust, Simplex, Stiefel и Grassmann.",driftModels:"Моделей дрейфа",driftList:"Stationary, linear, random walk, cyclic, jump, adaptive, sparse и Stiefel.",oracleModes:"Режимов оракула",oracleList:"First-order, zero-order, hybrid, scheduled и offline replay.",optimizerCatalog:"Примеров оптимизаторов",optimizerList:"12 first-order и 13 zero-order реализаций.",metrics:"Метрик",metricsList:"Tracking error, regret, Lyapunov, adaptivity и query efficiency.",geometries:"Геометрии действий",geometryList:"Euclidean, Simplex, Stiefel и Grassmann.",
    benchmarkPipeline:"Конвейер бенчмарка",configure:"Настроить",environment:"Среда",oracle:"Оракул",optimizer:"Пример оптимизатора",runner:"Runner",result:"Результат",
    environmentTitle:"Динамическая среда",environmentDescription:"Ландшафт задаёт геометрию задачи, drift — движение скрытого оптимума θₜ.",landscape:"Ландшафт",dimension:"Размерность",drift:"Дрейф",bounds:"Границы x",
    oracleTitle:"Оракул",oracleDescription:"Режим и состав информации, доступной алгоритму на каждом шаге.",oracleMode:"Режим оракула",blindValue:"Скрывать значение функции",schedule:"Расписание",recordedTrajectory:"Файл с траекторией θ",noiseChannels:"Каналы шума",valueNoise:"Шум значения",gradientNoise:"Шум градиента",
    optimizersTitle:"Примеры оптимизаторов",optimizersDescription:"Можно выбрать несколько реализаций или оставить список пустым.",clearSelection:"Снять выбор",selectedCount:"Выбрано: {count}",updateFormula:"Формула обновления",formulaHintFO:"gₜ — наблюдаемый градиент; η — шаг.",formulaHintZO:"Используются только значения f; ĝₜ — оценка градиента.",noOptimizers:"Не выбраны",chooseOptimizer:"Выберите хотя бы один оптимизатор для запуска",optimizerCount:"Оптимизаторов: {count}",runnerTitle:"Параметры запуска",runnerDescription:"Шаги, seeds, выходной каталог, запись траектории и набор метрик.",execution:"Выполнение",steps:"Шаги",outputDirectory:"Каталог результатов",recordTrajectory:"Записывать траекторию",normalizeRegret:"Нормировать regret",metricSet:"Набор метрик",readyToRun:"Готово к запуску",runBenchmark:"Запустить",stop:"Остановить",
    resultsTitle:"Результаты",resultsDescription:"Сохранённые JSON, CSV, метрики, траектории и параметры среды.",refresh:"Обновить",loadingResults:"Загрузка результатов…",loadForAnalysis:"Загрузить для анализа",analysisTitle:"Анализ",analysisDescription:"Временные ряды, сравнение алгоритмов, распределения и траектории.",gymTitle:"RL: Gymnasium",gymDescription:"Параметры action space, reward и геометрии ограничений.",currentConfiguration:"Текущая конфигурация",liveSync:"Синхронизировано",
    connected:"подключён · версия {version}",offline:"офлайн",staticMode:"статическое пособие",viewOverview:"Обзор",viewEnvironment:"Среда",viewOracle:"Оракул",viewOptimizers:"Примеры оптимизаторов",viewRunner:"Запуск",viewResults:"Результаты",viewAnalysis:"Анализ",viewGym:"RL: Gymnasium",copied:"Скопировано",runStarted:"Запуск начат",runCompleted:"Запуск завершён",runFailed:"Ошибка запуска",runCancelled:"Запуск остановлен",noResults:"Результаты пока не найдены",staticResults:"В статическом режиме загрузите локальный JSON или CSV для анализа.",analysisLoaded:"Данные загружены",invalidAnalysisFile:"Не удалось прочитать JSON или CSV",incompatible:"Режимы обратной связи алгоритма и оракула несовместимы",csvNeedsTrajectory:"Для CSV нужна запись траектории",adaptiveMode:"Режим адаптации",trackingNorm:"Норма tracking error",normalizeByDim:"Нормировать по размерности",jumpThreshold:"Порог скачка",recoveryEpsilon:"Радиус восстановления ε",oracleTtr:"Идеальный TTR",holderRho:"Показатель Гёльдера ρ",gymBounds:"Границы Gym x",initialTheta:"Начальное θ₀",initialX:"Стартовая точка x₀",gymInitialX:"Начальное Gym x₀",invalidVector:"Введите одно число или {dim} координат",formula:"Формула",parameters:"Параметры",downloadJson:"СКАЧАТЬ",downloadConfiguration:"Скачать конфигурацию",configurationDownloaded:"Конфигурация скачана",scenarioApplied:"Пример применён",staticLaunchDetail:"Статический режим: настройте параметры и скачайте JSON конфигурации.",
  },
  en: {
    navOverview:"Home",navEnvironment:"Environment",navOracle:"Oracle",navOptimizers:"Optimizer examples",navRunner:"Run",navResults:"Results",navAnalysis:"Analysis",navGym:"RL: Gymnasium",documentation:"Documentation ↗",workspaceReady:"Configuration ready",
    controlCenter:"CONFIGURATION OVERVIEW",overviewTitle:"WIND Benchmark",overviewDescription:"Environment, oracle, algorithm, runner, metrics and output configuration.",
    landscapes:"Landscapes",landscapesList:"Quadratic, P-norm, Rosenbrock, Multi-extremal, Robust, Simplex, Stiefel and Grassmann.",driftModels:"Drift models",driftList:"Stationary, linear, random walk, cyclic, jump, adaptive, sparse and Stiefel.",oracleModes:"Oracle modes",oracleList:"First-order, zero-order, hybrid, scheduled and offline replay.",optimizerCatalog:"Optimizer examples",optimizerList:"12 first-order and 13 zero-order implementations.",metrics:"Metrics",metricsList:"Tracking error, regret, Lyapunov, adaptivity and query efficiency.",geometries:"Action geometries",geometryList:"Euclidean, Simplex, Stiefel and Grassmann.",
    benchmarkPipeline:"Benchmark pipeline",configure:"Configure",environment:"Environment",oracle:"Oracle",optimizer:"Optimizer example",runner:"Runner",result:"Result",
    environmentTitle:"Dynamic environment",environmentDescription:"The landscape defines task geometry; drift moves the hidden optimum θₜ.",landscape:"Landscape",dimension:"Dimension",drift:"Drift",bounds:"x bounds",
    oracleTitle:"Oracle",oracleDescription:"Feedback mode and information available to the algorithm at each step.",oracleMode:"Oracle mode",blindValue:"Hide function value",schedule:"Schedule",recordedTrajectory:"Recorded θ file",noiseChannels:"Noise channels",valueNoise:"Value noise",gradientNoise:"Gradient noise",
    optimizersTitle:"Optimizer examples",optimizersDescription:"Select multiple implementations or leave the selection empty.",clearSelection:"Clear selection",selectedCount:"Selected: {count}",updateFormula:"Update formula",formulaHintFO:"gₜ is the observed gradient; η is the step size.",formulaHintZO:"Only f values are used; ĝₜ denotes a gradient estimate.",noOptimizers:"None selected",chooseOptimizer:"Select at least one optimizer to run",optimizerCount:"Optimizers: {count}",runnerTitle:"Run parameters",runnerDescription:"Steps, seeds, output directory, trajectory recording and metrics.",execution:"Execution",steps:"Steps",outputDirectory:"Results directory",recordTrajectory:"Record trajectory",normalizeRegret:"Normalize regret",metricSet:"Metric set",readyToRun:"Ready to run",runBenchmark:"Run",stop:"Stop",
    resultsTitle:"Results",resultsDescription:"Saved JSON, CSV, metrics, trajectories and environment parameters.",refresh:"Refresh",loadingResults:"Loading results…",loadForAnalysis:"Load for analysis",analysisTitle:"Analysis",analysisDescription:"Time series, algorithm comparisons, distributions and trajectories.",gymTitle:"RL: Gymnasium",gymDescription:"Action space, reward and constrained geometry parameters.",currentConfiguration:"Current configuration",liveSync:"Synchronized",
    connected:"connected · version {version}",offline:"offline",staticMode:"static handbook",viewOverview:"Overview",viewEnvironment:"Environment",viewOracle:"Oracle",viewOptimizers:"Optimizer examples",viewRunner:"Run",viewResults:"Results",viewAnalysis:"Analysis",viewGym:"RL: Gymnasium",copied:"Copied",runStarted:"Run started",runCompleted:"Run completed",runFailed:"Run failed",runCancelled:"Run stopped",noResults:"No saved results yet",staticResults:"In static mode, load a local JSON or CSV file for analysis.",analysisLoaded:"Data loaded",invalidAnalysisFile:"Could not read the JSON or CSV file",incompatible:"The algorithm and oracle feedback modes are incompatible",csvNeedsTrajectory:"CSV export requires trajectory recording",adaptiveMode:"Adaptive mode",trackingNorm:"Tracking-error norm",normalizeByDim:"Normalize by dimension",jumpThreshold:"Jump threshold",recoveryEpsilon:"Recovery radius ε",oracleTtr:"Ideal TTR",holderRho:"Hölder exponent ρ",gymBounds:"Gym x bounds",initialTheta:"Initial θ₀",initialX:"Initial point x₀",gymInitialX:"Initial Gym x₀",invalidVector:"Enter one value or {dim} coordinates",formula:"Formula",parameters:"Parameters",downloadJson:"DOWNLOAD",downloadConfiguration:"Download configuration",configurationDownloaded:"Configuration downloaded",scenarioApplied:"Example applied",staticLaunchDetail:"Static mode: configure the benchmark and download its JSON.",
  },
  zh: {
    navOverview:"首页",navEnvironment:"环境",navOracle:"预言机",navOptimizers:"优化器示例",navRunner:"运行",navResults:"结果",navAnalysis:"分析",navGym:"RL: Gymnasium",documentation:"文档 ↗",workspaceReady:"配置就绪",
    controlCenter:"配置概览",overviewTitle:"WIND Benchmark",overviewDescription:"环境、预言机、算法、运行参数、指标和输出配置。",
    landscapes:"损失景观",landscapesList:"Quadratic、P-norm、Rosenbrock、Multi-extremal、Robust、Simplex、Stiefel 和 Grassmann。",driftModels:"漂移模型",driftList:"Stationary、linear、random walk、cyclic、jump、adaptive、sparse 和 Stiefel。",oracleModes:"预言机模式",oracleList:"一阶、零阶、混合、调度和离线回放。",optimizerCatalog:"优化器示例",optimizerList:"12 个一阶和 13 个零阶实现。",metrics:"指标",metricsList:"跟踪误差、regret、Lyapunov、自适应性和查询效率。",geometries:"动作几何",geometryList:"Euclidean、Simplex、Stiefel 和 Grassmann。",
    benchmarkPipeline:"基准流程",configure:"配置",environment:"环境",oracle:"预言机",optimizer:"优化器示例",runner:"Runner",result:"结果",
    environmentTitle:"动态环境",environmentDescription:"景观定义任务几何，漂移控制隐藏最优点 θₜ 的运动。",landscape:"景观",dimension:"维度",drift:"漂移",bounds:"x 边界",
    oracleTitle:"预言机",oracleDescription:"每一步提供给算法的反馈模式和信息。",oracleMode:"预言机模式",blindValue:"隐藏函数值",schedule:"调度",recordedTrajectory:"记录的 θ 文件",noiseChannels:"噪声通道",valueNoise:"数值噪声",gradientNoise:"梯度噪声",
    optimizersTitle:"优化器示例",optimizersDescription:"可选择多个实现，也可以保留空列表。",clearSelection:"清除选择",selectedCount:"已选择：{count}",updateFormula:"更新公式",formulaHintFO:"gₜ 是观测梯度；η 是步长。",formulaHintZO:"仅使用函数值 f；ĝₜ 表示梯度估计。",noOptimizers:"未选择",chooseOptimizer:"至少选择一个优化器后才能运行",optimizerCount:"优化器：{count}",runnerTitle:"运行参数",runnerDescription:"步数、种子、输出目录、轨迹记录和指标。",execution:"执行",steps:"步数",outputDirectory:"结果目录",recordTrajectory:"记录轨迹",normalizeRegret:"归一化 regret",metricSet:"指标集",readyToRun:"可以运行",runBenchmark:"运行",stop:"停止",
    resultsTitle:"结果",resultsDescription:"保存的 JSON、CSV、指标、轨迹和环境参数。",refresh:"刷新",loadingResults:"正在加载结果…",loadForAnalysis:"加载用于分析",analysisTitle:"分析",analysisDescription:"时间序列、算法比较、分布和轨迹。",gymTitle:"RL: Gymnasium",gymDescription:"Action space、reward 和约束几何参数。",currentConfiguration:"当前配置",liveSync:"已同步",
    connected:"已连接 · 版本 {version}",offline:"离线",staticMode:"静态指南",viewOverview:"概览",viewEnvironment:"环境",viewOracle:"预言机",viewOptimizers:"优化器示例",viewRunner:"运行",viewResults:"结果",viewAnalysis:"分析",viewGym:"RL: Gymnasium",copied:"已复制",runStarted:"运行已开始",runCompleted:"运行已完成",runFailed:"运行失败",runCancelled:"运行已停止",noResults:"暂无保存结果",staticResults:"静态模式下，请加载本地 JSON 或 CSV 文件进行分析。",analysisLoaded:"数据已加载",invalidAnalysisFile:"无法读取 JSON 或 CSV 文件",incompatible:"算法与预言机的反馈模式不兼容",csvNeedsTrajectory:"CSV 导出需要记录轨迹",adaptiveMode:"自适应模式",trackingNorm:"跟踪误差范数",normalizeByDim:"按维度归一化",jumpThreshold:"跳变阈值",recoveryEpsilon:"恢复半径 ε",oracleTtr:"理想 TTR",holderRho:"Hölder 指数 ρ",gymBounds:"Gym x 边界",initialTheta:"初始 θ₀",initialX:"初始点 x₀",gymInitialX:"Gym 初始 x₀",invalidVector:"请输入一个值或 {dim} 个坐标",formula:"公式",parameters:"参数",downloadJson:"下载",downloadConfiguration:"下载配置",configurationDownloaded:"配置已下载",scenarioApplied:"示例已应用",staticLaunchDetail:"静态模式：配置基准并下载 JSON。",
  },
};

const els = {};
let language = "en";
let activeView = "overview";
let landscapeType = "quadratic";
let driftType = "random_walk";
let oracleType = "first-order";
let inspectedOptimizerName = null;
const selectedOptimizerNames = new Set();
const optimizerParamsByName = Object.fromEntries(
  OPTIMIZERS.map((optimizer) => [optimizer.name, { ...optimizer.params }]),
);
function freshNoiseParams() {
  return Object.fromEntries(Object.entries(NOISE_MODELS).map(([type, params]) => [type, Object.fromEntries(params)]));
}
const noiseParamsByChannel = { value: freshNoiseParams(), grad: freshNoiseParams() };
const metricParameterState = {
  tracking_norm: "l2",
  normalize_tracking: false,
  jump_threshold: 1,
  recovery_epsilon: 0.1,
  oracle_ttr: 1,
  rho: 1,
};
let orderFilter = "all";
let activeJobId = null;
let pollTimer;
let toastTimer;
let currentJson = "";
let analysisDataLoaded = false;
let mathTypesetChain = Promise.resolve();
let engineAvailable = false;
let runtimeChecked = false;
let engineVersion = null;
const collapsedGuidePages = new Set(
  Array.from(document.querySelectorAll("[data-guide-page]"), (container) => container.dataset.guidePage).filter(Boolean),
);

function tr(key, variables = {}) {
  let value = I18N[language]?.[key] ?? I18N.en[key] ?? key;
  Object.entries(variables).forEach(([name, replacement]) => { value = value.replaceAll(`{${name}}`, String(replacement)); });
  return value;
}

function mathCard(formula, extraClass = "") {
  return `<div class="model-formula ${extraClass}"><span>${tr("formula")} · LATEX</span><div class="math-expression">\\[${escapeHtml(formula)}\\]</div></div>`;
}

function typesetMath(container) {
  if (!container || !window.MathJax?.typesetPromise) return;
  mathTypesetChain = mathTypesetChain.then(() => {
    if (!container.isConnected) return undefined;
    window.MathJax.typesetClear?.([container]);
    return window.MathJax.typesetPromise([container]);
  }).catch(() => {});
}

function renderMathSlot(container, formula) {
  if (container.dataset.formula === formula) return;
  container.dataset.formula = formula;
  container.innerHTML = mathCard(formula);
  typesetMath(container);
}

function guideLocale() {
  return window.WIND_GUIDES?.[language] ?? window.WIND_GUIDES?.en;
}

function renderGuideTopic(topic, index, labels) {
  const bullets = topic.bullets?.length
    ? `<ul>${topic.bullets.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
    : "";
  const formula = topic.formula ? mathCard(topic.formula, "guide-formula") : "";
  const code = topic.code
    ? `<div class="guide-code"><header><span>${labels.source}</span><button type="button" data-copy-guide-code>${labels.copy}</button></header><pre><code>${escapeHtml(topic.code)}</code></pre></div>`
    : "";
  return `<article class="guide-topic"><header><span>${String(index + 1).padStart(2, "0")}</span><h3>${escapeHtml(topic.title)}</h3></header><div class="guide-topic-body"><p>${escapeHtml(topic.body)}</p>${bullets}${formula}${code}</div></article>`;
}

function renderGuides(openingPage = null) {
  const locale = guideLocale();
  if (!locale) return;
  const sourceByPage = {
    overview: "README.md",
    environment: "src/core.py",
    oracle: "src/oracle.py",
    optimizers: "src/experiment.py",
    runner: "src/benchmark.py",
    results: "README.md#reproducibility",
    analysis: "src/visualization.py",
    gym: "src/gym_env.py",
  };
  document.querySelectorAll("[data-guide-page]").forEach((container) => {
    const page = container.dataset.guidePage;
    const guide = locale.pages[page];
    if (!guide) return;
    const scenarios = guide.scenarios
      ? `<div class="scenario-grid"><span>${locale.labels.scenarios}</span>${(window.WIND_SCENARIOS ?? []).map((scenario) => `<button type="button" data-scenario="${scenario.id}"><b>${escapeHtml(scenario.labels[language] ?? scenario.labels.en)}</b><small>${escapeHtml(scenario.descriptions[language] ?? scenario.descriptions.en)}</small></button>`).join("")}</div>`
      : "";
    const source = sourceByPage[page];
    const collapsed = collapsedGuidePages.has(page);
    const toggleLabel = collapsed ? locale.labels.expand : locale.labels.collapse;
    const opening = openingPage === page && !collapsed;
    container.innerHTML = `<section class="guide-panel ${collapsed ? "collapsed" : ""} ${opening ? "opening" : ""}" data-guide-panel><header><div><span>${locale.labels.guide}</span><h2>${escapeHtml(guide.title)}</h2><p>${escapeHtml(guide.intro)}</p></div><div class="guide-panel-actions"><a href="https://github.com/muffin003/WIND/blob/main/${source}" target="_blank" rel="noopener noreferrer">${locale.labels.source} ↗</a><button type="button" data-guide-toggle aria-expanded="${!collapsed}" aria-label="${escapeHtml(toggleLabel)}" title="${escapeHtml(toggleLabel)}">${collapsed ? "+" : "−"}</button></div></header><div class="guide-panel-body" ${collapsed ? "hidden" : ""}><div class="guide-panel-content"><div class="guide-topics">${guide.topics.map((topic, index) => renderGuideTopic(topic, index, locale.labels)).join("")}</div>${scenarios}</div></div></section>`;
    typesetMath(container);
  });
}

function entityGuide(kind, id, parameterNames = []) {
  const locale = guideLocale();
  const entry = kind === "optimizers" ? window.WIND_OPTIMIZER_TEXT?.[id] : window.WIND_ENTITY_TEXT?.[kind]?.[id];
  const summary = entry?.[language] ?? entry?.en;
  if (!summary && !parameterNames.length) return "";
  const help = window.WIND_PARAMETER_HELP?.[language] ?? window.WIND_PARAMETER_HELP?.en ?? {};
  const parameters = parameterNames.length
    ? `<div class="entity-parameters"><span>${locale?.labels.parameters ?? "Parameters"}</span><dl>${parameterNames.map((name) => `<div><dt>${escapeHtml(name)}</dt><dd>${escapeHtml(help[name] ?? name.replaceAll("_", " "))}</dd></div>`).join("")}</dl></div>`
    : "";
  return `<div class="entity-guide">${summary ? `<p>${escapeHtml(summary)}</p>` : ""}${parameters}</div>`;
}

function renderStaticTranslations() {
  document.documentElement.lang = language === "zh" ? "zh-CN" : language;
  if (els.runnerFormula) delete els.runnerFormula.dataset.formula;
  if (els.gymFormula) delete els.gymFormula.dataset.formula;
  document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = tr(node.dataset.i18n); });
  document.querySelectorAll("[data-lang]").forEach((button) => button.classList.toggle("active", button.dataset.lang === language));
  if (els.landscapeGrid?.children.length) renderChoices(els.landscapeGrid, LANDSCAPES, landscapeType, "choice-card", els.landscapeFields);
  if (els.driftGrid?.children.length) renderChoices(els.driftGrid, DRIFTS, driftType, "choice-card", els.driftFields);
  if (els.oracleGrid?.children.length) renderOracleChoices();
  if (els.valueNoiseType?.options.length) { renderNoiseParams("value"); renderNoiseParams("grad"); }
  if (els.optimizerCatalog) renderOptimizerCatalog();
  if (els.metricsGrid?.children.length) renderMetrics(new Set(selectedMetricIds()));
  renderGuides();
  updateRuntimeUi();
  updateViewLabel();
}

function setView(view) {
  if (view === "analysis" && !analysisDataLoaded) view = "results";
  activeView = view;
  document.querySelectorAll("[data-view]").forEach((node) => node.classList.toggle("active", node.dataset.view === view));
  document.querySelectorAll("[data-view-link]").forEach((node) => node.classList.toggle("active", node.dataset.viewLink === view));
  updateViewLabel();
  if (view === "results") loadResults();
  history.replaceState(null, "", `#${view}`);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function updateViewLabel() {
  const key = `view${activeView[0].toUpperCase()}${activeView.slice(1)}`;
  document.title = `${tr(key)} — WIND Lab`;
}

function renderChoices(container, items, activeId, className = "choice-card", detailNode = null) {
  const guideKind = container === els.landscapeGrid ? "landscapes" : "drifts";
  container.innerHTML = items.map((item) => {
    const active = item.id === activeId;
    const formula = container === els.driftGrid && item.id === "adaptive"
      ? String.raw`\theta_{t+1}=\theta_t${els.driftMode?.value === "evasion" ? "-" : "+"}\alpha\,\operatorname{sign}(x_t-\theta_t),\qquad \|x_t-\theta_t\|_2\le\tau`
      : item.formula;
    return `<div class="choice-stack ${active ? "expanded" : ""}" data-choice-stack="${item.id}"><button type="button" class="${className} ${active ? "active" : ""}" data-choice="${item.id}"><b>${item.name}</b><small>${item.desc}</small></button>${active && formula ? `<div class="choice-inline-detail">${mathCard(formula, "inline-choice-formula")}${entityGuide(guideKind, item.id)}</div>` : ""}</div>`;
  }).join("");
  if (detailNode) container.querySelector(`[data-choice-stack="${activeId}"] .choice-inline-detail`)?.append(detailNode);
  typesetMath(container);
}

function renderOracleChoices() {
  els.oracleGrid.innerHTML = ORACLES.map((item) => {
    const active = item.id === oracleType;
    const formula = item.id === "first-order" && els.blindValue.checked
      ? String.raw`\mathcal O_t(x)=\nabla f_t(x)`
      : ORACLE_FORMULAS[item.id];
    return `<div class="choice-stack ${active ? "expanded" : ""}" data-oracle-stack="${item.id}"><button type="button" class="oracle-card ${active ? "active" : ""}" data-oracle="${item.id}"><em>${item.tag}</em><b>${item.name}</b><small>${item.desc}</small></button>${active ? `<div class="choice-inline-detail">${mathCard(formula, "inline-choice-formula")}${entityGuide("oracles", item.id)}</div>` : ""}</div>`;
  }).join("");
  els.scheduleField.style.display = oracleType === "scheduled" ? "flex" : "none";
  els.offlineField.style.display = oracleType === "offline" ? "flex" : "none";
  els.blindValue.closest("label").style.display = oracleType === "first-order" ? "flex" : "none";
  els.oracleGrid.querySelector(`[data-oracle-stack="${oracleType}"] .choice-inline-detail`)?.append(els.oracleExtra);
  typesetMath(els.oracleGrid);
}

function renderNoiseOptions() {
  const options = NOISES.map((name) => `<option value="${name}">${name.replaceAll("_", " ")}</option>`).join("");
  els.valueNoiseType.innerHTML = options;
  els.gradNoiseType.innerHTML = options;
  els.valueNoiseType.value = "gaussian";
  els.gradNoiseType.value = "gaussian";
  renderNoiseParams("value");
  renderNoiseParams("grad");
}

function renderNoiseParams(channel) {
  const type = els[`${channel}NoiseType`].value;
  const container = els[`${channel}NoiseParams`];
  const params = noiseParamsByChannel[channel][type];
  const parameterFields = Object.entries(params).map(([name, value]) => {
    const constraints = name === "phi" ? 'min="-0.999999" max="0.999999"' : name === "p" ? 'min="0" max="1"' : ["alpha", "scale", "sigma", "sigma_rel", "delta"].includes(name) ? 'min="0"' : "";
    return `<label><span>${name}</span><input type="number" step="any" ${constraints} data-noise-channel="${channel}" data-noise-param="${name}" value="${value}" /></label>`;
  }).join("");
  container.innerHTML = mathCard(NOISE_FORMULAS[type], "noise-formula") + entityGuide("noises", type, Object.keys(params)) + parameterFields;
  typesetMath(container);
}

function renderOptimizerCatalog() {
  const search = els.optimizerSearch.value.trim().toLowerCase();
  els.optimizerCatalog.innerHTML = OPTIMIZERS.map((optimizer) => {
    const hidden = (orderFilter !== "all" && optimizer.order !== orderFilter) || !optimizer.name.toLowerCase().includes(search);
    const selected = selectedOptimizerNames.has(optimizer.name);
    const button = `<button type="button" class="optimizer-item ${selected ? "active" : ""} ${optimizer.name === inspectedOptimizerName ? "inspected" : ""} ${hidden ? "hidden" : ""}" data-optimizer="${optimizer.name}" aria-pressed="${selected}"><i>${optimizer.order === "first-order" ? "FO" : "ZO"}</i><span class="optimizer-copy"><span class="optimizer-heading"><b>${optimizer.name}</b><small>${optimizer.order}</small></span><span class="optimizer-card-formula"><span class="math-expression">\\[${escapeHtml(optimizer.formula)}\\]</span></span></span><em>${selected ? "✓" : "+"}</em></button>`;
    if (optimizer.name !== inspectedOptimizerName || hidden) return button;
    const params = optimizerParamsByName[optimizer.name];
    const parameterFields = Object.entries(params).map(([name, value]) => {
      const attributes = name === "population_size" ? 'min="0" step="1"' : 'step="any"';
      return `<label><span>${name}${name === "population_size" ? " (0 = auto)" : ""}</span><input type="number" ${attributes} data-optimizer-name="${optimizer.name}" data-optimizer-param="${name}" value="${value}" /></label>`;
    }).join("");
    const detail = `<article class="optimizer-inline-detail"><header><span>${tr("parameters")}</span><b>${optimizer.name}</b><em>${optimizer.order.toUpperCase()}</em></header>${entityGuide("optimizers", optimizer.name, Object.keys(params))}<p class="optimizer-detail-note">${tr(optimizer.order === "first-order" ? "formulaHintFO" : "formulaHintZO")}</p><div class="optimizer-params">${parameterFields}</div></article>`;
    return button + detail;
  }).join("");
  typesetMath(els.optimizerCatalog);
}

function selectedOptimizerConfigs() {
  return OPTIMIZERS.filter((optimizer) => selectedOptimizerNames.has(optimizer.name))
    .map((optimizer) => {
      const params = { ...optimizerParamsByName[optimizer.name] };
      if (optimizer.name === "CMAES" && !(params.population_size > 0)) delete params.population_size;
      return { name: optimizer.name, params };
    });
}

function metricFormula(metric) {
  if (metric.id === "dynamic_regret" && els.normalizeRegret.checked) {
    return String.raw`\widetilde R_T=\frac{\sum_{t=1}^{T}[f_t(x_t)-f_t(\theta_t)]}{\sum_{t=2}^{T}\|\theta_t-\theta_{t-1}\|_2}`;
  }
  if (metric.id !== "tracking_error") return metric.formula;
  if (landscapeType === "stiefel") return String.raw`e_t=\|X_t-\Theta_t\|_F`;
  if (landscapeType === "grassmann") return String.raw`e_t=\left\|\arccos\!\left(\sigma(X_t^\top\Theta_t)\right)\right\|_2`;
  const norm = metricParameterState.tracking_norm;
  const base = norm === "mahalanobis"
    ? String.raw`\sqrt{(x_t-\theta_t)^\top A(x_t-\theta_t)}`
    : String.raw`\|x_t-\theta_t\|_${norm === "linf" ? String.raw`\infty` : norm.slice(1)}`;
  return metricParameterState.normalize_tracking ? String.raw`e_t=\frac{${base}}{\sqrt d}` : String.raw`e_t=${base}`;
}

function metricParameterFields(metricId) {
  if (metricId === "tracking_error" && !["stiefel", "grassmann"].includes(landscapeType)) {
    return `<div class="metric-inline-params"><label><span>${tr("trackingNorm")}</span><select data-metric-param="tracking_norm"><option value="l2" ${metricParameterState.tracking_norm === "l2" ? "selected" : ""}>ℓ2</option><option value="l1" ${metricParameterState.tracking_norm === "l1" ? "selected" : ""}>ℓ1</option><option value="linf" ${metricParameterState.tracking_norm === "linf" ? "selected" : ""}>ℓ∞</option><option value="mahalanobis" ${metricParameterState.tracking_norm === "mahalanobis" ? "selected" : ""}>Mahalanobis</option></select></label><label class="parameter-toggle"><span><b>${tr("normalizeByDim")}</b><small>tracking_error</small></span><input type="checkbox" data-metric-param="normalize_tracking" ${metricParameterState.normalize_tracking ? "checked" : ""} /></label></div>`;
  }
  if (["time_to_recovery", "adaptivity"].includes(metricId)) {
    const oracle = metricId === "adaptivity" ? `<label><span>${tr("oracleTtr")}</span><input type="number" min="0.000001" step="any" data-metric-param="oracle_ttr" value="${metricParameterState.oracle_ttr}" /></label>` : "";
    return `<div class="metric-inline-params"><label><span>${tr("jumpThreshold")}</span><input type="number" min="0.000001" step="any" data-metric-param="jump_threshold" value="${metricParameterState.jump_threshold}" /></label><label><span>${tr("recoveryEpsilon")}</span><input type="number" min="0.000001" step="any" data-metric-param="recovery_epsilon" value="${metricParameterState.recovery_epsilon}" /></label>${oracle}</div>`;
  }
  if (["lyapunov", "asymptotic_bound"].includes(metricId)) {
    return `<div class="metric-inline-params"><label><span>${tr("holderRho")}</span><input type="number" min="0.000001" max="1" step="any" data-metric-param="rho" value="${metricParameterState.rho}" /></label></div>`;
  }
  if (metricId === "dynamic_regret") {
    return `<div class="metric-inline-params"><label class="parameter-toggle"><span><b>${tr("normalizeRegret")}</b><small>Rₜ / path variation</small></span><input type="checkbox" data-runner-param="normalize_regret" ${els.normalizeRegret.checked ? "checked" : ""} /></label></div>`;
  }
  return "";
}

function renderMetrics(selectedIds = null) {
  const selected = selectedIds ?? new Set(["tracking_error", "instant_loss", "dynamic_regret", "drift_adaptation", "query_efficiency"]);
  els.metricsGrid.innerHTML = METRICS.map((metric) => {
    const checked = selected.has(metric.id);
    return `<div class="metric-stack"><label class="metric-option ${checked ? "active" : ""}"><span><b>${metric.name}</b><small>${metric.desc}</small></span><input type="checkbox" value="${metric.id}" data-metric-select ${checked ? "checked" : ""} /></label>${checked ? `<div class="metric-inline-detail">${mathCard(metricFormula(metric), "metric-math")}${metricParameterFields(metric.id)}</div>` : ""}</div>`;
  }).join("");
  typesetMath(els.metricsGrid);
}

function selectedMetricIds() {
  return [...els.metricsGrid.querySelectorAll("input[data-metric-select]:checked")].map((input) => input.value);
}

function parseSeeds() {
  return [...new Set(els.runnerSeeds.value.split(/[;,\s]+/).map(Number).filter((value) => Number.isInteger(value) && value >= 0))];
}

function parseVectorInput(raw, dim) {
  const text = raw.trim();
  if (!text) return null;
  const values = text.replace(/[\[\]]/g, "").split(/[;,\s]+/).filter(Boolean).map(Number);
  if (!values.length || values.some((value) => !Number.isFinite(value))) return null;
  if (values.length === 1) return Array(dim).fill(values[0]);
  return values.length === dim ? values : null;
}

function vectorInputIsValid(raw, dim) {
  return !raw.trim() || parseVectorInput(raw, dim) !== null;
}

function landscapeConfig() {
  const a = Number(els.landscapeParamA.value);
  const b = Number(els.landscapeParamB.value);
  if (landscapeType === "quadratic") return { type: landscapeType, condition_number: a };
  if (landscapeType === "pnorm") return { type: landscapeType, p: a, condition_number: b };
  if (landscapeType === "multiextremal") return { type: landscapeType, k_centers: Math.max(1, Math.round(a)), width: b };
  if (landscapeType === "robust") return { type: landscapeType, delta: a };
  if (["stiefel", "grassmann"].includes(landscapeType)) {
    const d = Math.max(2, Math.round(a));
    return { type: landscapeType, d, r: Math.min(d, Math.max(1, Math.round(b))) };
  }
  return { type: landscapeType };
}

function environmentDimension() {
  if (["stiefel", "grassmann"].includes(landscapeType)) {
    const landscape = landscapeConfig();
    return landscape.d * landscape.r;
  }
  return Math.max(1, Math.round(Number(els.environmentDim.value)));
}

function driftConfig(dim) {
  const a = Number(els.driftParamA.value);
  const b = Number(els.driftParamB.value);
  const c = Number(els.driftParamC.value);
  if (driftType === "linear") return { type: driftType, velocity: Array(dim).fill(a) };
  if (driftType === "random_walk") return { type: driftType, sigma: a, sparsity: b };
  if (driftType === "cyclic") return { type: driftType, amplitude: a, period: Math.max(1, Math.round(b)), center: Array(dim).fill(c) };
  if (driftType === "jump") return { type: driftType, jump_magnitude: a, interval: Math.max(1, Math.round(b)) };
  if (driftType === "adaptive") return { type: driftType, alpha: a, threshold: b, mode: els.driftMode.value };
  if (driftType === "sparse") return { type: driftType, sigma: a, k: Math.min(dim, Math.max(1, Math.round(b))) };
  if (driftType === "stiefel") {
    const landscape = landscapeConfig();
    return { type: driftType, sigma: a, d: landscape.d ?? dim, r: landscape.r ?? 1 };
  }
  return { type: driftType };
}

function noiseConfig(channel, type) {
  if (type === "none") return { type: "none" };
  return { type, ...noiseParamsByChannel[channel][type] };
}

function scheduleConfig() {
  const schedule = els.oracleSchedule.value.split(",").map((segment) => {
    const [mode, duration] = segment.trim().split(":");
    return [mode?.trim(), Number(duration)];
  }).filter(([mode, duration]) => ["first-order", "zero-order"].includes(mode) && duration > 0);
  return schedule.length ? schedule : [["first-order", 100], ["zero-order", 50]];
}

function workbenchConfig() {
  const dim = environmentDimension();
  const manifold = ["stiefel", "grassmann"].includes(landscapeType);
  const initialTheta = parseVectorInput(els.initialTheta.value, dim);
  const initialX = parseVectorInput(els.runnerInitialX.value, dim);
  const gymInitialX = parseVectorInput(els.gymInitialX.value, dim);
  const environment = {
    dim,
    x_bounds: manifold ? null : [Number(els.boundLow.value), Number(els.boundHigh.value)],
    landscape: landscapeConfig(),
    drift: driftConfig(dim),
  };
  if (initialTheta) environment.initial_theta = initialTheta;
  const oracle = {
    type: oracleType,
    blind_value: els.blindValue.checked,
    value_noise: noiseConfig("value", els.valueNoiseType.value),
    grad_noise: noiseConfig("grad", els.gradNoiseType.value),
  };
  if (oracleType === "scheduled") oracle.schedule = scheduleConfig();
  if (oracleType === "offline") oracle.recorded_path = els.offlinePath.value.trim();
  return {
    environment,
    oracle,
    optimizers: selectedOptimizerConfigs(),
    runner: {
      steps: Math.max(1, Math.round(Number(els.runnerSteps.value))),
      seeds: parseSeeds().length ? parseSeeds() : [42],
      output_dir: els.runnerOutput.value.trim() || "results/workbench",
      tail_fraction: Number(els.tailFraction.value),
      record_trajectory: els.recordTrajectory.checked,
      export_csv: els.exportCsv.checked,
      normalize_regret: els.normalizeRegret.checked,
      tracking_norm: metricParameterState.tracking_norm,
      normalize_tracking: metricParameterState.normalize_tracking,
      jump_threshold: Number(metricParameterState.jump_threshold),
      recovery_epsilon: Number(metricParameterState.recovery_epsilon),
      oracle_ttr: Number(metricParameterState.oracle_ttr),
      rho: Number(metricParameterState.rho),
      initial_x: initialX,
    },
    metrics: selectedMetricIds().length ? selectedMetricIds() : ["tracking_error"],
    gym: {
      horizon: Number(els.gymHorizon.value),
      action_mode: els.gymAction.value,
      reward: els.gymReward.value,
      geometry: els.gymGeometry.value,
      max_step: Number(els.gymMaxStep.value),
      x_bounds: [Number(els.gymBoundLow.value), Number(els.gymBoundHigh.value)],
      x0: gymInitialX,
      terminate_on_diverge: els.gymDiverge.value ? Number(els.gymDiverge.value) : null,
    },
  };
}

function escapeHtml(value) { return value.replace(/[&<>]/g, (char) => ({"&":"&amp;","<":"&lt;",">":"&gt;"})[char]); }
function highlightJson(json) {
  const pattern = /("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*"(?=\s*:))|("(?:\\u[a-fA-F0-9]{4}|\\[^u]|[^\\"])*")|\b(true|false|null)\b|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?/g;
  let result = "", last = 0;
  for (const match of json.matchAll(pattern)) {
    result += escapeHtml(json.slice(last, match.index));
    const token = match[0];
    const className = match[1] ? "json-key" : match[2] ? "json-string" : ["true","false","null"].includes(token) ? "json-null" : "json-number";
    result += `<span class="${className}">${escapeHtml(token)}</span>`;
    last = match.index + token.length;
  }
  return result + escapeHtml(json.slice(last));
}

function updateParamFields(kind) {
  const item = (kind === "landscape" ? LANDSCAPES : DRIFTS).find((entry) => entry.id === (kind === "landscape" ? landscapeType : driftType));
  const labelA = kind === "landscape" ? els.landscapeParamALabel : els.driftParamALabel;
  const labelB = kind === "landscape" ? els.landscapeParamBLabel : els.driftParamBLabel;
  const inputA = kind === "landscape" ? els.landscapeParamA : els.driftParamA;
  const inputB = kind === "landscape" ? els.landscapeParamB : els.driftParamB;
  labelA.textContent = item.a[0]; labelB.textContent = item.b[0]; inputA.value = item.a[1]; inputB.value = item.b[1];
  inputA.disabled = item.a[0] === "Unused"; inputB.disabled = item.b[0] === "Unused";
  if (kind === "landscape") els.environmentDim.disabled = ["stiefel", "grassmann"].includes(landscapeType);
  if (kind === "drift") {
    els.driftParamCField.hidden = !item.c;
    els.driftModeField.hidden = !item.mode;
    if (item.c) { els.driftParamCLabel.textContent = item.c[0]; els.driftParamC.value = item.c[1]; }
  }
}

function updateWorkspace() {
  const config = workbenchConfig();
  currentJson = JSON.stringify(config, null, 2);
  els.workspaceJson.innerHTML = highlightJson(currentJson);
  els.workspaceLines.textContent = `${currentJson.split("\n").length} lines`;
  els.signatureLandscape.textContent = `${LANDSCAPES.find((item) => item.id === landscapeType).name} / ${config.environment.dim}D`;
  els.signatureOracle.textContent = ORACLES.find((item) => item.id === oracleType).name;
  const selectedNames = config.optimizers.map((optimizer) => optimizer.name);
  const optimizerLabel = selectedNames.length ? selectedNames.join(", ") : tr("noOptimizers");
  els.signatureOptimizer.textContent = optimizerLabel;
  els.pipelineEnvironment.textContent = `${LANDSCAPES.find((item) => item.id === landscapeType).name} · ${config.environment.dim}D`;
  els.pipelineOracle.textContent = ORACLES.find((item) => item.id === oracleType).name;
  els.pipelineOptimizer.textContent = optimizerLabel;
  els.pipelineRunner.textContent = `${config.runner.steps} steps`;
  const totalRuns = config.runner.seeds.length * config.optimizers.length;
  renderMathSlot(els.runnerFormula, String.raw`N_{\mathrm{runs}}=${config.optimizers.length}\times${config.runner.seeds.length}=${totalRuns},\qquad \widehat M=\operatorname{aggregate}\!\left(M_t\,\middle|\,t\ge(1-${config.runner.tail_fraction})T\right)`);
  const staticMode = runtimeChecked && !engineAvailable;
  els.runModeLabel.textContent = staticMode ? "STATIC" : totalRuns > 1 ? "BATCH" : "SINGLE";
  els.optimizerSelectionCount.textContent = tr("selectedCount", { count: config.optimizers.length });
  els.launchSummary.textContent = `${tr("optimizerCount", { count: config.optimizers.length })} · ${LANDSCAPES.find((item) => item.id === landscapeType).name} · ${ORACLES.find((item) => item.id === oracleType).name}`;
  els.launchDetail.textContent = `${config.optimizers.length} × ${config.runner.seeds.length} seed${config.runner.seeds.length === 1 ? "" : "s"} × ${config.runner.steps} steps`;
  const scheduleModes = oracleType === "scheduled" ? config.oracle.schedule.map(([mode]) => mode) : [];
  const selectedOptimizers = OPTIMIZERS.filter((optimizer) => selectedOptimizerNames.has(optimizer.name));
  const incompatible = selectedOptimizers.some((optimizer) =>
    (optimizer.order === "first-order" && oracleType === "zero-order") ||
    (optimizer.order === "first-order" && scheduleModes.includes("zero-order")) ||
    (optimizer.order === "zero-order" && oracleType === "first-order" && config.oracle.blind_value));
  const noOptimizers = config.optimizers.length === 0;
  const invalidVector = !vectorInputIsValid(els.initialTheta.value, config.environment.dim) || !vectorInputIsValid(els.runnerInitialX.value, config.environment.dim);
  const csvWithoutTrajectory = config.runner.export_csv && !config.runner.record_trajectory;
  els.runBenchmark.disabled = staticMode
    ? invalidVector
    : noOptimizers || incompatible || invalidVector || csvWithoutTrajectory || Boolean(activeJobId);
  if (staticMode) els.launchDetail.textContent = tr("staticLaunchDetail");
  else if (noOptimizers) els.launchDetail.textContent = tr("chooseOptimizer");
  else if (incompatible) els.launchDetail.textContent = tr("incompatible");
  else if (invalidVector) els.launchDetail.textContent = tr("invalidVector", { dim: config.environment.dim });
  else if (csvWithoutTrajectory) els.launchDetail.textContent = tr("csvNeedsTrajectory");
  updateGymCode(config);
  try { localStorage.setItem("wind-workbench-view", activeView); } catch { /* optional */ }
}

function updateGymCode(config) {
  const gym = config.gym;
  const diverge = gym.terminate_on_diverge === null ? "None" : gym.terminate_on_diverge;
  const x0 = gym.x0 === null ? "None" : `np.array(${JSON.stringify(gym.x0)}, dtype=float)`;
  const transition = gym.action_mode === "absolute"
    ? String.raw`x_{t+1}=\Pi_{\mathcal G}(a_t)`
    : String.raw`x_{t+1}=\Pi_{\mathcal G}\!\left(x_t+\operatorname{clip}(a_t,-\Delta_{\max},\Delta_{\max})\right)`;
  const reward = gym.reward === "neg_error"
    ? String.raw`r_t=-d_{\mathcal G}(x_t,\theta_t)`
    : String.raw`r_t=-\left[f_t(x_t)-f_t(\theta_t)\right]`;
  renderMathSlot(els.gymFormula, `${transition}\\qquad ${reward}`);
  els.gymCode.textContent = `import numpy as np\n\nfrom wind_benchmark import make_environment\nfrom wind_benchmark.oracle import ${oracleType === "zero-order" ? "ZeroOrderOracle" : "FirstOrderOracle"}\nfrom wind_benchmark.gym_env import WindGymEnv\n\nenvironment = make_environment(environment_config, seed=42)\noracle = ${oracleType === "zero-order" ? "ZeroOrderOracle" : "FirstOrderOracle"}(environment)\nenv = WindGymEnv(\n    environment,\n    oracle=oracle,\n    T=${gym.horizon},\n    action_mode="${gym.action_mode}",\n    reward="${gym.reward}",\n    geometry="${gym.geometry}",\n    x_bounds=(${gym.x_bounds[0]}, ${gym.x_bounds[1]}),\n    max_step=${gym.max_step},\n    x0=${x0},\n    terminate_on_diverge=${diverge},\n)`;
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => els.toast.classList.remove("visible"), 2100);
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); }
  catch {
    const area = document.createElement("textarea"); area.value = text; area.style.position = "fixed"; area.style.opacity = "0"; document.body.appendChild(area); area.select(); document.execCommand("copy"); area.remove();
  }
  showToast(tr("copied"));
}

function downloadText(filename, text, type = "application/json") {
  const blob = new Blob([text], { type: `${type};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadConfiguration() {
  downloadText("wind-configuration.json", currentJson || JSON.stringify(workbenchConfig(), null, 2));
  showToast(tr("configurationDownloaded"));
}

function updateRuntimeUi() {
  if (!els.engineStatus || !els.runBenchmark) return;
  const staticMode = runtimeChecked && !engineAvailable;
  els.engineStatus.classList.toggle("offline", !engineAvailable);
  els.engineStatus.classList.toggle("static", staticMode);
  if (staticMode) els.engineLabel.textContent = tr("staticMode");
  else if (engineAvailable && engineVersion) els.engineLabel.textContent = tr("connected", { version: engineVersion });
  const actionLabel = els.runBenchmark.querySelector("span");
  if (actionLabel) actionLabel.textContent = staticMode ? tr("downloadConfiguration") : tr("runBenchmark");
  els.refreshResults.disabled = staticMode;
  els.refreshResults.hidden = staticMode;
}

async function checkApi() {
  if (location.protocol === "file:" || /\.github\.io$/i.test(location.hostname)) {
    engineAvailable = false;
    runtimeChecked = true;
    updateRuntimeUi();
    updateWorkspace();
    return;
  }
  try {
    const response = await fetch("/api/status", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const payload = await response.json();
    engineAvailable = true;
    engineVersion = payload.version;
    els.engineLabel.textContent = tr("connected", { version: payload.version });
    if (payload.job && ["starting","running"].includes(payload.job.status)) {
      renderJob(payload.job); pollJob(payload.job.id);
    }
  } catch {
    engineAvailable = false;
    engineVersion = null;
    els.engineLabel.textContent = tr("staticMode");
  } finally {
    runtimeChecked = true;
    updateRuntimeUi();
    updateWorkspace();
  }
}

async function runBenchmark() {
  if (runtimeChecked && !engineAvailable) {
    downloadConfiguration();
    return;
  }
  const config = workbenchConfig();
  els.runBenchmark.disabled = true;
  try {
    const response = await fetch("/api/benchmark/run", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(config) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Unable to start benchmark");
    activeJobId = payload.job.id;
    renderJob(payload.job);
    pollJob(activeJobId);
    showToast(tr("runStarted"));
  } catch (error) { activeJobId = null; updateWorkspace(); showToast(error.message); }
}

function renderJob(job) {
  const running = ["starting","running"].includes(job.status);
  activeJobId = running ? job.id : null;
  els.jobPanel.hidden = false;
  els.jobState.textContent = job.status.toUpperCase();
  els.jobPercent.textContent = `${Math.round(job.progress || 0)}%`;
  els.jobProgress.style.width = `${job.progress || 0}%`;
  els.jobLog.textContent = (job.logs || []).slice(-4).join("\n") || "Preparing benchmark…";
  els.cancelJob.disabled = !running;
  updateWorkspace();
  if (!running) {
    clearTimeout(pollTimer);
    showToast(job.status === "completed" ? tr("runCompleted") : job.status === "cancelled" ? tr("runCancelled") : tr("runFailed"));
    loadResults();
  }
}

async function pollJob(jobId) {
  try {
    const response = await fetch(`/api/jobs/${jobId}`, { cache:"no-store" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error);
    renderJob(payload.job);
    if (["starting","running"].includes(payload.job.status)) pollTimer = setTimeout(() => pollJob(jobId), 850);
  } catch (error) { showToast(error.message); pollTimer = setTimeout(() => pollJob(jobId), 1800); }
}

function formatBytes(bytes) {
  if (bytes > 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
  if (bytes > 1000) return `${(bytes / 1000).toFixed(1)} KB`;
  return `${bytes} B`;
}

async function loadAnalysisFile(event) {
  const file = event.target.files?.[0];
  if (!file) return;

  try {
    const extension = file.name.split(".").pop()?.toLowerCase();
    const text = await file.text();
    let details;

    if (extension === "json") {
      const data = JSON.parse(text);
      if (data === null || typeof data !== "object") throw new Error();
      const steps = data.metadata?.total_steps;
      details = `JSON · ${formatBytes(file.size)}${Number.isInteger(steps) ? ` · ${steps} steps` : ""}`;
    } else if (extension === "csv") {
      const rows = text.split(/\r?\n/).filter((row) => row.trim()).length - 1;
      if (rows < 1) throw new Error();
      details = `CSV · ${formatBytes(file.size)} · ${rows} rows`;
    } else {
      throw new Error();
    }

    analysisDataLoaded = true;
    els.analysisNav.hidden = false;
    els.analysisSourceName.textContent = file.name;
    els.analysisSourceMeta.textContent = details;
    setView("analysis");
    showToast(tr("analysisLoaded"));
  } catch {
    event.target.value = "";
    showToast(tr("invalidAnalysisFile"));
  }
}

async function loadResults() {
  els.resultsList.innerHTML = `<div class="empty-state">${tr("loadingResults")}</div>`;
  if (runtimeChecked && !engineAvailable) {
    els.resultsList.innerHTML = `<div class="empty-state">${tr("staticResults")}</div>`;
    return;
  }
  try {
    const response = await fetch("/api/results?limit=40", { cache:"no-store" });
    const payload = await response.json();
    if (!payload.results?.length) { els.resultsList.innerHTML = `<div class="empty-state">${tr("noResults")}</div>`; return; }
    els.resultsList.innerHTML = payload.results.map((result) => `<article class="result-row"><i>${result.kind === "summary" ? "SUM" : "JSON"}</i><span><b title="${result.path}">${result.name}</b><small>${result.optimizer || result.path.replace(result.name,"")} ${result.steps ? `· ${result.steps} steps` : ""}</small></span><span>${formatBytes(result.size)}</span></article>`).join("");
  } catch { els.resultsList.innerHTML = `<div class="empty-state">${tr("noResults")}</div>`; }
}

function bindElements() {
  const ids = ["engine-status","engine-label","landscape-grid","landscape-fields","drift-grid","drift-fields","environment-dim","landscape-param-a-label","landscape-param-a","landscape-param-b-label","landscape-param-b","initial-theta","drift-param-a-label","drift-param-a","drift-param-b-label","drift-param-b","drift-param-c-field","drift-param-c-label","drift-param-c","drift-mode-field","drift-mode","bound-low","bound-high","oracle-grid","oracle-extra","blind-value","schedule-field","oracle-schedule","offline-field","offline-path","value-noise-type","value-noise-params","grad-noise-type","grad-noise-params","optimizer-search","optimizer-catalog","optimizer-selection-count","clear-optimizers","runner-steps","runner-seeds","runner-output","runner-initial-x","tail-fraction","record-trajectory","export-csv","normalize-regret","runner-formula","run-mode-label","metrics-grid","launch-summary","launch-detail","run-benchmark","job-panel","job-state","job-percent","job-progress","job-log","cancel-job","results-list","refresh-results","analysis-nav","analysis-file","analysis-source-name","analysis-source-meta","gym-horizon","gym-action","gym-reward","gym-geometry","gym-max-step","gym-diverge","gym-bound-low","gym-bound-high","gym-initial-x","gym-formula","gym-code","copy-gym-code","workspace-json","workspace-lines","signature-landscape","signature-oracle","signature-optimizer","pipeline-environment","pipeline-oracle","pipeline-optimizer","pipeline-runner","download-workspace-json","copy-workspace-json","toast"];
  ids.forEach((id) => { const key = id.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase()); els[key] = document.getElementById(id); });
}

function applyScenario(id) {
  const scenario = (window.WIND_SCENARIOS ?? []).find((item) => item.id === id);
  if (!scenario) return;
  landscapeType = scenario.landscape;
  driftType = scenario.drift;
  oracleType = scenario.oracle;
  selectedOptimizerNames.clear();
  selectedOptimizerNames.add(scenario.optimizer);
  inspectedOptimizerName = scenario.optimizer;
  renderChoices(els.landscapeGrid, LANDSCAPES, landscapeType, "choice-card", els.landscapeFields);
  renderChoices(els.driftGrid, DRIFTS, driftType, "choice-card", els.driftFields);
  updateParamFields("landscape");
  updateParamFields("drift");
  renderOracleChoices();
  els.valueNoiseType.value = scenario.valueNoise;
  els.gradNoiseType.value = scenario.gradNoise;
  renderNoiseParams("value");
  renderNoiseParams("grad");
  renderOptimizerCatalog();
  renderMetrics(new Set(["tracking_error", "instant_loss", "dynamic_regret", "drift_adaptation", "query_efficiency"]));
  updateWorkspace();
  showToast(tr("scenarioApplied"));
}

function bindEvents() {
  document.addEventListener("click", (event) => {
    const viewTarget = event.target.closest("[data-view-link],[data-open-view]");
    if (viewTarget) { event.preventDefault(); setView(viewTarget.dataset.viewLink || viewTarget.dataset.openView); }
    const landscape = event.target.closest("#landscape-grid [data-choice]");
    if (landscape) { landscapeType = landscape.dataset.choice; renderChoices(els.landscapeGrid,LANDSCAPES,landscapeType,"choice-card",els.landscapeFields); updateParamFields("landscape"); renderMetrics(new Set(selectedMetricIds())); updateWorkspace(); }
    const drift = event.target.closest("#drift-grid [data-choice]");
    if (drift) { driftType = drift.dataset.choice; renderChoices(els.driftGrid,DRIFTS,driftType,"choice-card",els.driftFields); updateParamFields("drift"); updateWorkspace(); }
    const oracle = event.target.closest("[data-oracle]");
    if (oracle) { oracleType = oracle.dataset.oracle; renderOracleChoices(); updateWorkspace(); }
    const optimizer = event.target.closest("[data-optimizer]");
    if (optimizer) {
      inspectedOptimizerName = optimizer.dataset.optimizer;
      if (selectedOptimizerNames.has(inspectedOptimizerName)) selectedOptimizerNames.delete(inspectedOptimizerName);
      else selectedOptimizerNames.add(inspectedOptimizerName);
      renderOptimizerCatalog(); updateWorkspace();
    }
    const filter = event.target.closest("[data-order-filter]");
    if (filter) { orderFilter = filter.dataset.orderFilter; document.querySelectorAll("[data-order-filter]").forEach((button) => button.classList.toggle("active",button === filter)); renderOptimizerCatalog(); }
    const languageButton = event.target.closest("[data-lang]");
    if (languageButton) { language = languageButton.dataset.lang; try { localStorage.setItem("wind-language-v2",language); } catch {} renderStaticTranslations(); updateWorkspace(); }
    const guideCopy = event.target.closest("[data-copy-guide-code]");
    if (guideCopy) copyText(guideCopy.closest(".guide-code")?.querySelector("code")?.textContent ?? "");
    const guideToggle = event.target.closest("[data-guide-toggle]");
    if (guideToggle) {
      const page = guideToggle.closest("[data-guide-page]")?.dataset.guidePage;
      if (page) {
        const opening = collapsedGuidePages.has(page);
        if (opening) collapsedGuidePages.delete(page);
        else collapsedGuidePages.add(page);
        renderGuides(opening ? page : null);
      }
    }
    const scenario = event.target.closest("[data-scenario]");
    if (scenario) applyScenario(scenario.dataset.scenario);
  });
  document.addEventListener("input", (event) => {
    if (event.target.matches("[data-optimizer-param]")) optimizerParamsByName[event.target.dataset.optimizerName][event.target.dataset.optimizerParam] = Number(event.target.value);
    if (event.target.matches("[data-noise-param]")) noiseParamsByChannel[event.target.dataset.noiseChannel][els[`${event.target.dataset.noiseChannel}NoiseType`].value][event.target.dataset.noiseParam] = Number(event.target.value);
    if (event.target.matches("[data-metric-param]")) metricParameterState[event.target.dataset.metricParam] = event.target.type === "checkbox" ? event.target.checked : event.target.value;
    if (event.target.matches('[data-runner-param="normalize_regret"]')) els.normalizeRegret.checked = event.target.checked;
    updateWorkspace();
  });
  document.addEventListener("change", (event) => {
    if (event.target === els.valueNoiseType) renderNoiseParams("value");
    if (event.target === els.gradNoiseType) renderNoiseParams("grad");
    if (event.target === els.blindValue) renderOracleChoices();
    if (event.target === els.driftMode) renderChoices(els.driftGrid, DRIFTS, driftType, "choice-card", els.driftFields);
    if (event.target.matches("[data-metric-select]")) renderMetrics(new Set(selectedMetricIds()));
    if (event.target.matches("[data-metric-param]")) {
      const name = event.target.dataset.metricParam;
      metricParameterState[name] = event.target.type === "checkbox" ? event.target.checked : event.target.value;
      els.metricsGrid.querySelectorAll(`[data-metric-param="${name}"]`).forEach((input) => {
        if (input !== event.target) input.type === "checkbox" ? input.checked = Boolean(metricParameterState[name]) : input.value = metricParameterState[name];
      });
      if (["tracking_norm", "normalize_tracking"].includes(name)) renderMetrics(new Set(selectedMetricIds()));
    }
    if (event.target === els.normalizeRegret || event.target.matches('[data-runner-param="normalize_regret"]')) {
      const checked = event.target.checked;
      els.normalizeRegret.checked = checked;
      els.metricsGrid.querySelectorAll('[data-runner-param="normalize_regret"]').forEach((input) => { input.checked = checked; });
      renderMetrics(new Set(selectedMetricIds()));
    }
    updateWorkspace();
  });
  els.optimizerSearch.addEventListener("input", renderOptimizerCatalog);
  els.clearOptimizers.addEventListener("click", () => { selectedOptimizerNames.clear(); renderOptimizerCatalog(); updateWorkspace(); });
  document.getElementById("select-core-metrics").addEventListener("click", () => { renderMetrics(new Set(["tracking_error","instant_loss","dynamic_regret","drift_adaptation","query_efficiency"])); updateWorkspace(); });
  els.runBenchmark.addEventListener("click", runBenchmark);
  els.cancelJob.addEventListener("click", async () => { if (!activeJobId) return; const response = await fetch(`/api/jobs/${activeJobId}/cancel`,{method:"POST"}); const payload = await response.json(); if (response.ok) renderJob(payload.job); });
  els.refreshResults.addEventListener("click", loadResults);
  els.analysisFile.addEventListener("change", loadAnalysisFile);
  els.downloadWorkspaceJson.addEventListener("click", downloadConfiguration);
  els.copyWorkspaceJson.addEventListener("click", () => copyText(currentJson));
  els.copyGymCode.addEventListener("click", () => copyText(els.gymCode.textContent));
}

function init() {
  bindElements();
  try { const savedLanguage = localStorage.getItem("wind-language-v2"); if (I18N[savedLanguage]) language = savedLanguage; } catch {}
  renderStaticTranslations();
  renderChoices(els.landscapeGrid, LANDSCAPES, landscapeType, "choice-card", els.landscapeFields);
  renderChoices(els.driftGrid, DRIFTS, driftType, "choice-card", els.driftFields);
  renderOracleChoices();
  renderNoiseOptions();
  renderOptimizerCatalog();
  renderMetrics();
  updateParamFields("landscape");
  updateParamFields("drift");
  bindEvents();
  const hashView = location.hash.replace("#", "");
  if (["overview","environment","oracle","optimizers","runner","results","analysis","gym"].includes(hashView)) setView(hashView);
  updateWorkspace();
  checkApi();
}

document.addEventListener("DOMContentLoaded", init);
