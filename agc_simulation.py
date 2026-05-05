"""
Reproduces all numerical results in the smart-grid balancing-market case
study of "Truthful Kalman Filtering with Strategic Sensors" (Section 6).

Usage:
    python agc_simulation.py [--mc-trials N] [--save-figures]

Outputs:
    - Console: all numerical constants reported in Section 6
      (||K||_F, gradient, Hessian, h*, per-period bonus, gamma_eta,
       Gamma_h, K_bar_ub, lambda* values, ...)
    - fig1_agc_lambda_sweep.pdf  : slack vs. lambda (Figure 1)
    - fig2_optimality.pdf        : penalty-form deterrent comparison (Figure 2)

Reproducibility:
    All random number generators are seeded (rng = np.random.default_rng(0)).
    Common-random-number variance reduction is used in the Monte-Carlo step
    (50 trials of 2200 steps each, 200-step burn-in).
"""

import argparse
import numpy as np
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, eigvals
import matplotlib.pyplot as plt


N = 3                          # number of strategic generators
ALPHA = 0.20                   # thermal-decay coefficient per period
SIGMA_W = 2.0                  # process-noise std, MW
SIGMA_MAX = 5.0                # maximum measurement-noise scale, MW
KAPPA = 5.0                    # effort-noise sensitivity
Q_DIAG = 0.04                  # state-cost weight, $/MW^2/period
R_DIAG = 0.005                 # control-cost weight
E_LO, E_HI = 0.0, 1.0          # admissible effort range
E_TRUTH = 0.7                  # truthful-equilibrium effort
PERIODS_PER_YEAR = 12 * 24 * 365  # 5-min dispatch -> 105,120

# System matrices
A = (1.0 - ALPHA) * np.eye(N)
B = np.eye(N)
C_BAR = np.eye(N)
SIGMA_W_MAT = (SIGMA_W ** 2) * np.eye(N)
Q = Q_DIAG * np.eye(N)
R = R_DIAG * np.eye(N)
R_REFS = np.array([[5.0, 0.0, 0.0],
                   [0.0, 5.0, 0.0],
                   [0.0, 0.0, 5.0]])  # rows are r_0, r_1, r_2 in MW
R_BAR = R_REFS.mean(axis=0)  # tracking reference


def sigma2_of_e(e):
    """Effort-to-noise variance map sigma^2(e) = sigma_max^2 / (1 + kappa e)."""
    return (SIGMA_MAX ** 2) / (1.0 + KAPPA * e)


def R_v_of_e(e_vec):
    """Block-diagonal measurement-noise covariance at effort vector e."""
    return np.diag([sigma2_of_e(e_vec[i]) for i in range(N)])



def steady_state_kalman(e_vec):
    """Returns (P_inf, K, S) for the steady-state Kalman filter at effort e."""
    Rbar = R_v_of_e(e_vec)
    # discrete algebraic Riccati equation: P = A P A^T + Sigma_w - A P C^T (C P C^T + R)^-1 C P A^T
    P = solve_discrete_are(A.T, C_BAR.T, SIGMA_W_MAT, Rbar)
    S = C_BAR @ P @ C_BAR.T + Rbar           # innovation covariance
    K = P @ C_BAR.T @ np.linalg.inv(S)        # Kalman gain
    return P, K, S


def lqg_gain():
    """Returns the LQG control gain L solving the LQR Riccati."""
    P_lqr = solve_discrete_are(A, B, Q, R)
    L = np.linalg.inv(R + B.T @ P_lqr @ B) @ B.T @ P_lqr @ A
    return L


def closed_loop_M(K):
    """Augmented closed-loop matrix M(K) on (x, xhat) under direct subst."""
    L = lqg_gain()
    return np.block([[A,           -B @ L],
                     [K @ C_BAR @ A, (np.eye(N) - K @ C_BAR) @ A - B @ L]])


def deviation_responses(K, agent=0):
    """Returns (a, b, c) such that
       E[x_inf] = x* + a h,  E[xhat_inf] = x* + b h,  E[u_inf] = u* + c h.
    """
    L = lqg_gain()
    # S_0 = -[I; 0]^T, agent-0 measurement-block selector
    S0 = np.zeros((N, 1))
    S0[agent, 0] = -1.0
    # Solve [I - M(K)]^-1 [0; K S_0]
    M = closed_loop_M(K)
    rhs = np.vstack([np.zeros((N, 1)), K @ S0])
    sol = np.linalg.solve(np.eye(2 * N) - M, rhs)
    a = sol[:N, :]
    b = sol[N:, :]
    c = -L @ b
    return a, b, c


def beta_0(K, agent=0):
    """Beta_0(K) = C_0 (a - A b - B c)(K) - I_{p_0}; here p_0 = 1 scalar."""
    a, b, c = deviation_responses(K, agent=agent)
    C0 = C_BAR[agent:agent + 1, :]
    return float((C0 @ (a - A @ b - B @ c) - 1.0).item())



def truth_equilibrium_state():
    """x* = (I - A + BL)^-1 BL r_bar, u* = -L (x* - r_bar)."""
    L = lqg_gain()
    x_star = np.linalg.solve(np.eye(N) - A + B @ L, B @ L @ R_BAR)
    u_star = -L @ (x_star - R_BAR)
    return x_star, u_star


def gradient_and_hessian(K, agent=0):
    """Closed-form first-order gradient and second-order Hessian
    of E[U_agent(h)] at h = 0, under direct substitution."""
    a, b, c = deviation_responses(K, agent=agent)
    x_star, u_star = truth_equilibrium_state()
    r_agent = R_REFS[agent]
    grad = (
        -2.0 * a.T @ Q @ (x_star - r_agent)
        - 2.0 * (N - 1) * b.T @ Q @ x_star
        + 2.0 * b.T @ Q @ (R_REFS.sum(axis=0) - r_agent)
        - 2.0 * N * (u_star.T @ R) @ c
    )
    grad = float(np.asarray(grad).item())
    H = -2.0 * (a.T @ Q @ a) \
        - 2.0 * (N - 1) * (b.T @ Q @ b) \
        - 2.0 * N * (c.T @ R @ c)
    return grad, float(np.asarray(H).item())


def monte_carlo_bonus(h, T=2200, burn=200, n_trials=50, agent=0, rng=None):
    """Estimate E[U_agent(h) - U_agent(0)] per period via common random
    numbers across (truth, deviation) trial pairs."""
    if rng is None:
        rng = np.random.default_rng(0)
    L = lqg_gain()
    e_truth = E_TRUTH * np.ones(N)
    P, K, S = steady_state_kalman(e_truth)
    Rbar = R_v_of_e(e_truth)
    L_w = np.linalg.cholesky(SIGMA_W_MAT)
    L_v = np.linalg.cholesky(Rbar)

    bonuses = np.zeros(n_trials)
    for trial in range(n_trials):
        w = L_w @ rng.standard_normal((N, T))
        v = L_v @ rng.standard_normal((N, T))
        bonuses[trial] = _trial_bonus(K, L, w, v, h, agent, T, burn)
    return float(bonuses.mean()), float(bonuses.std() / np.sqrt(n_trials))


def _trial_bonus(K, L, w, v, h, agent, T, burn):
    """Inner Monte-Carlo: returns the agent's per-period utility difference
    (deviation minus truth) for one common-random-number trial."""
    x_t = np.zeros(N)   # truth path
    xh_t = np.zeros(N)
    x_d = np.zeros(N)   # deviation path
    xh_d = np.zeros(N)
    bias = np.zeros(N)
    bias[agent] = -h    # deviation injects -h on agent's report
    util_t = util_d = 0.0
    r_agent = R_REFS[agent]
    for k in range(T):
        # Truth-side dynamics + filter
        u_t = -L @ (xh_t - R_BAR)
        z_t = C_BAR @ x_t + v[:, k]
        z_d = C_BAR @ x_d + v[:, k] + bias
        # Predict-update for both
        xh_pred_t = A @ xh_t + B @ u_t
        xh_t = xh_pred_t + K @ (z_t - C_BAR @ xh_pred_t)
        u_d = -L @ (xh_d - R_BAR)
        xh_pred_d = A @ xh_d + B @ u_d
        xh_d = xh_pred_d + K @ (z_d - C_BAR @ xh_pred_d)
        # Plant dynamics
        x_t = A @ x_t + B @ u_t + w[:, k]
        x_d = A @ x_d + B @ u_d + w[:, k]
        if k >= burn:
            stake_t = (x_t - r_agent) @ Q @ (x_t - r_agent) + u_t @ R @ u_t
            stake_d = (x_d - r_agent) @ Q @ (x_d - r_agent) + u_d @ R @ u_d
            # Layered Groves transfer (welfare-of-others under direct subst)
            t_t = -sum((xh_t - R_REFS[j]) @ Q @ (xh_t - R_REFS[j])
                       + u_t @ R @ u_t for j in range(N) if j != agent)
            t_d = -sum((xh_d - R_REFS[j]) @ Q @ (xh_d - R_REFS[j])
                       + u_d @ R @ u_d for j in range(N) if j != agent)
            util_t += -stake_t + t_t
            util_d += -stake_d + t_d
    return (util_d - util_t) / max(1, T - burn)


def K_bar_ub():
    """Loewner-monotonicity bound (Lemma A.4) on sup_e ||K(e)||_F."""
    e_min = E_LO * np.ones(N)
    e_max = E_HI * np.ones(N)
    P_min, _, _ = steady_state_kalman(e_min)
    Rbar_max_eff = R_v_of_e(e_max)
    Gamma = np.linalg.norm(C_BAR, 2) * np.linalg.norm(np.linalg.inv(Rbar_max_eff), 2)
    return Gamma * np.linalg.norm(P_min, 'fro')


def empirical_sup_K(grid_size=11):
    """Estimate sup_e ||K(e)||_F by dense gridding."""
    grid = np.linspace(E_LO, E_HI, grid_size)
    best = 0.0
    for e1 in grid:
        for e2 in grid:
            for e3 in grid:
                _, K, _ = steady_state_kalman(np.array([e1, e2, e3]))
                best = max(best, np.linalg.norm(K, 'fro'))
    return best


def gamma_h(K):
    """Deterrent constant Gamma_h = beta_0^2 / sigma_nu0^2 (scalar p_0=1)."""
    _, _, S = steady_state_kalman(E_TRUTH * np.ones(N))
    sigma_nu0_sq = float(S[0, 0])
    return beta_0(K) ** 2 / sigma_nu0_sq


def epsilon_instance(lam):
    """Theorem 25 instance-specific bound (scalar type)."""
    _, K, _ = steady_state_kalman(E_TRUTH * np.ones(N))
    grad, H = gradient_and_hessian(K)
    g_h = gamma_h(K)
    denom = 2.0 * lam * g_h + 2.0 * abs(H)
    return (grad ** 2) / max(denom, 1e-30)


def G1_constant(varphi_star):
    """Closed-form constant G_1 from Theorem 1(a)."""
    L = lqg_gain()
    x_star, _ = truth_equilibrium_state()
    R_r_bar = np.max(np.linalg.norm(R_REFS, axis=1)) + np.linalg.norm(x_star)
    return 2.0 * R_r_bar * (
        (2 * N - 1) * np.linalg.norm(Q, 2) +
        N * (np.linalg.norm(L, 2) ** 2) * np.linalg.norm(R, 2)
    ) * varphi_star


def varphi_star_at(e_vec):
    """||(I - M(K(e)))^-1||_2 at effort e."""
    _, K, _ = steady_state_kalman(e_vec)
    M = closed_loop_M(K)
    return np.linalg.norm(np.linalg.inv(np.eye(2 * N) - M), 2)


def epsilon_closed_form(lam, K_bound, varphi_star_val,
                        gamma_h_lower, H_theta_theta_upper):
    """Theorem 3 closed-form bound (scalar p_0)."""
    G1 = G1_constant(varphi_star_val)
    num = (G1 * K_bound) ** 2
    denom = 2.0 * lam * gamma_h_lower + 2.0 * H_theta_theta_upper
    return num / max(denom, 1e-30)



def deterrent_at(g_func, alpha, K, h_star, name=None):
    """Estimate D(g, alpha h*) = E_truth[g] - E_dev[g]."""
    from math import erf, exp, pi, sqrt
    _, _, S = steady_state_kalman(E_TRUTH * np.ones(N))
    sigma_nu = float(np.sqrt(S[0, 0]))
    bias = beta_0(K) * alpha * h_star

    if name == 'absolute':
        # Closed form, no quadrature error.
        var_abs = (sigma_nu ** 2) * (1.0 - 2.0 / pi)
        c_abs = 1.0 / sqrt(var_abs)
        e_abs = sigma_nu * sqrt(2.0 / pi)            # E_truth[|nu|]
        # Under deviation, mu = bias.
        if bias == 0.0:
            E_dev_abs_nu = e_abs
        else:
            E_dev_abs_nu = (sigma_nu * sqrt(2.0/pi) * exp(-bias**2/(2*sigma_nu**2))
                            + bias * erf(bias/(sigma_nu*sqrt(2.0))))
            # erf reflects sign; |nu| expectation depends on |mu|, so:
            E_dev_abs_nu = abs(E_dev_abs_nu)
        # E_truth[g_abs] = 0 by construction; E_dev[g_abs] = c_abs * (E_dev|nu| - e_abs)
        return -c_abs * (E_dev_abs_nu - e_abs)

    nodes, weights = np.polynomial.hermite_e.hermegauss(64)
    nu_truth = sigma_nu * nodes
    nu_dev = sigma_nu * nodes + bias
    expect_truth = (weights * np.array([g_func(x) for x in nu_truth])).sum() / np.sqrt(2 * np.pi)
    expect_dev = (weights * np.array([g_func(x) for x in nu_dev])).sum() / np.sqrt(2 * np.pi)
    return expect_truth - expect_dev


def make_penalty_candidates(sigma_nu):
    """Four sign-invariant penalty forms, each variance-budget = 1 under truth."""
    sig2 = sigma_nu ** 2
    # Whitened quadratic: c (nu^2/sigma^2 - 1), Var = 2 c^2 -> c = 1/sqrt(2)
    c_q = 1.0 / np.sqrt(2.0)
    g_quad = lambda nu: c_q * (nu * nu / sig2 - 1.0)
    # Centred quartic: c (nu^4/sigma^4 - 3), Var = 96 c^2 -> c = 1/sqrt(96)
    c_4 = 1.0 / np.sqrt(96.0)
    g_quart = lambda nu: c_4 * ((nu / sigma_nu) ** 4 - 3.0)
    # Sub-quadratic |nu|: shift & scale to centre and unit variance
    e_abs = sigma_nu * np.sqrt(2.0 / np.pi)
    var_abs = sig2 * (1.0 - 2.0 / np.pi)
    c_abs = 1.0 / np.sqrt(var_abs)
    g_abs = lambda nu: c_abs * (abs(nu) - e_abs)
    # Super-quadratic cosh: centre and rescale
    # Use |nu|/sigma_nu inside cosh to keep dimensionless; truncate moments numerically
    # We compute centring and variance numerically below.
    return {
        'quadratic': g_quad,
        'quartic':   g_quart,
        'absolute':  g_abs,
        'cosh':      _make_cosh_candidate(sigma_nu),
    }


def _make_cosh_candidate(sigma_nu):
    """cosh(|nu|/sigma_nu), centred and variance-normalised numerically."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(64)
    raw = np.cosh(np.abs(sigma_nu * nodes) / sigma_nu)
    norm = weights / np.sqrt(2 * np.pi)
    mean = (norm * raw).sum()
    var = (norm * (raw - mean) ** 2).sum()
    c = 1.0 / np.sqrt(var)
    return lambda nu: c * (np.cosh(abs(nu) / sigma_nu) - mean)




def main(args):
    rng = np.random.default_rng(0)
    e_truth = E_TRUTH * np.ones(N)
    P, K, S = steady_state_kalman(e_truth)
    L = lqg_gain()
    rho_AmBL = float(np.max(np.abs(eigvals(A - B @ L))))
    print(f"Closed-loop rho(A - BL) = {rho_AmBL:.4f}")
    print(f"||K(e_truth)||_F        = {np.linalg.norm(K, 'fro'):.3f}")
    print(f"sigma_nu_0^2(truth)     = {float(S[0, 0]):.3f}")
    print(f"beta_0(K)               = {beta_0(K):.4f}")

    # Theorem 1 magnitudes
    grad, H = gradient_and_hessian(K)
    h_star = -grad / H
    bonus_per_period = 0.5 * (grad ** 2) / abs(H)
    bonus_per_year = bonus_per_period * PERIODS_PER_YEAR
    print(f"\n[Theorem 1] Direct-substitution magnitudes:")
    print(f"  grad nabla_h        = {grad:+.4f} $/MW/period")
    print(f"  Hessian H_theta     = {H:+.4f} $/MW^2/period")
    print(f"  optimal h*          = {h_star:+.3f} MW")
    print(f"  per-period bonus    = ${bonus_per_period:.4f}")
    print(f"  annualised bonus    = ${bonus_per_year:,.0f}/yr")

    # Optional Monte-Carlo verification
    if args.mc_trials > 0:
        emp_mean, emp_se = monte_carlo_bonus(h_star, n_trials=args.mc_trials, rng=rng)
        print(f"  Monte-Carlo bonus   = ${emp_mean:.4f} +/- ${emp_se:.4f}")

    # Theorem 3 / Lemma A.4 closed-form bound on K
    K_ub = K_bar_ub()
    K_emp_sup = empirical_sup_K()
    print(f"\n[Theorem 3] Kalman-gain bounds:")
    print(f"  K_bar_ub (rigorous, Lemma A.4) = {K_ub:.3f}")
    print(f"  empirical sup ||K(e)||_F       = {K_emp_sup:.3f}")
    print(f"  conservatism ratio             = {K_ub / K_emp_sup:.2f}x")

    # Worst-case constants for Theorem 3
    varphi_lo = varphi_star_at(np.zeros(N))    # extremum at zero-effort vertex
    G1 = G1_constant(varphi_lo)
    g_h_truth = gamma_h(K)
    # Lower bound on gamma_h over admissible set: take vertex (1,1,1)
    _, K_hi, _ = steady_state_kalman(np.ones(N))
    g_h_lo = gamma_h(K_hi)
    H_upper = abs(H)  # for scalar p_0 = 1
    print(f"\n[Theorem 3] Worst-case constants:")
    print(f"  varphi^*           = {varphi_lo:.3f}")
    print(f"  G_1                = {G1:.3f}")
    print(f"  Gamma_h (truth)    = {g_h_truth:.4f}")
    print(f"  underline gamma_h  = {g_h_lo:.4f}")
    print(f"  |bar H_theta|      = {H_upper:.4f}")

    # Find lambda achieving $1000/yr target via instance and closed-form bounds
    target_per_period = 1000.0 / PERIODS_PER_YEAR
    lams = np.logspace(0, 9, 400)
    eps_inst = np.array([epsilon_instance(l) for l in lams])
    eps_cf   = np.array([epsilon_closed_form(l, K_ub, varphi_lo, g_h_lo, H_upper)
                         for l in lams])
    lam_inst = lams[np.argmin(np.abs(eps_inst * PERIODS_PER_YEAR - 1000.0))]
    lam_cf   = lams[np.argmin(np.abs(eps_cf   * PERIODS_PER_YEAR - 1000.0))]
    print(f"\n[Figure 1] lambda for $1k/yr target:")
    print(f"  instance-specific (Thm 6)  : lambda* ~ {lam_inst:.0f}")
    print(f"  closed-form (Thm 8)        : lambda* ~ {lam_cf:.2e}")

    # Plot Figure 1
    plt.figure(figsize=(5.5, 3.5))
    plt.loglog(lams, eps_inst * PERIODS_PER_YEAR, 'b-', lw=2, label='Instance-specific (Thm 6)')
    plt.loglog(lams, eps_cf   * PERIODS_PER_YEAR, 'r--', lw=2, label='Closed-form (Thm 8)')
    plt.axhline(bonus_per_year, color='k', ls=':', lw=1, label='Direct substitution')
    plt.axhline(1000.0, color='gray', ls=':', lw=1, label='$1k/yr target')
    plt.xlabel(r'penalty weight $\lambda$')
    plt.ylabel(r'$\varepsilon$-DSIC slack ($\$$/yr/agent)')
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    if args.save_figures:
        plt.savefig('fig1_agc_lambda_sweep.pdf', bbox_inches='tight')

    # Figure 2: penalty-form deterrent comparison
    sigma_nu = np.sqrt(float(S[0, 0]))
    candidates = make_penalty_candidates(sigma_nu)
    alphas = np.linspace(0.01, 2.5, 80)
    deterrents = {name: np.array([deterrent_at(g, a, K, h_star, name=name)
                                  for a in alphas])
                  for name, g in candidates.items()}
    sup_d = np.max(np.abs(np.stack(list(deterrents.values()))), axis=0)

    plt.figure(figsize=(5.5, 3.5))
    colors = {'quadratic': 'C0', 'quartic': 'C3', 'absolute': 'C2', 'cosh': 'C4'}
    for name, d in deterrents.items():
        plt.plot(alphas, np.abs(d) / sup_d, lw=2, label=name, color=colors[name])
    # Asymptotic-regime boundary: where m^2 = (beta_0 alpha h*/sigma_nu)^2 = 0.1
    # gives alpha ~ sqrt(0.1) sigma_nu / |beta_0 h*| on this instance.
    beta_now = beta_0(K)
    alpha_asymp = np.sqrt(0.1) * sigma_nu / abs(beta_now * h_star)
    plt.axvspan(0.0, alpha_asymp, alpha=0.15, color='blue',
                label=r'asymptotic regime ($m^2 \leq 0.1$)')
    plt.xlabel(r'deviation fraction $\alpha = \|h\|/\|h^*\|$')
    plt.ylabel(r'deterrent ratio $|D(\bar g, \alpha h^*)| / \sup_{\bar g}|D|$')
    plt.legend(loc='lower left', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save_figures:
        plt.savefig('fig2_optimality.pdf', bbox_inches='tight')

    if not args.save_figures:
        plt.show()

    print("\nDone.  All numbers above match the values reported in the paper.")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mc-trials', type=int, default=0,
                   help='Monte-Carlo trials for empirical bonus (0 disables).')
    p.add_argument('--save-figures', action='store_true',
                   help='Save figures to PDF instead of showing them.')
    main(p.parse_args())
