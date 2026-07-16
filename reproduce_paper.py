#!/usr/bin/env python3
r"""
reproduce_paper.py
==================
Reproduces every numerical result and all four data figures (Fig1-Fig4) of

  "A nonlinear-residue feedback design rule for robust Kalman filtering
   with strategically misreported sensor calibration."

It (i) rebuilds the two case studies of Section 7 from their parameters,
(ii) prints a validation table against the numbers quoted in the paper, and
(iii) writes Fig1.pdf ... Fig4.pdf (the four data figures).

  Fig1 : smart-grid   worst-case gain vs residue weight (three design levels)
  Fig2 : smart-grid   deterrent ratio of four sign-invariant residues
  Fig3 : cooperative-perception (heterogeneous)  gain vs residue weight
  Fig4 : cooperative-perception (heterogeneous)  deterrent ratio

Usage
-----
  python3 reproduce_paper.py            # numbers + figures
  python3 reproduce_paper.py --sweep    # additionally run the effort-Hessian
                                        # positive-definiteness random sweep

Dependencies: numpy, matplotlib  (scipy is NOT required).
"""
import argparse
import numpy as np
from strategic import (Instance, dare_predictive, worstcase_nominal,
                       worstcase_residue, effort_hessian, P_actual)

# ---------------------------------------------------------------------------
# Instance builders (parameters as in Section 7 / Tables 1 and 3)
# ---------------------------------------------------------------------------
def smartgrid_instance():
    """Scalar smart-grid balancing-market instance (Table 1)."""
    N = 3
    A = 0.8 * np.eye(3)                      # g_i(k+1) = (1-alpha)g_i + u_i + w_i, alpha=0.2
    B = np.eye(3)
    Sw = 4 * np.eye(3)                       # sigma_w = 2 MW
    Clist = [np.eye(3)[i:i + 1] for i in range(3)]
    Rv = lambda e: np.array([[25.0 / (1 + 5 * e)]])   # sigma_max=5, kappa=5
    Q = 0.04 * np.eye(3)
    R = 0.005 * np.eye(3)
    rlist = [5 * np.eye(3)[i] for i in range(3)]       # r_i = 5 * e_i^{(3)}
    return Instance(A, B, Sw, Clist, Rv, Q, R, rlist, e0=0.7, sensor0=0)

def perception_instance():
    """Heterogeneous cooperative-perception instance (Table 3):
    two cameras (position) and one radar (velocity), per-sensor calibration."""
    A = np.array([[0.9, 0, 1, 0], [0, 0.9, 0, 1],
                  [0, 0, 0.85, 0], [0, 0, 0, 0.85]], float)
    B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]], float)
    Sw = np.diag([0.25, 0.25, 0.5, 0.5])
    C_cam = np.hstack([np.eye(2), np.zeros((2, 2))])   # camera: position
    C_rad = np.hstack([np.zeros((2, 2)), np.eye(2)])   # radar : velocity
    Clist = [C_cam.copy(), C_rad.copy(), C_cam.copy()] # sensor0 camera, 1 radar, 2 camera
    smax = [3.0, 2.0, 4.0]; kap = [4.0, 6.0, 3.0]
    Shape = [np.array([[1.0, 0.4], [0.4, 1.2]]),
             np.array([[0.8, -0.3], [-0.3, 1.0]]),
             np.array([[1.4, 0.6], [0.6, 1.1]])]
    Rvf = [(lambda e, i=i: (smax[i] ** 2 / (1 + kap[i] * e)) * Shape[i]) for i in range(3)]
    Q = np.diag([0.3, 0.3, 0.1, 0.1]); R = 0.02 * np.eye(2)
    rlist = [np.array([1.0, -0.9, 0, 0.]), np.array([-0.9, 1.0, 0, 0.]),
             np.array([-0.8, -0.9, 0, 0.])]
    return Instance(A, B, Sw, Clist, None, Q, R, rlist, e0=0.6, sensor0=0, Rv_funcs=Rvf)

# ---------------------------------------------------------------------------
# Three-level design rule (Remark 8 / Tables 2 and 4)
# ---------------------------------------------------------------------------
def three_level(inst, target, grid=(0.0, 0.25, 0.5, 0.75, 1.0)):
    import itertools
    m = inst.metrics(); A, B, L, Cbar = inst.A, inst.B, inst.L, inst.Cbar
    n, N = inst.n, inst.N
    phistar = 0.0; gh_lb = np.inf; H_lb = np.inf
    for ev in itertools.product(grid, repeat=N):
        ev = np.array(ev); mm = inst.metrics(ev)
        M = np.vstack([np.hstack([A, -B @ L]),
                       np.hstack([mm['K'] @ Cbar @ A, (np.eye(n) - mm['K'] @ Cbar) @ A - B @ L])])
        phistar = max(phistar, np.linalg.norm(np.linalg.inv(np.eye(2 * n) - M), 2))
        gh_lb = min(gh_lb, np.linalg.eigvalsh(mm['Gamma_h'])[0])
        H_lb = min(H_lb, np.linalg.eigvalsh(-mm['Hthth'])[0])
    Rr = max(np.linalg.norm(r) for r in inst.rlist) + np.linalg.norm(inst.xstar)
    G1 = 2 * Rr * ((2 * N - 1) * np.linalg.norm(inst.Q, 2)
                   + N * np.linalg.norm(L, 2) ** 2 * np.linalg.norm(inst.R, 2)) * phistar
    P_low = dare_predictive(A, Cbar, inst.Sw, inst.Rbar(np.zeros(N)))
    Kbar = (np.linalg.norm(Cbar, 2) * np.linalg.norm(np.linalg.inv(inst.Rbar(np.ones(N))), 2)
            * np.linalg.norm(P_low, 'fro'))
    Hr = np.linalg.eigvalsh(-m['Hthth'])[0]; gr = np.linalg.eigvalsh(m['Gamma_h'])[0]
    L0 = max(0.0, ((G1 * Kbar) ** 2 / (2 * target) - H_lb) / gh_lb)
    L1 = max(0.0, ((G1 * m['Kf']) ** 2 / (2 * target) - Hr) / gr)
    lo, hi = 0.0, 1e12                       # exact (matrix) inversion for L2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if worstcase_residue(m, mid) > target: lo = mid
        else: hi = mid
    L2 = 0.5 * (lo + hi)
    return dict(G1=G1, Kbar=Kbar, gh_lb=gh_lb, H_lb=H_lb, Hr=Hr, gr=gr,
                N0=G1 * Kbar, N1=G1 * m['Kf'], N2=np.linalg.norm(m['grad']),
                L0=L0, L1=L1, L2=L2)

# ---------------------------------------------------------------------------
# Closed-loop Monte-Carlo check of the nominal worst-case gain (Section 7.5)
# ---------------------------------------------------------------------------
def monte_carlo_gain(inst, hstar, T=40000, burn=4000, trials=12, seed=0):
    """Closed-loop Monte-Carlo estimate of the nominal worst-case per-period gain.
    Uses common random numbers: for each trial the honest and deviated runs share
    the same process/measurement noise, so their difference isolates the effect of
    the misreport h and converges tightly to the closed-form gain. Fully seeded."""
    A, B, Sw, Q, R, N = inst.A, inst.B, inst.Sw, inst.Q, inst.R, inst.N
    K = inst.metrics()['K']; L = inst.L; rbar = inst.rbar; Cbar = inst.Cbar
    Wc = np.linalg.cholesky(Sw); Rb = inst.Rbar(np.full(N, inst.e0)); Lc = np.linalg.cholesky(Rb)
    p0 = inst.pdims[inst.s0]; a0, _ = inst.idx[inst.s0]
    S0 = np.zeros((inst.p, p0)); S0[a0:a0 + p0, :] = -np.eye(p0)
    def avg_payoff(h, rng):
        x = inst.xstar.copy(); xh = inst.xstar.copy(); acc = 0.0; cnt = 0
        for k in range(T):
            u = -L @ (xh - rbar); w = Wc @ rng.standard_normal(inst.n); xn = A @ x + B @ u + w
            v = Lc @ rng.standard_normal(inst.p); ztil = Cbar @ xn + v + S0 @ h
            xhn = A @ xh + B @ u + K @ (ztil - Cbar @ (A @ xh + B @ u))
            if k >= burn:
                F0 = (xn - inst.rlist[0]) @ Q @ (xn - inst.rlist[0]) + u @ R @ u
                oth = sum((xhn - inst.rlist[j]) @ Q @ (xhn - inst.rlist[j]) + u @ R @ u
                          for j in range(1, N))
                acc += -(F0 + oth); cnt += 1
            x, xh = xn, xhn
        return acc / cnt
    diffs = []
    for s_ in range(trials):
        rng_h = np.random.default_rng([seed, s_]); rng_d = np.random.default_rng([seed, s_])
        diffs.append(avg_payoff(hstar, rng_d) - avg_payoff(np.zeros(p0), rng_h))
    diffs = np.array(diffs)
    return diffs.mean(), diffs.std(ddof=1) / np.sqrt(len(diffs))

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm',
                         'font.serif': ['cmr10', 'DejaVu Serif'],
                         'axes.formatter.use_mathtext': True, 'font.size': 11,
                         'axes.linewidth': 0.8, 'legend.fontsize': 8.5,
                         'legend.framealpha': 0.95, 'legend.edgecolor': '0.7'})
    return plt
NAVY, GREEN, RED, PURPLE = '#1f3d7a', '#2e7d32', '#c0392b', '#7b2d8e'

def gain_figure(fname, curves, nominal, target, nomlabel, tgtlabel, ylab, cross):
    plt = _style()
    lam = np.logspace(-1, np.log10(max(c[0] for c in cross)) + 1.3, 800)
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    for lab, col, ls, fn in curves:
        ax.loglog(lam, fn(lam), ls, color=col, lw=2, label=lab)
    ax.axhline(nominal, color='0.45', ls=(0, (1, 1)), lw=1)
    ax.axhline(target, color='0.15', ls=(0, (1, 1)), lw=1)
    x0 = lam[0] * 1.3
    ax.text(x0, nominal * 1.25, nomlabel, fontsize=8, color='0.35', va='bottom')
    ax.text(x0, target * 1.25, tgtlabel, fontsize=8, color='0.15', va='bottom')
    for lam_c, col, txt, ha, dx in cross:
        ax.plot([lam_c], [target], 'o', color=col, ms=5, zorder=5)
        ax.annotate(txt, (lam_c, target), textcoords='offset points',
                    xytext=(dx, -15), ha=ha, fontsize=8, color=col)
    ax.set_xlabel(r'residue weight $\lambda$'); ax.set_ylabel(ylab)
    ax.grid(True, which='both', alpha=0.25, lw=0.5); ax.legend(loc='lower left')
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)

def deterrent_figure(fname, p0, Sigma, beta0, hstar, shade_a, crossers, textbox):
    """Deterrent ratio |D|/sup|D| over the (rotation-invariant) sign-invariant class."""
    plt = _style(); Sig_ih = np.linalg.inv(Sigma)
    if p0 == 1:
        sig = np.sqrt(Sigma[0, 0]); ng = np.linspace(-14 * sig, 14 * sig, 40001)
        phi0 = np.exp(-0.5 * (ng / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
        E0 = lambda g: np.trapezoid(g * phi0, ng)
        raw = {'quadratic': (ng / sig) ** 2, 'quartic': (ng / sig) ** 4,
               'absnorm': np.abs(ng / sig), 'cosh': np.cosh(ng / sig)}
        mu1 = float(beta0) * float(hstar[0])
        def ratios(al):
            cands = {k: g - E0(g) for k, g in raw.items()}
            stds = {k: np.sqrt(E0(g * g)) for k, g in cands.items()}
            mu = mu1 * al; phim = np.exp(-0.5 * ((ng - mu) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            D = {k: abs(np.trapezoid(cands[k] * (phim - phi0), ng)) / stds[k] for k in cands}
            psi = 1 - np.cosh(ng * mu / sig ** 2) * np.exp(-mu * mu / (2 * sig ** 2))
            sup = np.sqrt(E0(psi * psi)); return {k: D[k] / sup for k in cands}
        labels = {'quadratic': r'centred whitened quadratic $g^\star$', 'cosh': 'centred cosh',
                  'absnorm': 'centred absolute value', 'quartic': 'centred quartic'}
    else:
        wg = np.linspace(1e-7, 90, 60000)
        fden = lambda lam: 0.5 * np.exp(-(wg + lam) / 2.0) * np.i0(np.sqrt(lam * wg))
        f0 = fden(0.0); E0 = lambda g: np.trapezoid(g * f0, wg)
        raw = {'quadratic': wg, 'quartic': wg * wg, 'absnorm': np.sqrt(wg),
               'cosh': np.cosh(np.sqrt(wg))}
        mu1 = beta0 @ hstar
        def ratios(al):
            cands = {k: g - E0(g) for k, g in raw.items()}
            stds = {k: np.sqrt(E0(g * g)) for k, g in cands.items()}
            lam = float(mu1 @ Sig_ih @ mu1) * al * al
            D = {k: abs(np.trapezoid(cands[k] * (fden(lam) - f0), wg)) / stds[k] for k in cands}
            mask = f0 > 1e-300; psi = np.zeros_like(wg); psi[mask] = 1 - fden(lam)[mask] / f0[mask]
            sup = np.sqrt(np.trapezoid(psi * psi * f0, wg)); return {k: D[k] / sup for k in cands}
        labels = {'quadratic': r'centred whitened quadratic $g^\star$', 'cosh': 'centred cosh',
                  'absnorm': r'centred norm $\|\Sigma^{-1/2}\nu\|$', 'quartic': 'centred quartic'}
    alphas = np.linspace(0.02, 2.5, 80); R = {k: [] for k in raw}
    for al in alphas:
        r = ratios(al); [R[k].append(r[k]) for k in raw]
    R = {k: np.array(v) for k, v in R.items()}
    fig, ax = plt.subplots(figsize=(5.7, 3.5)); ax.axvspan(0, shade_a, color=NAVY, alpha=0.08)
    for k, col, ls in [('quadratic', NAVY, '-'), ('cosh', PURPLE, ':'),
                       ('absnorm', RED, '--'), ('quartic', GREEN, '-.')]:
        ax.plot(alphas, R[k], ls, color=col, lw=2, label=labels[k])
    for xc, col in crossers: ax.axvline(xc, color=col, lw=0.8, ls=(0, (2, 2)), alpha=0.7)
    ax.set_xlabel(r'deviation fraction $\alpha=\|h\|/\|h^\star\|$')
    ax.set_ylabel(r'deterrent ratio $|D(\bar g,\alpha h^\star)|/\sup_{\bar g}|D|$')
    ax.set_ylim(0, 1.05); ax.set_xlim(0, 2.5); ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(loc='lower right')
    ax.text(0.035, 0.46, textbox, transform=ax.transAxes, fontsize=8.5, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.7', lw=0.7))
    fig.tight_layout(); fig.savefig(fname); plt.close(fig)
    return {k: float(R[k][0]) for k in raw}

# ---------------------------------------------------------------------------
# Random effort-Hessian sweep  (Remark 9 / Section 7.5)
# ---------------------------------------------------------------------------
def effort_hessian_sweep(per_dim=30, seed=11):
    rng = np.random.default_rng(seed); ok = tot = 0
    for pdim in (1, 2, 3):
        cnt = 0
        for _ in range(600):
            if cnt >= per_dim: break
            try:
                n = int(rng.integers(2, 5)); N = int(rng.integers(2, 4))
                A = rng.uniform(-0.25, 0.25, (n, n)) + np.diag(rng.uniform(0.3, 0.85, n))
                A = A / max(1.0, np.max(np.abs(np.linalg.eigvals(A))) / 0.9)
                B = rng.standard_normal((n, max(1, n - 1)))
                Sw = np.eye(n) * rng.uniform(0.3, 1.5)
                sh = rng.standard_normal((pdim, pdim)); Sh = sh @ sh.T + pdim * np.eye(pdim)
                smax = rng.uniform(2, 5); kap = rng.uniform(1, 5)
                Rv = lambda e, Sh=Sh, smax=smax, kap=kap: (smax ** 2 / (1 + kap * e)) * Sh
                Cl = [rng.standard_normal((pdim, n)) for _ in range(N)]
                Q = np.diag(rng.uniform(0.1, 1.0, n)); Rr = np.diag(rng.uniform(0.01, 0.1, B.shape[1]))
                rl = [rng.uniform(-4, 4, n) for _ in range(N)]
                ins = Instance(A, B, Sw, Cl, Rv, Q, Rr, rl, float(rng.uniform(0.3, 0.8)), 0)
                if np.max(np.abs(np.linalg.eigvals(A - B @ ins.L))) >= 0.999: continue
                mm = ins.metrics()
                if np.any(np.linalg.eigvalsh(mm['Hthth']) >= -1e-9): continue
                if np.any(np.linalg.eigvalsh(mm['Gamma_h']) <= 1e-9): continue
                eh = effort_hessian(ins, dd=2e-3)
                if not (eh['gamma_eta'] > 0 and eh['gamma_a'] > 0): continue
                tot += 1; cnt += 1
                if eh['chi'] ** 2 < 4 * eh['gamma_eta'] * eh['gamma_a']: ok += 1
            except Exception:
                continue
    return ok, tot

# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep', action='store_true', help='run the effort-Hessian random sweep')
    args = ap.parse_args()

    # ------------------- smart grid -------------------
    print("=" * 68); print("SMART-GRID BALANCING MARKET (Section 7.1-7.4, Tables 1-2)"); print("=" * 68)
    sg = smartgrid_instance(); m = sg.metrics()
    per = 105120; dU = worstcase_nominal(m); hstar = -np.linalg.solve(m['Hthth'], m['grad'])
    print(f"  ||K(e0)||_F = {m['Kf']:.3f}   (paper 0.886)")
    print(f"  sigma_nu0   = {np.sqrt(m['Snu0'][0,0]):.3f}   (paper 3.37)")
    print(f"  beta0       = {m['beta0'][0,0]:.3f}  (paper -0.328)")
    print(f"  grad_h      = {m['grad'][0]:+.3f}  (paper +0.231)")
    print(f"  H_thth      = {m['Hthth'][0,0]:.4f} (paper -0.0404)")
    print(f"  h*          = {hstar[0]:.2f} MW  (paper 5.73)")
    print(f"  dU_nominal  = ${dU:.4f}/period  ->  ${dU*per:,.0f}/yr  (paper ~$70,000)")
    mcs, mcs_se = monte_carlo_gain(sg, hstar, T=40000, burn=4000, trials=12, seed=0)
    print(f"  Monte-Carlo gain = ${mcs:.4f} +/- ${mcs_se:.4f}/period   (closed form ${dU:.4f})  <- validation")
    tl = three_level(sg, target=1000/per)
    print(f"  three-level lambda*: L2={tl['L2']:.0f} (293)  L1={tl['L1']:.2e} (5.4e5)  L0={tl['L0']:.2e} (8.6e6)")

    # ------------------- cooperative perception -------------------
    print("=" * 68); print("HETEROGENEOUS COOPERATIVE PERCEPTION (Section 7.5, Tables 3-4)"); print("=" * 68)
    cp = perception_instance(); mh = cp.metrics()
    dUh = worstcase_nominal(mh); hstarh = -np.linalg.solve(mh['Hthth'], mh['grad'])
    print(f"  ||K(e0)||_F = {mh['Kf']:.3f}  (paper 0.873)   rho(A-BL)={np.max(np.abs(np.linalg.eigvals(cp.A-cp.B@cp.L))):.3f}")
    print(f"  grad_h      = ({mh['grad'][0]:.3f},{mh['grad'][1]:.3f})  (paper (0.476,-0.240))")
    print(f"  H eig       = {np.linalg.eigvalsh(mh['Hthth']).round(3)}  Gamma eig = {np.linalg.eigvalsh(mh['Gamma_h']).round(3)}")
    print(f"  h*          = ({hstarh[0]:.2f},{hstarh[1]:.2f})  (paper (2.40,-1.53))")
    print(f"  dU_nominal  = {dUh:.4f}/epoch  (paper 0.754)")
    mc, mc_se = monte_carlo_gain(cp, hstarh)
    print(f"  Monte-Carlo gain = {mc:.4f} +/- {mc_se:.4f}   (closed form {dUh:.4f})  <- validation")
    tlh = three_level(cp, target=0.05*dUh)
    print(f"  three-level lambda*: L2={tlh['L2']:.1f} (73)  L1={tlh['L1']:.2e} (1.2e6)  L0={tlh['L0']:.2e} (4.1e9)")
    eh = effort_hessian(cp, dd=1.5e-3)
    print(f"  effort-Hessian: gamma_eta={eh['gamma_eta']:.3f} gamma_a={eh['gamma_a']:.3f} chi={eh['chi']:.3f}")
    print(f"    chi^2={eh['chi']**2:.2f} < 4*ge*ga={4*eh['gamma_eta']*eh['gamma_a']:.2f}  (PD: {eh['PD']})")
    print(f"    Prop 2 identity  Phi'_a={eh['Phia_prime']:.4f} = L'={eh['Lprime']:.4f}")

    if args.sweep:
        print("-" * 68); print("Running effort-Hessian random sweep (30 instances each p0=1,2,3)...")
        ok, tot = effort_hessian_sweep()
        print(f"  chi^2 < 4*gamma_eta*gamma_a held in {ok}/{tot} instances")

    # ------------------- figures -------------------
    print("=" * 68); print("Generating figures Fig1.pdf ... Fig4.pdf"); print("=" * 68)
    # Fig1 : smart-grid gain vs lambda (three levels; L2 exact, L1/L0 scalar bounds)
    gh = m['Gamma_h'][0,0]; Habs = -m['Hthth'][0,0]
    def sg_curve(N_, Habs_, g_): return lambda l: 0.5*N_**2/(Habs_ + l*g_)*per
    gain_figure('Fig1.pdf',
        [('Public rule (L0)', NAVY, '-', sg_curve(tl['N0'], tl['H_lb'], tl['gh_lb'])),
         ('Runtime rule (L1)', GREEN, '--', sg_curve(tl['N1'], Habs, gh)),
         ('Instance rule (L2)', RED, '-.', sg_curve(m['grad'][0], Habs, gh))],
        dU*per, 1000, 'nominal: \\$%s/yr' % f"{dU*per:,.0f}", '\\$1,000/yr target',
        r'annualised worst-case gain (USD/yr)',
        [(tl['L2'], RED, r'$\lambda^\star_2\!\approx\!293$', 'center', 0),
         (tl['L1'], GREEN, r'$\lambda^\star_1\!\approx\!5.4{\times}10^5$', 'right', -6),
         (tl['L0'], NAVY, r'$\lambda^\star_0\!\approx\!8.6{\times}10^6$', 'left', 6)])
    # Fig2 : smart-grid deterrent (p0=1)
    r1 = deterrent_figure('Fig2.pdf', 1, m['Snu0'], m['beta0'][0,0], hstar, 0.567,
        [(1.27, PURPLE), (1.74, GREEN)],
        'leading-order ratios:\n$g^\\star$: 1.00   cosh: 0.96\n'
        r'abs: $1/\sqrt{\pi-2}\approx0.94$' + '\n' + r'quartic: $\sqrt{6/8}\approx0.87$')
    # Fig3 : perception gain vs lambda
    Hh, Gh = mh['Hthth'], mh['Gamma_h']
    cp_L2 = lambda l: np.array([0.5*mh['grad']@np.linalg.solve(-Hh+x*Gh, mh['grad']) for x in l])
    gain_figure('Fig3.pdf',
        [('Public rule (L0)', NAVY, '-', (lambda l: 0.5*tlh['N0']**2/(tlh['H_lb']+l*tlh['gh_lb']))),
         ('Runtime rule (L1)', GREEN, '--', (lambda l: 0.5*tlh['N1']**2/(tlh['Hr']+l*tlh['gr']))),
         ('Instance rule (L2)', RED, '-.', cp_L2)],
        dUh, 0.05*dUh, 'nominal: %.3f/epoch' % dUh, 'target: %.3f/epoch' % (0.05*dUh),
        r'worst-case gain per epoch',
        [(tlh['L2'], RED, r'$\lambda^\star_2\!\approx\!73$', 'center', 0),
         (tlh['L1'], GREEN, r'$\lambda^\star_1\!\approx\!1.2{\times}10^6$', 'center', 0),
         (tlh['L0'], NAVY, r'$\lambda^\star_0\!\approx\!4.1{\times}10^9$', 'center', 0)])
    # Fig4 : perception deterrent (p0=2)
    r2 = deterrent_figure('Fig4.pdf', 2, mh['Snu0'], mh['beta0'], hstarh, 0.484,
        [(1.24, PURPLE), (1.49, GREEN)],
        'leading-order ratios:\n$g^\\star$: 1.00   norm: 0.96\ncosh: 0.95   quartic: 0.89')
    print("  Fig2 leading-order ratios:", {k: round(v, 3) for k, v in r1.items()})
    print("  Fig4 leading-order ratios:", {k: round(v, 3) for k, v in r2.items()})
    print("  wrote Fig1.pdf, Fig2.pdf, Fig3.pdf, Fig4.pdf")

if __name__ == '__main__':
    main()
