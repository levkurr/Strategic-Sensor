import numpy as np
np.set_printoptions(precision=5, suppress=True)

def dare_predictive(A, C, Sw, Rv, iters=200000, tol=1e-13):
    """Steady-state PREDICTIVE covariance P^- : P = A P A^T + Sw - A P C^T (C P C^T + Rv)^{-1} C P A^T."""
    n=A.shape[0]; P=Sw.copy()
    for _ in range(iters):
        S=C@P@C.T+Rv
        K_=A@P@C.T@np.linalg.inv(S)
        Pn=A@P@A.T+Sw-K_@C@P@A.T
        Pn=0.5*(Pn+Pn.T)
        if np.max(np.abs(Pn-P))<tol:
            P=Pn; break
        P=Pn
    return P

def dare_control(A,B,Q,R,iters=200000,tol=1e-13):
    """Control DARE for LQR: X = A^T X A - A^T X B (R+B^T X B)^{-1} B^T X A + Q ; L=(R+B^T X B)^{-1}B^T X A."""
    X=Q.copy()
    for _ in range(iters):
        BtXB=R+B.T@X@B
        L=np.linalg.inv(BtXB)@B.T@X@A
        Xn=A.T@X@A-A.T@X@B@L+Q
        Xn=0.5*(Xn+Xn.T)
        if np.max(np.abs(Xn-X))<tol:
            X=Xn; break
        X=Xn
    L=np.linalg.inv(R+B.T@X@B)@B.T@X@A
    return X,L

class Instance:
    def __init__(self, A,B,Sw, Clist, Rv_func, Q,R, rlist, e0, sensor0=0, Rv_funcs=None):
        self.A=A;self.B=B;self.Sw=Sw;self.Clist=Clist;self.Rv_func=Rv_func
        # per-sensor calibration: list of callables e->SPD; default = same Rv_func for all
        self.Rv_funcs = Rv_funcs if Rv_funcs is not None else [Rv_func]*len(Clist)
        self.Q=Q;self.R=R;self.rlist=rlist;self.e0=e0;self.s0=sensor0
        self.n=A.shape[0]; self.N=len(Clist)
        self.pdims=[C.shape[0] for C in Clist]
        self.Cbar=np.vstack(Clist)
        self.p=self.Cbar.shape[0]
        _,self.L=dare_control(A,B,Q,R)
        # references
        self.rbar=np.mean(np.stack(rlist),axis=0)
        # honest steady-state mean state x*: (I - A + BL) x* = BL rbar
        M1=np.eye(self.n)-A+B@self.L
        self.xstar=np.linalg.solve(M1, B@self.L@self.rbar)
        self.ustar=-self.L@(self.xstar-self.rbar)
        # block index ranges
        self.idx=[]; c=0
        for pd in self.pdims:
            self.idx.append((c,c+pd)); c+=pd

    def Rbar(self, evec):
        return np.block([[ (self.Rv_funcs[i](evec[i]) if i==j else np.zeros((self.pdims[i],self.pdims[j]))) for j in range(self.N)] for i in range(self.N)])

    def solve_filter(self, evec):
        Rb=self.Rbar(evec)
        P=dare_predictive(self.A,self.Cbar,self.Sw,Rb)
        S=self.Cbar@P@self.Cbar.T+Rb        # innovation cov Sigma_nu (full)
        K=P@self.Cbar.T@np.linalg.inv(S)     # filter gain (paper's K)
        return dict(P=P,S=S,K=K,Rb=Rb)

    def responses(self, K):
        """a,b,c (n x p0), beta0 (p0 x p0) at sensor0 for filter gain K."""
        A,B,L,Cbar=self.A,self.B,self.L,self.Cbar
        n=self.n; s0=self.s0; a0,b0=self.idx[s0]; p0=self.pdims[s0]
        Mtop=np.hstack([A, -B@L])
        Mbot=np.hstack([K@Cbar@A, (np.eye(n)-K@Cbar)@A - B@L])
        Mmat=np.vstack([Mtop,Mbot])       # 2n x 2n
        S0=np.zeros((self.p,p0)); S0[a0:b0,:]=-np.eye(p0)   # -I on sensor0 block
        d1=np.vstack([np.zeros((n,p0)), K@S0])             # 2n x p0
        ab=np.linalg.solve(np.eye(2*n)-Mmat, d1)
        a=ab[:n,:]; b=ab[n:,:]; c=-L@b
        C0=self.Clist[s0]
        beta0=C0@(a-A@b-B@c)-np.eye(p0)
        return a,b,c,beta0

    def metrics(self, evec=None):
        if evec is None: evec=np.full(self.N,self.e0)
        f=self.solve_filter(evec); K=f['K']
        a,b,c,beta0=self.responses(K)
        s0=self.s0; a0,b0=self.idx[s0]; p0=self.pdims[s0]
        Snu0=f['S'][a0:b0,a0:b0]
        Gamma_h=beta0.T@np.linalg.inv(Snu0)@beta0
        Q,R=self.Q,self.R; xstar,ustar=self.xstar,self.ustar
        r0=self.rlist[s0]; N=self.N
        rsum_others=sum(self.rlist[j] for j in range(N) if j!=s0)
        grad = (-2*a.T@Q@(xstar-r0) - 2*(N-1)*b.T@Q@xstar
                + 2*b.T@Q@rsum_others - 2*N*c.T@R@ustar)
        Hthth = -2*(a.T@Q@a + (N-1)*b.T@Q@b + N*c.T@R@c)
        Kf=np.linalg.norm(K,'fro')
        return dict(K=K,Kf=Kf,S=f['S'],Snu0=Snu0,beta0=beta0,Gamma_h=Gamma_h,
                    grad=grad,Hthth=Hthth,a=a,b=b,c=c,P=f['P'],p0=p0)

def worstcase_nominal(m):
    g=m['grad']; negH=-m['Hthth']
    return 0.5*g@np.linalg.solve(negH,g)
def worstcase_residue(m,lam):
    g=m['grad']; Mmat=-m['Hthth']+lam*m['Gamma_h']
    return 0.5*g@np.linalg.solve(Mmat,g)

def stein(Acl, Qmat, iters=200000, tol=1e-14):
    """Solve P = Acl P Acl^T + Qmat (Acl Schur)."""
    P=Qmat.copy()
    for _ in range(iters):
        Pn=Acl@P@Acl.T+Qmat; Pn=0.5*(Pn+Pn.T)
        if np.max(np.abs(Pn-P))<tol: P=Pn; break
        P=Pn
    return P

def P_actual(inst, K, evec_true):
    """Actual steady-state prediction-error cov under gain K, true noise Rbar(evec_true)."""
    A,Cbar=inst.A,inst.Cbar; n=inst.n
    Rt=inst.Rbar(evec_true)
    Acl=A@(np.eye(n)-K@Cbar)
    Qm=A@K@Rt@K.T@A.T+inst.Sw
    return stein(Acl,Qm)

def Phi_expected(inst, ereport, eactual):
    """E_dev[Phi_0] for sensor0, given full report and actual effort vectors."""
    s0=inst.s0; a0,b0=inst.idx[s0]; p0=inst.pdims[s0]
    fr=inst.solve_filter(ereport); K=fr['K']
    Snu0_design=fr['S'][a0:b0,a0:b0]
    Pact=P_actual(inst,K,eactual)
    Sact_full=inst.Cbar@Pact@inst.Cbar.T+inst.Rbar(eactual)
    Snu0_act=Sact_full[a0:b0,a0:b0]
    # honest centring
    fh=inst.solve_filter(np.full(inst.N,inst.e0))
    Snu0_e0=fh['S'][a0:b0,a0:b0]
    val=np.trace(np.linalg.solve(Snu0_design,Snu0_act))-p0 \
        +np.linalg.slogdet(Snu0_design)[1]-np.linalg.slogdet(Snu0_e0)[1]
    return val

def effort_hessian(inst, dd=1e-3):
    """Return gamma_eta, gamma_a, chi (cross), and Prop2 check (Phi'_a vs L')."""
    e0=inst.e0; N=inst.N; s0=inst.s0
    base=np.full(N,e0)
    def rep(dr): v=base.copy(); v[s0]=e0+dr; return v
    def act(da): v=base.copy(); v[s0]=e0+da; return v
    def Phi(dr,da): return Phi_expected(inst, rep(dr), act(da))
    # second derivatives via finite differences
    f00=Phi(0,0)
    grr=(Phi(dd,0)-2*f00+Phi(-dd,0))/dd**2      # d2/dr2
    gaa=(Phi(0,dd)-2*f00+Phi(0,-dd))/dd**2      # d2/da2
    cross=(Phi(dd,dd)-Phi(dd,-dd)-Phi(-dd,dd)+Phi(-dd,-dd))/(4*dd**2)
    gamma_eta=0.5*grr; gamma_a=0.5*gaa; chi=cross
    # Prop2: Phi'_a (first deriv on actual axis) vs L' = d/de logdet Snu0(e)
    Phia_prime=(Phi(0,dd)-Phi(0,-dd))/(2*dd)
    a0,b0=inst.idx[s0]
    def logdetSnu(e):
        v=np.full(N,e0); v[s0]=e; f=inst.solve_filter(v); return np.linalg.slogdet(f['S'][a0:b0,a0:b0])[1]
    Lprime=(logdetSnu(e0+dd)-logdetSnu(e0-dd))/(2*dd)
    # report-axis first deriv (should be ~0 by Loewner)
    Ar_prime=(Phi(dd,0)-Phi(-dd,0))/(2*dd)
    return dict(gamma_eta=gamma_eta,gamma_a=gamma_a,chi=chi,
                Phia_prime=Phia_prime,Lprime=Lprime,Ar_prime=Ar_prime,
                PD=(chi**2 < 4*gamma_eta*gamma_a))
