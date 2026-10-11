"""
Stage E7 - SINDy with a library: which formula does the data choose, step by step?
================================================================================
This is SINDy as on the KTH slide: give the method a LIBRARY of 12 candidate terms and let a
sparse optimisation decide which ones belong to the governing equation.

   u  =  xi_1 * du/dx + xi_2 * du/dy + xi_3 * d2u/dx2 + ... + xi_12 * lap(lap(u))

The physics we simulated is the 2D wave (Helmholtz) equation  G*(d2u/dx2 + d2u/dy2) + rho w^2 u = 0,
so the CORRECT answer is: keep only d2u/dx2 and d2u/dy2, with equal coefficients
xi = -G*/(rho w^2), and throw away the other 10 terms. Because the brains are synthetic we
know the true stiffness, so we can check both the chosen formula and its numbers.

Steps (literature in brackets):
 1. Weak form (Messenger & Bortz 2021): every equation is a weighted sum under a smooth bump phi.
    sum(phi * Du) equals sum(u * D^T phi) exactly on the grid (summation by parts), so the noisy
    data is only ever weighted by smooth kernels. One bump per spot = one equation (40-150 per region).
 2. Two stages. (a) Choosing terms: plain normalised least squares (stable even when some
    columns are mostly noise). (b) The numbers: refit only the kept terms with the noise
    correction (method of moments / errors-in-variables: subtract the average effect of
    pure-noise copies, same idea as stage_f3), which is well-conditioned for 2-3 terms.
 3. Sparse optimisation, STLSQ (Brunton et al. 2016; Rudy et al. 2017): fit all 12 terms,
    drop terms whose normalised coefficient is below lambda OR not significant (|coef| < 2 SE),
    refit, repeat. This gives the "path": first formula -> what changed -> final formula.
 4. Choosing lambda (Pareto front; Mangan et al. 2017): try 25 values of lambda; the final model
    is the SPARSEST one whose fit (R^2) is within 0.01 of the best fit, plus the R^2 that 12
    random terms would gain just by fitting noise (about 12/n of the unexplained part).
 5. Trust (Ensemble-SINDy, Fasel et al. 2022): refit on 100 bootstrap resamples of the
    equations. Inclusion probability of each term = share of resamples that keep it
    (>= 0.9 = trusted). Bootstrap 95 % interval for the stiffness read from the formula.
    Also reported: R^2 (share of the data the formula explains), isotropy (coef of d2u/dx2
    divided by coef of d2u/dy2, physics says 1), and error vs the true stiffness.
 6. 3 mm grid bias: the discrete second derivative reads waves slightly too short (tissue too
    stiff); removed with the exact plane-wave factor (same idea as the C1 table).

Regions: a full uniform square is rare in a real brain slice (sulci and CSF cut through it), so
a "region" here = all spots in one brain that share the same stiffness: each spot's 21 mm
neighbourhood is fully inside the brain and uniform (true rel. std < 5 %), and spots are grouped
by stiffness in 10 % bins. Same stiffness -> same equation, so one formula is fitted to all of
them (40-150 equations). Up to 3 groups per test brain.
(Ground truth is used only to pick validation regions and to grade.)

Usage:  python stage_e7_sindy_library.py        (all 75 test brains)
        python stage_e7_sindy_library.py 10     (first 10 brains, quick check)
Output: mre_project/stage_e7_sindy/results.json, fig_sindy_path.png, fig_sindy_pareto.png,
        fig_sindy_trust.png
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from mre_solver import RHO, FREQ

warnings.filterwarnings("ignore")
W2 = (2 * np.pi * FREQ) ** 2
H = 3e-3
CONDS = ["clean", 50, 20, 10]
FOOT = 7                      # each equation uses a 7 x 7-voxel (21 mm) footprint
UNI_MAX = 0.05                # spot counts as uniform if true rel. std of G' in its footprint < 5 %
G_BIN = 1.10                  # spots grouped by stiffness in 10 % bins
MIN_EQ, MAX_EQ = 40, 150      # equations per group
MAX_REG = 3                   # groups per brain
N_NOISE = 8                   # pure-noise copies for the method-of-moments correction
LAMS = np.logspace(-3, 0, 25) # lambda values for the Pareto front
R2_TOL = 0.01                 # final = sparsest model within 0.01 R^2 of the best
Z_MIN = 2.0                   # keep a term only if |coef| > 2 standard errors
N_BOOT = 100                  # Ensemble-SINDy resamples
TRUST = 0.9                   # inclusion probability for a "trusted" term
MAX_IT = 10

# (plain name, math label, derivative, weight, derivative order)
TERMS = [("du/dx",       r"$\partial_x u$",        "x",    None, 1),
         ("du/dy",       r"$\partial_y u$",        "y",    None, 1),
         ("d2u/dx2",     r"$\partial_{xx} u$",     "xx",   None, 2),
         ("d2u/dy2",     r"$\partial_{yy} u$",     "yy",   None, 2),
         ("d2u/dxdy",    r"$\partial_{xy} u$",     "xy",   None, 2),
         ("x*u",         r"$x\,u$",                "",     "x",  0),
         ("y*u",         r"$y\,u$",                "",     "y",  0),
         ("x*lap(u)",    r"$x\,\nabla^2 u$",       "lap",  "x",  2),
         ("y*lap(u)",    r"$y\,\nabla^2 u$",       "lap",  "y",  2),
         ("d3u/dx3",     r"$\partial_{xxx} u$",    "xxx",  None, 3),
         ("d3u/dy3",     r"$\partial_{yyy} u$",    "yyy",  None, 3),
         ("lap(lap(u))", r"$\nabla^4 u$",          "lap2", None, 4)]
NT = len(TERMS)
IXX, IYY = 2, 3
TRUE_SET = np.zeros(NT, bool); TRUE_SET[[IXX, IYY]] = True

BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
PRED = BASE / "mre_project" / "stage_d" / "predictions"
OUT = BASE / "mre_project" / "stage_e7_sindy"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


# ---------------------------------------------------------------- test function ---
R_TEST = 2.0
_o = np.arange(-1, 2)
PHI = np.clip(1 - (_o[:, None] ** 2 + _o[None, :] ** 2) / R_TEST ** 2, 0, None) ** 3   # 3 x 3 bump


# ---------------------------------------------------------- grid derivatives ---
def _s(f, dy, dx):
    p = np.pad(f, 3, constant_values=np.nan)
    ny, nx = f.shape
    return p[3 + dy:3 + dy + ny, 3 + dx:3 + dx + nx]


def d_x(f):  return (_s(f, 0, 1) - _s(f, 0, -1)) / (2 * H)
def d_y(f):  return (_s(f, 1, 0) - _s(f, -1, 0)) / (2 * H)
def d_xx(f): return (_s(f, 0, 1) - 2 * f + _s(f, 0, -1)) / H ** 2
def d_yy(f): return (_s(f, 1, 0) - 2 * f + _s(f, -1, 0)) / H ** 2
def d_xy(f): return (_s(f, 1, 1) - _s(f, 1, -1) - _s(f, -1, 1) + _s(f, -1, -1)) / (4 * H ** 2)


def all_fields(u):
    """Every derivative the library needs (NaN outside the brain spreads, so invalid spots show)."""
    F = {"": u, "x": d_x(u), "y": d_y(u), "xx": d_xx(u), "yy": d_yy(u), "xy": d_xy(u)}
    F["lap"] = F["xx"] + F["yy"]
    F["xxx"] = d_x(F["xx"])
    F["yyy"] = d_y(F["yy"])
    F["lap2"] = d_xx(F["lap"]) + d_yy(F["lap"])
    return F


def test_sum(f):
    """Weighted sum under the 3 x 3 bump, centred on every voxel (NaN if any value is missing)."""
    out = np.zeros(f.shape, complex)
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            out = out + PHI[a + 1, b + 1] * _s(f, a, b)
    return out


def columns(F):
    """Test-function sums of every library term (and of u) at every voxel."""
    ny, nx = F[""].shape
    Y, X = np.mgrid[0:ny, 0:nx] * H
    C = {"u": test_sum(F[""])}
    for op in {t[2] for t in TERMS}:
        C[op] = test_sum(F[op])
    for op in {t[2] for t in TERMS if t[3] is not None}:
        C[("x", op)] = test_sum(X * F[op])
        C[("y", op)] = test_sum(Y * F[op])
    return C


def region_rows(C, reg):
    """Library matrix Theta (n x 12) and target y (n) for one group of spots.
    x, y are measured from the group's centre: sum(phi*(X - x0)*f) = S[X f] - x0 S[f]."""
    ys, xs = reg["ys"], reg["xs"]
    Th = np.empty((len(ys), NT), complex)
    for k, (_, _, op, w, _) in enumerate(TERMS):
        if w is None:
            Th[:, k] = C[op][ys, xs]
        elif w == "x":
            Th[:, k] = C[("x", op)][ys, xs] - reg["x0"] * C[op][ys, xs]
        else:
            Th[:, k] = C[("y", op)][ys, xs] - reg["y0"] * C[op][ys, xs]
    return Th, C["u"][ys, xs]


# ------------------------------------------------------------ 3 mm grid bias ---
def _bias_table():
    Gt = np.linspace(300, 12000, 400)
    k = np.sqrt(RHO * W2 / Gt)
    th = np.linspace(0, np.pi / 2, 24)
    kh_c, kh_s = np.outer(k * H, np.cos(th)), np.outer(k * H, np.sin(th))
    kd2 = (4 - 2 * np.cos(kh_c) - 2 * np.cos(kh_s)) / H ** 2
    return Gt, Gt * np.mean(k[:, None] ** 2 / kd2, axis=1)        # true G, G read by the grid


GT_TAB, GRAW_TAB = _bias_table()


def unbias(g_raw):
    return float(np.interp(g_raw, GRAW_TAB, GT_TAB))


# ------------------------------------------------------------- optimisation ---
def build(Th, y, Thn, yn, idx, cols, correct):
    """Normal equations for the chosen rows/columns. correct=True subtracts the average effect of
    pure noise (method of moments / errors-in-variables)."""
    A = Th[np.ix_(idx, cols)]
    M, b = A.conj().T @ A, A.conj().T @ y[idx]
    if correct and Thn:
        K = len(Thn)
        for B_, yb in zip(Thn, yn):
            Bi = B_[np.ix_(idx, cols)]
            M = M - Bi.conj().T @ Bi / K
            b = b - Bi.conj().T @ yb[idx] / K
    return M, b


def normal_eq(Th, y, Thn, yn, idx):
    """Stage 1 (choosing terms): plain normalised normal equations, no noise subtraction.
    Every column is scaled to length 1 and the target too, so coefficients are comparable and
    R^2 stays between 0 and 1. (Subtracting noise from all 12 columns is unstable: the 3rd and
    4th derivative columns are almost pure noise, so the subtraction leaves ~0 and blows up.)"""
    M, b = build(Th, y, Thn, yn, idx, np.arange(NT), False)
    yy = float(np.vdot(y[idx], y[idx]).real)
    s = np.sqrt(np.maximum(np.diag(M).real, 1e-300))
    return M / np.outer(s, s), b / s / np.sqrt(yy), s, yy


def final_coefs(Th, y, Thn, yn, idx, act):
    """Stage 2 (the numbers): refit ONLY the kept terms with the noise subtraction, which is
    well-conditioned for 2-3 terms. Falls back to the plain fit if the correction is unsafe."""
    cols = np.flatnonzero(act)
    xi = np.zeros(NT, complex)
    if len(cols) == 0:
        return xi
    M0, b0 = build(Th, y, Thn, yn, idx, cols, False)
    M, b = build(Th, y, Thn, yn, idx, cols, True)
    if np.any(np.diag(M).real < 0.05 * np.diag(M0).real):
        M, b = M0, b0
    try:
        xi[cols] = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        xi[cols] = np.linalg.lstsq(M0, b0, rcond=None)[0]
    return xi


def stlsq(Mt, bt, lam, n):
    """Sequentially thresholded least squares. Returns the path (one entry per step)."""
    act = np.ones(NT, bool)
    path = []
    for _ in range(MAX_IT):
        xi = np.zeros(NT, complex)
        z = np.zeros(NT)
        Ma = Mt[np.ix_(act, act)]
        try:
            xi[act] = np.linalg.solve(Ma, bt[act])
        except np.linalg.LinAlgError:
            xi[act] = np.linalg.lstsq(Ma, bt[act], rcond=None)[0]
        exc = 1 - 2 * np.real(np.vdot(xi, bt)) + np.real(np.vdot(xi, Mt @ xi))
        sig2 = max(exc, 1e-12) / max(n - act.sum(), 1)
        se = np.sqrt(np.abs(np.diag(np.linalg.pinv(Ma))) * sig2) + 1e-30
        z[act] = np.abs(xi[act]) / se
        path.append(dict(xi=xi.copy(), act=act.copy(), r2=float(1 - exc), z=z.copy()))
        new = act & (np.abs(xi) >= lam) & (z >= Z_MIN)
        if not new.any():
            new = np.zeros(NT, bool); new[np.argmax(np.abs(xi))] = True
        if (new == act).all():
            break
        act = new
    return path


def stiffness(xi_phys, act):
    if not (act[IXX] and act[IYY]):
        return float("nan"), float("nan")
    lap = (xi_phys[IXX] + xi_phys[IYY]) / 2
    g_raw = float((-RHO * W2 * lap).real)
    iso = float((xi_phys[IXX] / xi_phys[IYY]).real) if abs(xi_phys[IYY]) > 0 else float("nan")
    return unbias(g_raw), iso


def formula(xi_phys, act):
    parts = ["rho*w^2*u"]
    for k in np.flatnonzero(act):
        c = float((-RHO * W2 * xi_phys[k]).real)
        name, _, _, w, order = TERMS[k]
        if order == 2 and w is None:
            parts.append(f"{'+' if c >= 0 else '-'} {abs(c) / 1000:.2f} kPa*{name}")
        else:
            parts.append(f"{'+' if c >= 0 else '-'} {abs(c):.2e}*{name}")
    return " ".join(parts) + " = 0"


def fit_region(Th, y, Thn, yn, rng):
    n = len(y)
    idx = np.arange(n)
    Mt, bt, s, yy = normal_eq(Th, y, Thn, yn, idx)
    paths = [stlsq(Mt, bt, lam, n) for lam in LAMS]
    pareto = [(int(p[-1]["act"].sum()), p[-1]["r2"], float(lam)) for p, lam in zip(paths, LAMS)]
    best = max(r for _, r, _ in pareto)
    # tolerance: 0.01, plus what 12 random terms could gain just by fitting noise (~ terms/equations)
    tol = R2_TOL + (1 - best) * NT / n
    ok = [i for i, (nt, r, _) in enumerate(pareto) if r >= best - tol]
    pick = min(ok, key=lambda i: (pareto[i][0], -pareto[i][1]))
    lam = LAMS[pick]
    path = paths[pick]
    fin = path[-1]
    to_phys = np.sqrt(yy) / s
    xi_phys = final_coefs(Th, y, Thn, yn, idx, fin["act"])
    G, iso = stiffness(xi_phys, fin["act"])
    # Ensemble-SINDy: bootstrap the equations, same lambda, same two stages
    incl = np.zeros(NT)
    Gb = []
    for _ in range(N_BOOT):
        ib = rng.integers(0, n, n)
        Mb, bb, sb, yyb = normal_eq(Th, y, Thn, yn, ib)
        act_b = stlsq(Mb, bb, lam, n)[-1]["act"]
        incl += act_b
        g, _ = stiffness(final_coefs(Th, y, Thn, yn, ib, act_b), act_b)
        if np.isfinite(g):
            Gb.append(g)
    incl /= N_BOOT
    lo, hi = (np.percentile(Gb, [2.5, 97.5]) if len(Gb) >= 10 else (np.nan, np.nan))
    return dict(act=fin["act"], xi_phys=xi_phys, r2=fin["r2"], lam=float(lam), incl=incl, G=G,
                G_lo=float(lo), G_hi=float(hi), iso=iso, pareto=pareto, pick=pick, tol=float(tol),
                path=path, best_r2=float(best), formula=formula(xi_phys, fin["act"]),
                path_formulas=[formula(p["xi"] * to_phys, p["act"]) for p in path])


# ------------------------------------------------------------------- regions ---
def find_regions(mask, G3, valid, rng):
    """Groups of spots with uniform surroundings and the same stiffness (10 % bins)."""
    mf = ndimage.uniform_filter(mask.astype(float), FOOT, mode="constant")
    g1 = ndimage.uniform_filter(np.where(mask, G3, 0.0), FOOT, mode="constant")
    g2 = ndimage.uniform_filter(np.where(mask, G3 ** 2, 0.0), FOOT, mode="constant")
    full = (mf > 1 - 1e-6) & valid
    rel = np.full(mask.shape, np.inf)
    rel[full] = np.sqrt(np.maximum(g2[full] - g1[full] ** 2, 0)) / g1[full]
    ok = rel < UNI_MAX
    if not ok.any():
        return []
    ys, xs = np.nonzero(ok)
    bins = np.floor(np.log(g1[ys, xs]) / np.log(G_BIN)).astype(int)
    groups = []
    for bval in np.unique(bins):
        sel = np.flatnonzero(bins == bval)
        if len(sel) >= MIN_EQ:
            groups.append(sel)
    groups.sort(key=len, reverse=True)
    regs = []
    for sel in groups[:MAX_REG]:
        if len(sel) > MAX_EQ:
            sel = rng.choice(sel, MAX_EQ, replace=False)
        gy, gx = ys[sel], xs[sel]
        gv = G3[gy, gx]
        regs.append(dict(ys=gy, xs=gx, y0=float(gy.mean() * H), x0=float(gx.mean() * H),
                         G_true=float(gv.mean()), rel=float(gv.std() / gv.mean()),
                         key=(int(gy[0]), int(gx[0]))))
    return regs


# ---------------------------------------------------------------------- main ---
if __name__ == "__main__":
    files = sorted(PRED.glob("sim_*.npz"))
    if not files:
        raise SystemExit(f"No test predictions in {PRED}. Run stage_d1_evaluate.py first.")
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    log(f"SINDy with a {NT}-term library on {len(files)} test brains, conditions {CONDS}")
    R = {c: [] for c in CONDS}
    example, ex_key = {}, None
    n_regions = 0
    for nb, f in enumerate(files):
        sid = int(f.stem.split("_")[1])
        d = np.load(f)
        b = np.load(B_DIR / f"sim_{sid:05d}.npz")
        mask, G3, u3 = d["mask3"], d["G3"].astype(float), b["u3"].astype(complex)
        C_clean = columns(all_fields(np.where(mask, u3, np.nan)))
        valid = np.all([np.isfinite(v) for v in C_clean.values()], axis=0)
        regions = find_regions(mask, G3, valid, np.random.default_rng([sid, 11]))
        if not regions:
            continue
        n_regions += len(regions)
        if ex_key is None:
            ex_key = (sid,) + regions[0]["key"]
        for c in CONDS:
            if c == "clean":
                C = C_clean
            else:
                C = columns(all_fields(np.where(mask, d[f"u_noisy_snr{c}"].astype(complex), np.nan)))
            Cn = []
            if c != "clean":
                sigma = np.median(np.abs(u3[mask])) / c
                rn = np.random.default_rng([sid, c, 99])
                for _ in range(N_NOISE):
                    nz = sigma * (rn.standard_normal(mask.shape) + 1j * rn.standard_normal(mask.shape)) / np.sqrt(2)
                    Cn.append(columns(all_fields(np.where(mask, nz, np.nan))))
            for reg in regions:
                Th, y = region_rows(C, reg)
                if not (np.isfinite(Th).all() and np.isfinite(y).all()):
                    continue
                Thn, yn = [], []
                for Ck in Cn:
                    a, bb = region_rows(Ck, reg)
                    Thn.append(a); yn.append(bb)
                r = fit_region(Th, y, Thn, yn, np.random.default_rng([sid, reg["key"][0], reg["key"][1], 5]))
                r.update(sid=sid, rel=reg["rel"], G_true=reg["G_true"], n_eq=len(y))
                R[c].append(r)
                if (sid,) + reg["key"] == ex_key:
                    example[c] = r
        if (nb + 1) % 5 == 0 or nb + 1 == len(files):
            log(f"  {nb + 1}/{len(files)} brains, {n_regions} uniform regions so far")
    if not R["clean"]:
        raise SystemExit("No uniform regions found. Raise UNI_MAX (e.g. 0.08) or lower MIN_EQ (e.g. 25).")

    # ------------------------------------------------------------ example path ---
    print(f"\n=== Example region: brain #{ex_key[0]}, {example['clean']['n_eq']} uniform spots, "
          f"true G' = {example['clean']['G_true'] / 1000:.2f} kPa (varies {100 * example['clean']['rel']:.1f} %) ===")
    print("    correct answer:  rho*w^2*u + G*d2u/dx2 + G*d2u/dy2 = 0   (plain wave equation)")
    for c in ("clean", 20):
        e = example[c]
        print(f"\n  [{c}]  lambda chosen = {e['lam']:.3g}  (sparsest model within {e['tol']:.3f} R^2 of the best, "
              f"best R^2 = {e['best_r2']:.3f})")
        for i, (p, fstr) in enumerate(zip(e["path"], e["path_formulas"])):
            tag = "start" if i == 0 else ("final" if i == len(e["path"]) - 1 else f"step {i}")
            print(f"    {tag:6s} {int(p['act'].sum()):2d} terms  R^2 = {p['r2']:.3f}   {fstr}")
        print(f"    final, noise-corrected numbers:  {e['formula']}")
        print(f"    stiffness from the formula: {e['G'] / 1000:.2f} kPa "
              f"[95 % {e['G_lo'] / 1000:.2f} - {e['G_hi'] / 1000:.2f}]  (true {e['G_true'] / 1000:.2f})")
        print("    inclusion probability: " + ", ".join(
            f"{TERMS[k][0]} {e['incl'][k]:.2f}" for k in np.argsort(-e["incl"])[:4]))

    # --------------------------------------------------------------- summary ---
    res = {}
    print(f"\n=== All uniform regions ({len(R['clean'])} regions in {len(files)} brains) ===")
    print(f"  {'data':>6} | {'exact wave eq.':>14} | {'has both d2 terms':>17} | {'median R^2':>10} | "
          f"{'stiffness error':>15} | {'95 % CI width':>13} | {'isotropy':>8}")
    for c in CONDS:
        rs = R[c]
        act = np.array([r["act"] for r in rs])
        exact = np.mean((act == TRUE_SET).all(1))
        wave = np.mean(act[:, IXX] & act[:, IYY])
        r2 = np.median([r["r2"] for r in rs])
        err = [abs(r["G"] - r["G_true"]) / r["G_true"] for r in rs if np.isfinite(r["G"])]
        ciw = [(r["G_hi"] - r["G_lo"]) / r["G_true"] for r in rs if np.isfinite(r["G_lo"])]
        iso = [r["iso"] for r in rs if np.isfinite(r["iso"])]
        incl = np.mean([r["incl"] for r in rs], axis=0)
        res[str(c)] = dict(n=len(rs), exact_pct=float(100 * exact), wave_pct=float(100 * wave),
                           median_r2=float(r2),
                           median_G_err_pct=float(100 * np.median(err)) if err else float("nan"),
                           median_CI_width_pct=float(100 * np.median(ciw)) if ciw else float("nan"),
                           median_isotropy=float(np.median(iso)) if iso else float("nan"),
                           inclusion={TERMS[k][0]: float(incl[k]) for k in range(NT)})
        q = res[str(c)]
        print(f"  {str(c):>6} | {q['exact_pct']:13.1f}% | {q['wave_pct']:16.1f}% | {q['median_r2']:10.3f} | "
              f"{q['median_G_err_pct']:14.1f}% | {q['median_CI_width_pct']:12.1f}% | {q['median_isotropy']:8.2f}")

    print(f"\n=== Ensemble-SINDy inclusion probability (share of bootstrap fits that keep the term; "
          f">= {TRUST} = trusted) ===")
    print(f"  {'term':>12}" + "".join(f"{str(c):>8}" for c in CONDS))
    for k in range(NT):
        print(f"  {TERMS[k][0]:>12}" + "".join(f"{res[str(c)]['inclusion'][TERMS[k][0]]:8.2f}" for c in CONDS))
    for c in CONDS:
        inc = res[str(c)]["inclusion"]
        trusted = [t for t, v in inc.items() if v >= TRUST]
        print(f"  [{c}] trusted terms: {', '.join(trusted) if trusted else 'none'}")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    # ------------------------------------------------------- figure 1: path ---
    rows = ["clean", 20]
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), squeeze=False)
    for i, c in enumerate(rows):
        e = example[c]
        path = e["path"]
        steps = list(range(len(path))) if len(path) <= 4 else [0, 1, len(path) - 2, len(path) - 1]
        scale = 1.0
        for j in range(4):
            ax = axes[i, j]
            if j >= len(steps):
                ax.axis("off"); continue
            sidx = steps[j]
            p = path[sidx]
            nxt = path[sidx + 1]["act"] if sidx + 1 < len(path) else p["act"]
            vals = np.abs(p["xi"]) * scale
            cols = []
            for k in range(NT):
                if not p["act"][k]:
                    cols.append("#D3D1C7")
                elif not nxt[k]:
                    cols.append("#D85A30")
                else:
                    cols.append("#378ADD")
            yk = np.arange(NT)[::-1]
            ax.barh(yk, np.where(p["act"], np.maximum(vals, 1e-5), np.nan), color=cols)
            ax.axvline(e["lam"], color="0.3", ls="--", lw=1)
            ax.set_xscale("log"); ax.set_xlim(1e-4, 3)
            ax.set_yticks(yk); ax.set_yticklabels([t[1] for t in TERMS], fontsize=9)
            last = sidx == len(path) - 1
            name = "Start: all 12 terms" if sidx == 0 else ("Final formula" if last else f"Step {sidx}")
            ax.set_title(f"{name}\n{int(p['act'].sum())} terms, R$^2$ = {p['r2']:.3f}", fontsize=10)
            ax.set_xlabel("|normalised coefficient|  (dashed = lambda)", fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
        axes[i, 0].set_ylabel(f"{'clean data' if c == 'clean' else f'SNR {c}'}", fontsize=12)
        fig.text(0.5, 0.505 if i == 0 else 0.015,
                 f"[{c}] final: {e['formula']}   ->   G' = {e['G'] / 1000:.2f} kPa "
                 f"[{e['G_lo'] / 1000:.2f}-{e['G_hi'] / 1000:.2f}], true {e['G_true'] / 1000:.2f} kPa",
                 ha="center", fontsize=9, family="monospace")
    fig.suptitle("SINDy path for one region: blue = kept, orange = dropped at the next step, grey = already dropped",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=4)
    fig.savefig(OUT / "fig_sindy_path.png", dpi=140)

    # ----------------------------------------------------- figure 2: Pareto ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    for j, c in enumerate(rows):
        e = example[c]
        nt = np.array([p[0] for p in e["pareto"]]); r2 = np.array([p[1] for p in e["pareto"]])
        ax[j].plot(nt, r2, "o", color="#888780", alpha=0.7, label="model for one lambda")
        ax[j].plot(nt[e["pick"]], r2[e["pick"]], "*", ms=18, color="#1D9E75", label="chosen (final)")
        ax[j].axhline(e["best_r2"] - e["tol"], color="0.4", ls=":", lw=1, label=f"best R$^2$ - {e['tol']:.3f}")
        ax[j].set(xlabel="number of terms in the formula", ylabel="R$^2$ (share of data explained)",
                  title=f"Pareto front ({'clean' if c == 'clean' else f'SNR {c}'}): fewer terms vs better fit",
                  xticks=range(0, NT + 1))
        ax[j].legend(frameon=False, fontsize=8, loc="lower right")
        ax[j].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_sindy_pareto.png", dpi=150)

    # ------------------------------------------------------ figure 3: trust ---
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2), gridspec_kw=dict(width_ratios=[1.1, 1, 1]))
    inc = np.array([[res[str(c)]["inclusion"][t[0]] for c in CONDS] for t in TERMS])
    ax[0].imshow(inc, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for k in range(NT):
        for j in range(len(CONDS)):
            ax[0].text(j, k, f"{inc[k, j]:.2f}", ha="center", va="center", fontsize=8,
                       color="white" if inc[k, j] > 0.6 else "black")
    ax[0].set_yticks(range(NT)); ax[0].set_yticklabels([t[1] for t in TERMS])
    ax[0].set_xticks(range(len(CONDS))); ax[0].set_xticklabels([str(c) for c in CONDS])
    ax[0].set(xlabel="data (noise level)", title=f"A  Inclusion probability (Ensemble-SINDy)\n>= {TRUST} = trusted term")
    xs = np.arange(len(CONDS))
    ax[1].bar(xs - 0.2, [res[str(c)]["exact_pct"] for c in CONDS], 0.4, color="#1D9E75", label="exactly the wave equation")
    ax[1].bar(xs + 0.2, [res[str(c)]["wave_pct"] for c in CONDS], 0.4, color="#9FE1CB", label="contains both d2u terms")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([str(c) for c in CONDS])
    ax[1].set(ylim=(0, 105), ylabel="% of regions", xlabel="data (noise level)", title="B  Right formula found?")
    ax[1].legend(frameon=False, fontsize=8)
    pts = [r for r in R[20] if np.isfinite(r["G"])]
    if pts:
        gt = np.array([r["G_true"] for r in pts]) / 1000
        gh = np.array([r["G"] for r in pts]) / 1000
        lo = np.array([r["G_lo"] for r in pts]) / 1000
        hi = np.array([r["G_hi"] for r in pts]) / 1000
        yerr = np.vstack([np.clip(gh - lo, 0, None), np.clip(hi - gh, 0, None)])
        yerr = np.nan_to_num(yerr)
        ax[2].errorbar(gt, gh, yerr=yerr, fmt="o", ms=4, color="#378ADD", ecolor="#B5D4F4", elinewidth=1)
        m = [0, max(gt.max(), gh.max()) * 1.1]
        ax[2].plot(m, m, color="0.5", lw=1)
        ax[2].set(xlim=m, ylim=m)
    ax[2].set(xlabel="true G' (kPa)", ylabel="G' read from the formula (kPa)",
              title=f"C  Stiffness from the formula, SNR 20\nmedian error {res['20']['median_G_err_pct']:.1f} %, "
                    f"bars = bootstrap 95 % interval")
    for a in ax[1:]:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_sindy_trust.png", dpi=150)
    log(f"Done. Results in {OUT}")