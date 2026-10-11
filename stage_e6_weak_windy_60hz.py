"""
Stage E6 (60 Hz) - Weak-form SINDy system identification: is the tissue uniform here or not?
==========================================================================================
Question: which wave equation does each small patch of tissue follow?
   M0 (uniform)   G* lap(u) + rho w^2 u = 0                 <- what direct inversion assumes
   M1 (varying)   div(G* grad u) + rho w^2 u = 0,  G* = G0 + gx x + gy y  (linear in the window)
Only 60 Hz data is used, so the "extra term" model (M3) is NOT tested: at one frequency an
extra c*u term cannot be told apart from rho w^2 u (both are "a number times u").

Weak form (Messenger & Bortz 2021): instead of taking 2nd derivatives of the noisy wave,
multiply the equation by a smooth bump phi and sum. Summation by parts moves every
derivative onto phi (which has no noise):
   sum u*div(G grad phi) + rho w^2 * sum u*phi = 0
   with G linear:  G0*S[u lap phi] + gx*S[u (x lap phi + phi_x)] + gy*S[u (y lap phi + phi_y)]
So the noisy data is only multiplied and summed, which averages noise away.

Ingredients (same as the multi-frequency version, minus M3):
 - discrete-consistent test functions: phi's derivatives use the SAME finite differences as
   the grid, so summation by parts is exact on the 3 mm grid
 - 3 mm grid-bias correction: computed exactly for plane waves at 60 Hz and removed
 - method-of-moments noise correction: subtract the average effect of pure-noise copies
 - STLSQ-style pruning: keep "varying" only if stiffness changes > 10 % over 6 mm AND the
   gradient is > 2 standard errors; otherwise fall back to "uniform"
 - "no model fits": the chosen model explains < 90 % of the (noise-corrected) data

Graded against the known truth (synthetic brains):
 H1  M1 is chosen where stiffness truly varies (true relative spread in the 5x5 window > 10 %)
 H4  where the physics says "uniform" the network is rarely badly wrong; where nothing fits it is
Conditions: clean, SNR 50, 20, 10 - test brains, the SAME noisy data as stage_d1_evaluate.py.

Usage:  python stage_e6_weak_sindy_60hz.py        (all 75 test brains)
        python stage_e6_weak_sindy_60hz.py 5      (first 5 brains, quick check)
Output: mre_project/stage_e6_60hz/results.json, fig_weak_sindy_60hz.png, fig_weak_sindy_maps.png
"""
import json, sys, time, warnings
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage
from sklearn.metrics import roc_auc_score
from mre_solver import RHO, FREQ

warnings.filterwarnings("ignore")
W2 = (2 * np.pi * FREQ) ** 2
H = 3e-3                     # voxel size (m)
CONDS = ["clean", 50, 20, 10]
HALF = 2                     # 5x5 window for the "true variation" label (15 mm)
R_TEST = 2.0                 # test-function radius (voxels)
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]   # 9 test functions per voxel
MIN_TF = 5                   # need at least 5 test functions fully inside the brain
HET_TRUE = 0.10              # truly varying window: relative std of true G' > 10 %
TAU_HET = 0.10               # keep "varying" only if G changes > 10 % over 6 mm ...
Z_MIN = 2.0                  # ... and the gradient is > 2 standard errors
R2_MIN = 0.90                # below this, "no model fits"
N_NOISE = 6                  # pure-noise copies for the method-of-moments correction
BAD = 0.20                   # network "badly wrong" = > 20 % error
MODELS = ["M0", "M1"]

BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
PRED = BASE / "mre_project" / "stage_d" / "predictions"
OUT = BASE / "mre_project" / "stage_e6_60hz"
OUT.mkdir(parents=True, exist_ok=True)
START = time.time()


def log(msg):
    print(f"{msg}   ({(time.time() - START) / 60:.1f} min)", flush=True)


# --------------------------------------------------------------- test functions ---
def test_kernels():
    """phi sampled on the grid; derivatives with the grid's own finite differences
    (5-point Laplacian, central differences) so sum(u * L phi) = sum(phi * L u) exactly."""
    n = int(np.ceil(R_TEST)) + 1
    jy, jx = np.mgrid[-n:n + 1, -n:n + 1].astype(float)
    s = (jx ** 2 + jy ** 2) / R_TEST ** 2
    phi = np.where(s < 1, (1 - s) ** 3, 0.0)
    p = np.pad(phi, 1)
    lap = (p[2:, 1:-1] + p[:-2, 1:-1] + p[1:-1, 2:] + p[1:-1, :-2] - 4 * phi) / H ** 2
    dx = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * H)
    dy = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * H)
    x, y = jx * H, jy * H
    k = dict(phi=phi, lap=lap, dx=-dx, dy=-dy, xlap=x * lap, ylap=y * lap)   # first derivatives flip sign
    keep = np.any([np.abs(v) > 0 for v in k.values()], axis=0)
    rows, cols = np.any(keep, 1), np.any(keep, 0)
    k = {name: v[rows][:, cols] for name, v in k.items()}
    k["support"] = (np.abs(k["lap"]) + k["phi"] > 0).astype(float)
    return k


K = test_kernels()


def _bias_table():
    """G_estimated / G_true for a 60 Hz plane wave in uniform tissue with these kernels
    (averaged over directions). Used to remove the 3 mm grid bias."""
    n = K["phi"].shape[0] // 2
    jy, jx = np.mgrid[-n:n + 1, -n:n + 1] * H
    Gs = np.linspace(400, 9000, 120)
    fac = []
    for G in Gs:
        k = np.sqrt(W2 * RHO / G)
        f = []
        for th in np.linspace(0, np.pi, 12, endpoint=False):
            e = np.exp(1j * k * (np.cos(th) * jx + np.sin(th) * jy))
            f.append(k ** 2 / (-(e * K["lap"]).sum() / (e * K["phi"]).sum()).real)
        fac.append(np.mean(f))
    return Gs, np.array(fac)


GS_TAB, FAC_TAB = _bias_table()


def bias_factor(G):
    return np.interp(np.clip(G, GS_TAB[0], GS_TAB[-1]), GS_TAB, FAC_TAB)


def corr(a, k):
    if np.iscomplexobj(a):
        return (ndimage.correlate(a.real, k, mode="constant")
                + 1j * ndimage.correlate(a.imag, k, mode="constant")) * H ** 2
    return ndimage.correlate(a, k, mode="constant") * H ** 2


def weak_fields(u):
    return {k: corr(u, K[k]) for k in ("phi", "lap", "dx", "dy", "xlap", "ylap")}


# ------------------------------------------------------------------- the fit ---
def fit_brain(u, mask, sigma=None, rng=None):
    noisy = sigma is not None
    valid_tf = (ndimage.correlate(mask.astype(float), K["support"], mode="constant")
                >= K["support"].sum() - 1e-6)                 # test function fully inside brain
    F = weak_fields(u)
    Fn = []
    if noisy:
        for _ in range(N_NOISE):
            nz = sigma * (rng.standard_normal(mask.shape) + 1j * rng.standard_normal(mask.shape)) / np.sqrt(2)
            Fn.append(weak_fields(nz * mask))
    scale = 1.0 / (W2 * RHO * np.median(np.abs(u[mask])))
    ny, nx = mask.shape
    out = {k: np.full(mask.shape, np.nan) for k in ["choice", "r2", "h", "G_phys"]}

    def rows(Ff, ys, xs, oy, ox, c):
        A0 = Ff["lap"][ys, xs] * c
        Ax = ox * H * A0 + (Ff["xlap"][ys, xs] + Ff["dx"][ys, xs]) * c
        Ay = oy * H * A0 + (Ff["ylap"][ys, xs] + Ff["dy"][ys, xs]) * c
        U = Ff["phi"][ys, xs]
        cols = {"M0": np.stack([A0], 1) * scale, "M1": np.stack([A0, Ax, Ay], 1) * scale}
        return cols, -RHO * W2 * U * scale

    for y0, x0 in zip(*np.nonzero(mask)):
        pts = [(y0 + oy, x0 + ox, oy, ox) for oy, ox in OFFSETS
               if 0 <= y0 + oy < ny and 0 <= x0 + ox < nx and valid_tf[y0 + oy, x0 + ox]]
        if len(pts) < MIN_TF:
            continue
        ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
        oy = np.array([p[2] for p in pts]); ox = np.array([p[3] for p in pts])

        # rough noise-corrected stiffness -> grid-bias factor for this voxel
        A0 = F["lap"][ys, xs]; b0 = -RHO * W2 * F["phi"][ys, xs]
        num, den = np.vdot(A0, b0).real, np.vdot(A0, A0).real
        for Fk in Fn:
            An = Fk["lap"][ys, xs]; bn = -RHO * W2 * Fk["phi"][ys, xs]
            num -= np.vdot(An, bn).real / len(Fn); den -= np.vdot(An, An).real / len(Fn)
        g0 = num / max(den, 0.05 * np.vdot(A0, A0).real, 1e-30)
        c = bias_factor(abs(g0))

        X, y = rows(F, ys, xs, oy, ox, c)
        Xn, yn = [], []
        for Fk in Fn:
            a, b = rows(Fk, ys, xs, oy, ox, c)
            Xn.append(a); yn.append(b)
        tot = np.sum(np.abs(y) ** 2) - (np.mean([np.sum(np.abs(v) ** 2) for v in yn]) if noisy else 0)

        def fit(m):
            A_ = X[m]; AhA = A_.conj().T @ A_; Ahy = A_.conj().T @ y
            for k in range(len(Fn)):                       # method of moments: remove noise energy
                AhA = AhA - Xn[k][m].conj().T @ Xn[k][m] / len(Fn)
                Ahy = Ahy - Xn[k][m].conj().T @ yn[k] / len(Fn)
            beta = np.linalg.lstsq(AhA, Ahy, rcond=None)[0]
            rss = np.sum(np.abs(y - A_ @ beta) ** 2)
            excess = max(rss - np.mean([np.sum(np.abs(yn[k] - Xn[k][m] @ beta) ** 2)
                                        for k in range(len(Fn))]), 0.0) if noisy else rss
            se = np.sqrt(np.abs(np.diag((rss / max(len(y) - beta.size, 1)) * np.linalg.pinv(AhA)))) + 1e-30
            return beta, excess, se

        bf, _, sf = fit("M1")
        G0 = abs(bf[0]) + 1e-12
        h = np.hypot(abs(bf[1]), abs(bf[2])) * HALF * H / G0          # relative change over 6 mm
        zh = max(abs(bf[1]) / sf[1], abs(bf[2]) / sf[2])
        m = "M1" if (h >= TAU_HET and zh >= Z_MIN) else "M0"          # STLSQ-style pruning
        beta, excess, _ = fit(m)
        out["choice"][y0, x0] = MODELS.index(m)
        out["r2"][y0, x0] = 1 - excess / max(tot, 1e-300)
        out["h"][y0, x0] = h
        out["G_phys"][y0, x0] = np.real(beta[0])
    return out


def true_labels(G3, mask):
    ny, nx = mask.shape
    relstd = np.full(mask.shape, np.nan)
    for y0, x0 in zip(*np.nonzero(mask)):
        ys = slice(max(y0 - HALF, 0), min(y0 + HALF + 1, ny))
        xs = slice(max(x0 - HALF, 0), min(x0 + HALF + 1, nx))
        g = G3[ys, xs][mask[ys, xs]]
        relstd[y0, x0] = g.std() / g.mean()
    return relstd


# ------------------------------------------------------------------- main ---
if __name__ == "__main__":
    files = sorted(PRED.glob("sim_*.npz"))
    if not files:
        raise SystemExit(f"No test predictions in {PRED}. Run stage_d1_evaluate.py first.")
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    log(f"Weak SINDy (60 Hz, M0 vs M1) on {len(files)} test brains, conditions {CONDS}")
    keys = ["choice", "r2", "h", "G_phys", "het", "G_true", "net_err"]
    pool = {c: {k: [] for k in keys} for c in CONDS}
    show = []
    for n, f in enumerate(files):
        sid = int(f.stem.split("_")[1])
        d = np.load(f)
        b = np.load(B_DIR / f"sim_{sid:05d}.npz")
        mask, G3, u3 = d["mask3"], d["G3"].astype(float), b["u3"].astype(complex)
        relstd = true_labels(G3, mask)
        net_err = np.abs(1000 * np.exp(np.nanmean(d["logG_ILI_snr20"], axis=0)) - G3) / G3
        maps = {}
        for c in CONDS:
            if c == "clean":
                r = fit_brain(u3, mask)
            else:
                u = d[f"u_noisy_snr{c}"].astype(complex)          # same noisy data as stage_d1
                sigma = np.median(np.abs(u3[mask])) / c
                r = fit_brain(u, mask, sigma, np.random.default_rng([sid, c, 77]))
            sel = mask & np.isfinite(r["choice"])
            for k in ["choice", "r2", "h", "G_phys"]:
                pool[c][k].append(r[k][sel])
            pool[c]["het"].append((relstd > HET_TRUE)[sel])
            pool[c]["G_true"].append(G3[sel]); pool[c]["net_err"].append(net_err[sel])
            maps[c] = r
        if len(show) < 3:
            show.append(dict(sid=sid, mask=mask, G3=G3, relstd=relstd, maps=maps))
        if (n + 1) % 5 == 0 or n + 1 == len(files):
            log(f"  {n + 1}/{len(files)} brains")

    P = {c: {k: np.concatenate(v) for k, v in pool[c].items()} for c in CONDS}
    res = {}
    print("\n=== Weak-form SINDy at 60 Hz: does it find where stiffness truly varies? (H1) ===")
    print(f"  {'data':>6} | {'H1 AUC':>6} | {'picks M1 where truly varying':>28} | {'picks M1 where uniform':>22} "
          f"| {'physics-only G error':>20} | {'no fit':>6} | voxels")
    for c in CONDS:
        p = P[c]
        het = p["het"].astype(bool)
        a1 = roc_auc_score(het, p["h"]) if 0 < het.mean() < 1 else float("nan")
        m1 = p["choice"] == 1
        gerr = float(100 * np.nanmedian(np.abs(p["G_phys"] - p["G_true"]) / p["G_true"]))
        res[str(c)] = dict(H1_auc=float(a1), M1_when_varying=float(100 * m1[het].mean()),
                           M1_when_uniform=float(100 * m1[~het].mean()), phys_G_median_err_pct=gerr,
                           M0_pct=float(100 * np.mean(~m1)), M1_pct=float(100 * np.mean(m1)),
                           no_fit_pct=float(100 * np.mean(p["r2"] < R2_MIN)), n=int(len(m1)))
        q = res[str(c)]
        print(f"  {str(c):>6} | {a1:6.2f} | {q['M1_when_varying']:27.1f}% | {q['M1_when_uniform']:21.1f}% "
              f"| {gerr:19.1f}% | {q['no_fit_pct']:5.1f}% | {q['n']}")
    print(f"  (true varying windows: {100 * P['clean']['het'].mean():.1f} % of voxels)")

    print("\n=== H4: network error (SNR 20) grouped by what the physics says (SNR 20 data) ===")
    p = P[20]; fits = p["r2"] >= R2_MIN
    groups = {"M0 fits (uniform)": (p["choice"] == 0) & fits,
              "M1 fits (varying)": (p["choice"] == 1) & fits,
              "no model fits": ~fits,
              "all voxels": np.ones(len(fits), bool)}
    res["H4"] = {}
    for name, m in groups.items():
        if m.sum():
            res["H4"][name] = dict(pct=float(100 * m.mean()), median_err=float(100 * np.median(p["net_err"][m])),
                                   bad_pct=float(100 * np.mean(p["net_err"][m] > BAD)))
            print(f"  {name:20s} {100 * m.mean():5.1f} % of voxels | network median error "
                  f"{100 * np.median(p['net_err'][m]):4.1f} % | >20 % wrong {100 * np.mean(p['net_err'][m] > BAD):4.1f} %")
    (OUT / "results.json").write_text(json.dumps(res, indent=1))

    # figure 1: summary
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    xc = [str(c) for c in CONDS]
    ax[0].plot(xc, [res[c]["H1_auc"] for c in xc], "o-", color="#1D9E75", lw=2.5)
    ax[0].axhline(0.5, color="0.7", lw=0.8)
    ax[0].set(ylim=(0.4, 1), ylabel="AUC", title="A  Finds where stiffness truly varies (H1)")
    ax[1].plot(xc, [res[c]["phys_G_median_err_pct"] for c in xc], "o-", color="#1D9E75", lw=2.5)
    ax[1].set(ylabel="median |error| (%)", title="B  Physics-only stiffness error")
    names = [k for k in res["H4"] if k != "all voxels"]
    ax[2].bar(range(len(names)), [res["H4"][k]["bad_pct"] for k in names],
              color=["#9FE1CB", "#378ADD", "#888780"][:len(names)])
    ax[2].axhline(res["H4"]["all voxels"]["bad_pct"], color="0.3", ls=":", lw=1)
    ax[2].text(len(names) - 0.5, res["H4"]["all voxels"]["bad_pct"], " all voxels", va="bottom", fontsize=8)
    ax[2].set_xticks(range(len(names))); ax[2].set_xticklabels(names, fontsize=8)
    ax[2].set(ylabel="% of voxels >20 % wrong", title="C  Network errors by physics verdict (SNR 20)")
    for a in ax[:2]:
        a.set_xlabel("data (noise level)")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "fig_weak_sindy_60hz.png", dpi=150)

    # figure 2: maps
    cmap = ListedColormap(["#9FE1CB", "#378ADD"])
    fig, ax = plt.subplots(len(show), 4, figsize=(13, 3.4 * len(show)), squeeze=False)
    for i, s in enumerate(show):
        m = s["mask"]
        ax[i, 0].imshow(np.where(m, s["G3"] / 1000, np.nan), cmap="magma", vmin=0, vmax=5)
        ax[i, 0].set_title(f"#{s['sid']} true G' (kPa)", fontsize=9)
        ax[i, 1].imshow(np.where(m, s["relstd"] > HET_TRUE, np.nan), cmap=cmap, vmin=0, vmax=1)
        ax[i, 1].set_title("TRUE: uniform (green) / varying (blue)", fontsize=9)
        for j, c in ((2, "clean"), (3, 20)):
            r = s["maps"][c]
            img = np.where(np.isfinite(r["choice"]), r["choice"], np.nan)
            ax[i, j].imshow(np.where(m, 0.5, np.nan), cmap="Greys", vmin=0, vmax=1)      # grey = not covered
            ax[i, j].imshow(img, cmap=cmap, vmin=0, vmax=1)
            nofit = np.where(np.isfinite(r["r2"]) & (r["r2"] < R2_MIN), 1.0, np.nan)
            ax[i, j].imshow(nofit, cmap=ListedColormap(["#D85A30"]), vmin=0, vmax=1)
            ax[i, j].set_title(f"physics picks ({c})\ngrey = edge, not tested; orange = no fit", fontsize=9)
        for a in ax[i]:
            a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "fig_weak_sindy_maps.png", dpi=140)
    log(f"Done. Results in {OUT}")