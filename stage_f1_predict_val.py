"""
Stage F1 - Run the trained ILI networks on the VALIDATION brains (60 Hz)
=======================================================================
Why: trust maps from now on are DEVELOPED on validation brains and tested on the
test brains only ONCE at the end. The validation brains were only used to decide
when training stops, so they are a fair practice ground.

Same networks, same phase cycling, same noise model as stage_d1_evaluate.py.
Noise seeds differ from the test set ([sid, snr, 7] vs [sid, snr]).

Output: mre_project/stage_f/predictions_val/sim_XXXXX.npz
        (same keys as the test predictions: u_noisy_snr*, logG_ILI_snr*, xi_ILI_snr*)
"""
import csv, time
from pathlib import Path
import numpy as np
import mre_patches as P
from stage_d1_evaluate import load_nets, predict

SNRS = [5, 10, 20, 50]
BASE = Path(__file__).resolve().parent
B_DIR = BASE / "mre_project" / "stage_b" / "variant_A"
OUT = BASE / "mre_project" / "stage_f" / "predictions_val"
OUT.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    t0 = time.time()
    nets = load_nets("ILI")
    print(f"Loaded {len(nets)} ILI networks", flush=True)
    with open(BASE / "mre_project" / "stage_b" / "catalog_b.csv", newline="") as f:
        ids = sorted(int(r["sim_id"]) for r in csv.DictReader(f) if r["split"] == "val")
    print(f"{len(ids)} validation brains", flush=True)

    errs = {snr: [] for snr in SNRS}
    for n, sid in enumerate(ids):
        out = OUT / f"sim_{sid:05d}.npz"
        d = np.load(B_DIR / f"sim_{sid:05d}.npz")
        u3, mask, G3 = d["u3"].astype(complex), d["mask3"], d["G3"]
        if out.exists():                                   # restartable
            old = np.load(out)
            for snr in SNRS:
                G = 1000 * np.exp(np.nanmean(old[f"logG_ILI_snr{snr}"], axis=0))
                errs[snr].append(np.abs(G[mask] - G3[mask]) / G3[mask])
            continue
        save = dict(G3=G3, xi3=d["xi3"], mask3=mask)
        for snr in SNRS:
            rng = np.random.default_rng([sid, snr, 7])     # own seeds, never reused for test
            un = P.add_noise(u3, mask, snr, rng)
            lg, xi = predict(nets, un, mask)
            save[f"u_noisy_snr{snr}"] = un.astype(np.complex64)
            save[f"logG_ILI_snr{snr}"] = lg
            save[f"xi_ILI_snr{snr}"] = xi
            G = 1000 * np.exp(np.nanmean(lg, axis=0))
            errs[snr].append(np.abs(G[mask] - G3[mask]) / G3[mask])
        np.savez_compressed(out, **save)
        if (n + 1) % 15 == 0 or n + 1 == len(ids):
            print(f"  {n + 1}/{len(ids)} brains ({(time.time() - t0) / 60:.1f} min)", flush=True)

    print("\nSanity check - ILI median |error| on validation brains (all brain voxels):")
    for snr in SNRS:
        e = np.concatenate(errs[snr])
        print(f"  SNR {snr:>2}: {100 * np.nanmedian(e):5.1f} %   "
              f"voxels >20 % wrong: {100 * np.nanmean(e > 0.20):4.1f} %")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Saved in {OUT}")