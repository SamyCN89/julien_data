import numpy as np
import os

# --- Config ---
old_npz_path = 'results/speed/speed2m4m_dist_03052024_lag=1_tau=3_wmax=100_wmin=5.npz'
new_npz_path = '/media/samy/Elements1/Proyectos/LauraHarsan/dataset/julien/results/speed/speed_dfc_lag=1_tau=3_wmax=100_wmin=5.npz'

# --- Load files ---
old_data = np.load(old_npz_path, allow_pickle=True)
new_data = np.load(new_npz_path, allow_pickle=True)

print("✅ Loaded files:")
print(f"  Old: {len(old_data['vel'])} animals")
print(f"  New: {len(new_data['vel'])} animals")

assert len(old_data['vel']) == len(new_data['vel']), "❌ Mismatch in number of animals"
assert old_data['speed_median'].shape == new_data['speed_median'].shape, "❌ Mismatch in speed_median shapes"

# --- Compare speed_median arrays ---
if np.allclose(old_data['speed_median'], new_data['speed_median'], atol=1e-6):
    print("✅ speed_median arrays are numerically equivalent")
else:
    diffs = np.abs(old_data['speed_median'] - new_data['speed_median'])
    print("⚠️ Minor differences in speed_median:")
    print(f"  Max diff:  {np.max(diffs)}")
    print(f"  Mean diff: {np.mean(diffs)}")

# --- Helper to flatten any object-array entry ---
def flatten_speed_entry(entry):
    if isinstance(entry, (list, np.ndarray)):
        flat = np.concatenate([np.ravel(e) for e in entry])
        return flat.astype(np.float64)
    else:
        raise ValueError("Speed entry is not array-like.")

# --- Compare all animals ---
print("\n🔍 Comparing speed distributions for all animals...\n")
n_mismatch = 0
for idx in range(len(old_data['vel'])):
    try:
        vel_old = flatten_speed_entry(old_data['vel'][idx])
        vel_new = flatten_speed_entry(new_data['vel'][idx])

        if vel_old.shape != vel_new.shape:
            print(f"⚠️ Animal {idx}: shape mismatch (old: {vel_old.shape}, new: {vel_new.shape})")
            n_mismatch += 1
        elif not np.allclose(vel_old, vel_new, atol=1e-6):
            diff = np.abs(vel_old - vel_new)
            print(f"⚠️ Animal {idx}: numerical mismatch — mean diff: {np.mean(diff):.3e}, max diff: {np.max(diff):.3e}")
            n_mismatch += 1
        else:
            print(f"✅ Animal {idx}: match")
    except Exception as e:
        print(f"❌ Animal {idx}: error during comparison — {e}")
        n_mismatch += 1

# --- Summary ---
print("\n✅ Comparison complete.")
if n_mismatch == 0:
    print("🎉 All animals match!")
else:
    print(f"⚠️ {n_mismatch} animal(s) had mismatches or errors.")
