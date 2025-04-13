# Spherical Field Theory (SFT) Quantum Simulator - N=3000 Run Details

This is a resultset from the simulation running N=3000 particles for 5000 frames.

## ⚙️ Configuration

```json
{
    "N": 3000,
    "planck_length": 5.0,
    "binding_energy": 800.0,
    "pauli_strength": 0.25,
    "initial_dt": 0.001,
    "max_frames": 5000,
    "use_gravity": true,
    "enable_color_flips": true
}
```

## 📁 Output Files
- `metadata.json`: Full config for reproduction
- `summary.csv`: Per-frame stats (KE, proton counts, distances)
- `proton_lifetimes_by_frame.csv`: Histogram of proton lifetimes every 10 frames
- `quantum_logs.csv`: Spin flips, color flips, DT and KE