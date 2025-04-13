# Spherical Field Theory (SFT) Quantum Simulator

This project is an interactive and headless-capable simulation of a theoretical model of quantum behavior based on **discrete interacting spheres** in a 3D grid — a working sandbox of **spin, color, force, and emergence** at quantum scales.

> Built from scratch in just 5 days using Python + Numba, this simulation showcases stable metastable behavior, emergent proton-like clusters, and self-regulating kinetic energy from first principles. No assumptions beyond 3D+time.

---

## 💡 Features

- **Metastability**: Proton birth and decay from interacting triplet clusters.
- **Emergent behavior** from Pauli exclusion, confinement, nuclear attraction, and gravity.
- **Self-regulating DT (timestep)** for consistent and stable simulation speeds.
- **Scalable**: Verified up to `N=3000` particles, cross-platform stable.
- **Headless mode**: Run high-N simulations without graphical overhead.
- **Logging**: Summary, proton lifetimes (histogram), quantum events per frame.
- **Replayable**: Supply `metadata.json` from a previous run to recreate identical behavior.

---

## 📦 Requirements

- Python 3.8+
- Numba
- NumPy
- SciPy
- Matplotlib

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Simulation

### Graphical mode (default):

```bash
python sft-cpu.py
```

### Headless mode with metadata replay:

```bash
python sft-cpu.py --config logs/run_YYYYMMDD_HHMMSS/metadata.json --headless
```

---

## 📁 Output Files

All logs are written to `logs/run_TIMESTAMP/`, including:

- `metadata.json`: Full config for reproduction
- `summary.csv`: Per-frame stats (KE, proton counts, distances)
- `proton_lifetimes_by_frame.csv`: Histogram of proton lifetimes every 10 frames
- `quantum_logs.csv`: Spin flips, color flips, DT and KE

---

## ⚙️ Configuration

Adjust settings either in `sft-cpu.py` before running or via `metadata.json`. Key parameters include:

```json
{
  "N": 1000,
  "planck_length": 5.0,
  "binding_energy": 800.0,
  "pauli_strength": 0.25,
  "initial_dt": 0.001,
  "max_frames": 5000,
  "use_gravity": true,
  "enable_color_flips": true
}
```

---

## 🧠 What Is This?

A testbed for **Spherical Field Theory** (SFT), a novel idea that quantum behavior may emerge from interactions between discrete spheres under force laws that reproduce exclusion, binding, and attraction. It offers a new route to model **zero point energy**, **proton formation**, and possibly **quantum fluctuations** from classical mechanics.

---

## 📍 Status

- ✅ Works on Linux and Windows (CLI and graphical)
- ✅ Reproducible metastability
- ✅ Fast multithreaded performance (via Numba)
- 🔜 Further analysis tooling and visualizers

---

## 🙏 Acknowledgements

Built with guidance from ChatGPT-4 and fueled by curiosity, persistence, and a deep love for physics.

---

## License

MIT License — use freely, cite if you build on it!
