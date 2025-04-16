# Spherical Field Theory (SFT) Quantum Simulator

This project is a simulation based on the observable forces we know of in 3 dimensional space plus time (3+1D) operating at the Planck/Quantum scale. The model is a computational toy model derived from an effective Lagrangian.

This was built as a way to answer a question that i had: "What if the quantum level was made of discrete interacting spheres?"

I made this simulation, hand-tuned it and was able to achieve metastability of proton-like clusters that were forming.

After that i was able to use the data i recorded and below are the equations that this model now incorporates. 

When switching from hand-tuned forces to Lagrangian based forces, the Kinetic Energy after about 2100+ frames out of 5000 would hit a perfect stability across increasingly larger particle count runs of the simulation.

Yes, this model uses a classical framework. Yes, this model uses force-field based interactions, which are related to force-field based Molecular Dynamics.

Yes, the model is showing possible quantum-like emergent behavior that i did not specifically code.

---

## 📐 Theoretical Foundation

This model incorporates:

- The **effective Lagrangian** captures repulsive, attractive, and confining components through a custom interaction potential.

  ```L = Σᵢ (½ m * |vᵢ|²) - Σ_{i<j} [ A / (3 * |xᵢ - xⱼ|³) - (B / 2) * |xᵢ - xⱼ|² + C * |xᵢ - xⱼ| ]```
  
- A **Hamiltonian diagnostic** tracking kinetic, potential, and total energy at each frame, below is the Hamiltonian equation:

  ```H = Σᵢ (pᵢ² / 2m) + Σ_{i<j} [ A / |xᵢ - xⱼ|³ - B * |xᵢ - xⱼ|² + C * |xᵢ - xⱼ| ]```

- A **Euler-Lagrange** Derived force law:

  ```m * aᵢ = Σ_{j ≠ i} [ (A / |xᵢ - xⱼ|⁴) - B * |xᵢ - xⱼ| - C ] * (xᵢ - xⱼ) / |xᵢ - xⱼ|```

These derivations bring classical modeling closer to quantum-like emergence — with potential for further theoretical analysis.

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

## ✅ Verified Platforms & Setup

This project has been successfully tested across a variety of systems and environments:

- 🐧 **Linux**  
  - Zorin OS  
  - Ubuntu Desktop  
  - Ubuntu Server (headless VM)  
- 🪟 **Windows 10 / 11**  
  - Python via Anaconda  
  - System-installed Python  
- 💻 **Hardware**  
  - Rackmount server (Dell R740)  
  - Gaming desktop (Intel Core i7-12700KF)  
  - Workstation laptop (HP ZBook 15v G5)

---

## 📊 Results & Data Reproducibility

All test environments used the same workflow:

```bash
git clone https://github.com/Beelzebarb/sft.git
cd sft
pip install -r requirements.txt
python sft-cpu.py --headless
```

No manual patching or system-specific modifications required.

✔️ Runs consistently across Python 3.8, 3.10, and 3.12.

Each run generates a full log folder containing:
- `summary.csv` — per-frame energy, cluster, and spatial statistics
- `proton_lifetimes_by_frame.csv` — histogram of proton lifetimes over time
- `quantum_logs.csv` — per-frame quantum activity: spin flips, color flips, timestep, and Hamiltonian diagnostics
- `metadata.json` — config snapshot for full reproducibility

These results have been validated across a range of particle counts (`N=180` to `N=3000`) and environments, demonstrating long-term metastability or full stability.

---

## 📂 Batch Results

Benchmark runs (`N=180` through `N=3000`) are stored under `/results`, each with logs and metadata for reproducibility:

```
results/
├── N180/
├── N250/
├── N500/
├── ...
└── N3000/
```

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

## 🧪 Analysis Tools

*(Coming Soon)*

- Scripts and notebooks to explore `summary.csv` trends (energy, stability)
- Visual diagnostics of proton lifetimes and cluster statistics
- Comparisons between Lagrangian and Hamiltonian dynamics

---

## 🙏 Acknowledgements

Built with guidance from ChatGPT-4 and fueled by curiosity, persistence, and a deep love for physics.

---

## License

MIT License — use freely, cite if you build on it!
