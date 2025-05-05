# Spherical Field Theory (SFT) Quantum Simulator

Spherical Field Theory is the idea that at the most basic and small scale, the Planck-scale, that there exist discrete interacting spheres that only interact with local neighbours.

The idea for this theory came from the brilliance of Paul Dirac's work, and his belief in total and complete locality. This principle is the entire thing that allows SFT to work as it does. 

This project has had a lot of ups and downs, I thought it was dead when I couldn't get the system to function without scaffolding holding it up in the code it was dead, but that has been resolved.
It now maintains stability through 100,000 frames deterministic across linux and windows and desktop and server hardware still, but the force equations have been completely redone.

The entire theory now operates on a custom usage of the Morse potential and a new implementation named Dirac Core Pressure, which acts as a short-range quantum exclusion force.

**Personal Note:** This entire idea is because of **Paul Dirac**, and his vision and philosophy of **locality**. He inspired me to have the idea, to make the simulation and now because of that I decided to name the short-range quantum exclusion force as the Dirac Core Pressure, his philosophy and work live on.

Also, Philip M. Morse, who developed the Morse potential now used in this theory and the simulation, thank you for your amazing and excellent work, it lives on today as well inside SFT.

---

## Observed Phenomena

- **Mass Emergence from Energy**
  - Particles slow as they cluster, converting kinetic energy into localized potential energy, perceived as “mass.”
  - Denser clusters experience stronger gravity-like attraction and resistance to acceleration.

- **Relativistic Behavior (Emergent, Not Imposed)**
  - Time dilation arises naturally from local density and force feedback.
  - No speed limit is programmed — but one emerges.
  - Energy conversion and mass-like inertia occur from local motion alone.

- **SpinStorm Events (Decoherence Storms)**
  - Massive bursts of spin flips triggered by force/density thresholds.
  - Only arise in **deterministic, sequential runs**.
  - Display repeatable start/end frames, cluster behavior, and energy shifts.

- **Black Hole and Neutron Star Analogs**
  - At extreme density, SFT simulates:
    - **Diracium (Black Hole analogs)**: Perfect KE suppression, total gravitational collapse.
    - **Morsium (Neutron Star analogs)**: Quasi-static structure with creeping energy rise, potential cause of **starquakes**.

---

## Technical Highlights

- Built in Python, Numba-accelerated, fully multithreaded.
- Cross-platform: confirmed on Windows/Linux, desktop/server, virtual/baremetal.
- Deterministic SHA256-locked execution with full per-run verification.
- Extensible sensor suite (PyQt-based) with real-time entropy, KE, cluster, and storm detection.

---

*Note: All emergent phenomena arise strictly from local force interaction. Nothing is hardcoded. Nothing is assumed.*

---

## 📐 Theoretical Foundation


The custom potential energy function combines attractive and repulsive terms through a Morse-like interaction with short-range Dirac Core Pressure:

	V(r) = D * (1 - exp(-α * (r - r₀)))² - k / r   (if r < cutoff)

The force derived from this potential (negative gradient of V) is split into two terms:

	F_total = (F_morse + F_dcp) * (xᵢ - xⱼ) / |xᵢ - xⱼ|
Where:

		F_morse = -2 * D * α * (1 - e^(-α * (r - r₀))) * e^(-α * (r - r₀))

		F_dcp = k / r² (only active if r < cutoff)

The adaptive timestep DT is dynamically calculated every frame using local velocity and force magnitudes:

	DT = 0.001 * min(1.0, 0.1 / max_force, 0.1 / max_velocity)

Then smoothed:
	
	self.DT = 0.9 * self.DT + 0.1 * DT_dynamic

---

## 💡 Features

- **Metastability**: Proton birth and decay from interacting triplet clusters.
- **Emergent behavior** from Dirac Core Pressure, Morse potential and gravity scaling by energy.
- **Self-regulating DT (timestep)** for consistent and stable simulation speeds.
- **Scalable**: Verified up to `N=3000` particles, cross-platform stable.
- **Headless mode**: Simulation engine is entirely headless, no GUI, GUI is separated and connects via NetworkBeam.
- **Logging**: Summary, quantum events per frame, proton binding energy per frame, cluster lifetimes by size.
- **Replayable**: Supply `metadata.json` from a previous run to recreate identical behavior.

---

## 📦 Requirements

- Python 3.8+
- Numba
- NumPy
- SciPy
- msgpack
- zstandard

Install with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Simulation

### Graphical mode (default):

```bash
python sft_qsim.py
```

### Headless mode with metadata replay:

```bash
python sft_qsim.py --config logs/run_YYYYMMDD_HHMMSS/metadata.json --headless
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
python sft_qsim.py --headless
```

No manual patching or system-specific modifications required.

✔️ Runs consistently across Python 3.8, 3.10, and 3.12.

Each run generates a full log folder containing:
- `summary.csv` — per-frame energy, cluster, and spatial statistics
- `cluster_lifetimes_size_x.csv` - Tracks cluster lifetimes by size x is replaced with cluster size, each gets its own csv for automation.
- `proton_binding_energy.csv` - per frame Proton Binding Energy statistics.
- `quantum_logs.csv` — per-frame quantum activity: spin flips, color flips, timestep, and Hamiltonian diagnostics
- `metadata.json` — config snapshot for full reproducibility

These results have been validated across a range of particle counts (`N=180` to `N=3000`) and environments, demonstrating long-term metastability or full stability.

---

## 📂 Batch Results

Benchmark runs will soon be found under the folder results_v2, once i get them uploaded.

---

## ⚙️ Configuration

Adjust settings either in `sft_qsim.py` before running or via `metadata.json`. Key parameters include:

```json
{
	"N": 2000,
	"D": 0.3,
	"r0": 0.3,
	"alpha": 3.0,
	"dcp_k": 0.001,
	"dcp_cutoff": 0.2,
	"use_seed": true,
	"seed_value": 42,
	"planck_length": 5.0,
	"initial_dt": 0.001,
	"max_frames": 50000,
	"use_gravity": true,
	"enable_color_flips": true,
	"cold_start": true,
	"fast": false,
	"cluster_energy_input": "np_array_int32",
	"reflected_list_mode": false,
    	"source_hash": "c4f1def8d926799da9e506b0fb5b2dc1e50386cccd8da6cf53a61322518257da"
}
```

source_hash -> Changes based on source code changes. Deterministic --config runs do not allow different versions to run them as results cannot be guaranteed.

---

## 🧠 What Is This?

A testbed for **Spherical Field Theory** (SFT), a novel idea that quantum behavior may emerge from interactions between discrete spheres under force laws that reproduce exclusion, binding, and attraction. It offers a new route to model **zero point energy**, **proton formation**, and possibly **quantum fluctuations** from classical mechanics.

---

## 📍 Status

- ✅ Works on Linux and Windows (CLI-based engine, GUI is SFT Q3S and is separate and connects to the siim via NetworkBeam)
- ✅ Reproducible metastability
- ✅ Fast multithreaded performance (via Numba)
- ✅ Further analysis tooling and visualizers (Done!) - SFT Q3S is being finalized, and CLI tools have already been made, first release will contain all of them.

---

## 🧪 Analysis Tools

*(Coming Soon)*

- Scripts and notebooks to explore `summary.csv` trends (energy, stability)
- Visual diagnostics of proton lifetimes and cluster statistics
- Comparisons between Lagrangian and Hamiltonian dynamics

---

## 🙏 Acknowledgements


- **Paul Dirac** - Discovered antimatter and the Dirac equation, his belief in **locality** inspired me to build this. I named the Dirac Core Pressure after him, which is a custom short-range quantum exclusion force.
- **Philip M. Morse** - Inventor of the Morse Potential which is used along the Dirac Core Pressure to stabilize the quantum universe. Without it, this would never work.
- **ChatGPT** helped me with the calculus, force derivations and very tedious debugging of issues in the early days.

---

## License

MIT License — use freely, cite if you build on it!
