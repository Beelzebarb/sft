from numba import njit, prange
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import cKDTree, distance
from scipy.spatial.distance import pdist
from collections import Counter
import os, json, csv

# Quantum Physics Parameters
L_unit = 0.12              	# PLANCK_LENGTH (used for scaling forces/distances?)
E_unit = 800.0             	# BINDING_ENERGY (used for normalization / maybe force constants)
PLANCK_LENGTH = 5.0        	# Actual working interaction scale
PAULI_STRENGTH = 0.25      	# Effective Pauli repulsion factor
N = 500                   	# Particle count
MAX_FRAMES = 5000			# Maximum Frames to run
HEADLESS = False			# Headless mode toggle

#Global proton state trackers
proton_lifetimes = {}
previous_proton_ids = set()

class QuantumUniverse:
	def __init__(self, config_path=None, output_dir=None, verbose=False):
		# Defaults
		self.use_gravity = True
		self.enable_color_flips = True
		self.DT = 0.001
		self.frame = 0
		self.verbose = verbose

		global N, PLANCK_LENGTH, E_unit, PAULI_STRENGTH, MAX_FRAMES

		# Load overrides from metadata.json if provided
		if config_path:
			with open(config_path, 'r') as f:
				config = json.load(f)
			N = config.get("N", N)
			PLANCK_LENGTH = config.get("planck_length", PLANCK_LENGTH)
			E_unit = config.get("binding_energy", E_unit)
			PAULI_STRENGTH = config.get("pauli_strength", PAULI_STRENGTH)
			self.DT = config.get("initial_dt", self.DT)
			MAX_FRAMES = config.get("max_frames", MAX_FRAMES)
			self.use_gravity = config.get("use_gravity", self.use_gravity)
			self.enable_color_flips = config.get("enable_color_flips", self.enable_color_flips)

		self.stable_proton_clusters = set()
		self.proton_birth_frames = {}
		self.proton_death_frames = {}
		self.frame = 0

		# Create log directory
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		if output_dir:
			self.log_dir = output_dir
		else:
			self.log_dir = os.path.join("logs", f"run_{timestamp}")
		os.makedirs(self.log_dir, exist_ok=True)

		# Initialize particle positions and states
		box_size = 6.0
		grid_spacing = 0.35
		num_per_axis = int(box_size // grid_spacing)
		coords = np.linspace(0, box_size, num=num_per_axis)
		positions = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
		np.random.shuffle(positions)
		self.positions = positions[:N]

		self.velocities = np.random.randn(N, 3) * 0.01
		self.spins = np.random.choice([-1, 1], N)
		self.colors = np.random.randint(0, 2, N)
		self.lifetimes = np.zeros(N)

		self.last_spin_flips = 0
		self.last_color_flip = 0

		# Write metadata (optional: only if no config was passed)
		if not config_path:
			meta = {
				"N": N,
				"planck_length": PLANCK_LENGTH,
				"binding_energy": E_unit,
				"pauli_strength": PAULI_STRENGTH,
				"initial_dt": self.DT,
				"max_frames": MAX_FRAMES,
				"use_gravity": self.use_gravity,
				"enable_color_flips": self.enable_color_flips
			}
			with open(os.path.join(self.log_dir, "metadata.json"), "w") as f:
				json.dump(meta, f, indent=4)

	def hamiltonian_energy(self, A=1.0, B=0.1, C=0.01):
		return compute_hamiltonian_energy(self.positions, self.velocities, N, A, B, C)

	def apply_energy_dissipation(self):
		ke = self.kinetic_energy()

		if ke > 10:
			factor = 0.99
			if self.verbose:
				print(f"High KE detected: {ke:.5f}, applying strong damping...")
		elif ke > 1:
			factor = 0.997
		else:
			factor = 0.999  # Minimal damping during calm periods

		self.velocities *= factor

	def gradual_proton_birth(self, rate=0.005):
		# Gradually add protons at a specific rate over time
		num_new_protons = int(rate * N)  # Adjust the rate as necessary
		new_protons = np.random.choice(N, num_new_protons, replace=False)
		# Modify the positions, spins, and colors of the new protons
		self.spins[new_protons] = np.random.choice([-1, 1], num_new_protons)
		self.colors[new_protons] = np.random.randint(0, 2, num_new_protons)
		return new_protons  # Return the new proton IDs

	# Gravity scaling: Increase gravity effect as spheres compress (based on density or distance)
	def scale_gravity_effect(self):
		avg_distance = np.mean(pdist(self.positions))
		scaling_factor = 1 / (avg_distance ** 2)  # Gravity effect increases as density increases

		# Limit gravity at high densities to prevent runaway acceleration
		max_scaling = 5  # Maximum scaling factor to limit gravity at high densities
		scaling_factor = min(scaling_factor, max_scaling)  # Cap the scaling factor

		# Apply a smoother scaling to avoid fluctuations
		if hasattr(self, 'previous_scaling_factor'):
			smoothing_factor = 0.95  # Smoother scaling factor
			scaling_factor = smoothing_factor * self.previous_scaling_factor + (1 - smoothing_factor) * scaling_factor
		self.previous_scaling_factor = scaling_factor  # Store for next time

		return scaling_factor

	def kinetic_energy(self):
		return 0.5 * np.sum(self.velocities ** 2)

	def update(self):
		self.lifetimes += self.DT
		decohere = self.lifetimes > 0.3 + np.random.rand(N) * 0.2
		self.last_spin_flips = np.count_nonzero(decohere)
		self.spins[decohere] *= -1
		self.lifetimes[decohere] = 0
		current_frame = self.frame
		
		self.last_color_flip = 0  # reset every frame

		if self.enable_color_flips:
			color_flip_chance = 0.002  # adjustable!
			flip_mask = np.random.rand(N) < color_flip_chance
			prev_colors = self.colors.copy()
			self.colors[flip_mask] = 1 - self.colors[flip_mask]
			self.last_color_flip = np.count_nonzero(prev_colors != self.colors)
		
		# Track proton births/deaths
		new_proton_ids = self.gradual_proton_birth()  # Track newly born protons
		if len(new_proton_ids) > 0:
			if self.verbose:
				print(f"New protons born: {len(new_proton_ids)}")
			# Gradual damping effect after protons are added
			self.velocities *= 0.995  # Apply stronger damping after protons are added
		
		#forces = compute_forces_numba(
		#	self.positions, self.spins, self.colors, N,
		#	PLANCK_LENGTH, PAULI_STRENGTH,
		#	self.use_gravity
		#)
		
		forces = compute_effective_lagrangian_forces(
			self.positions, N, A=1.0, B=0.1, C=0.01
		)
		
		# Apply gravity scaling (more gradual change during proton events)
		scaling_factor = self.scale_gravity_effect()
		forces *= scaling_factor

		# Symplectic integration to update positions and velocities
		force_mags = np.linalg.norm(forces, axis=1)
		max_force = 50.0
		too_high_force = force_mags > max_force
		forces[too_high_force] *= (max_force / force_mags[too_high_force])[:, None]
		
		#print(f"Mean velocity: {np.linalg.norm(self.velocities, axis=1).mean():.5f}")
		
		# Adaptive DT calculation
		force_scale = np.max(np.linalg.norm(forces, axis=1)) + 1e-8
		velocity_scale = np.max(np.linalg.norm(self.velocities, axis=1)) + 1e-8

		# Calculate candidate DT based on current forces and velocities
		DT_dynamic = 0.001 * min(1.0, 0.1 / force_scale, 0.1 / velocity_scale)

		# Smooth the change to avoid jitter
		self.DT = 0.9 * getattr(self, 'DT', 0.001) + 0.1 * DT_dynamic
		
		self.velocities += forces * self.DT
		velocity_mags = np.linalg.norm(self.velocities, axis=1)
		max_velocity = 2.0
		too_fast = velocity_mags > max_velocity
		self.velocities[too_fast] *= (max_velocity / velocity_mags[too_fast])[:, None]

		# Apply a smaller overall damping factor to prevent runaway energy
		#self.velocities *= 0.999  # Less aggressive damping - commented out, possible stacking dampening happening from this and apply_energy_dissipation().
		
		#print(f"Mean velocity: {np.linalg.norm(self.velocities, axis=1).mean():.5f}")
		
		self.apply_energy_dissipation() # Moved here after velocities are set but before positions are set.

		self.positions += self.velocities * self.DT
		self.positions = np.clip(self.positions, 0, 6)

		return self.kinetic_energy()

# Initialize quantum universe
quantum  = QuantumUniverse()

@njit(parallel=True)
def compute_hamiltonian_energy(positions, velocities, N, A, B, C):
    # Kinetic Energy
    ke = 0.0
    for i in prange(N):
        ke += 0.5 * np.dot(velocities[i], velocities[i])

    # Potential Energy
    pe = 0.0
    for i in prange(N):
        for j in range(i + 1, N):
            delta = positions[i] - positions[j]
            dist = np.sqrt(np.dot(delta, delta)) + 1e-8
            pe += A / dist**3 - B * dist**2 + C * dist

    return ke, pe, ke + pe

@njit(parallel=True)
def compute_effective_lagrangian_forces(positions: np.ndarray, N: int, A: float, B: float, C: float) -> np.ndarray:
	forces = np.zeros_like(positions)
	for i in prange(N):
		for j in range(N):
			if i == j:
				continue

			delta = positions[i] - positions[j]
			dist = np.linalg.norm(delta) + 1e-8  # Avoid div by zero
			dir_ = delta / dist

			# Effective Lagrangian-derived forces
			f1 = 3 * A * dir_ / dist**4         # Repulsive inverse-cube
			f2 = 2 * B * dist * dir_             # Quadratic attractive
			f3 = -C * dir_                       # Linear attractive

			force = f1 + f2 + f3
			forces[i] += force
	return forces

def append_event_log(frame, spin_flips, color_flip):
	file_path = "quantum_events.csv"
	write_header = not os.path.exists(file_path) or os.path.getsize(file_path) == 0

	with open(file_path, "a") as f:
		if write_header:
			f.write("frame,event,value\n")
		if spin_flips > 0:
			f.write(f"{frame},spin,{spin_flips}\n")
		if color_flip:
			f.write(f"{frame},color,1\n")

def write_summary_row(
	logdir, frame, ke, stable_count, unstable_count,
	just_born, just_died, spin_flips, color_flips,
	cluster_radius_stats, min_dist, max_dist, mean_dist
):
	summary_path = os.path.join(logdir, "summary.csv")
	file_exists = os.path.isfile(summary_path)

	with open(summary_path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow([
				"Frame", "KE",
				"StableProtons", "UnstableParticles",
				"NewBorn", "Dissolved",
				"SpinFlips", "ColorFlips",
				"AvgClusterSize", "MinClusterSize", "MaxClusterSize",
				"MinDistance", "MaxDistance", "MeanDistance"
			])
		writer.writerow([
			frame, f"{ke:.5f}",
			stable_count, unstable_count,
			len(just_born), len(just_died),
			spin_flips, color_flips,
			round(np.mean(cluster_radius_stats), 2),
			min(cluster_radius_stats),
			max(cluster_radius_stats),
			round(min_dist, 5), round(max_dist, 5), round(mean_dist, 5)
		])

def write_lifetime_histogram_row(logdir, frame, lifetime_bins):
	path = os.path.join(logdir, "proton_lifetimes_by_frame.csv")
	file_exists = os.path.isfile(path)

	with open(path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["Frame", "LifetimeBin", "ProtonCount"])

		for bucket, count in sorted(lifetime_bins.items()):
			writer.writerow([frame, bucket, count])

def write_quantum_log_row(logdir, frame, spin_flips, color_flips, dt, ke, pe, hamiltonian):
    path = os.path.join(logdir, "quantum_logs.csv")
    file_exists = os.path.isfile(path)

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Frame", "SpinFlips", "ColorFlips", "DT",
                "KineticEnergy", "PotentialEnergy", "HamiltonianEnergy"
            ])
        writer.writerow([
            frame, spin_flips, color_flips, round(dt, 6),
            round(ke, 6), round(pe, 6), round(hamiltonian, 6)
        ])

def update(frame):
	global previous_proton_ids
	
	ke = quantum.update()
	
	radius = 0.1 * PLANCK_LENGTH
	
	append_event_log(frame, quantum.last_spin_flips, quantum.last_color_flip)
	
	tree = cKDTree(quantum.positions)
	proton_clusters = []
	cluster_radius_stats = []
	unstable_clusters = set()
	new_proton_ids = set()
	for i in range(N):
		cluster = tree.query_ball_point(quantum.positions[i], radius)
		cluster = [j for j in cluster if j != i]  # NOW back in
		cluster_radius_stats.append(len(cluster))

		if len(cluster) == 2:
			full_cluster = [i] + cluster
			cluster_id = frozenset(full_cluster)
			spins = quantum.spins[full_cluster]
			colors = quantum.colors[full_cluster]
			unique_colors = set(colors)
			
			if len(unique_colors) in (1, 2) and abs(sum(spins)) <= 1:
				counts = [list(colors).count(c) for c in unique_colors]
				if sorted(counts) == [1, 2]:
					new_proton_ids.add(cluster_id)
					proton_clusters.extend(full_cluster)
					continue

			if cluster_id in quantum.stable_proton_clusters:
				if quantum.verbose or HEADLESS:
					print("Already a stable proton!")
			unstable_clusters.update(full_cluster)
#		elif len(cluster) == 2:
#			unstable_clusters.update([i] + cluster)
	if quantum.verbose or HEADLESS:
		print(f"Min: {min(cluster_radius_stats)}, Max: {max(cluster_radius_stats)}, Avg: {np.mean(cluster_radius_stats):.2f}")
	# Log counts
	stable_count = len(set(proton_clusters)) // 3
	unstable_count = len(unstable_clusters)

	# Lifetimes
	for pid in new_proton_ids:
		proton_lifetimes[pid] = proton_lifetimes.get(pid, 0) + 1

	distances = pdist(quantum.positions)
	if quantum.verbose or HEADLESS:
		print("Min distance:", np.min(distances))
		print("Max distance:", np.max(distances))
		print("Mean distance:", np.mean(distances))

	# Track proton birth/death
	just_died = previous_proton_ids - new_proton_ids
	
	# Record proton death frames
	for cluster in just_died:
		if cluster not in quantum.proton_death_frames:
			quantum.proton_death_frames[cluster] = frame
	
	just_born = new_proton_ids - previous_proton_ids
	previous_proton_ids = new_proton_ids
	
	#Store stable proton clusters
	quantum.stable_proton_clusters = new_proton_ids
	
	# Track proton birth frames
	for cluster in just_born:
		if cluster not in quantum.proton_birth_frames:
			quantum.proton_birth_frames[cluster] = frame

	# Compute lifetime of existing protons
	lifetimes = {
		cluster: frame - quantum.proton_birth_frames[cluster]
		for cluster in quantum.stable_proton_clusters
		if cluster in quantum.proton_birth_frames
	}
	if quantum.verbose or HEADLESS:
		# Terminal logging
		print(f"Frame {frame:4}: Stable Protons: {stable_count:2} | Unstable Particles: {unstable_count:3} | KE: {quantum.kinetic_energy():.5f}")
		print(f"New Protons: {len(just_born)} | Dissolved: {len(just_died)} | Long-lived: {sum(1 for v in proton_lifetimes.values() if v > 20)}")

		if quantum.last_spin_flips > 0 or quantum.last_color_flip > 0:
			print(f"Spin Flips: {quantum.last_spin_flips} | Color Flip: {quantum.last_color_flip}")

		cluster_sizes = Counter(len(tree.query_ball_point(quantum.positions[i], 2.2 * PLANCK_LENGTH)) for i in range(N))
		print(f"Cluster sizes: {dict(cluster_sizes)}")

		# Optionally show detailed changes
		if just_born:
			print(f" Born: {sorted(just_born)}")
		if just_died:
			print(f" Died: {sorted(just_died)}")

	# Color handling as before
	colors_array = np.array(['blue' if c == 1 else 'red' for c in quantum.colors], dtype='<U5')
	for idx in set(proton_clusters):
		colors_array[idx] = 'green'

	if not HEADLESS:
		scatter._offsets3d = (quantum.positions[:, 0],
							  quantum.positions[:, 1],
							  quantum.positions[:, 2])
		scatter.set_color(colors_array)
		scatter.set_sizes(np.where(colors_array == 'green', 800, 400))
	
	write_summary_row(
		quantum.log_dir, frame, ke,
		stable_count, unstable_count,
		just_born, just_died,
		quantum.last_spin_flips, quantum.last_color_flip,
		cluster_radius_stats,
		np.min(distances), np.max(distances), np.mean(distances)
	)
	
	ke, pe, h_total = compute_hamiltonian_energy(
		quantum.positions, quantum.velocities, N,
		A=0.005, B=0.002, C=0.001
	)
	write_quantum_log_row(
		quantum.log_dir,
		frame,
		quantum.last_spin_flips,
		quantum.last_color_flip,
		quantum.DT,
		ke, pe, h_total
	)
		
	if frame % 10 == 0:
		lifetimes = {
			cluster: frame - quantum.proton_birth_frames[cluster]
			for cluster in quantum.stable_proton_clusters
			if cluster in quantum.proton_birth_frames
		}
		bin_size = 10
		lifetime_bins = defaultdict(int)
		for life in lifetimes.values():
			bucket = (life // bin_size) * bin_size
			lifetime_bins[bucket] += 1

		write_lifetime_histogram_row(quantum.log_dir, frame, lifetime_bins)
	
	cal = MAX_FRAMES - 1
	if frame == cal:
		print("Simulation complete. Closing animation.")
		if not HEADLESS:
			plt.close()  # This will close the window when the last frame is rendered
	
	return (scatter,) if not HEADLESS else None

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run the Quantum Sphere Simulation")
	parser.add_argument('--config', type=str, help="Path to metadata.json config file")
	parser.add_argument('--headless', action='store_true', help="Run simulation without visualization")
	parser.add_argument('--outdir', type=str, help="Optional output directory to save logs.")
	parser.add_argument('--verbose', action='store_true', help="Print live simulation output to terminal.")
	args = parser.parse_args()
	
	if args.config and not os.path.isfile(args.config):
		print(f"[Error] Metadata file not found: {args.config}")
		sys.exit(1)

	quantum = QuantumUniverse(config_path=args.config, output_dir=args.outdir, verbose=args.verbose)

	if args.headless:
		HEADLESS = args.headless
		for frame in range(MAX_FRAMES):
			update(frame)  # Executes simulation + logging
	else:
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim(0, 6)
		ax.set_ylim(0, 6)
		ax.set_zlim(0, 6)
		scatter = ax.scatter(
			quantum.positions[:, 0],
			quantum.positions[:, 1],
			quantum.positions[:, 2],
			s=600, c=quantum.spins, cmap='coolwarm', alpha=0.9
		)
		ani = FuncAnimation(fig, update, frames=MAX_FRAMES, interval=20, blit=False)
		plt.show()
