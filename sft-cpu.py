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
import os, json, csv, time, math

# Quantum Physics Parameters
L_unit = 0.12              	# PLANCK_LENGTH (used for scaling forces/distances?)
E_unit = 800.0             	# BINDING_ENERGY (used for normalization / maybe force constants)
PLANCK_LENGTH = 5.0        	# Actual working interaction scale
PAULI_STRENGTH = 0.25      	# Effective Pauli repulsion factor

# Simulation Parameters
MAX_FRAMES = 5000			# Maximum Frames to run
BOX_SIZE = 6.0				# Size of box/cube
HEADLESS = False			# Headless mode toggle

#Global proton state trackers
proton_lifetimes = {}
previous_proton_ids = set()
start_time = time.time()

class QuantumUniverse:
	def __init__(self, config_path=None, output_dir=None, verbose=False):
		# Defaults
		self.use_gravity = True
		self.enable_color_flips = False
		self.DT = 0.001
		self.frame = 0
		self.verbose = verbose
		self.BOX_SIZE = BOX_SIZE
		
		self.A = 1.0
		self.B = +0.01
		self.C = +0.005
		
		self.epsilon = 1.0    # Depth of potential well (how strong binding is)
		self.sigma = 1.0      # Preferred separation distance

		global PLANCK_LENGTH, E_unit, PAULI_STRENGTH, MAX_FRAMES
		
		#Default frames, will change with loaded config automatically.#
		N = 500

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
		
		# For stable proton-like clusters (3-sphere clusters with charge of +1 or superprotons, which are +2.
		self.stable_proton_clusters = set()
		self.proton_birth_frames = {}
		self.proton_death_frames = {}
		self.proton_charges = {}
		
		# For non-proton cluster tracking by cluster size
		self.cluster_birth_frames = defaultdict(dict)     # {size: {cluster_id: frame}}
		self.cluster_death_frames = defaultdict(dict)     # {size: {cluster_id: frame}}
		self.cluster_lifetimes = defaultdict(dict)        # {size: {cluster_id: lifetime}}

		# Current active clusters by size (used per frame)
		self.current_clusters_by_size = defaultdict(set)
		self.previous_clusters_by_size = defaultdict(set)
		
		self.superproton_ids = set()
		self.proton_ids = set()
		self.unstable_clusters = set()

		# Create log directory
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		if output_dir:
			self.log_dir = output_dir
		else:
			self.log_dir = os.path.join("logs", f"run_{timestamp}")
		os.makedirs(self.log_dir, exist_ok=True)

		# Initialize particle positions and states
		grid_spacing = self.sigma * 0.75
		num_per_axis = int(self.BOX_SIZE // grid_spacing)
		coords = np.linspace(0, self.BOX_SIZE, num=num_per_axis)
		positions = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
		np.random.shuffle(positions)
		self.positions = positions[:N]
		self.N = self.positions.shape[0]

		#self.velocities = np.random.randn(N, 3) * 0.01
		self.velocities = np.zeros_like(self.positions)
		self.spins = np.random.choice([-1, 1], self.N)
		self.colors = np.random.randint(0, 2, self.N)
		self.lifetimes = np.zeros(self.N)
		
		assert self.positions.shape[0] == self.velocities.shape[0] == self.spins.shape[0], "Mismatch in particle array lengths!"

		self.last_spin_flips = 0
		self.last_color_flip = 0

		# Write metadata (optional: only if no config was passed)
		if not config_path:
			meta = {
				"N": self.N,
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
		assert self.positions.shape[0] == self.N, f"[CRITICAL] N mismatch: positions has {self.positions.shape[0]}, but N = {self.N}"
		
		
		self.lifetimes += self.DT
		#decohere = self.lifetimes > 0.3 + np.random.rand(N) * 0.2 - allows for spin flips, removed to debug energy accumulation issues.
		decohere = np.zeros(self.N, dtype=bool)
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
		
		forces = compute_lj_forces(
			self.positions, self.N,
			epsilon=self.epsilon,
			sigma=self.sigma,
			BOX_SIZE=self.BOX_SIZE,
			PLANCK_LENGTH=PLANCK_LENGTH
		)
		
		print("Forces shape:", forces.shape)
		if not np.all(np.isfinite(forces)):
			print("[ERROR] Force array contains NaN or inf!")
			exit(1)
		
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
		
		# Local emergent damping (force-responsive)
		#for i in range(self.positions.shape[0]):
		#	force = forces[i]
		#	velocity = self.velocities[i]
		#
		#	force_magnitude = np.linalg.norm(force)
		#	if force_magnitude > 0:
		#		# Apply damping based on local force magnitude
		#		damping_strength = 0.002  # You can tune this lower or higher
		#		damping_force = -velocity * force_magnitude * damping_strength
		#		self.velocities[i] += damping_force * self.DT

		self.positions += self.velocities * self.DT
		self.positions %= BOX_SIZE
		
		self.frame += 1
		
		return self.kinetic_energy()

# Initialize quantum universe
quantum  = QuantumUniverse()

# Lennard-Jones pairwise force calculation with periodic boundaries and cutoff
@njit(parallel=True)
def compute_lj_forces(positions, N, epsilon, sigma, BOX_SIZE, PLANCK_LENGTH):
	forces = np.zeros_like(positions)
	cutoff_radius_squared = (2.5 * PLANCK_LENGTH) ** 2
	min_r2 = 1e-4  # Prevent r2 from being too small

	for i in prange(N):
		for j in range(i + 1, N):
			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dz = positions[i][2] - positions[j][2]

			# Periodic boundary correction
			if dx > 0.5 * BOX_SIZE:
				dx -= BOX_SIZE
			elif dx < -0.5 * BOX_SIZE:
				dx += BOX_SIZE
			if dy > 0.5 * BOX_SIZE:
				dy -= BOX_SIZE
			elif dy < -0.5 * BOX_SIZE:
				dy += BOX_SIZE
			if dz > 0.5 * BOX_SIZE:
				dz -= BOX_SIZE
			elif dz < -0.5 * BOX_SIZE:
				dz += BOX_SIZE

			r2 = dx*dx + dy*dy + dz*dz
			if r2 < min_r2 or r2 > cutoff_radius_squared:
				continue

			inv_r2 = 1.0 / r2
			inv_r6 = (sigma * sigma * inv_r2) ** 3
			inv_r12 = inv_r6 * inv_r6

			f_mag = 24 * epsilon * inv_r2 * (2 * inv_r12 - inv_r6)
			
			if not math.isfinite(f_mag):
				print(f"[FATAL] Bad f_mag at r²={r2}, f_mag={f_mag}, dx={dx}, dy={dy}, dz={dz}")
				continue

			if not math.isfinite(f_mag):
				continue  # extra protection

			fx = f_mag * dx
			fy = f_mag * dy
			fz = f_mag * dz

			forces[i, 0] += fx
			forces[i, 1] += fy
			forces[i, 2] += fz
			forces[j, 0] -= fx
			forces[j, 1] -= fy
			forces[j, 2] -= fz

	return forces

# Lennard-Jones Hamiltonian energy computation
@njit
def compute_lj_hamiltonian(positions: np.ndarray, velocities: np.ndarray, N: int, epsilon: float, sigma: float, BOX_SIZE: float) -> tuple:
	ke = 0.5 * np.sum(velocities**2)
	pe = 0.0

	for i in range(N):
		for j in range(i + 1, N):
			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dz = positions[i][2] - positions[j][2]

			if dx > 0.5 * BOX_SIZE:
				dx -= BOX_SIZE
			elif dx < -0.5 * BOX_SIZE:
				dx += BOX_SIZE

			if dy > 0.5 * BOX_SIZE:
				dy -= BOX_SIZE
			elif dy < -0.5 * BOX_SIZE:
				dy += BOX_SIZE

			if dz > 0.5 * BOX_SIZE:
				dz -= BOX_SIZE
			elif dz < -0.5 * BOX_SIZE:
				dz += BOX_SIZE

			r2 = dx*dx + dy*dy + dz*dz
			if r2 < 1e-4:
				continue

			inv_r2 = 1.0 / r2
			inv_r6 = (sigma * sigma * inv_r2) ** 3
			inv_r12 = inv_r6 * inv_r6

			pe += 4 * epsilon * (inv_r12 - inv_r6)

	return ke, pe, ke + pe

@njit
def compute_cluster_energy(cluster_indices, positions, velocities, epsilon, sigma, BOX_SIZE):
	ke = 0.0
	for i in range(len(cluster_indices)):
		idx = cluster_indices[i]
		v = velocities[idx]
		ke += 0.5 * (v[0]**2 + v[1]**2 + v[2]**2)

	pe = 0.0
	for i in range(len(cluster_indices)):
		for j in range(i + 1, len(cluster_indices)):
			i_idx = cluster_indices[i]
			j_idx = cluster_indices[j]

			dx = positions[i_idx][0] - positions[j_idx][0]
			dy = positions[i_idx][1] - positions[j_idx][1]
			dz = positions[i_idx][2] - positions[j_idx][2]

			# PBC wrap
			if dx > 0.5 * BOX_SIZE:
				dx -= BOX_SIZE
			elif dx < -0.5 * BOX_SIZE:
				dx += BOX_SIZE
			if dy > 0.5 * BOX_SIZE:
				dy -= BOX_SIZE
			elif dy < -0.5 * BOX_SIZE:
				dy += BOX_SIZE
			if dz > 0.5 * BOX_SIZE:
				dz -= BOX_SIZE
			elif dz < -0.5 * BOX_SIZE:
				dz += BOX_SIZE

			r2 = dx*dx + dy*dy + dz*dz
			if r2 < 1e-4:
				continue  # avoid singularity

			inv_r2 = 1.0 / r2
			inv_r6 = (sigma * sigma * inv_r2) ** 3
			inv_r12 = inv_r6 * inv_r6

			pe += 4 * epsilon * (inv_r12 - inv_r6)

	return ke + pe


def color_to_charge(color):
	return {0: +1, 1: 0, 2: -1}[color]
	
def cluster_charge(cluster_indices, colors):
	return sum(color_to_charge(colors[i]) for i in cluster_indices)

######################## Logging functions for writing to CSV files. ########################

def write_summary_row(
	logdir, frame, ke, stable_count, unstable_count,
	just_born, just_died, spin_flips, color_flips,
	cluster_radius_stats, min_dist, max_dist, mean_dist,
	ke_loss, clusters_damped, particles_damped
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
				"MinDistance", "MaxDistance", "MeanDistance",
				"KELossRadiative", "ClustersDamped", "ParticlesDamped"
			])
		writer.writerow([
			frame, f"{ke:.5f}",
			stable_count, unstable_count,
			len(just_born), len(just_died),
			spin_flips, color_flips,
			round(np.mean(cluster_radius_stats), 2),
			min(cluster_radius_stats),
			max(cluster_radius_stats),
			round(min_dist, 5), round(max_dist, 5), round(mean_dist, 5),
			round(ke_loss, 5), clusters_damped, particles_damped
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

def write_proton_charges(logdir, frame, charge_data):
	path = os.path.join(logdir, "proton_charges.csv")
	file_exists = os.path.isfile(path)

	with open(path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["Frame", "ClusterID", "Charge"])

		for cluster_id, charge in charge_data:
			cluster_label = "-".join(map(str, sorted(cluster_id)))
			writer.writerow([frame, cluster_label, charge])

def write_lifetimes_by_charge(logdir, frame, birth_frames, death_frames, charges):
	path = os.path.join(logdir, "proton_lifetimes_by_charge.csv")
	file_exists = os.path.isfile(path)

	with open(path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["Frame", "ClusterID", "Charge", "Lifetime"])

		for cluster in birth_frames:
			if cluster in death_frames and cluster in charges:
				birth = birth_frames[cluster]
				death = death_frames[cluster]
				charge = charges[cluster]
				lifetime = death - birth
				cluster_label = "-".join(map(str, sorted(cluster)))
				writer.writerow([frame, cluster_label, charge, lifetime])
				
def write_charge_lifetimes_framewise(logdir, frame, birth_frames, active_clusters, charges):
    path = os.path.join(logdir, "proton_charge_lifetimes.csv")
    file_exists = os.path.isfile(path)

    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Frame", "ClusterID", "Charge", "Lifetime"])

        for cluster in active_clusters:
            if cluster in birth_frames and cluster in charges:
                birth = birth_frames[cluster]
                charge = charges[cluster]
                lifetime = frame - birth
                cluster_label = "-".join(map(str, sorted(cluster)))
                writer.writerow([frame, cluster_label, charge, lifetime])

def write_binding_energy(logdir, energy_log_data):
	path = os.path.join(logdir, "proton_binding_energy.csv")
	file_exists = os.path.isfile(path)
	with open(path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["Frame", "ClusterID", "Charge", "BindingEnergy"])
		for frame, cluster, charge, energy in energy_log_data:
			cluster_str = "-".join(map(str, sorted(cluster)))
			writer.writerow([frame, cluster_str, charge, energy])

def write_unstable_particles(logdir, unstable_log_data):
    path = os.path.join(logdir, "unstable_particles.csv")
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Frame", "ClusterID", "FailureReason"])
        for frame, cluster_id, reason in unstable_log_data:
            cluster_str = "-".join(map(str, sorted(cluster_id)))
            writer.writerow([frame, cluster_str, reason])

def write_unknown_particles(logdir, data):
    path = os.path.join(logdir, "unknown_particles.csv")
    file_exists = os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Frame", "ClusterID", "Reason"])
        for frame, cluster_id, reason in data:
            writer.writerow([frame, "-".join(map(str, sorted(cluster_id))), reason])

def write_cluster_lifetimes(logdir, cluster_lifetimes):
	"""
	Writes cluster lifetime data to one CSV per cluster size.
	Each file contains: ClusterID, BirthFrame, DeathFrame, Lifetime
	"""
	for size, lifetimes in cluster_lifetimes.items():
		filename = f"cluster_lifetimes_size_{size}.csv"
		filepath = os.path.join(logdir, filename)

		file_exists = os.path.isfile(filepath)
		with open(filepath, 'a', newline='') as f:
			writer = csv.writer(f)
			if not file_exists:
				writer.writerow(["ClusterID", "BirthFrame", "DeathFrame", "Lifetime"])

			for cid, lifetime in lifetimes.items():
				birth = quantum.cluster_birth_frames[size].get(cid, "unknown")
				death = quantum.cluster_death_frames[size].get(cid, "unknown")

				# Format ClusterID as a readable sorted list
				cid_str = "-".join(map(str, sorted(cid)))
				writer.writerow([cid_str, birth, death, lifetime])

######################## Logging functions for writing to CSV files. ########################

def update(frame):
	global previous_proton_ids, start_time
	
	frame_start = time.time()
	
	ke = quantum.update()

	radius = 0.1 * PLANCK_LENGTH

	tree = cKDTree(quantum.positions)
	proton_clusters = []
	charge_log_data = []
	energy_log_data = []
	unstable_log_data = []
	cluster_radius_stats = []
	unstable_clusters = set()
	new_proton_ids = set()
	charge_log_ids = set()
	seen_cluster_ids = set()
	cluster_size_histogram = defaultdict(int)
	clusters_by_size = defaultdict(list)
	unknown_particle_log = []  # New: catch-all for non-proton-like clusters
	quantum.unstable_clusters.clear()

	for i in range(quantum.N):
		cluster = tree.query_ball_point(quantum.positions[i], radius)
		cluster = [j for j in cluster if j != i]  # NOW back in
		cluster_radius_stats.append(len(cluster))

		full_cluster = [i] + cluster
		cluster_id = frozenset(full_cluster)
		
		if cluster_id in seen_cluster_ids:
			continue  # Already processed

		seen_cluster_ids.add(cluster_id)
		cluster_size = len(full_cluster)
		quantum.current_clusters_by_size[cluster_size].add(cluster_id)

		if cluster_id not in quantum.previous_clusters_by_size.get(cluster_size, set()):
			#print(f"[DEBUG] Frame {frame}: registering birth for cluster {cluster_id} (size {cluster_size})")
			quantum.cluster_birth_frames[cluster_size][cluster_id] = quantum.frame
		
		# Track this cluster size
		cluster_size_histogram[cluster_size] += 1

		# Optionally: track cluster ID by size for future analysis
		clusters_by_size.setdefault(cluster_size, []).append(full_cluster)
		
		# You can still filter 3-spheres for proton/superproton rules below
		if cluster_size == 3:
			spins = quantum.spins[full_cluster]
			colors = quantum.colors[full_cluster]
			unique_colors = set(colors)

			counts = [list(colors).count(c) for c in unique_colors]

			# Check if it's a valid proton or superproton
			if len(unique_colors) in (1, 2) and abs(sum(spins)) <= 1 and sorted(counts) == [1, 2]:
				if cluster_id not in charge_log_ids:
					charge_log_ids.add(cluster_id)
					charge = cluster_charge(full_cluster, quantum.colors)
					cluster_energy = compute_cluster_energy(
						full_cluster,
						quantum.positions,
						quantum.velocities,
						quantum.epsilon,
						quantum.sigma,
						quantum.BOX_SIZE
					)
					
					binding_energy = cluster_energy - sum(0.5 * np.sum(quantum.velocities[i]**2) for i in full_cluster)
					
					if quantum.verbose or HEADLESS:
						print(f"Frame {frame}: Proton Cluster {full_cluster} → Net Charge: {charge} | Binding Energy: {cluster_energy:.5f}")

					quantum.proton_charges[cluster_id] = charge
					charge_log_data.append((cluster_id, charge))
					if charge == 1:
						quantum.proton_ids.add(cluster_id)
					elif charge == 2:
						quantum.superproton_ids.add(cluster_id)

					energy_log_data.append((frame, cluster_id, charge, cluster_energy))

				new_proton_ids.add(cluster_id)
				proton_clusters.extend(full_cluster)
				continue  # skip the unstable logic below
			else:
				# Failed 3-sphere classification
				if len(unique_colors) not in (1, 2):
					reason = "color_mismatch"
				elif abs(sum(spins)) > 1:
					reason = "spin_violation"
				elif sorted(counts) != [1, 2]:
					reason = "color_ratio_wrong"
				else:
					reason = "unknown_failure"

				unstable_clusters.update(full_cluster)
				unstable_log_data.append((frame, cluster_id, reason))
		else:
			reason = f"size_{cluster_size}"
			unknown_particle_log.append((frame, cluster_id, reason))
	
	frame_ke_loss = 0.0  # RESET just before applying damping!
	clusters_damped = 0
	particles_damped = 0

	for size, clusters in clusters_by_size.items():
		if size >= 3:
			for cluster in clusters:
				cluster_id = frozenset(cluster)

				birth_frame = quantum.cluster_birth_frames[size].get(cluster_id, None)
				if birth_frame is None:
					continue  # unknown cluster

				lifetime = frame - birth_frame
				if lifetime < 3:
					continue  # too young to safely dampen

				# Stronger damp based on size
				base_damp = 0.998
				extra = min(0.001 * (size - 3), 0.010)
				damp_factor = base_damp - extra

				# ✅ Log damped cluster and particle count
				clusters_damped += 1
				particles_damped += len(cluster)

				for idx in cluster:
					v_before = quantum.velocities[idx].copy()
					quantum.velocities[idx] *= damp_factor
					v_after = quantum.velocities[idx]
					frame_ke_loss += 0.5 * (np.dot(v_before, v_before) - np.dot(v_after, v_after))
	
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
			quantum.proton_death_frames[cluster] = quantum.frame
	
	just_born = new_proton_ids - previous_proton_ids
	previous_proton_ids = new_proton_ids
	
	#Store stable proton clusters
	quantum.stable_proton_clusters = new_proton_ids
	
	# Track proton birth frames
	for cluster in just_born:
		if cluster not in quantum.proton_birth_frames:
			quantum.proton_birth_frames[cluster] = quantum.frame
	
	#print(f"[DEBUG] Total birth frames tracked: {sum(len(v) for v in quantum.cluster_birth_frames.values())}")
	
	# Compute lifetime of existing protons
	lifetimes = {
		cluster: frame - quantum.proton_birth_frames[cluster]
		for cluster in quantum.stable_proton_clusters
		if cluster in quantum.proton_birth_frames
	}
	
	# Detect deaths and update lifetimes
	for size in quantum.previous_clusters_by_size:
		just_died = quantum.previous_clusters_by_size[size] - quantum.current_clusters_by_size[size]
		for cid in just_died:
			if cid not in quantum.cluster_death_frames[size]:
				birth = quantum.cluster_birth_frames[size].get(cid, quantum.frame - 1)
				quantum.cluster_death_frames[size][cid] = quantum.frame
				quantum.cluster_lifetimes[size][cid] = quantum.frame - birth
		
	if quantum.verbose or HEADLESS:
		# Terminal logging
		print(f"Frame {frame:4}: Stable Protons: {stable_count:2} | Unstable Particles: {unstable_count:3} | KE: {quantum.kinetic_energy():.5f}")
		print(f"New Protons: {len(just_born)} | Dissolved: {len(just_died)} | Long-lived: {sum(1 for v in proton_lifetimes.values() if v > 20)}")

		if quantum.last_spin_flips > 0 or quantum.last_color_flip > 0:
			print(f"Spin Flips: {quantum.last_spin_flips} | Color Flip: {quantum.last_color_flip}")

		cluster_sizes = Counter(len(tree.query_ball_point(quantum.positions[i], 2.2 * PLANCK_LENGTH)) for i in range(quantum.N))
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
		np.min(distances), np.max(distances), np.mean(distances),
		frame_ke_loss, clusters_damped, particles_damped
	)
	
	ke, pe, h_total = compute_lj_hamiltonian(
		quantum.positions, quantum.velocities, quantum.N,
		epsilon=quantum.epsilon,
		sigma=quantum.sigma,
		BOX_SIZE=quantum.BOX_SIZE
	)
	
	write_quantum_log_row(
		quantum.log_dir,
		frame,
		quantum.last_spin_flips,
		quantum.last_color_flip,
		quantum.DT,
		ke, pe, h_total
	)
	
	if quantum.verbose or HEADLESS:
		print(f"Frame {frame}: KE Lost via Radiative Damping = {frame_ke_loss:.5f}")
	
	if frame > 0 and frame % 100 == 0:
		write_unstable_particles(quantum.log_dir, unstable_log_data)
		write_unknown_particles(quantum.log_dir, unknown_particle_log)
	
	if frame % 10 == 0:
		write_charge_lifetimes_framewise(
			quantum.log_dir,
			frame,
			quantum.proton_birth_frames,
			quantum.stable_proton_clusters,
			quantum.proton_charges
		)
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
		
		write_binding_energy(quantum.log_dir, energy_log_data)
		
	cal = MAX_FRAMES - 1
	if frame == cal:
		#print("[DEBUG] Final frame reached, checking cluster survival...")
		#print(f"[DEBUG] Final cluster size map: {dict((k, len(v)) for k, v in quantum.current_clusters_by_size.items())}")
		for size, active_clusters in quantum.current_clusters_by_size.items():
			#print(f"[DEBUG] Size {size}: {len(active_clusters)} clusters")
			for cid in active_clusters:
				if cid not in quantum.cluster_birth_frames[size]:
					#print(f"[DEBUG] Cluster {cid} has no recorded birth.")
					continue
				birth = quantum.cluster_birth_frames[size][cid]
				lifetime = MAX_FRAMES - birth
				quantum.cluster_death_frames[size][cid] = MAX_FRAMES
				quantum.cluster_lifetimes[size][cid] = lifetime

		total_lifetimes = sum(len(v) for v in quantum.cluster_lifetimes.values())
		#print(f"[DEBUG] Total lifetimes recorded: {total_lifetimes}")

		write_cluster_lifetimes(quantum.log_dir, quantum.cluster_lifetimes)
		print("Simulation complete. Closing animation.")
		if not HEADLESS:
			plt.close()  # This will close the window when the last frame is rendered
	else:
		# Swap trackers for next frame
		quantum.previous_clusters_by_size = {
			size: set(clusters)
			for size, clusters in quantum.current_clusters_by_size.items()
		}
		quantum.current_clusters_by_size = defaultdict(set)
	
	frame_end = time.time()
	frame_duration = frame_end - frame_start

	elapsed_total = frame_end - start_time
	remaining_frames = MAX_FRAMES - frame
	est_remaining_time = frame_duration * remaining_frames
	est_total_time = elapsed_total + est_remaining_time
	
	real = frame + 1
	if real % 100 == 0:  # or % 100 to reduce spam
		print(f"[Frame {frame}] Time/frame: {frame_duration:.3f}s | "
			  f"Elapsed: {elapsed_total/60:.1f} min | "
			  f"ETA: {est_remaining_time/60:.1f} min | "
			  f"Est Total: {est_total_time/60:.1f} min")
	
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
