### ──────────────────────── Imports and Globals ───────────────────────── ###

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
from itertools import combinations
import os, json, csv, time, math, sys, hashlib

# Planck Length
PLANCK_LENGTH = 5.0        	# Actual working interaction scale

# Simulation Parameters
MAX_FRAMES = 50000			# Maximum Frames to run
BOX_SIZE = 6.0				# Size of box/cube
HEADLESS = False			# Headless mode toggle

#Global proton state trackers
proton_lifetimes = {}
previous_proton_ids = set()
start_time = time.time()

#GUI plot definitions
scatter = None

### ──────────────────────── Imports and Globals ───────────────────────── ###

### ──────────────────────── QuantumUniverse Class ─────────────────────── ###

class QuantumUniverse:
	def __init__(self, config_path=None, output_dir=None, verbose=False, fast=False):
		"""
		Initialize a new simulation instance of the Spherical Field Theory universe.

		This constructor configures all physical constants, initializes particle
		arrays and quantum states, handles deterministic seeding, sets up the
		simulation domain with periodic boundary conditions, and prepares the output
		directory and metadata for logging and post-analysis.

		It loads optional configuration overrides from a JSON metadata file, and
		ensures reproducibility for safe (non-parallel) deterministic runs.

		Key features initialized:
		- Particle positions, velocities, spins, and colors
		- Cluster tracking systems for protons, superprotons, and general clusters
		- Simulation constants: Morse potential, DCP repulsion, Planck length, etc.
		- Runtime flags: headless, fast mode, cold start, quantum color flips
		- Log directory creation and metadata output
		- Sanity checks for grid generation, particle count, and layout validity

		Parameters:
			config_path (str or None): Optional path to a metadata JSON file for overrides
			output_dir (str or None): Optional path to store simulation logs and metadata
			verbose (bool): If True, enables terminal output (frame-level logs, debug)
			fast (bool): If True, enables non-deterministic multithreaded mode (Numba parallel)

		Raises:
			ValueError: If N is too large for the available 3D grid spacing
		"""
		
		#Simulation state
		self.frame = 0						# Frame counter.
		self.ke = 1.0						# Kinetic Energy (normalized units) per frame.
		self.DT = 0.001						# Default DT, to be used in calculation with dynamic velocities in a local configuration.
		self.frame_force_magnitude = 0.0	# Mean force magnitude applied this frame
		
		#Config toggles
		self.verbose = verbose				# Toggles terminal spam/debug mode.
		self.fast = fast					# Parallel/multithreaded toggle, enabling this sets the simulation into non-deterministic mode.
		self.cold_start = True				# No initial movement of particles, allow interactions to start reactions.
		self.use_gravity = True				# Toggles scaling gravity effect
		self.enable_color_flips = True		# Toggles quantum-like flucuations (color flips)
		self.use_seed = True 				# Single threaded deterministic mode, disable for parallel as default, parallel is non-deterministic.
		self.seed_value = 42				# Default seed value, allows deterministic bit-for-bit results.
		
		#Spatial Structure
		self.BOX_SIZE = BOX_SIZE			# Size of toroidal structure.
		
		#Interaction constants
		self.D = 0.3         				# Morse well depth
		self.alpha = 3.0     				# Morse sharpness
		self.r0 = 0.3        				# Morse equilibrium distance
		self.dcp_k = 0.001 					# Dirac Core Pressure - Soft repulsion strength
		self.dcp_cutoff = 0.2  				# Only repels at very close range
		
		#Runtime utility
		self.live_stats = {}  				# Populated per frame for zstd + msg output.
		
		global PLANCK_LENGTH, MAX_FRAMES
		
		#Default frames, will change with loaded config automatically.#
		N = 2000

		# Load overrides from metadata.json if provided
		if config_path:
			with open(config_path, 'r') as f:
				config = json.load(f)
			N = config.get("N", N)
			self.D = config.get("D", self.D)
			self.r0 = config.get("r0", self.r0)
			self.alpha = config.get("alpha", self.alpha)
			self.dcp_k = config.get("dcp_k", self.dcp_k)
			self.dcp_cutoff = config.get("dcp_cutoff", self.dcp_cutoff)
			self.use_seed = config.get("use_seed", self.use_seed)
			self.seed_value = config.get("seed_value", self.seed_value)
			PLANCK_LENGTH = config.get("planck_length", PLANCK_LENGTH)
			self.DT = config.get("initial_dt", self.DT)
			MAX_FRAMES = config.get("max_frames", MAX_FRAMES)
			self.use_gravity = config.get("use_gravity", self.use_gravity)
			self.enable_color_flips = config.get("enable_color_flips", self.enable_color_flips)
			self.cold_start = config.get("cold_start", self.cold_start)
			self.fast = config.get("fast", self.fast)
		
		if self.fast:
			self.use_seed = False
		
		if self.use_seed:
			np.random.seed(self.seed_value)
		
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
		grid_spacing = self.r0 * 0.75
		num_per_axis = int(self.BOX_SIZE // grid_spacing)
		coords = np.linspace(0, self.BOX_SIZE, num=num_per_axis)
		positions = np.array(np.meshgrid(coords, coords, coords)).T.reshape(-1, 3)
		np.random.shuffle(positions)
		self.positions = positions[:N]
		self.N = N
		
		if positions.shape[0] < N:
			raise ValueError(f"Only generated {positions.shape[0]} particles, but N = {N}")
		
		if N > len(np.unique(positions, axis=0)):
			raise ValueError(f"N = {N} is too large for the available unique positions: {len(np.unique(positions, axis=0))}")

		if self.cold_start:
			self.velocities = np.zeros_like(self.positions)
		else:
			self.velocities = np.random.randn(N, 3) * 0.01
		
		self.velocities += np.random.normal(0, 0.01, size=self.velocities.shape)
		
		self.spins = np.random.choice([-1, 1], self.N)
		self.colors = np.random.randint(0, 2, self.N)
		self.lifetimes = np.zeros(self.N)
		
		assert self.positions.shape[0] == self.velocities.shape[0] == self.spins.shape[0], "Mismatch in particle array lengths!"

		self.last_spin_flips = 0
		self.last_color_flip = 0

		# Write metadata
		meta = {
			"N": self.N,
			"D": self.D,
			"r0": self.r0,
			"alpha": self.alpha,
			"dcp_k": self.dcp_k,
			"dcp_cutoff": self.dcp_cutoff,
			"use_seed": self.use_seed,
			"seed_value": self.seed_value,
			"planck_length": PLANCK_LENGTH,
			"initial_dt": self.DT,
			"max_frames": MAX_FRAMES,
			"use_gravity": self.use_gravity,
			"enable_color_flips": self.enable_color_flips,
			"cold_start": self.cold_start,
			"fast": self.fast,
			"cluster_energy_input": "np_array_int32",
			"reflected_list_mode": False,
			"source_hash": hash_self()
		}
		with open(os.path.join(self.log_dir, "metadata.json"), "w") as f:
			json.dump(meta, f, indent=4)

	def scale_gravity_effect(self):
		"""
		Compute a dynamic scaling factor for gravity based on system density.

		This function calculates a gravity scaling factor that adjusts dynamically
		based on the system's particle density. It uses the **average pairwise distance** 
		between particles to determine the system's **density**. As the particles get closer 
		(i.e., higher density), the gravitational effects become stronger. However, the 
		scaling factor is capped at a maximum value and smoothed between frames to ensure 
		stability and avoid extreme fluctuations.

		Gravity scaling behavior:
		- Inversely proportional to the **square of the average distance** between particles.
		- **Capped at a maximum value** to prevent runaway gravitational effects at high densities.
		- **Smoothed over time** to avoid sharp fluctuations or jitter from one frame to the next.

		This method enables gravity to scale naturally with the particle distribution while 
		maintaining a **stable system**, ensuring that clusters can form and evolve without 
		causing numerical instability.

		Returns:
			float: The **smoothed gravitational scaling factor** for the current frame. This factor 
				   adjusts the gravitational force applied to the system based on the current density.
		"""
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
		"""
		Compute the total kinetic energy of the system.

		Uses the classical formula:
			KE = 0.5 * Σ(v²)
		across all particle velocity vectors in the simulation.

		Returns:
			float: Total kinetic energy of all particles
		"""
		return 0.5 * np.sum(self.velocities ** 2)

	def update(self):
		"""
		Advance the Spherical Field Theory simulation by one frame.

		This is the primary per-frame update method that drives all time evolution
		in the simulation. It performs the following major operations:

		- Ensures particle count integrity
		- Applies quantum decoherence to spin states (with random lifetime-based flipping)
		- Performs reproducible color flips if enabled
		- Computes net force on each particle using either:
			- Safe (deterministic) Lagrangian force function, or
			- Fast (parallel, non-deterministic) mode
		- Scales gravitational influence dynamically based on local density
		- Applies symplectic integration to update velocities and positions
		- Performs local, emergent damping when forces and velocities align
		- Tracks per-frame energy, work, and force/velocity alignment
		- Dynamically adjusts timestep (`DT`) to remain stable under changing system state.
		- Accounts for force and velocity before and after DT calculation. Time is Relative.
		- Wraps positions using toroidal boundary conditions
		- Updates simulation frame count and total kinetic energy

		This function is the central simulation loop, called once per animation or
		data frame, and is responsible for real-time physics integration of all 
		particles in the quantum field.

		Returns:
			float: Total kinetic energy after the update (for logging or visualization)
		"""
		assert self.positions.shape[0] == self.N, f"[CRITICAL] N mismatch: positions has {self.positions.shape[0]}, but N = {self.N}"
		
		ke_factor = min(self.ke, 1.0)
		self.lifetimes += self.DT
		decohere = self.lifetimes > 0.3 + np.random.rand(self.N) * 0.2 
		#decohere = np.zeros(self.N, dtype=bool)
		self.last_spin_flips = np.count_nonzero(decohere)
		self.spins[decohere] *= -1
		self.lifetimes[decohere] = 0
		current_frame = self.frame
		
		self.last_color_flip = 0  # reset every frame

		if self.enable_color_flips:
			np.random.seed(self.seed_value + self.frame)  # <-- REPRODUCIBLE flips
			color_flip_chance = 0.002
			flip_mask = np.random.rand(self.N) < color_flip_chance
			prev_colors = self.colors.copy()
			self.colors[flip_mask] = 1 - self.colors[flip_mask]
			self.last_color_flip = np.count_nonzero(prev_colors != self.colors)
		
		if self.fast:
			forces = compute_effective_lagrangian_forces_fast_dcp(
				self.positions,
				self.N,
				self.D,
				self.alpha,
				self.r0,
				self.dcp_k,
				self.dcp_cutoff,
				self.BOX_SIZE
			)
		else:
			forces = compute_effective_lagrangian_forces_safe_dcp(
				self.positions,
				self.N,
				self.D,
				self.alpha,
				self.r0,
				self.dcp_k,
				self.dcp_cutoff,
				self.BOX_SIZE
			)

		
		if hasattr(self, "frame_force_magnitude"):
			quantum.frame_force_magnitude = np.mean(np.linalg.norm(forces, axis=1))
		
		#print("Forces shape:", forces.shape)
		if self.verbose or HEADLESS:
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

		frame_work = 0.0
		cos_sum = 0.0
		count = 0
		
		# Local emergent damping (force-responsive)
		for i in range(self.positions.shape[0]):
			force = forces[i]
			velocity = self.velocities[i]
			
			force_mag = np.linalg.norm(force)
			vel_mag = np.linalg.norm(velocity)
			
			if force_mag > 0 and vel_mag > 0:
				cos_theta = np.dot(force, velocity) / (force_mag * vel_mag)
				cos_sum += cos_theta
				count += 1
			
			dot = np.dot(force, velocity)
			if dot > 0:
				unit_force = force / (force_mag + 1e-10)
				aligned_component = np.dot(velocity, unit_force)
				
				# Only damp the aligned part of velocity
				if cos_theta > 0.9:  # Only when strongly aligned
					force_damp = (cos_theta - 0.9) * 0.01  # Scale how aligned it is
					self.velocities[i] *= 1.0 - force_damp * self.DT
		
			frame_work += force[0] * velocity[0] + force[1] * velocity[1] + force[2] * velocity[2]
		
		self.frame_avg_force_velocity_cos = cos_sum / max(count, 1)
		
		self.frame_force_work = frame_work
		
		self.positions += self.velocities * self.DT
		self.positions %= BOX_SIZE
		
		self.frame += 1
		
		self.ke = self.kinetic_energy()
		
		return self.ke

### ──────────────────────── QuantumUniverse Class ─────────────────────── ###

### ──────────────────────── Computational Functions ───────────────────── ###

@njit(parallel=True, fastmath=True)
def compute_hamiltonian_energy_fast_dcp(
	positions: np.ndarray,
	velocities: np.ndarray,
	N: int,
	D: float,
	alpha: float,
	r0: float,
	dcp_k: float,
	dcp_cutoff: float,
	BOX_SIZE: float
):
	"""
	Compute the total Hamiltonian energy (KE + PE) of the full system.

	This function calculates:
	- Total kinetic energy from per-particle velocities
	- Total potential energy using pairwise Morse + DCP (repulsive) interactions
	- Applies toroidal (periodic) boundary conditions for space wrapping

	The Morse potential captures quantum attraction:
		PE_morse = D * (1 - exp(-alpha * (r - r0)))^2

	While the Dirac Core Pressure (DCP) term introduces a soft repulsion:
		PE_dcp = -dcp_k / r (if r < dcp_cutoff)

	This function is used for energy tracking, stability validation, and 
	Hamiltonian analysis at every frame.

	Parameters:
		positions (np.ndarray): Nx3 array of particle positions
		velocities (np.ndarray): Nx3 array of particle velocities
		N (int): Number of particles in the system
		D (float): Morse well depth (attraction strength)
		alpha (float): Morse sharpness (controls steepness)
		r0 (float): Morse equilibrium distance
		dcp_k (float): Dirac repulsion constant
		dcp_cutoff (float): Distance cutoff for DCP interaction
		BOX_SIZE (float): Length of the toroidal simulation cube

	Returns:
		tuple:
			- ke (float): Total kinetic energy
			- pe (float): Total potential energy
			- h_total (float): Hamiltonian = KE + PE
	"""
	ke = 0.0
	for i in prange(N):
		ke += 0.5 * np.dot(velocities[i], velocities[i])

	pe = 0.0
	for i in prange(N):
		for j in range(i + 1, N):
			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dz = positions[i][2] - positions[j][2]

			# Toroidal wrapping
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

			r = math.sqrt(dx * dx + dy * dy + dz * dz)
			r = max(r, 1e-5)  # Prevent zero distance
			delta = r - r0
			exp_input = min(alpha * delta, 50.0)
			exp_term = math.exp(-exp_input)
			morse = D * (1 - exp_term) ** 2
			pe += morse

			# Repulsive potential term
			if r < dcp_cutoff:
				pe += -dcp_k / r

	return ke, pe, ke + pe

@njit(parallel=True, fastmath=True)
def compute_effective_lagrangian_forces_fast_dcp(
	positions: np.ndarray,
	N: int,
	D: float,         # Morse depth
	alpha: float,     # Morse sharpness
	r0: float,        # Morse equilibrium distance
	dcp_k: float,   # Repulsive force constant
	dcp_cutoff: float,  # Repulsive force cutoff distance
	BOX_SIZE: float
) -> np.ndarray:
	"""
	Compute the net classical forces on all particles using a local Lagrangian model.

	This function calculates the per-particle net force from pairwise interactions
	using a combination of:

	- Morse potential (attractive):  
	  F_morse = -2 * D * alpha * (1 - exp(-alpha(r - r0))) * exp(-alpha(r - r0))

	- Dirac Core Pressure (repulsive):  
	  F_dcp = dcp_k / r^2  (only applied within cutoff)

	Forces are symmetrically applied to both particles (Newton’s 3rd Law), and
	toroidal (periodic) boundary conditions are enforced for spatial wrapping.

	This is the core per-frame force model that governs all local interactions in
	the Spherical Field Theory simulation.

	Parameters:
		positions (np.ndarray): Nx3 array of particle positions
		N (int): Number of particles
		D (float): Morse potential well depth
		alpha (float): Morse sharpness parameter
		r0 (float): Morse equilibrium distance
		dcp_k (float): Dirac-like repulsive force strength
		dcp_cutoff (float): Cutoff distance for DCP force activation
		BOX_SIZE (float): Simulation space side length (for toroidal wrap)

	Returns:
		np.ndarray: Nx3 array of net force vectors per particle
	"""
	forces = np.zeros_like(positions)
	cutoff_radius_squared = (2.5 * r0) ** 2

	for i in prange(N):
		for j in range(N):
			if i == j:
				continue

			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dz = positions[i][2] - positions[j][2]

			# Toroidal wrapping
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

			r2 = dx * dx + dy * dy + dz * dz
			if r2 > cutoff_radius_squared:
				continue

			r = math.sqrt(r2)
			r = max(r, 1e-5)
			inv_r = 1.0 / r

			dir_x = dx * inv_r
			dir_y = dy * inv_r
			dir_z = dz * inv_r

			# Morse force
			delta = r - r0
			exp_input = min(alpha * delta, 50.0)
			exp_term = math.exp(-exp_input)
			f_morse = -2 * D * alpha * (1 - exp_term) * exp_term

			# Repulsive short-range force (only active when r < repel_cutoff)
			f_dcp = 0.0
			if r < dcp_cutoff:
				f_dcp = dcp_k / (r * r)

			# Combine forces
			fx = (f_morse + f_dcp) * dir_x
			fy = (f_morse + f_dcp) * dir_y
			fz = (f_morse + f_dcp) * dir_z

			# Apply forces
			forces[i, 0] += fx
			forces[i, 1] += fy
			forces[i, 2] += fz

			forces[j, 0] -= fx
			forces[j, 1] -= fy
			forces[j, 2] -= fz

	return forces

@njit(fastmath=True)
def compute_hamiltonian_energy_safe_dcp(
	positions: np.ndarray,
	velocities: np.ndarray,
	N: int,
	D: float,
	alpha: float,
	r0: float,
	dcp_k: float,
	dcp_cutoff: float,
	BOX_SIZE: float
):
	"""
	Compute the total Hamiltonian energy (KE + PE) of the full system.

	This function calculates:
	- Total kinetic energy from per-particle velocities
	- Total potential energy using pairwise Morse + DCP (repulsive) interactions
	- Applies toroidal (periodic) boundary conditions for space wrapping

	The Morse potential captures quantum attraction:
		PE_morse = D * (1 - exp(-alpha * (r - r0)))^2

	While the Dirac Core Pressure (DCP) term introduces a soft repulsion:
		PE_dcp = -dcp_k / r (if r < dcp_cutoff)

	This function is used for energy tracking, stability validation, and 
	Hamiltonian analysis at every frame.

	Parameters:
		positions (np.ndarray): Nx3 array of particle positions
		velocities (np.ndarray): Nx3 array of particle velocities
		N (int): Number of particles in the system
		D (float): Morse well depth (attraction strength)
		alpha (float): Morse sharpness (controls steepness)
		r0 (float): Morse equilibrium distance
		dcp_k (float): Dirac repulsion constant
		dcp_cutoff (float): Distance cutoff for DCP interaction
		BOX_SIZE (float): Length of the toroidal simulation cube

	Returns:
		tuple:
			- ke (float): Total kinetic energy
			- pe (float): Total potential energy
			- h_total (float): Hamiltonian = KE + PE
	"""
	ke = 0.0
	for i in prange(N):
		ke += 0.5 * np.dot(velocities[i], velocities[i])

	pe = 0.0
	for i in prange(N):
		for j in range(i + 1, N):
			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dz = positions[i][2] - positions[j][2]

			# Toroidal wrapping
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

			r = math.sqrt(dx * dx + dy * dy + dz * dz)
			r = max(r, 1e-5)  # Prevent zero distance
			delta = r - r0
			exp_input = min(alpha * delta, 50.0)
			exp_term = math.exp(-exp_input)
			morse = D * (1 - exp_term) ** 2
			pe += morse

			# Repulsive potential term
			if r < dcp_cutoff:
				pe += -dcp_k / r

	return ke, pe, ke + pe

@njit(fastmath=True)
def compute_effective_lagrangian_forces_safe_dcp(
	positions: np.ndarray,
	N: int,				# N count of objects
	D: float,         	# Morse depth
	alpha: float,     	# Morse sharpness
	r0: float,        	# Morse equilibrium distance
	dcp_k: float,   	# Repulsive force constant
	dcp_cutoff: float,  # Repulsive force cutoff distance
	BOX_SIZE: float		# Size of cube.
) -> np.ndarray:
	"""
	Compute the net classical forces on all particles using a local Lagrangian model.

	This function calculates the per-particle net force from pairwise interactions
	using a combination of:

	- Morse potential (attractive):  
	  F_morse = -2 * D * alpha * (1 - exp(-alpha(r - r0))) * exp(-alpha(r - r0))

	- Dirac Core Pressure (repulsive):  
	  F_dcp = dcp_k / r^2  (only applied within cutoff)

	Forces are symmetrically applied to both particles (Newton’s 3rd Law), and
	toroidal (periodic) boundary conditions are enforced for spatial wrapping.

	This is the core per-frame force model that governs all local interactions in
	the Spherical Field Theory simulation.

	Parameters:
		positions (np.ndarray): Nx3 array of particle positions
		N (int): Number of particles
		D (float): Morse potential well depth
		alpha (float): Morse sharpness parameter
		r0 (float): Morse equilibrium distance
		dcp_k (float): Dirac-like repulsive force strength
		dcp_cutoff (float): Cutoff distance for DCP force activation
		BOX_SIZE (float): Simulation space side length (for toroidal wrap)

	Returns:
		np.ndarray: Nx3 array of net force vectors per particle
	"""
	forces = np.zeros_like(positions)
	cutoff_radius_squared = (2.5 * r0) ** 2

	for i in prange(N):
		for j in range(N):
			if i == j:
				continue

			dx = positions[i][0] - positions[j][0]
			dy = positions[i][1] - positions[j][1]
			dz = positions[i][2] - positions[j][2]

			# Toroidal wrapping
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

			r2 = dx * dx + dy * dy + dz * dz
			if r2 > cutoff_radius_squared:
				continue

			r = math.sqrt(r2)
			r = max(r, 1e-5)
			inv_r = 1.0 / r

			dir_x = dx * inv_r
			dir_y = dy * inv_r
			dir_z = dz * inv_r

			# Morse force
			delta = r - r0
			exp_input = min(alpha * delta, 50.0)
			exp_term = math.exp(-exp_input)
			f_morse = -2 * D * alpha * (1 - exp_term) * exp_term

			# Repulsive short-range force (only active when r < dcp_cutoff)
			f_dcp = 0.0
			if r < dcp_cutoff:
				f_dcp = dcp_k / (r * r)

			# Combine forces
			fx = (f_morse + f_dcp) * dir_x
			fy = (f_morse + f_dcp) * dir_y
			fz = (f_morse + f_dcp) * dir_z

			# Apply forces
			forces[i, 0] += fx
			forces[i, 1] += fy
			forces[i, 2] += fz

			forces[j, 0] -= fx
			forces[j, 1] -= fy
			forces[j, 2] -= fz

	return forces

@njit(fastmath=True)
def compute_cluster_energy(cluster_indices, positions, velocities, D, alpha, r0, repel_k, repel_cutoff, BOX_SIZE):
	"""
	Compute the Hamiltonian energy of a single cluster.

	This function calculates the total kinetic + potential energy of a
	cluster of particles using a Morse potential and short-range repulsion
	under periodic boundary conditions.

	The potential energy is pairwise-computed for each unique particle pair
	within the cluster, and includes:
	- Morse attraction term: D * (1 - exp(-alpha * (r - r0)))^2
	- Dirac-like soft repulsion term: -repel_k / r (active within repel_cutoff)

	Periodic boundary conditions are applied to simulate a toroidal space.

	Parameters:
		cluster_indices (np.ndarray[int]): Array of particle indices in the cluster
		positions (np.ndarray[float]): Nx3 array of particle positions
		velocities (np.ndarray[float]): Nx3 array of particle velocities
		D (float): Morse potential well depth
		alpha (float): Morse potential sharpness
		r0 (float): Morse equilibrium distance
		repel_k (float): Dirac core pressure constant (soft repulsion)
		repel_cutoff (float): Distance cutoff for repulsion
		BOX_SIZE (float): Size of the simulation box (assumes periodic)

	Returns:
		float: Total Hamiltonian energy (KE + PE) of the cluster
	"""
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

			# Periodic boundary conditions
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

			r = max((dx*dx + dy*dy + dz*dz)**0.5, 1e-5)

			delta = r - r0
			exp_input = min(alpha * delta, 50.0)
			exp_term = math.exp(-exp_input)
			morse = D * (1 - exp_term) ** 2
			pe += morse

			if r < repel_cutoff:
				pe += -repel_k / r

	return ke + pe

### ──────────────────────── Computational Functions ───────────────────── ###

### ──────────────────────── Color and Charge Functions ────────────────── ###

def color_to_charge(color):
	"""
	Convert a particle's color state to its corresponding quantum charge.

	This mapping defines how color states translate to charge values:
	- 0 → +1 (positive)
	- 1 →  0 (neutral)
	- 2 → -1 (negative)

	Used as the basis for computing net charge of clusters.

	Parameters:
		color (int): The color index of a particle (0, 1, or 2)

	Returns:
		int: The corresponding quantum charge
	"""
	return {0: +1, 1: 0, 2: -1}[color]
	
def cluster_charge(cluster_indices, colors):
	"""
	Calculate the net charge of a cluster based on member colors.

	Uses `color_to_charge()` to convert each particle's color state
	into its associated charge, then sums the results.

	Parameters:
		cluster_indices (list[int]): Indices of the particles in the cluster
		colors (np.ndarray): Array of particle color states (0, 1, or 2)

	Returns:
		int: Net charge of the cluster
	"""
	return sum(color_to_charge(colors[i]) for i in cluster_indices)

### ──────────────────────── Color and Charge Functions ────────────────── ###

### ──────────────────────── Logging Functions ─────────────────────────── ###

def write_summary_row(
	logdir, frame, ke, just_born, just_died, spin_flips,
	color_flips, cluster_radius_stats, ke_loss, clusters_damped,
	particles_damped, avg_force_mag, work_done, force_velocity_alignment,
	total_clusters, max_cluster_size, velocity_std
):
	"""
	Write a single summary row for the current simulation frame.

	Creates or appends to 'summary.csv' in the specified log directory.
	This file tracks frame-wise scalar data used for global time-series analysis.

	Columns include:
	- Frame number
	- Kinetic energy
	- Number of new and dissolved protons
	- Spin and color flip counts
	- Cluster radius stats (avg/min/max)
	- KE lost via cluster damping
	- Total clusters and cluster size
	- Force/Work/Alignment metrics
	- Velocity standard deviation

	Parameters:
		logdir (str): Path to output directory
		frame (int): Current frame number
		ke (float): Kinetic energy
		just_born (set): Proton cluster IDs born this frame
		just_died (set): Proton cluster IDs dissolved this frame
		spin_flips (int): Spin flips this frame
		color_flips (int): Color flips this frame
		cluster_radius_stats (list[int]): Per-particle neighbor counts
		ke_loss (float): Total KE lost to damping
		clusters_damped (int): Damped cluster count
		particles_damped (int): Damped particle count
		avg_force_mag (float): Mean per-particle force magnitude
		work_done (float): Frame-level dot(F, v) work
		force_velocity_alignment (float): Mean cos(F·v) across particles
		total_clusters (int): Total cluster count this frame
		max_cluster_size (int): Largest cluster this frame
		velocity_std (float): Std dev of particle velocity magnitudes
	"""
	summary_path = os.path.join(logdir, "summary.csv")
	file_exists = os.path.isfile(summary_path)

	with open(summary_path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow([
				"Frame", "KE",
				"NewBorn", "Dissolved",
				"SpinFlips", "ColorFlips",
				"AvgClusterSize", "MinClusterSize", "MaxClusterSize",
				"KELossClusterDamp", "ClustersDamped", "ParticlesDamped",
				"AvgForceMag", "WorkDone", "ForceVelocityAlignment",
				"TotalClusters", "MaxClusterSize", "VelocityStd"
			])
		writer.writerow([
			frame, f"{ke:.5f}",
			len(just_born), len(just_died),
			spin_flips, color_flips,
			round(np.mean(cluster_radius_stats), 2),
			min(cluster_radius_stats),
			max(cluster_radius_stats),
			round(ke_loss, 5), clusters_damped, particles_damped,
			avg_force_mag, work_done, force_velocity_alignment,
			total_clusters, max_cluster_size, round(velocity_std, 5)
		])

def write_quantum_log_row(logdir, frame, spin_flips, color_flips, dt, ke, pe, hamiltonian):
	"""
	Write a single row to the quantum system log.

	Appends frame-level energy and flip counts to 'quantum_logs.csv'.

	Columns:
	- Frame
	- SpinFlips / ColorFlips
	- DT (time step)
	- Kinetic, Potential, and Hamiltonian energy

	Parameters:
		logdir (str): Log output directory
		frame (int): Frame number
		spin_flips (int): Spin flips this frame
		color_flips (int): Color flips this frame
		dt (float): Timestep value used
		ke (float): Kinetic energy
		pe (float): Potential energy
		hamiltonian (float): Total system Hamiltonian energy
	"""
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
	"""
	Write per-frame proton charge observations.

	Appends to 'proton_charges.csv' with one row per charged cluster.

	Each entry includes:
	- Frame number
	- Cluster ID (dash-separated)
	- Net proton charge

	Parameters:
		logdir (str): Output directory
		frame (int): Current frame number
		charge_data (list of tuples): (cluster_id, charge)
	"""
	path = os.path.join(logdir, "proton_charges.csv")
	file_exists = os.path.isfile(path)

	with open(path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["Frame", "ClusterID", "Charge"])

		for cluster_id, charge in charge_data:
			writer.writerow([frame, cluster_id, charge])

def write_lifetimes_by_charge(logdir, frame, birth_frames, death_frames, charges):
	"""
	Write proton lifetimes by charge at the time of death.

	Used at the end of a simulation to write total lifespan of
	clusters that died during the run.

	Each row includes:
	- Frame (usually final)
	- ClusterID (dash-separated)
	- Charge
	- Lifetime in frames

	Parameters:
		logdir (str): Output directory
		frame (int): Frame of death report
		birth_frames (dict): ClusterID → birth frame
		death_frames (dict): ClusterID → death frame
		charges (dict): ClusterID → integer charge
	"""
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
				cluster_label = generate_cluster_id(cluster)
				writer.writerow([frame, cluster_label, charge, lifetime])
				
def write_charge_lifetimes_framewise(logdir, frame, birth_frames, active_clusters, charges):
	"""
	Log per-frame lifespan of all charged clusters still alive.

	Useful for real-time charge lifespan analysis. Called every frame.

	Each row includes:
	- Frame
	- Cluster ID (dash-separated)
	- Charge
	- Lifetime in frames since birth

	Parameters:
		logdir (str): Output directory
		frame (int): Current frame
		birth_frames (dict): ClusterID → birth frame
		active_clusters (iterable): ClusterIDs currently active
		charges (dict): ClusterID → charge
	"""
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
				cluster_label = generate_cluster_id(cluster)
				writer.writerow([frame, cluster_label, charge, lifetime])

def write_cluster_lifetimes(logdir, cluster_lifetimes):
	"""
	Write cluster lifetime data to separate CSV files per cluster size.

	For each cluster size (e.g., size=3), a CSV file named
	'cluster_lifetimes_size_<size>.csv' is created or appended to
	in the specified log directory.

	Each file contains the following columns:
	- ClusterID (str): Particle indices joined by dashes (sorted)
	- BirthFrame (int or str): The frame when the cluster first formed
	- DeathFrame (int or str): The frame when the cluster dissolved
	- Lifetime (int): Number of frames the cluster persisted

	Parameters:
		logdir (str): Path to the directory where files will be written.
		cluster_lifetimes (dict): Dictionary keyed by cluster size (int),
			with values as {ClusterID: Lifetime} mappings.
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

				writer.writerow([cid, birth, death, lifetime])

def write_binding_energy(logdir, energy_log_data):
	"""
	Write proton binding energy data to a CSV file.

	Appends new binding energy records to 'proton_binding_energy.csv'
	in the specified log directory. If the file does not exist, a header
	row is written first.

	Each row contains:
	- Frame number (int)
	- ClusterID (str) : particle indices joined by dashes (sorted)
	- Charge (int)
	- BindingEnergy (float)

	Parameters:
		logdir (str): The directory where the CSV file is stored.
		energy_log_data (list): A list of tuples in the form:
			(frame: int, cluster: list[int], charge: int, energy: float)
	"""
	path = os.path.join(logdir, "proton_binding_energy.csv")
	file_exists = os.path.isfile(path)
	with open(path, 'a', newline='') as f:
		writer = csv.writer(f)
		if not file_exists:
			writer.writerow(["Frame", "ClusterID", "Charge", "BindingEnergy"])
		for frame, cluster, charge, energy in energy_log_data:
			cluster_str = generate_cluster_id(cluster)
			writer.writerow([frame, cluster_str, charge, energy])

### ──────────────────────── Logging Functions ─────────────────────────── ###

### ──────────────────────── Cluster ID Functions ──────────────────────── ###

def generate_cluster_id(cluster_indices):
	"""
	Generate a unique, reproducible string ID from a list of particle indices.

	Used to create a canonical identifier for a cluster based on the indices
	of the particles it contains. The resulting string is:
	- Deterministic (sorted)
	- Compact (colon-delimited)
	- Hashable (used as a dictionary key)

	Example:
		[5, 2, 9] → "2:5:9"

	Parameters:
		cluster_indices (list[int]): List of particle indices in the cluster.

	Returns:
		str: Cluster ID string.
	"""
	return ":".join(map(str, sorted(cluster_indices)))

def parse_cluster_id(cluster_id):
	"""
	Parse a cluster ID string into a list of particle indices.

	Used to convert a stored or logged ClusterID string back into its
	original list of particle indices. If the input is already a list, it is
	returned unchanged to ensure compatibility with mixed ID usage.

	Example:
		"3:7:8" → [3, 7, 8]

	Parameters:
		cluster_id (str or list[int]): Cluster ID in string or list form.

	Returns:
		list[int]: Particle indices representing the cluster.
	"""
	if isinstance(cluster_id, str):
		return list(map(int, cluster_id.split(":")))
	return cluster_id  # Already a list — just return it as-is
	
def get_cluster_birth_frame(size, cid):
	"""
	Retrieve the birth frame of a cluster from the quantum tracking dictionary.

	This function looks up when a specific cluster (by size and ID) first formed.
	If the cluster ID is not yet a string, it will be generated from the list
	using `generate_cluster_id`.

	Parameters:
		size (int): Size of the cluster (number of particles).
		cid (str or list[int]): Cluster ID string or list of particle indices.

	Returns:
		int or None: Frame number when the cluster was first recorded, or None
	if not found.
	"""
	if not isinstance(cid, str):
		cid = generate_cluster_id(cid)
	return quantum.cluster_birth_frames[size].get(cid, None)

### ──────────────────────── Cluster ID Functions ──────────────────────── ###

### ──────────────────────── SHA256 File Hash Functions ────────────────── ###

def hash_self():
	"""
	Compute the SHA-256 hash of the currently executing Python script.

	This function reads the full contents of the current source file 
	(as identified by `__file__`) and calculates its SHA-256 hash.
	The result is a deterministic, fixed-length string that uniquely 
	represents the exact code used for the simulation run.

	This allows precise identification of the simulation version 
	used to produce a given output dataset, even if no formal version 
	control system (e.g., Git) is being used.

	The hash can be included in logs or metadata.json to ensure full 
	reproducibility and code integrity across different platforms, 
	machines, or historical runs.

	Returns:
		str: A 64-character hexadecimal SHA-256 digest of the current script.
	"""
	path = os.path.abspath(__file__)
	with open(path, "rb") as f:
		return hashlib.sha256(f.read()).hexdigest()

def sha256_file(path):
	"""
	Compute the SHA256 hash of a file.

	Reads the file at the given path in binary mode and returns its
	SHA256 checksum as a hexadecimal string. This is used for verifying
	data integrity or comparing file content across machines.

	Parameters:
		path (str): Absolute or relative path to the file.

	Returns:
		str: SHA256 hash of the file's contents, in hexadecimal format.
	"""
	with open(path, "rb") as f:
		return hashlib.sha256(f.read()).hexdigest()

def write_hash_manifest(log_dir):
	"""
	Generate and write SHA256 hashes for all output files in a directory.

	Recursively scans the specified log directory and computes a SHA256
	hash for each file, skipping the output hash file itself ("hashes.json").
	Writes the results to a JSON file (`hashes.json`) that maps relative file
	paths to their corresponding hashes.

	This manifest can be used to:
	- Verify file integrity after simulation runs
	- Detect tampering or corruption
	- Compare identical runs across systems or environments

	Parameters:
		log_dir (str): Path to the root output directory where files are stored.

	Outputs:
		hashes.json (in log_dir): A JSON file mapping relative file paths to SHA256 hashes.
	"""
	hashes = {}
	for root, _, files in os.walk(log_dir):
		for name in files:
			# Skip the hash file itself
			if name == "hashes.json":
				continue
			full_path = os.path.join(root, name)
			rel_path = os.path.relpath(full_path, log_dir)
			hashes[rel_path] = sha256_file(full_path)

	with open(os.path.join(log_dir, "hashes.json"), "w") as f:
		json.dump(hashes, f, indent=2)
		
### ──────────────────────── SHA256 File Hash Functions ────────────────── ###

### ──────────────────────── Primary Update Function ───────────────────── ###

def update(frame):
	"""
	Primary per-frame update function for the Spherical Field Theory simulation.

	This function is the core engine of the simulation, handling:
	- Sphere position updates and force integration
	- Neighbor lookup and cluster detection via KDTree
	- Proton/superproton classification (based on spin and color rules)
	- Charge and binding energy tracking
	- Radiative damping of valid clusters
	- Logging of cluster birth/death, spin flips, energy metrics, KE trends
	- Per-frame CSV and quantum log writes
	- GUI scatter plot updates (if not in headless mode)
	- Terminal output (if verbose or headless mode enabled)

	The result is stored in quantum.live_stats for optional GUI display.

	Parameters:
		frame (int): Current simulation frame number.

	Returns:
		tuple or None: Updated matplotlib scatter plot, or None in headless mode.
	"""
	global previous_proton_ids, start_time

	frame_start = time.time()

	ke = quantum.update()


	radius = 0.1 * PLANCK_LENGTH

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

	tree = cKDTree(quantum.positions)

	for i in range(quantum.N):
		cluster = tree.query_ball_point(quantum.positions[i], radius)
		cluster = [j for j in cluster if j != i]  # NOW back in
		cluster_radius_stats.append(len(cluster))

		full_cluster = [i] + cluster
		cluster_id = generate_cluster_id(full_cluster)
		
		if cluster_id in seen_cluster_ids:
			continue  # Already processed
		
		seen_cluster_ids.add(cluster_id)
		cluster_size = len(full_cluster)
		quantum.current_clusters_by_size[cluster_size].add(cluster_id)

		if cluster_id not in quantum.previous_clusters_by_size.get(cluster_size, set()):
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
					cluster_indices_np = np.array(full_cluster, dtype=np.int32)
					cluster_energy = compute_cluster_energy(
						cluster_indices_np,
						quantum.positions,
						quantum.velocities,
						quantum.D,
						quantum.alpha,
						quantum.r0,
						quantum.dcp_k,
						quantum.dcp_cutoff,
						quantum.BOX_SIZE
					)

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

	# Radiative dampening section

	frame_ke_loss = 0.0  # RESET just before applying damping!
	clusters_damped = 0
	particles_damped = 0

	for size, clusters in clusters_by_size.items():
		if size >= 3:
			for cluster_id in clusters:
				cluster_members = parse_cluster_id(cluster_id)

				birth_frame = get_cluster_birth_frame(size, cluster_id)
				if birth_frame is None:
					birth_frame = quantum.frame - 1

				lifetime = frame - birth_frame
				if lifetime < 2:
					continue  # too young to safely damped
				
				# Stronger damp based on size
				base_damp = 0.995
				extra = min(0.002 * (size - 3), 0.025)
				damp_factor = base_damp - extra
				
				if lifetime > 20:
					damp_factor -= 0.002  # stackable damping for old clusters

				# Log damped cluster and particle count
				clusters_damped += 1
				particles_damped += len(cluster_members)
				
				for idx in cluster_members:
					v_before = quantum.velocities[idx].copy()
					quantum.velocities[idx] *= damp_factor
					v_after = quantum.velocities[idx]
					frame_ke_loss += 0.5 * (np.dot(v_before, v_before) - np.dot(v_after, v_after))


	total_clusters = sum(len(v) for v in clusters_by_size.values())
	max_cluster_size = max(clusters_by_size.keys(), default=0)
	velocity_mags = np.linalg.norm(quantum.velocities, axis=1)
	velocity_std = np.std(velocity_mags)

	# Lifetimes
	for pid in new_proton_ids:
		proton_lifetimes[pid] = proton_lifetimes.get(pid, 0) + 1

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

	# Compute lifetime of existing protons
	lifetimes = {
		cluster: frame - quantum.proton_birth_frames[cluster]
		for cluster in quantum.stable_proton_clusters
		if cluster in quantum.proton_birth_frames
	}

	# Make sure IDs are sorted and consistent
	generate_id = lambda cluster: ":".join(map(str, sorted(cluster)))

	prev_ids = set(generate_id(c) for c in quantum.previous_clusters_by_size.get(size, []))
	curr_ids = set(generate_id(c) for c in quantum.current_clusters_by_size[size])

	just_died = prev_ids - curr_ids

	for cid in just_died:
		if cid not in quantum.cluster_death_frames[size]:
			birth = quantum.cluster_birth_frames[size].get(cid, quantum.frame - 1)
			quantum.cluster_death_frames[size][cid] = quantum.frame
			quantum.cluster_lifetimes[size][cid] = quantum.frame - birth

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
		just_born, just_died,
		quantum.last_spin_flips, quantum.last_color_flip,
		cluster_radius_stats,
		frame_ke_loss, clusters_damped, particles_damped,
		quantum.frame_force_magnitude,
		quantum.frame_force_work,
		quantum.frame_avg_force_velocity_cos,
		total_clusters, max_cluster_size, velocity_std
	)

	if quantum.fast:
		ke, pe, h_total = compute_hamiltonian_energy_fast_dcp(
			quantum.positions,
			quantum.velocities,
			quantum.N,
			quantum.D,
			quantum.alpha,
			quantum.r0,
			quantum.dcp_k,
			quantum.dcp_cutoff,
			quantum.BOX_SIZE
		)
		
	else:
		ke, pe, h_total = compute_hamiltonian_energy_safe_dcp(
			quantum.positions,
			quantum.velocities,
			quantum.N,
			quantum.D,
			quantum.alpha,
			quantum.r0,
			quantum.dcp_k,
			quantum.dcp_cutoff,
			quantum.BOX_SIZE
		)

	write_quantum_log_row(
		quantum.log_dir,
		frame,
		quantum.last_spin_flips,
		quantum.last_color_flip,
		quantum.DT,
		ke, pe, h_total
	)

	if frame % 500 == 0:
		write_binding_energy(quantum.log_dir, energy_log_data)
	
	frame_end = time.time()
	frame_duration = frame_end - frame_start

	elapsed_total = frame_end - start_time
	remaining_frames = MAX_FRAMES - frame
	est_remaining_time = frame_duration * remaining_frames
	est_total_time = elapsed_total + est_remaining_time

	real = frame + 1

	if quantum.verbose or HEADLESS:
		log_line = (
			f"[Frame {frame:5}] KE={quantum.kinetic_energy():.5f}"
			f" | Spin={quantum.last_spin_flips} Color={quantum.last_color_flip}"
		)

		if just_born:
			log_line += f" | +{len(just_born)} Born"
		if just_died:
			log_line += f" | -{len(just_died)} Died"

		log_line += f" | KE Loss={frame_ke_loss:.5f}"
		log_line += (
			f"\n           Time/frame: {frame_duration:.3f}s"
			f" | Elapsed: {elapsed_total/60:.1f} min"
			f" | ETA: {est_remaining_time/60:.1f} min"
			f" | Total: {est_total_time/60:.1f} min"
		)
		log_line += (
			f"\n           Cluster sizes → Min: {min(cluster_radius_stats)}, "
			f"Max: {max(cluster_radius_stats)}, Avg: {np.mean(cluster_radius_stats):.2f}"
		)

		print(log_line)
	
	cal = MAX_FRAMES - 1
	if frame == cal:
		for size, active_clusters in quantum.current_clusters_by_size.items():
			for cid in active_clusters:
				if cid not in quantum.cluster_birth_frames[size]:
					continue
				birth = quantum.cluster_birth_frames[size][cid]
				lifetime = MAX_FRAMES - birth
				quantum.cluster_death_frames[size][cid] = MAX_FRAMES
				quantum.cluster_lifetimes[size][cid] = lifetime

		total_lifetimes = sum(len(v) for v in quantum.cluster_lifetimes.values())

		write_cluster_lifetimes(quantum.log_dir, quantum.cluster_lifetimes)
		write_hash_manifest(quantum.log_dir)
		if quantum.verbose or HEADLESS:
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

	quantum.live_stats = {
		# Existing stats...
		"KE": f"{quantum.ke:.5f}",
		"KE Loss": f"{frame_ke_loss:.5f}",
		"Clusters Damped": f"{clusters_damped}",
		"Particles Damped": f"{particles_damped}",
		"Total Clusters": f"{total_clusters}",
		"Max Cluster Size": f"{max_cluster_size}",
		"Spin Flips": f"{quantum.last_spin_flips}",
		"Color Flips": f"{quantum.last_color_flip}",
		"cos(F·v)": f"{quantum.frame_avg_force_velocity_cos:.3f}",
		"Work": f"{quantum.frame_force_work:.5f}",
		"Avg Force": f"{quantum.frame_force_magnitude:.5f}",

		# Timing stats
		"Elapsed Time": f"{elapsed_total:.2f}",
		"ETA": f"{est_remaining_time:.2f}",
		"Est Total": f"{est_total_time:.2f}",
		"Frame Time": f"{frame_duration:.3f}",
		"Frame": real  # this will be used for progress bar too
	}
	
	return (scatter,) if not HEADLESS else None

### ──────────────────────── Primary Update Function ───────────────────── ###

### ──────────────────────── __main__ Runtime Execution ────────────────── ###

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Run the Quantum Sphere Simulation")
	parser.add_argument('--config', type=str, help="Path to metadata.json config file")
	parser.add_argument('--headless', action='store_true', help="Run simulation without visualization")
	parser.add_argument('--outdir', type=str, help="Optional output directory to save logs.")
	parser.add_argument('--verbose', action='store_true', help="Print live simulation output to terminal.")
	parser.add_argument('--fast', action='store_true', help="Enabled Numba parallel=True, this is non-deterministic, results will never match, seed or not.")
	args = parser.parse_args()
	
	if args.config and not os.path.isfile(args.config):
		print(f"[Error] Metadata file not found: {args.config}")
		sys.exit(1)

	quantum = QuantumUniverse(config_path=args.config, output_dir=args.outdir, verbose=args.verbose, fast=args.fast)

	if args.headless:
		HEADLESS = args.headless
		for frame in range(MAX_FRAMES):
			update(frame)
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

### ──────────────────────── __main__ Runtime Execution ────────────────── ###