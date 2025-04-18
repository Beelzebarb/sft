# Archived original force function. Replaced by Lagrangian-based version.

@njit(parallel=True)
def compute_forces_numba(positions, spins, colors, N, planck_length, pauli_strength, use_gravity):
	forces = np.zeros_like(positions)
	G_scaled = (6.67430e-11 * 1.0**2) / (0.12**3 * 800.0)  # Assuming L_unit = 0.12, E_unit = 800.0

	for i in prange(N):
		for j in range(N):
			if i == j:
				continue

			delta = positions[i] - positions[j]
			dist = np.linalg.norm(delta)
			if dist < 1e-10:
				continue
			dir_ = delta / dist

			# Pauli Exclusion
			if dist < planck_length * 0.3 and spins[i] == spins[j]:
				forces[i] += pauli_strength * dir_ / (dist ** 4 + 1e-12)
				continue

			# Quantum forces
			nuclear_attraction = -8.0 * np.exp(-(dist - 0.8) ** 2)
			confinement = 0.2 * dist
			color_factor = 1 if colors[i] == colors[j] else -1

			total_force = color_factor * (nuclear_attraction + confinement) * dir_
			forces[i] += total_force

			# Gravity (mirrored)
			if use_gravity:
				capped_dist = max(dist, 2.0)
				g_force_mag = G_scaled / (capped_dist ** 2)
				gravity_force = g_force_mag * dir_
				forces[i] += gravity_force
				#forces[j] -= gravity_force  # NOTE: remove for strict symmetry or track separately

	return forces

# Archived energy dissipation/dampening, old global dampening, replaced with local interaction based dampening.

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
	
#Use: #self.apply_energy_dissipation() # Moved here after velocities are set but before positions are set. - put right before self.positions at end of update(self) in quantum class.
	
#Archived proton injection function, now done entirely spontaneously by the equations and environment.

def gradual_proton_birth(self, rate=0.005):
	# Gradually add protons at a specific rate over time
	num_new_protons = int(rate * N)  # Adjust the rate as necessary
	new_protons = np.random.choice(N, num_new_protons, replace=False)
	# Modify the positions, spins, and colors of the new protons
	self.spins[new_protons] = np.random.choice([-1, 1], num_new_protons)
	self.colors[new_protons] = np.random.randint(0, 2, num_new_protons)
	return new_protons  # Return the new proton IDs


#Archived custom Lagrangian and Hamiltonian/Cluster energy compute functions, replaced with Lennard-Jones pairwise force calculations

@njit(parallel=True)
def compute_hamiltonian_energy(positions, velocities, N, A, B, C, BOX_SIZE):
	# Kinetic Energy
	ke = 0.0
	for i in prange(N):
		ke += 0.5 * np.dot(velocities[i], velocities[i])

	# Potential Energy
	pe = 0.0
	for i in prange(N):
		for j in range(i + 1, N):
			delta = positions[i] - positions[j]
			# Apply minimum image convention (PBC-aware direction)
			for k in range(3):  # x, y, z dimensions
				if delta[k] > 0.5 * BOX_SIZE:
					delta[k] -= BOX_SIZE
				elif delta[k] < -0.5 * BOX_SIZE:
					delta[k] += BOX_SIZE
			dist = max(np.sqrt(np.dot(delta, delta)), 0.2)
			pe += -A / dist**3 + B * dist**2 - C * dist

	return ke, pe, ke + pe

@njit(parallel=True)
def compute_effective_lagrangian_forces(positions: np.ndarray, N: int, A: float, B: float, C: float, BOX_SIZE: float) -> np.ndarray:
	forces = np.zeros_like(positions)
	cutoff_radius_squared = (2.5 * PLANCK_LENGTH) ** 2
	
	for i in prange(N):
		for j in range(N):
			if i == j:
				continue

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

			if r2 > cutoff_radius_squared:
				continue  # Too far

			r = math.sqrt(r2)
			r = max(r, PLANCK_LENGTH * 0.5)  # ← Clamps minimum distance
			inv_r = 1.0 / r
			dir_x = dx * inv_r
			dir_y = dy * inv_r
			dir_z = dz * inv_r

			# Now assuming V = -A / r³ + B·r² - C·r
			# So force is -dV/dr = +3A/r⁴ + 2B·r - C

			f1x = -3 * A * dir_x / r**4
			f1y = -3 * A * dir_y / r**4
			f1z = -3 * A * dir_z / r**4

			f2x = -2 * B * r * dir_x
			f2y = -2 * B * r * dir_y
			f2z = -2 * B * r * dir_z

			f3x = +C * dir_x
			f3y = +C * dir_y
			f3z = +C * dir_z

			forces[i, 0] += f1x + f2x + f3x
			forces[i, 1] += f1y + f2y + f3y
			forces[i, 2] += f1z + f2z + f3z

	return forces

@njit
def compute_cluster_energy(cluster_indices, positions, velocities, A, B, C, BOX_SIZE):
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

            r = max((dx*dx + dy*dy + dz*dz)**0.5, 0.2)  # Avoid unrealistically small r
            pe += A / (r**3) - B * (r**2) + C * r

    return ke + pe