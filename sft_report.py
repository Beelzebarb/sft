import os
import csv
from collections import defaultdict
import statistics
import json

def load_metadata(folder):
	metadata_file = os.path.join(folder, "metadata.json")
	if not os.path.exists(metadata_file):
		return None
	with open(metadata_file) as f:
		try:
			return json.load(f)
		except json.JSONDecodeError:
			return None

def load_cluster_lifetimes(folder):
	lifetimes = defaultdict(list)
	for fname in os.listdir(folder):
		if fname.startswith("cluster_lifetimes_size_") and fname.endswith(".csv"):
			size = int(fname.split("_")[-1].replace(".csv", ""))
			with open(os.path.join(folder, fname)) as f:
				reader = csv.DictReader(f)
				for row in reader:
					try:
						lifetime = int(row["Lifetime"])
						lifetimes[size].append(lifetime)
					except:
						continue
	return lifetimes

def load_binding_energy(folder):
	energy_by_charge = defaultdict(list)
	fname = os.path.join(folder, "proton_binding_energy.csv")
	if not os.path.exists(fname):
		return energy_by_charge
	with open(fname) as f:
		reader = csv.reader(f)
		next(reader)
		for row in reader:
			if len(row) >= 4:
				_, _, charge, energy = row
				try:
					charge = int(charge)
					energy = float(energy)
					energy_by_charge[charge].append(energy)
				except:
					continue
	return energy_by_charge

def load_quantum_logs(folder):
	fname = os.path.join(folder, "quantum_logs.csv")
	if not os.path.exists(fname):
		return []
	data = []
	with open(fname) as f:
		reader = csv.DictReader(f)
		for row in reader:
			try:
				frame = int(row["Frame"])
				ke = float(row["KineticEnergy"])
				pe = float(row["PotentialEnergy"])
				h = float(row["HamiltonianEnergy"])
				data.append((frame, ke, pe, h))
			except:
				continue
	return data

def report(folder):
	print(f"\n=== SFT Simulation Report ===\nSource: {folder}\n")

	metadata = load_metadata(folder)

	if metadata:
		print("Metadata\n--------")
		for key in sorted(metadata.keys()):
			print(f"{key}: {metadata[key]}")
		print("")
	else:
		print("Metadata: (not found)\n")

	lifetimes = load_cluster_lifetimes(folder)
	energies = load_binding_energy(folder)
	qlogs = load_quantum_logs(folder)

	print("1. Cluster Lifetimes\n-------------------")
	total_clusters = 0
	for size in sorted(lifetimes):
		cluster_lives = lifetimes[size]
		total_clusters += len(cluster_lives)
		avg_life = statistics.mean(cluster_lives)
		min_life = min(cluster_lives)
		max_life = max(cluster_lives)
		immortals = sum(1 for l in cluster_lives if l == max_life)
		print(f"Size {size}: {len(cluster_lives)} clusters, Avg = {avg_life:.2f}, Max = {max_life}, Immortal = {immortals}")

	print(f"\nTotal Clusters: {total_clusters}")

	print("\n2. Binding Energy by Charge\n---------------------------")
	for charge in sorted(energies):
		e_list = energies[charge]
		avg_e = statistics.mean(e_list)
		print(f"Charge {charge:+}: {len(e_list)} clusters, Avg Energy = {avg_e:.3f}")

	print("\n3. Energy Log Summary\n---------------------")
	if qlogs:
		final_ke = qlogs[-1][1]
		final_pe = qlogs[-1][2]
		final_h = qlogs[-1][3]
		print(f"Final Frame: KE = {final_ke:.3f}, PE = {final_pe:.3f}, H = {final_h:.3f}")
	else:
		print("No quantum_logs.csv found.")

	print("\n=== End of Report ===")

# Entry point
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python sft_report.py path/to/log_folder")
        exit(1)
    folder = sys.argv[1]
    report(folder)