import numpy as np


class CuttingStockGenerator:
	"""
	Generator item untuk skenario cutting-stock style.

	Karakteristik:
	- Banyak item kecil-menengah (offcuts)
	- Sesekali item besar (primary cuts)
	- Reproducible dengan seed
	"""

	def __init__(self, seed=None):
		self.seed = seed
		self.rng = np.random.RandomState(seed)

	def set_seed(self, seed):
		self.seed = seed
		self.rng = np.random.RandomState(seed)

	def generate_episode(self, num_items=50):
		"""
		Generate episode items dalam format (length, width, height).
		"""
		if num_items <= 0:
			raise ValueError(f"num_items harus positive, got {num_items}")

		items = []

		for _ in range(num_items):
			u = self.rng.rand()

			if u < 0.65:
				# Offcuts: dominan kecil-menengah
				l = self.rng.randint(2, 9)
				w = self.rng.randint(2, 8)
				h = self.rng.randint(2, 7)
			elif u < 0.90:
				# Mid pieces
				l = self.rng.randint(6, 13)
				w = self.rng.randint(5, 10)
				h = self.rng.randint(4, 9)
			else:
				# Primary pieces: relatif besar
				l = self.rng.randint(10, 16)
				w = self.rng.randint(8, 13)
				h = self.rng.randint(6, 11)

			items.append((int(l), int(w), int(h)))

		# Sort descending by volume seperti order cutting plan.
		items.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
		return items


def generate_episode(num_items=50, seed=None):
	gen = CuttingStockGenerator(seed=seed)
	return gen.generate_episode(num_items=num_items)
