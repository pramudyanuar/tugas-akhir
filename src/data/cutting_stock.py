import numpy as np


class CuttingStockGenerator:
	"""
	Generator item untuk skenario cutting-stock style.

	Karakteristik:
	- Banyak item kecil-menengah (offcuts)
	- Sesekali item besar (primary cuts)
	- Reproducible dengan seed
	- Optional target utilization untuk mendekati full container
	"""

	def __init__(self, seed=None, container_dims=(60, 24, 26), target_utilization=1.0, min_dim=1):
		self.seed = seed
		self.rng = np.random.RandomState(seed)
		self.L, self.W, self.H = container_dims
		self.target_utilization = target_utilization
		self.min_dim = max(1, int(min_dim))

	def set_seed(self, seed):
		self.seed = seed
		self.rng = np.random.RandomState(seed)

	def generate_episode(self, num_items=None):
		"""
		Generate episode items dalam format (length, width, height).
		Jika target_utilization diset, total volume diarahkan mendekati container penuh.
		"""
		if num_items is not None and num_items <= 0:
			raise ValueError(f"num_items harus positive, got {num_items}")

		if self.target_utilization is None:
			if num_items is None:
				raise ValueError("num_items wajib untuk generator non-full")
			items = self._generate_random_items(num_items)
			items.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
			return items

		items = self._generate_full_items(num_items)
		items.sort(key=lambda x: x[0] * x[1] * x[2], reverse=True)
		return items

	def _generate_random_items(self, num_items):
		items = []

		for _ in range(num_items):
			u = self.rng.rand()

			if u < 0.65:
				# Offcuts: dominan kecil-menengah
				l = self.rng.randint(2, min(9, self.L) + 1)
				w = self.rng.randint(2, min(8, self.W) + 1)
				h = self.rng.randint(2, min(7, self.H) + 1)
			elif u < 0.90:
				# Mid pieces
				l = self.rng.randint(6, min(13, self.L) + 1)
				w = self.rng.randint(5, min(10, self.W) + 1)
				h = self.rng.randint(4, min(9, self.H) + 1)
			else:
				# Primary pieces: relatif besar
				l = self.rng.randint(10, min(16, self.L) + 1)
				w = self.rng.randint(8, min(13, self.W) + 1)
				h = self.rng.randint(6, min(11, self.H) + 1)

			items.append((int(l), int(w), int(h)))

		return items

	def _generate_full_items(self, num_items):
		container_volume = int(self.L * self.W * self.H)
		target_volume = int(round(container_volume * float(self.target_utilization)))
		target_volume = max(self.min_dim, min(target_volume, container_volume))
		remaining_volume = target_volume
		items = []
		min_fill_volume = self.min_dim ** 3

		if num_items is not None:
			for i in range(num_items):
				remaining_items = num_items - i
				if remaining_items == 1:
					item = self._make_filler_item(remaining_volume)
					items.append(item)
					remaining_volume -= item[0] * item[1] * item[2]
					break

				max_volume_for_item = remaining_volume - (remaining_items - 1) * min_fill_volume
				if max_volume_for_item <= 0:
					break

				item = self._sample_item(max_volume_for_item)
				items.append(item)
				remaining_volume -= item[0] * item[1] * item[2]
		else:
			while remaining_volume > 0:
				item = self._sample_item(remaining_volume)
				items.append(item)
				remaining_volume -= item[0] * item[1] * item[2]

		while remaining_volume > 0:
			item = self._make_filler_item(remaining_volume)
			items.append(item)
			remaining_volume -= item[0] * item[1] * item[2]

		return items

	def _sample_item(self, max_volume):
		for _ in range(50):
			u = self.rng.rand()
			if u < 0.65:
				l = self.rng.randint(max(self.min_dim, 2), min(9, self.L) + 1)
				w = self.rng.randint(max(self.min_dim, 2), min(8, self.W) + 1)
				h = self.rng.randint(max(self.min_dim, 2), min(7, self.H) + 1)
			elif u < 0.90:
				l = self.rng.randint(max(self.min_dim, 6), min(13, self.L) + 1)
				w = self.rng.randint(max(self.min_dim, 5), min(10, self.W) + 1)
				h = self.rng.randint(max(self.min_dim, 4), min(9, self.H) + 1)
			else:
				l = self.rng.randint(max(self.min_dim, 10), min(16, self.L) + 1)
				w = self.rng.randint(max(self.min_dim, 8), min(13, self.W) + 1)
				h = self.rng.randint(max(self.min_dim, 6), min(11, self.H) + 1)
			volume = int(l) * int(w) * int(h)
			if volume <= max_volume:
				return (int(l), int(w), int(h))

		return self._make_filler_item(max_volume)

	def _make_filler_item(self, remaining_volume):
		max_l = max(self.min_dim, self.L)
		max_w = max(self.min_dim, self.W)

		for l in range(min(max_l, self.L), 0, -1):
			for w in range(min(max_w, self.W), 0, -1):
				base = l * w
				if base <= 0:
					continue
				if remaining_volume % base == 0:
					h = remaining_volume // base
					if h <= self.H:
						return (int(l), int(w), int(max(self.min_dim, h)))

		l = max(self.min_dim, min(self.L, int(np.sqrt(remaining_volume))))
		w = max(self.min_dim, min(self.W, max(1, remaining_volume // max(1, l * self.H))))
		h = max(self.min_dim, min(self.H, max(1, remaining_volume // max(1, l * w))))
		return (int(l), int(w), int(h))


def generate_episode(num_items=None, seed=None, container_dims=(60, 24, 26), target_utilization=1.0, min_dim=1):
	gen = CuttingStockGenerator(
		seed=seed,
		container_dims=container_dims,
		target_utilization=target_utilization,
		min_dim=min_dim,
	)
	return gen.generate_episode(num_items=num_items)
