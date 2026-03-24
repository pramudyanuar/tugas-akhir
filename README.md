![alt text](image.png)

## Ringkasan Proyek

Repositori ini berisi sistem online 3D container loading berbasis kontrol hierarkis.
Sistem menggabungkan:

1. High-level controller untuk keputusan strategi makro.
2. Low-level policy (actor-critic/PPO) untuk pemilihan aksi penempatan detail.
3. Rearrangement planner berbasis MCTS saat sistem mengalami deadlock (tidak ada posisi legal).

## Alur Algoritma Utama

### 1. Hierarchical Control

Setiap item diproses dengan urutan berikut:

1. Amati state kontainer saat ini (height map + item saat ini).
2. High-level agent menghasilkan macro decision:
	- orientasi/arah strategi,
	- prioritas zona placement,
	- apakah repacking boleh dipicu.
3. Candidate generator membuat kandidat aksi berdasarkan macro decision.
4. Feasibility masking menghapus aksi tidak valid (out-of-bound, overflow, instabil).
5. Jika kandidat tersedia, low-level policy memilih aksi terbaik dari kandidat valid.
6. Jika kandidat kosong, masuk deadlock handler:
	- coba rearrangement/repacking,
	- jika masih gagal, fallback ke search MCTS aksi placement.
7. Environment diupdate, lanjut ke item berikutnya.

### 2. Low-Level Placement Policy (Actor-Critic)

Low-level policy berjalan dengan langkah:

1. Logits actor dimasking agar hanya aksi legal yang punya probabilitas.
2. Sampling aksi dilakukan dari distribusi masked policy.
3. Critic mengestimasi value state untuk update advantage.
4. PPO update dilakukan berkala dari trajectory yang terkumpul.

Intinya: actor memilih aksi placement, critic menilai kualitas state, dan PPO menjaga update tetap stabil.

### 3. Rearrangement dan Repacking (MCTS)

Saat deadlock terjadi, MCTS rearrangement dijalankan dengan 4 fase:

1. Selection: pilih node menggunakan UCB.
2. Expansion: buat child dari aksi unpack top-most k items.
3. Simulation: coba repack item yang di-unpack + item gagal, hitung reward (utilization + success).
4. Backpropagation: propagasi nilai reward sepanjang path tree.

Jika urutan terbaik valid, snapshot hasil bisa langsung diaplikasikan ke environment utama.

## Struktur Folder Inti

1. env
	- Environment, masking, height map, dan utilitas stabilitas.
2. rl
	- High-level dan low-level model, termasuk PPO.
3. planning
	- MCTS search dan mekanisme rearrangement/repacking.
4. dataset
	- Generator instance item.
5. utils
	- Logging dan metrics helper.
6. tests
	- Unit tests untuk komponen penting (termasuk MCTS rearrangement).

## Menjalankan dengan Make

Gunakan Makefile untuk perintah harian:

1. Lihat semua target:
	make help
2. Preview cleanup (dry-run):
	make clean
3. Terapkan cleanup:
	make clean-apply
4. Cleanup penuh + hapus folder output kosong:
	make clean-full
5. Jalankan unit test:
	make test
6. Smoke test evaluasi:
	make eval-smoke
7. Smoke test training:
	make train-smoke
8. Smoke test evaluasi (cutting stock):
	make eval-cutting-smoke
9. Smoke test training (cutting stock):
	make train-cutting-smoke
10. Jalankan semua smoke checks sekaligus:
	make all-smoke

## Cleanup Manual

Jika ingin langsung pakai script:

1. Preview:
	scripts/clean.sh
2. Apply:
	scripts/clean.sh --apply
3. Opsi tambahan:
	scripts/clean.sh --apply --no-outputs
	scripts/clean.sh --apply --no-caches