SHELL := /bin/bash

# ====== VENV CONFIG ======
VENV_DIR := .venv
PY := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := . $(VENV_DIR)/bin/activate

# ====== PATHS ======
DATA_DIR := data/synthetic
PREVIEW_OUT := results/dataset_preview

# ====== DEFAULT PARAMS ======
N_SEQ ?= 1000
SEQ_LEN ?= 500
SEED ?= 42

BIN_L ?= 1.0
BIN_W ?= 1.0
BIN_H ?= 1.0

LOOKAHEAD ?= 3
DELTA_COG ?= 0.05

SAME_HEIGHT ?= 0.12

GRID_W ?= 10
GRID_H ?= 10
TARGET_RECTS ?= 35

.PHONY: help venv install freeze check \
        gen gen-random3d gen-same-height gen-fill100 gen-semi \
        preview preview-all \
        clean clean-preview clean-data clean-venv

help:
	@echo "==== Stuffing Optimizer Makefile (with venv) ===="
	@echo ""
	@echo "Setup:"
	@echo "  make venv        -> create python venv in $(VENV_DIR)"
	@echo "  make install     -> install dependencies"
	@echo "  make freeze      -> write requirements.txt"
	@echo ""
	@echo "Dataset generation:"
	@echo "  make gen         -> generate ALL datasets"
	@echo "  make gen-random3d"
	@echo "  make gen-same-height"
	@echo "  make gen-fill100"
	@echo "  make gen-semi"
	@echo ""
	@echo "Preview:"
	@echo "  make preview DATA=<jsonl path>"
	@echo "  make preview-all"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean-data"
	@echo "  make clean-preview"
	@echo "  make clean-venv"
	@echo "  make clean"
	@echo ""
	@echo "Example:"
	@echo "  make venv install"
	@echo "  make gen-random3d N_SEQ=200 SEQ_LEN=150"
	@echo "  make preview DATA=data/synthetic/random3d/train.jsonl"

venv:
	python3 -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip wheel setuptools
	@echo "✅ venv created at $(VENV_DIR)"

install: venv
	$(ACTIVATE) && $(PIP) install -r requirements.txt
	@echo "✅ deps installed"

freeze: venv
	$(ACTIVATE) && $(PIP) freeze > requirements.txt
	@echo "✅ requirements.txt updated"

check: venv
	$(ACTIVATE) && $(PY) --version
	$(ACTIVATE) && $(PIP) --version

# ====== GENERATION ======
gen: gen-random3d gen-same-height gen-fill100 gen-semi
	@echo "✅ All datasets generated."

gen-random3d: venv
	$(ACTIVATE) && $(PY) scripts/generate_dataset/generate_dataset_random3d.py \
		--out_dir $(DATA_DIR) \
		--n_sequences $(N_SEQ) \
		--seq_len $(SEQ_LEN) \
		--L $(BIN_L) --W $(BIN_W) --H $(BIN_H) \
		--lookahead_k $(LOOKAHEAD) \
		--delta_cog $(DELTA_COG) \
		--seed $(SEED)

gen-same-height: venv
	$(ACTIVATE) && $(PY) scripts/generate_dataset/generate_dataset_same_height.py \
		--out_dir $(DATA_DIR) \
		--n_sequences $(N_SEQ) \
		--seq_len $(SEQ_LEN) \
		--L $(BIN_L) --W $(BIN_W) --H $(BIN_H) \
		--same_height $(SAME_HEIGHT) \
		--lookahead_k $(LOOKAHEAD) \
		--delta_cog $(DELTA_COG) \
		--seed $(SEED)

gen-fill100: venv
	$(ACTIVATE) && $(PY) scripts/generate_dataset/generate_dataset_fill100_2d_to_3d.py \
		--out_dir $(DATA_DIR) \
		--n_sequences 500 \
		--grid_W $(GRID_W) \
		--grid_H $(GRID_H) \
		--target_rects $(TARGET_RECTS) \
		--bin_H $(BIN_H) \
		--height_mode random \
		--same_height $(SAME_HEIGHT) \
		--lookahead_k $(LOOKAHEAD) \
		--delta_cog $(DELTA_COG) \
		--seed $(SEED)

gen-semi: venv
	$(ACTIVATE) && $(PY) scripts/generate_dataset/generate_dataset_semi_online.py \
		--out_dir $(DATA_DIR) \
		--n_sequences $(N_SEQ) \
		--seq_len $(SEQ_LEN) \
		--L $(BIN_L) --W $(BIN_W) --H $(BIN_H) \
		--accessible_k 5 \
		--known_total 30 \
		--delta_cog $(DELTA_COG) \
		--seed $(SEED)

# ====== PREVIEW ======
preview: venv
ifndef DATA
	$(error DATA is not set. Example: make preview DATA=data/synthetic/random3d/train.jsonl)
endif
	$(ACTIVATE) && $(PY) scripts/generate_dataset/preview_dataset.py \
		--path $(DATA) \
		--plot \
		--save_report \
		--out_dir $(PREVIEW_OUT)
	@mkdir -p $(PREVIEW_OUT)

preview-all: venv
	@echo "🔎 Preview random3d/train.jsonl"
	$(MAKE) preview DATA=$(DATA_DIR)/random3d/train.jsonl PREVIEW_OUT=$(PREVIEW_OUT)/random3d
	@echo "🔎 Preview same-height/train.jsonl"
	$(MAKE) preview DATA=$(DATA_DIR)/same_height_h012/train.jsonl PREVIEW_OUT=$(PREVIEW_OUT)/same_height_h012
	@echo "🔎 Preview fill100/train.jsonl"
	$(MAKE) preview DATA=$(DATA_DIR)/fill100_2d_to_3d/train.jsonl PREVIEW_OUT=$(PREVIEW_OUT)/fill100_2d_to_3d
	@echo "🔎 Preview semi-online/train.jsonl"
	$(MAKE) preview DATA=$(DATA_DIR)/semi_online/train.jsonl PREVIEW_OUT=$(PREVIEW_OUT)/semi_online
	@echo "✅ Preview all done."

# ====== CLEANUP ======
clean-data:
	rm -rf $(DATA_DIR)/random3d \
	       $(DATA_DIR)/same_height_h012 \
	       $(DATA_DIR)/fill100_2d_to_3d \
	       $(DATA_DIR)/semi_online
	@echo "🧹 Dataset folder cleaned."

clean-preview:
	rm -rf $(PREVIEW_OUT)
	@echo "🧹 Preview results cleaned."

clean-venv:
	rm -rf $(VENV_DIR)
	@echo "🧹 venv deleted."

clean: clean-data clean-preview
	@echo "🧹 All cleaned (except venv)."
