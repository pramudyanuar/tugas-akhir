.PHONY: help clean clean-apply clean-no-outputs clean-no-caches clean-full test eval-smoke train-smoke eval-cutting-smoke train-cutting-smoke all-smoke

PYTHON ?= venv/bin/python

help:
	@echo "Available targets:"
	@echo "  make clean            # Dry-run cleanup preview"
	@echo "  make clean-apply      # Apply cleanup"
	@echo "  make clean-no-outputs # Apply cleanup for caches only"
	@echo "  make clean-no-caches  # Apply cleanup for outputs only"
	@echo "  make clean-full       # Apply cleanup and remove empty output dirs"
	@echo "  make test             # Run unit tests"
	@echo "  make eval-smoke       # Run 1-episode evaluation smoke test"
	@echo "  make train-smoke      # Run tiny training smoke test"
	@echo "  make eval-cutting-smoke  # Run 1-episode evaluation on cutting_stock dataset"
	@echo "  make train-cutting-smoke # Run tiny training on cutting_stock dataset"
	@echo "  make all-smoke        # Run all smoke checks (test + eval/train random + eval/train cutting)"

clean:
	@./scripts/clean.sh

clean-apply:
	@./scripts/clean.sh --apply

clean-no-outputs:
	@./scripts/clean.sh --apply --no-outputs

clean-no-caches:
	@./scripts/clean.sh --apply --no-caches

clean-full: clean-apply
	@find logs -type d -empty -delete 2>/dev/null || true
	@find . -maxdepth 1 -type d \( -name visualizations -o -name visualizations_optimized \) -empty -delete 2>/dev/null || true

test:
	@$(PYTHON) -m unittest tests/test_mcts_rearrangement.py

eval-smoke:
	@$(PYTHON) -c "from evaluate import evaluate; evaluate(num_episodes=1, use_mcts=True, output_csv='logs/evaluation/eval_smoke.csv')"

train-smoke:
	@$(PYTHON) -c "from train import train; train(num_epochs=1, n_steps=8, max_items=4, seed=5, device='cpu')"

eval-cutting-smoke:
	@$(PYTHON) -c "from evaluate import evaluate; evaluate(num_episodes=1, use_mcts=True, output_csv='logs/evaluation/eval_cutting_smoke.csv', dataset_type='cutting_stock')"

train-cutting-smoke:
	@$(PYTHON) -c "from train import train; train(num_epochs=1, n_steps=8, max_items=4, seed=5, device='cpu', dataset_type='cutting_stock')"

all-smoke: test eval-smoke train-smoke eval-cutting-smoke train-cutting-smoke
