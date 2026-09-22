# PI-SASR: reproduction targets.  Run `make help` for a summary.
#
# Variables (override on the command line, e.g. `make reproduce-tx123bt OUT=/scratch/pisasr`):
#   PYTHON       Python interpreter of the environment        (default: python)
#   OUT          top-level folder for reruns                    (default: outputs)
#   S_LIST       scenario counts of the scaling study           (default: 100 200 500 1000 2500)
#   SUMMARY_FROM results folder summarized by the experiments targets (default: $(OUT))
#   ASSETS_FROM  results folder used to build paper_assets/       (default: results)

PYTHON       ?= python
OUT          ?= outputs
S_LIST       ?= 100 200 500 1000 2500
SUMMARY_FROM ?= $(OUT)
ASSETS_FROM  ?= results
RUNS          = deterministic $(addprefix full_S,$(S_LIST)) $(addprefix pisasr_S,$(S_LIST))

.PHONY: help install test test-fast check-s20 data-check data-tx123bt data-texas2k \
        reproduce-tx123bt reproduce-texas2k experiments-tx123bt experiments-texas2k \
        merit-order certified-gap assets clean

help:
	@echo "install              editable install of the package with the test extra"
	@echo "test                 all tests except those marked slow (needs Gurobi; about 4 min)"
	@echo "test-fast            tests that need no solver (about 40 s)"
	@echo "check-s20            short end-to-end check: full model and PI-SASR on TX-123BT at S=20 (about 40 s)"
	@echo "data-check           compare data/ with data/SHA256SUMS"
	@echo "data-tx123bt         download TX-123BT and regenerate its inputs (checked)"
	@echo "data-texas2k         regenerate the Texas2k inputs from data/raw/texas2k (checked)"
	@echo "reproduce-tx123bt    deterministic UC, full model, PI-SASR and evaluation (about 2 h)"
	@echo "reproduce-texas2k    the same for Texas2k (about 13 h)"
	@echo "experiments-tx123bt  additional experiments, merit-order validation and summaries (about 6 h)"
	@echo "experiments-texas2k  the same for Texas2k (about 15 h)"
	@echo "merit-order          merit-order check of both systems, compared with results/ (about 1 min)"
	@echo "certified-gap        certified gaps from the stored lower-bound batches (no MILP, seconds)"
	@echo "assets               tables, figures and numbers of the paper in paper_assets/"
	@echo "clean                remove $(OUT)/, caches and build files (never results/ or data/)"

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -m "not slow"

test-fast:
	$(PYTHON) -m pytest -m "not gurobi and not slow"

check-s20:
	$(PYTHON) scripts/run_full_model.py --system tx123bt --S 20 --output-dir $(OUT)/check-s20
	$(PYTHON) scripts/run_pisasr.py --system tx123bt --S 20 --output-dir $(OUT)/check-s20

data-check:
	$(PYTHON) -c "import sys; from pisasr.paths import verify_checksums; sys.exit(0 if verify_checksums() else 1)"

data-tx123bt:
	$(PYTHON) scripts/data/download_tx123bt.py
	$(PYTHON) scripts/data/prepare_tx123bt.py --out-dir $(OUT)/data/tx123bt --check

data-texas2k:
	$(PYTHON) scripts/data/prepare_texas2k.py --out-dir $(OUT)/data/texas2k --check

reproduce-tx123bt:
	$(PYTHON) scripts/run_full_model.py --system tx123bt --deterministic --S $(S_LIST) --output-dir $(OUT)
	$(PYTHON) scripts/run_pisasr.py --system tx123bt --S $(S_LIST) --output-dir $(OUT)
	$(PYTHON) scripts/evaluate.py --system tx123bt --runs $(RUNS) --output-dir $(OUT)

reproduce-texas2k:
	$(PYTHON) scripts/run_full_model.py --system texas2k --deterministic --S $(S_LIST) --output-dir $(OUT)
	$(PYTHON) scripts/run_pisasr.py --system texas2k --S $(S_LIST) --output-dir $(OUT)
	$(PYTHON) scripts/evaluate.py --system texas2k --runs $(RUNS) --output-dir $(OUT)

experiments-tx123bt:
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage component-timing --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage criticality-only --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage gp-kernels --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage embedding --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage scenario-weights --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage weights-s2500 --S 2500 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage lower-bound --S 200 500 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage budget-sensitivity --S 200 500 1000 2500 --output-dir $(OUT)
	$(PYTHON) scripts/run_replication.py --system tx123bt --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/validate_merit_order.py --system tx123bt --output-dir $(OUT)
	$(PYTHON) scripts/summarize_evaluation.py --system tx123bt --results-dir $(SUMMARY_FROM)
	$(PYTHON) scripts/paired_cost_difference.py --system tx123bt --results-dir $(SUMMARY_FROM)

experiments-texas2k:
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage criticality-only --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage gp-kernels --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage embedding --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage scenario-weights --S 1000 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage weights-s2500 --S 2500 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage lower-bound --S 200 --output-dir $(OUT)
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage budget-sensitivity --S 500 1000 2500 --output-dir $(OUT)
	$(PYTHON) scripts/validate_merit_order.py --system texas2k --output-dir $(OUT)
	$(PYTHON) scripts/summarize_evaluation.py --system texas2k --results-dir $(SUMMARY_FROM)
	$(PYTHON) scripts/paired_cost_difference.py --system texas2k --results-dir $(SUMMARY_FROM)

merit-order:
	$(PYTHON) scripts/validate_merit_order.py --system tx123bt --output-dir $(OUT) --check
	$(PYTHON) scripts/validate_merit_order.py --system texas2k --output-dir $(OUT) --check

certified-gap:
	$(PYTHON) scripts/run_experiments.py --system tx123bt --stage lower-bound --output-dir $(OUT) \
		--batches-file results/tx123bt/experiments/lower_bound_batches.csv
	$(PYTHON) scripts/run_experiments.py --system texas2k --stage lower-bound --output-dir $(OUT) \
		--batches-file results/texas2k/experiments/lower_bound_batches.csv

assets:
	$(PYTHON) scripts/make_paper_assets.py --results-dir $(ASSETS_FROM) --out paper_assets

clean:
	@for d in $(abspath . results data src scripts tests docs paper_assets); do \
		if [ "$(abspath $(OUT))" = "$$d" ]; then \
			echo "clean: OUT=$(OUT) is a folder of the repository; nothing removed"; exit 1; \
		fi; \
	done
	rm -rf $(OUT) .pytest_cache build dist src/*.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name ".DS_Store" -delete
	find . -name "*.lp" -not -path "./data/*" -delete
