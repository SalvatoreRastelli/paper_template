PDFS = paper/aaai/main.pdf paper/supplementary/supplementary.pdf
FIGURE_STAMP = results/.generated

# The three figures main.tex includes. Everything else rendered into
# results/ belongs to the supplementary.
MAIN_FIGURES = \
	Regret/relay_regret_er_N100_K5_T5000.pdf \
	BAI/merw_ucb_bai_er_K10.pdf \
	FaultTolerance/fault_tolerance_er_N20_K10_T2000_at500-1000-1500.pdf

# Default target: build the PDFs from the figures, which are rendered from
# the *committed* data/*.csv. This never re-runs the Monte Carlo
# experiments, so it is safe and fast to run in CI.
all: $(PDFS)

# results/ is the staging tree every script writes to. It is split in
# two here so each document carries only what it actually includes: opening
# paper/aaai/results/ shows exactly the figures in the main paper, and nothing
# a supplementary rerun touches can silently change one of them. rsync -a is
# portable across macOS (BSD cp has no -u flag) and Linux CI runners alike.
paper/aaai/results: $(FIGURE_STAMP)
	rm -rf paper/aaai/results
	for f in $(MAIN_FIGURES); do \
		mkdir -p "paper/aaai/results/$$(dirname $$f)"; \
		cp "results/$$f" "paper/aaai/results/$$f"; \
	done

paper/supplementary/results: $(FIGURE_STAMP)
	mkdir -p paper/supplementary/results
	rsync -a $(foreach f,$(MAIN_FIGURES),--exclude '$(f)') \
		results/ paper/supplementary/results/

paper/aaai/main.pdf: paper/aaai/main.tex paper/aaai/results
	cd paper/aaai && latexmk -pdf -interaction=nonstopmode main.tex

paper/supplementary/supplementary.pdf: paper/supplementary/supplementary.tex paper/supplementary/results
	cd paper/supplementary && latexmk -pdf -interaction=nonstopmode supplementary.tex

# Renders figures from data/*.csv (committed to the repo). Cheap;
# this is the step CI runs as part of `make all`.
$(FIGURE_STAMP): $(wildcard data/*.csv)
	uv run python scripts/generate_figures.py --mode plot
	touch $(FIGURE_STAMP)

# Local-only, expensive: runs the Monte Carlo experiments / eigenvector
# computations and (re)writes data/*.csv. NOT run by CI -- run this
# locally and commit the resulting CSVs whenever the experiments change.
# Named `compute`, not `data`: data/ is a real directory now, and a phony
# target sharing its name would always look up to date and never run.
compute:
	uv run python scripts/generate_figures.py --mode compute

clean:
	cd paper/aaai && latexmk -C
	cd paper/supplementary && latexmk -C
	rm -f $(FIGURE_STAMP)

clean-data:
	rm -rf data

.PHONY: all compute clean clean-data
