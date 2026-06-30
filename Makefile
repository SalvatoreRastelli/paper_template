PDFS = paper/merw_ucb.pdf paper/merw_fault_tolerant.pdf
FIGURE_STAMP = paper/results/.generated

# Default target: build the PDFs from the figures, which are rendered from
# the *committed* paper/data/*.csv. This never re-runs the Monte Carlo
# experiments, so it is safe and fast to run in CI.
all: $(PDFS)

paper/merw_ucb.pdf: paper/merw_ucb.tex $(FIGURE_STAMP)
	cd paper && latexmk -pdf -interaction=nonstopmode merw_ucb.tex

paper/merw_fault_tolerant.pdf: paper/merw_fault_tolerant.tex
	cd paper && latexmk -pdf -interaction=nonstopmode merw_fault_tolerant.tex

# Renders figures from paper/data/*.csv (committed to the repo). Cheap;
# this is the step CI runs as part of `make all`.
$(FIGURE_STAMP): paper/data
	uv run python scripts/generate_figures.py --mode plot
	touch $(FIGURE_STAMP)

# Local-only, expensive: runs the Monte Carlo experiments / eigenvector
# computations and (re)writes paper/data/*.csv. NOT run by CI -- run this
# locally and commit the resulting CSVs whenever the experiments change.
data:
	uv run python scripts/generate_figures.py --mode compute

clean:
	cd paper && latexmk -C
	rm -f $(FIGURE_STAMP)

clean-data:
	rm -rf paper/data

.PHONY: all data clean clean-data
