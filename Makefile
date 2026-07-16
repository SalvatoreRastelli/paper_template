PDFS = paper/aaai/main.pdf paper/arxiv/main.pdf
FIGURE_STAMP = paper/results/.generated

# Default target: build the PDFs from the figures, which are rendered from
# the *committed* paper/data/*.csv. This never re-runs the Monte Carlo
# experiments, so it is safe and fast to run in CI.
all: $(PDFS)

# paper/aaai/ and paper/arxiv/ each keep a self-contained copy of the
# figures (required for arXiv submission, and convenient for the AAAI
# submission bundle too). rsync -a syncs from the shared paper/results/,
# only touching files that changed, and is portable across macOS (BSD
# cp has no -u flag) and Linux CI runners alike.
paper/aaai/results: $(FIGURE_STAMP)
	mkdir -p paper/aaai/results
	rsync -a paper/results/ paper/aaai/results/

paper/arxiv/results: $(FIGURE_STAMP)
	mkdir -p paper/arxiv/results
	rsync -a paper/results/ paper/arxiv/results/

aaai: paper/aaai/main.tex paper/aaai/results
	cd paper/aaai && latexmk -pdf -interaction=nonstopmode main.tex

paper/arxiv/main.pdf: paper/arxiv/main.tex paper/arxiv/results
	cd paper/arxiv && latexmk -pdf -interaction=nonstopmode main.tex

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
	cd paper/aaai && latexmk -C
	cd paper/arxiv && latexmk -C
	rm -f $(FIGURE_STAMP)

clean-data:
	rm -rf paper/data

.PHONY: all data clean clean-data
