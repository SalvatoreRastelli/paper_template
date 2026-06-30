PDF = paper/paper.pdf
STAMP = paper/figures/.generated

all: $(PDF)

$(PDF): paper/paper.tex $(STAMP)
	cd paper && latexmk -pdf -interaction=nonstopmode paper.tex

$(STAMP):
	uv run python scripts/generate_figures.py
	touch $(STAMP)

clean:
	cd paper && latexmk -C
	rm -f $(STAMP)

.PHONY: all clean
