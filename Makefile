LATEXMK = latexmk

all: paper/paper.pdf

paper/paper.pdf: $(wildcard paper/*.tex) $(wildcard paper/*.bib)
	cd paper && $(LATEXMK) -pdf paper.tex

clean:
	cd paper && $(LATEXMK) -C

.PHONY: all clean
