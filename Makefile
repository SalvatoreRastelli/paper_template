all:
	cd paper && latexmk -pdf paper.tex

clean:
	cd paper && latexmk -C

.PHONY: all clean
