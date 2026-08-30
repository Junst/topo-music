# Citations resolve only after bibtex has run and pdflatex has been re-run
# twice; a single pdflatex pass reports all 15 as undefined, which is the
# build sequence talking rather than a problem with refs.bib.
#
#   make          build main.pdf the whole way through
#   make clean    remove build products, keep main.pdf
#   make check    report anything still unresolved in the final pass

main.pdf: main.tex refs.bib table1.tex figs/fig1_averaging.pdf figs/fig2_octave_peak.pdf
	latexmk -pdf -interaction=nonstopmode main.tex

.PHONY: manual clean check
manual:
	pdflatex -interaction=nonstopmode main.tex
	bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	latexmk -c

check: main.pdf
	@echo "undefined citations: `grep -c 'Citation .* undefined' main.log || true`"
	@echo "undefined refs:      `grep -c 'Reference .* undefined' main.log || true`"
	@grep -E 'Output written' main.log | tail -1
