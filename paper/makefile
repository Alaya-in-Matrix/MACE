mainfile   = paper
suppfile   = supp
posterfile = ICML_poster
PDF        = ${mainfile}.pdf
SUPP       = ${suppfile}.pdf
POSTER     = ${posterfile}.pdf

all : ${PDF} ${SUPP} ${POSTER}

${posterfile}.pdf: ${posterfile}.tex baposter.cls
	pdflatex $<

${mainfile}.pdf: *.tex ref.bib
	pdflatex ${mainfile}.tex
	- bibtex ${mainfile}
	pdflatex ${mainfile}.tex
	pdflatex ${mainfile}.tex

${suppfile}.pdf: *.tex ref.bib
	pdflatex ${suppfile}.tex
	- bibtex ${suppfile}
	pdflatex ${suppfile}.tex
	pdflatex ${suppfile}.tex

tidy:
	rm  -vf *.log *.aux \
	*.cfg *.glo *.idx *.toc \
	*.ilg *.ind *.out *.lof \
	*.lot *.bbl *.blg *.gls *.cut *.hd \
	*.dvi *.ps *.thm *.tgz *.zip *.rpi


clean: tidy
	rm -f ${PDF}
	rm -f ${SUPP}
	rm -f ${POSTER}
