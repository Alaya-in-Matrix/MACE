mainfile = main
PDF      = ${mainfile}.pdf

all : ${PDF}

${mainfile}.pdf: *.tex ref.bib
	pdflatex ${mainfile}.tex
	- bibtex ${mainfile}
	pdflatex ${mainfile}.tex
	pdflatex ${mainfile}.tex

tidy:
	rm  -vf *.log *.aux \
	*.cfg *.glo *.idx *.toc \
	*.ilg *.ind *.out *.lof \
	*.lot *.bbl *.blg *.gls *.cut *.hd \
	*.dvi *.ps *.thm *.tgz *.zip *.rpi


clean: tidy
	rm -f ${PDF}
