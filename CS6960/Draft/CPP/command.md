cd ~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Draft/CPP && \
rm -f document.aux document.bbl document.blg document.log document.out document.toc document.fls document.fdb_latexmk && \
pdflatex document.tex && bibtex document && pdflatex document.tex && pdflatex document.tex
