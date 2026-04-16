 # Paper
 rm -f RecSys.aux RecSys.out RecSys.log RecSys.fls RecSys.fdb_latexmk && pdflatex -interaction=nonstopmode -halt-on-error RecSys.tex && pdflatex -interaction=nonstopmode -halt-on-error RecSys.tex


 # Presentation

 marp presentation.md --pdf -o presentation.pdf 