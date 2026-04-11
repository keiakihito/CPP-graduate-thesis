# ISMIR
cd ~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Draft/ISMIR && rm -f ISMIR2026_template.aux ISMIR2026_template.bbl ISMIR2026_template.blg ISMIR2026_template.log ISMIR2026_template.out && pdflatex -interaction=nonstopmode -halt-on-error ISMIR2026_template.tex && bibtex ISMIR2026_template && pdflatex -interaction=nonstopmode -halt-on-error ISMIR2026_template.tex && pdflatex -interaction=nonstopmode -halt-on-error ISMIR2026_template.tex

# CPP Thesis
cd ~/Dropbox/Academic/CalPolyPomona/Thesis/CPP-graduate-thesis/CS6960/Draft/CPP && rm -f document.aux document.bbl document.blg document.log document.out document.toc && pdflatex document.tex && bibtex document && pdflatex document.tex && pdflatex document.tex

