```
if need,
rm -f ISMIR2026_template.aux ISMIR2026_template.log ISMIR2026_template.out
```

pdflatex -interaction=nonstopmode -halt-on-error ISMIR2026_template.tex && bibtex ISMIR2026_template && pdflatex -interaction=nonstopmode -halt-on-error ISMIR2026_template.tex && pdflatex -interaction=nonstopmode -halt-on-error ISMIR2026_template.tex
