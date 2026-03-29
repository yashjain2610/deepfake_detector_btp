IEEE Paper (LaTeX) — How to Use

This folder contains an IEEE-conference style paper in LaTeX.

Files
  - main.tex    : IEEEtran conference manuscript
  - refs.bib    : BibTeX references
  - paper.txt   : Plain-text backup version (same content)

How to build a PDF
  Option A (recommended): Overleaf
    1) Create a new Overleaf project
    2) Upload main.tex and refs.bib
    3) Set compiler to pdfLaTeX
    4) Compile

  Option B: Local LaTeX (if you install TeX Live / MiKTeX)
    pdflatex main.tex
    bibtex main
    pdflatex main.tex
    pdflatex main.tex

Notes
  - The paper includes your latest evaluation numbers from results/20260315_145241.
  - It explicitly labels FaceForensics++ “validation” as an internal split, matching your repo’s split method.
  - If you later run more experiments (DFDC, compression robustness), you can update Tables/Results sections.

