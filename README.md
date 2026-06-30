# LaTeX Paper Template with GitHub Actions

A minimal template for writing LaTeX papers with:

- local editing
- Python-generated figures managed with `uv`
- reproducible builds via `Make`
- automatic PDF generation on GitHub Actions
- no generated PDFs committed to the repository

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── paper.yml
├── paper/
│   ├── figures/
│   ├── paper.tex
│   └── references.bib
├── scripts/
│   └── generate_figures.py
├── pyproject.toml
├── uv.lock
├── Makefile
└── README.md
```

The intended responsibilities are:

- `paper/`
  - contains all LaTeX sources
  - generated figures should end up in `paper/figures/`
- `scripts/`
  - contains Python scripts for generating figures
- `Makefile`
  - defines the complete build process
- `.github/workflows/paper.yml`
  - executes the build on GitHub Actions and uploads the resulting PDF

---

## Local setup

Install:

- Python
- `uv`
- a LaTeX distribution with `latexmk`

Create the environment:

```bash
uv sync
```

---

## Building the paper

Simply run

```bash
make
```

This will

1. generate all figures
2. compile the paper

The resulting PDF is

```text
paper/paper.pdf
```

---

## Cleaning

Remove LaTeX auxiliary files with

```bash
make clean
```

---

## Figure generation

All figure generation should happen inside

```text
scripts/
```

The main entry point is

```text
scripts/generate_figures.py
```

This script should generate **all** figures required by the paper and write them into

```text
paper/figures/
```

For example,

```python
plt.savefig("paper/figures/accuracy.pdf")
```

The LaTeX document can then simply include

```latex
\includegraphics{figures/accuracy.pdf}
```

---

## Python dependencies

Python dependencies are managed with `uv`.

Add a package with

```bash
uv add numpy
```

or

```bash
uv add matplotlib
```

Development dependencies can be added with

```bash
uv add --dev pytest
```

Commit both

```text
pyproject.toml
uv.lock
```

to version control.

---

## GitHub Actions

Every push to the repository triggers a workflow that

1. checks out the repository
2. installs Python
3. installs `uv`
4. installs project dependencies (`uv sync`)
5. installs LaTeX
6. runs `make`
7. uploads `paper.pdf` as a workflow artifact

The PDF is available from the corresponding workflow run under **Artifacts**.

---

## Build philosophy

The repository follows a simple principle:

> The Makefile is the single source of truth for how the paper is built.

The GitHub Action only prepares the environment and executes

```bash
make
```

This keeps local builds and CI builds identical.

---

## Extending the build

If additional preprocessing steps become necessary (e.g., simulations, table generation, bibliography preprocessing, or data downloads), they should be added to the `Makefile`, not to the GitHub workflow.

This keeps the build process reproducible and avoids duplicating build logic.

---

## Notes

- Do **not** commit generated PDFs.
- Do **not** commit LaTeX auxiliary files (`.aux`, `.log`, `.fls`, etc.).
- Generated figures should be reproducible from the Python scripts in `scripts/`.
- All generated figures should be written into `paper/figures/`.
