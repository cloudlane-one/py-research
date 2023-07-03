# How to generate docs

To generate one consolidated HTML documentation, which includes both API references from docstrings and manual docs in `/docs`, run these commands in your terminal:

```bash
sphinx-apidoc -efM -o ./docs/api/py/ ./src
sphinx-build -b html ./ ./docs-html/
```

The resulting website code will be put into `/docs-html`.

> If you do not have the standalone [Pandoc](https://pandoc.org) binary installed and available on `$PATH`, the docs generation might fail.

You can also generate a PDF version of the documentation via `sphinx-build -M latexpdf ./ ./docs-pdf/`. Yet this will only work if you have [Miktex](https://miktex.org/) and [Latexmk](https://mg.readthedocs.io/latexmk.html) installed.
