# Probabilistic piping

This package contains core functionality for probabilistic piping calculations, utilizing [OpenTURNS](http://openturns.github.io), [pandas](https://pandas.pydata.org), [numpy](https://numpy.org) and [tqdm](https://tqdm.github.io). Probabilistic piping is developed by HKV and is published under the GNU GPL-3 license.


## Installing the package

In case you just want to use the package, simply install the package from pypi:

run `pip install probabilistic_piping`

## Setting up a development environment

In case you want to actively contribute to the source code, you need to set up a development environment. This can be done by installing [Pixi](https://pixi.sh/latest/).

Once Pixi is installed, checkout the source code from the repository and run the  `pixi` command from the root of the source code:

```bash
  pixi install
```

Python support is declared once in `project.requires-python`. Runtime Python
dependencies live in `project.dependencies`, and development tools live in
`dependency-groups.dev`. Pixi installs these from PyPI and locks their versions.
It provisions the Python interpreter and Quarto through conda-forge. Supported
development platforms are Windows x86-64, Linux x86-64 and macOS ARM64 (14+).

Run `pixi run pre-commit` to check all tracked files. The hooks use the tools
from the locked Pixi development environment. Run `pixi run install` to install
the Git hook so these checks also run when committing.
