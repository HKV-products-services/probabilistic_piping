# Probabilistic piping

This package contains core functionality for probabilistic piping calculations, utilizing [OpenTURNS](http://openturns.github.io), [pandas](https://pandas.pydata.org), [numpy](https://numpy.org) and [tqdm](https://tqdm.github.io). Probabilistic piping is developed by HKV and is published under the GNU GPL-3 license.

Timedep pipe growth uses a cached Numba kernel. Its first calculation can take
longer while the numerical loop compiles; later calls reuse the compiled code.
Use `pipe_growth(settings)` to inspect growth for the series held in the
settings. It returns a `PipeGrowth` with one pipe-length matrix shaped
`(time points, samples)`, including constant columns for samples without erosion.
