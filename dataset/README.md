# California Housing

`cal_housing.tgz` is the California Housing dataset: 20,640 census block groups from
the 1990 US census, with 8 features and the median house value. It's the archive
scikit-learn's `fetch_california_housing` downloads, byte for byte. Its SHA-256 is
`aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`, which
scikit-learn and `task/dataset.py` both check.

**Why it's here.** scikit-learn downloads the archive from figshare, which refuses
requests from some cloud networks (HTTP 403 from some AWS regions). So `train.py`
copies this file to the Hugging Face repository's `data` branch, and training nodes
load it from there.

**Source.** R. Kelley Pace and Ronald Barry, "Sparse Spatial Autoregressions",
*Statistics and Probability Letters*, 33 (1997), 291–297. It's distributed by the
StatLib archive at Carnegie Mellon University.
