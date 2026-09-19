# Velociscuti
## Ages of young $\delta$ Scuti stars from their pulsation frequencies, with a neural network emulator of rotating stellar models

Asteroseismology of $\delta$ Scuti stars has strong potential for determining the ages of young stars through individual mode fitting. `velociscuti` is a neural network that emulates a grid of rotating stellar models (MESA) and their pulsation frequencies (StORM). Given a mass, a metallicity, an age and a rotation velocity, it returns 14 stellar and seismic quantities (such as $T_{\rm eff}$ and the large frequency separation) and 218 pulsation frequencies.

This repo holds the trained emulator and the inference code of the paper below. The inference code samples the posterior of a star's mass, metallicity, age and rotation.

Notebooks:
- [`paper-stars.ipynb`](paper-stars.ipynb) fits the five Pleiades stars of the paper and combines their ages into an age for the cluster
- [`new-star.ipynb`](new-star.ipynb) goes through everything needed for a star of your own: the mode table, the priors, the covariance matrix and the fit

The emulator covers only the range of its training grid: initial masses of 1.4 to 2.5 $M_\odot$, $Z$ of 0.001 to 0.026, and evolution from the early pre-main sequence to the end of the contraction that follows the main sequence. There is more detail in [`velociscuti_info.md`](velociscuti/model/velociscuti_info.md).

`velociscuti` is published with the associated paper:
- Gautam & Murphy, **An asteroseismic age estimate for the Pleiades using a neural-network emulator of rotating $\delta$ Scuti pulsation models** (link to be added when the paper is out)

## Getting started:
```
git clone https://github.com/gautam-404/velociscuti-inference
cd velociscuti-inference
pip install -e .
```
Run the notebooks from the root of the repo, because the star folders are read from [`stars/`](stars).

## Licence:
MIT, see [`LICENSE`](LICENSE). It covers the code, the trained emulator, the mode tables and the calibration data.
