# Velociscuti: further information

## Inputs and trained range

The emulator takes four inputs, always in this order:

| input | meaning | trained range |
|---|---|---|
| `m` | initial mass ($M_\odot$) | 1.4 to 2.5 |
| `z` | initial metal mass fraction | 0.001 to 0.026 |
| `Myr` | age (Myr) | 1 to 2376 |
| `v_eq` | equatorial rotation velocity at that age (km/s) | 0 to 316 |

`predict` raises an error for inputs outside this box, because a prediction there would be an extrapolation. Passing this check does not guarantee that a point lies inside the grid, because the grid is not a box: its tracks run from the early pre-main sequence to the end of the contraction that follows the main sequence, so the oldest ages exist only for the lowest masses. Keep the age prior inside the ages that the grid covers for the masses you allow.

## Outputs

`predict` returns 232 columns, named in `emulator.output_cols`. Frequencies are in cycles per day, in the observer's (inertial) frame.

| columns | meaning |
|---|---|
| `teff` | effective temperature (K) |
| `log_L`, `log_R`, `log_g` | $\log_{10}$ of luminosity ($L_\odot$), radius ($R_\odot$) and surface gravity (cgs) |
| `density` | mean density in units of the solar mean density |
| `center_h1` | central hydrogen mass fraction |
| `omega_c`, `w_ov_w_c` | critical rotation frequency (c/d) and the ratio $\Omega/\Omega_{\rm crit}$ |
| `R_eq`, `R_polar` | equatorial and polar radius ($R_\odot$) |
| `pp`, `cno` | $\log_{10}$ of the luminosity from the pp chains and from the CNO cycle ($L_\odot$) |
| `Dnu`, `eps` | large frequency separation (c/d) and the phase term $\epsilon$ |
| `n{n}ell{l}m{m}` | p modes, $n$ = 1 to 11, $\ell$ = 0 to 3, all $m$ (176 columns) |
| `ng0ell{l}m{m}` | f modes, $\ell$ = 2 and 3, all $m$ (12 columns) |
| `ng{n}ell{l}m{m}` | g modes, $n$ = 1 and 2, $\ell$ = 1 to 3, all $m$ (30 columns) |

Positive $m$ is prograde.

## The grid

The stellar models were computed with MESA (r24.03.1) and their pulsation frequencies with StORM, which treats the Coriolis force directly and includes the centrifugal deformation to second order. The emulator was trained on approximately 46 million of these models. The paper describes the grid and the training.

## The neural network

The four inputs pass through five shared blocks of 1024, 768, 512, 384 and 384 neurons. Each block multiplies its input by a learned weight matrix, adds a learned bias and applies a GELU activation, and a shortcut adds the block's input to the result (through a learned linear map where the number of neurons changes). Three separate output branches follow, one for the 14 non-frequency outputs, one for the 176 p modes and one for the 42 f and g modes, each with one hidden layer of 256 neurons. The network has 3,269,224 parameters.

Inputs and outputs are transformed before and after the network with Yeo-Johnson power transforms and linear scalings. The fitted constants are stored in `velociscuti.json` and the weights in `velociscuti_weights.npz`, both as plain numbers, so loading the emulator does not unpickle anything.

## Emulator error

`sigma_model.npy` holds the standard deviation of the emulator's error on grid models that were held out of training (the recalibrated values used in the paper).

| outputs | median | smallest | largest |
|---|---|---|---|
| p modes (c/d) | 0.057 | 0.020 | 0.094 |
| f modes (c/d) | 0.024 | 0.019 | 0.029 |
| g modes (c/d) | 0.028 | 0.019 | 0.033 |
| `teff` (K) | 11.3 | | |

These standard deviations are an order of magnitude larger than the mean absolute error of the same outputs (for the p modes, 0.004 c/d). The two differ because the errors are heavy-tailed: a small minority of models, almost all in the final pre-main-sequence contraction, have much larger errors than the rest and dominate the variance.

The errors of different modes are correlated. `build_covariance` estimates those correlations for the modes of one star from the held-out models inside that star's priors, and rescales the matrix so that its diagonal holds the squares of the values above. Notebook 2 shows the steps.

## Calibration data

`calibration/` holds the held-out residuals (grid model minus emulator) that `build_covariance` needs. `pleiades_box.npz` contains the held-out models inside the priors of the paper and is enough for any priors within them. `calibration/full/` contains all 2,259,106 held-out models, is about 2 GB, and is kept in Git LFS. A normal clone does not download it. Fetch it with

    git lfs pull --include="calibration/full" --exclude=""
