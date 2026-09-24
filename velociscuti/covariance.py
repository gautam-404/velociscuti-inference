import json
import os

import numpy as np

from .utils import PARAM_NAMES, load_modes, paper_priors, prior_limits

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_DIR = os.path.join(os.path.dirname(PACKAGE_DIR), 'calibration')
if not os.path.isdir(CALIBRATION_DIR):
    ## an install that is not editable copies only the package and leaves calibration/ in the
    ## repo, so look for it in the folder python was started from, as the sampler does for stars/
    CALIBRATION_DIR = os.path.join(os.getcwd(), 'calibration')
SIGMA_MODEL_PATH = os.path.join(PACKAGE_DIR, 'model', 'sigma_model.npy')


def load_box():
    """Held-out residuals inside the prior box of the Pleiades paper (about 9 MB, always present)."""
    with np.load(os.path.join(CALIBRATION_DIR, 'pleiades_box.npz'), allow_pickle=False) as box:
        return box['inputs'], [box['residuals']], [str(col) for col in box['output_cols']]


def load_full():
    """The full held-out set (about 2 GB, kept in Git LFS), memory-mapped so that only the needed rows are read."""
    full_dir = os.path.join(CALIBRATION_DIR, 'full')
    with open(os.path.join(full_dir, 'columns.json'), 'r') as fp:
        columns = json.load(fp)

    paths = [os.path.join(full_dir, 'inputs.npy')] + [os.path.join(full_dir, chunk['file']) for chunk in columns['chunks']]
    for path in paths:
        ## a clone without the LFS files holds small text pointers in their place
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            raise FileNotFoundError(
                f'[build_covariance] {path} has not been downloaded. These priors are wider than the '
                'Pleiades box, so the full calibration set is needed. Fetch it (about 2 GB) with:\n'
                '    git lfs pull --include="calibration/full" --exclude=""'
            )

    inputs = np.load(paths[0], mmap_mode='r')
    residual_chunks = [np.load(path, mmap_mode='r') for path in paths[1:]]
    return inputs, residual_chunks, list(columns['output_cols'])


def load_calibration(priors):
    limits = np.array([prior_limits(prior) for prior in priors], dtype=np.float64)
    with np.load(os.path.join(CALIBRATION_DIR, 'pleiades_box.npz'), allow_pickle=False) as box:
        inside_box = np.all(limits[:, 0] >= box['box_min']) and np.all(limits[:, 1] <= box['box_max'])

    if inside_box:
        print('[build_covariance] priors fit inside the Pleiades box, using calibration/pleiades_box.npz')
        return load_box()

    print('[build_covariance] priors are wider than the Pleiades box, using calibration/full')
    return load_full()


def take_rows(residual_chunks, rows, col_idxs):
    ## rows come in the random order of the draw. each chunk is read in file order, which is
    ## much faster on a memory map, and the rows are then put back in draw order
    out = np.empty((len(rows), len(col_idxs)), dtype=np.float64)
    start = 0
    for chunk in residual_chunks:
        stop = start + chunk.shape[0]
        here = np.flatnonzero((rows >= start) & (rows < stop))
        order = np.argsort(rows[here])
        out[here[order]] = np.asarray(chunk[rows[here][order] - start][:, col_idxs], dtype=np.float64)
        start = stop
    return out


def build_covariance(star_name, priors=None, save_as=None, n_samples=100000, seed=42, shrinkage=0.1):
    """Build the covariance for stars/<star_name> and save it next to the mode table.

    C_nu = C_obs + C_emu. C_emu is the covariance of held-out emulator residuals (grid model minus
    emulator) over the prior box, weighted by the prior density, shrunk towards its diagonal, and
    rescaled so that its diagonal holds the squared emulator errors in sigma_model.npy. Use the same priors here and in the
    sampler, because C_emu describes the emulator over the region those priors allow.
    """
    if not priors:
        priors = paper_priors()
    if not save_as:
        save_as = f'{star_name}_covariance.npz'

    with open(f'stars/{star_name}/{star_name}.json', 'r') as fp:
        star_dict = json.load(fp)
    teff_unc = star_dict['teff'][1]

    mode_cols, _, freq_errs = load_modes(star_name)
    n_modes = len(mode_cols)

    inputs, residual_chunks, output_cols = load_calibration(priors)

    missing = [col for col in mode_cols if col not in output_cols]
    if missing:
        raise ValueError(f'[build_covariance] the emulator does not predict {missing}.')
    col_idxs = [output_cols.index(col) for col in mode_cols]

    ## from here to the nugget the arithmetic follows the estimator of the paper line by line.
    ## do not tidy it: rebuilt matrices are compared with the ones the paper used

    # -----------------------
    # rows inside the prior box
    # -----------------------
    mask = np.ones(inputs.shape[0], dtype=bool)
    for j, (name, prior) in enumerate(zip(PARAM_NAMES, priors)):
        values = np.asarray(inputs[:, j], dtype=np.float64)
        lo, hi = prior_limits(prior)
        if name in ('m', 'z'):
            ## mass and metallicity are grid nodes stored in float32, so 1.4 reads as 1.39999998
            values = np.round(values, 5)
            lo, hi = np.round([lo, hi], 5)
        mask &= (values >= lo) & (values <= hi)
    candidates = np.flatnonzero(mask)

    ## a covariance of n modes has about n^2/2 free entries, so ask for at least n^2 rows
    min_rows = n_modes ** 2
    if candidates.size < min_rows:
        raise RuntimeError(f'[build_covariance] only {candidates.size} calibration rows lie inside the priors, '
                           f'and {n_modes} modes need at least {min_rows}. Widen the priors.')

    rng = np.random.default_rng(seed)
    draw = rng.choice(candidates, size=min(int(n_samples), candidates.size), replace=False)
    selected_inputs = np.asarray(inputs[draw], dtype=np.float64)
    residuals = take_rows(residual_chunks, draw, col_idxs)

    ## a grid model that lacks one of the modes has NaN there, and the whole row is dropped
    finite = np.all(np.isfinite(residuals), axis=1)
    residuals = residuals[finite]
    selected_inputs = selected_inputs[finite]
    if residuals.shape[0] < min_rows:
        raise RuntimeError(f'[build_covariance] only {residuals.shape[0]} calibration rows have all {n_modes} modes, '
                           f'and at least {min_rows} are needed.')

    # -----------------------
    # prior-density weights
    # -----------------------
    weights = np.ones(residuals.shape[0], dtype=np.float64)
    for j, (name, prior) in enumerate(zip(PARAM_NAMES, priors)):
        values = selected_inputs[:, j]
        if name in ('m', 'z'):
            ## weigh a row where the mask found it, at the rounded node. the stored 2.5 node
            ## reads 2.50000008, where a prior ending at 2.5 is zero, and a limit such as
            ## 2.499996 rounds to admit that node too, hence the clip into the prior's limits
            lo, hi = prior_limits(prior)
            values = np.clip(np.round(values, 5), lo, hi)
        weights *= prior.pdf(values)
    total = float(weights.sum())
    if total <= 0.0 or not np.all(np.isfinite(weights)):
        raise RuntimeError('[build_covariance] the prior density is zero or undefined on every calibration row.')
    weights = weights / total

    effective_n = float(1.0 / np.sum(weights**2))
    if effective_n < min_rows:
        raise RuntimeError(f'[build_covariance] the priors leave an effective {effective_n:.0f} calibration rows, '
                           f'and at least {min_rows} are needed. Widen the priors.')

    # -----------------------
    # weighted covariance, shrinkage, diagonal from sigma_model
    # -----------------------
    mean = weights @ residuals
    centered = residuals - mean
    denominator = 1.0 - float(np.sum(weights**2))
    raw = (centered * weights[:, None]).T @ centered / denominator
    C_emu = (1.0 - shrinkage) * raw + shrinkage * np.diag(np.diag(raw))

    ## correlations come from the prior box, the size of the error from the whole held-out set
    sigma_model = np.load(SIGMA_MODEL_PATH, allow_pickle=False)
    sigma = np.asarray(sigma_model[col_idxs], dtype=np.float64)
    local_std = np.sqrt(np.diag(C_emu))
    correlation = C_emu / np.outer(local_std, local_std)
    C_emu = np.outer(sigma, sigma) * correlation

    ## a small nugget keeps the matrix safely positive definite
    nugget = 1.0e-8 * float(np.median(np.diag(C_emu)))
    C_emu += nugget * np.eye(n_modes)

    C_obs = np.diag(freq_errs**2)
    C_nu = C_obs + C_emu

    # -----------------------
    # teff block
    # -----------------------
    teff_var = float(sigma_model[output_cols.index('teff')]) ** 2
    C_classical = np.asarray([[float(teff_unc) ** 2]]) + np.asarray([[teff_var + 1.0e-8 * teff_var]])

    save_path = f'stars/{star_name}/{save_as}'
    np.savez_compressed(save_path, mode_cols=np.asarray(mode_cols, dtype=str),
                        C_obs=C_obs, C_emu=C_emu, C_nu=C_nu, C_classical=C_classical)

    print(f'[build_covariance] {candidates.size} calibration rows inside the priors, '
          f'{residuals.shape[0]} used, {effective_n:.0f} effective')
    print(f'[build_covariance] saved {save_path}')

    return {'mode_cols': mode_cols, 'C_obs': C_obs, 'C_emu': C_emu, 'C_nu': C_nu, 'C_classical': C_classical,
            'n_rows_in_prior_box': int(candidates.size), 'n_rows_used': int(residuals.shape[0]),
            'effective_n': effective_n}
