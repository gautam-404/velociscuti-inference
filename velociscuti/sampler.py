import json
import os
import time

import numpy as np
import ultranest

from .utils import PARAM_NAMES, load_modes, paper_priors, prior_limits, save_results

R_SUN_M = 6.957e8
SECONDS_PER_DAY = 86400.0

## log-likelihood given to parameters the emulator or the rotation limit rules out
FAIL_LOGLIKE = -1.0e100


def check_covariance(name, C, size):
    C = np.asarray(C, dtype=np.float64)
    if C.shape != (size, size):
        raise ValueError(f'[velociscuti_sampler] {name} has shape {C.shape}, but the mode table needs {(size, size)}.')
    if not np.all(np.isfinite(C)) or not np.allclose(C, C.T, atol=1e-12):
        raise ValueError(f'[velociscuti_sampler] {name} must be finite and symmetric.')
    try:
        return np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        raise ValueError(f'[velociscuti_sampler] {name} is not positive definite.') from None


class velociscuti_sampler():
    def __init__(self, emulator, priors=None, omega_ratio_max=0.5):
        self.emulator = emulator
        self.omega_ratio_max = omega_ratio_max

        if not priors:
            self.priors = paper_priors()
        else:
            self.priors = priors

        ## a prior wider than the trained range would stop the run the first time a live point
        ## lands outside it, so check the corners of the prior box now
        limits = np.array([prior_limits(prior) for prior in self.priors], dtype=np.float32)
        try:
            self.emulator.check_range(limits.T)
        except ValueError as error:
            raise ValueError(f'[velociscuti_sampler] the priors reach outside the emulator. {error}') from None

        output_cols = self.emulator.output_cols
        self.teff_idx = output_cols.index('teff')
        self.omega_c_idx = output_cols.index('omega_c')
        self.log_R_idx = output_cols.index('log_R')

    def load_star_data(self, star_name, covariance=None):
        with open(f'stars/{star_name}/{star_name}.json', 'r') as fp:
            star_dict = json.load(fp)
        self.teff = star_dict['teff'][0]

        # -----------------------
        # identified modes
        # -----------------------
        self.mode_cols, self.freqs, _ = load_modes(star_name)

        missing = [col for col in self.mode_cols if col not in self.emulator.output_cols]
        if missing:
            raise ValueError(f'[velociscuti_sampler] the emulator does not predict {missing}. '
                             'See velociscuti/model/velociscuti_info.md for the modes it covers.')
        self.mode_idxs = np.array([self.emulator.output_cols.index(col) for col in self.mode_cols])

        # -----------------------
        # covariance matrix
        # -----------------------
        ## the file holds C_nu = C_obs + C_emu for the frequencies and a separate 1x1 block for teff.
        ## the teff uncertainty in the star file is already inside C_classical, so rebuild the
        ## covariance after changing it
        if not covariance:
            covariance = f'{star_name}_covariance.npz'
        cov_path = f'stars/{star_name}/{covariance}'
        if not os.path.exists(cov_path):
            raise FileNotFoundError(f'[velociscuti_sampler] {cov_path} not found. '
                                    f"For a new star, make it first with build_covariance('{star_name}').")

        with np.load(cov_path, allow_pickle=False) as cov:
            cov_mode_cols = [str(col) for col in cov['mode_cols']]
            C_nu = cov['C_nu']
            C_classical = cov['C_classical']

        if cov_mode_cols != self.mode_cols:
            raise ValueError(f'[velociscuti_sampler] the modes in {cov_path} do not match the mode table, '
                             'in content or in order. Rebuild the covariance after editing the mode table.')

        ###------------- cholesky ----------------------
        self.L_nu = check_covariance('C_nu', C_nu, len(self.mode_cols))
        self.L_classical = check_covariance('C_classical', C_classical, 1)
        self.logdet_nu = 2.0 * float(np.log(np.diag(self.L_nu)).sum())
        self.logdet_classical = 2.0 * float(np.log(np.diag(self.L_classical)).sum())

    def ptform(self, u):
        theta = np.array([self.priors[i].ppf(u[:,i]) for i in range(len(self.priors))]).T
        return theta

    def mvn_logl(self, residual, L, logdet):
        ## do not tidy the arithmetic in this function or in logl: fits are compared with frozen values
        whitened = np.linalg.solve(L, residual.T).T
        quadratic = np.sum(whitened**2, axis=1)
        return -0.5 * (quadratic + logdet + residual.shape[1] * np.log(2.0 * np.pi))

    def logl(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        preds = self.emulator.predict(theta)
        valid = np.all(np.isfinite(preds), axis=1)

        ## rotation limit. v_eq and the predicted radius give the rotation frequency in cycles per
        ## day, which is compared with the critical value the emulator predicts
        radius_m = np.power(10.0, preds[:, self.log_R_idx]) * R_SUN_M
        omega = theta[:, 3] * 1000.0 * SECONDS_PER_DAY / (radius_m * 2.0 * np.pi)
        ratio = np.abs(omega) / np.maximum(preds[:, self.omega_c_idx], 1.0e-100)
        valid &= np.isfinite(ratio) & (ratio <= self.omega_ratio_max)

        frequency_residual = preds[:, self.mode_idxs] - self.freqs
        teff_residual = preds[:, [self.teff_idx]] - self.teff
        valid &= np.all(np.isfinite(frequency_residual), axis=1)
        valid &= np.all(np.isfinite(teff_residual), axis=1)

        ll = np.full(theta.shape[0], FAIL_LOGLIKE, dtype=np.float64)
        good = np.flatnonzero(valid)
        if good.size:
            ll[good] = self.mvn_logl(
                frequency_residual[good], self.L_nu, self.logdet_nu
            ) + self.mvn_logl(teff_residual[good], self.L_classical, self.logdet_classical)
        return ll

    def __call__(self, star_name, covariance=None, save_as=None, seed=42, **run_kwargs):
        print(f'[velociscuti_sampler] sampling posterior of {star_name}!')

        self.load_star_data(star_name, covariance=covariance)

        print(f'[velociscuti_sampler] {len(self.mode_cols)} identified modes, teff = {self.teff:.0f} K, '
              f'rotation limit = {self.omega_ratio_max}')

        ## the run settings of the paper. anything passed as a keyword replaces them
        run_settings = dict(min_num_live_points=1000, dlogz=0.1, min_ess=1000, frac_remain=0.01, max_ncalls=10000000)
        run_settings.update(run_kwargs)

        ## UltraNest draws from numpy's global generator, so a fixed seed repeats a run on the same machine
        np.random.seed(seed)

        sampler_start_time = time.perf_counter()

        self.sampler = ultranest.ReactiveNestedSampler(PARAM_NAMES,
                                                       self.logl,
                                                       transform=self.ptform,
                                                       vectorized=True,
                                                      )

        results = self.sampler.run(**run_settings)

        self.sampler_elapsed_time = time.perf_counter() - sampler_start_time

        results['elapsed_time'] = self.sampler_elapsed_time
        results['priors'] = self.priors
        results['star_name'] = star_name

        if self.sampler_elapsed_time < 60:
            print(f'[velociscuti_sampler] finished in {self.sampler_elapsed_time:.1f}s!')
        else:
            mins, secs = divmod(self.sampler_elapsed_time, 60)
            print(f'[velociscuti_sampler] finished in {int(mins)}m {secs:.1f}s!')

        if save_as:
            save_results(results, f'stars/{star_name}/{save_as}')

        return results
