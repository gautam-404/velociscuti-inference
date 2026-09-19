import csv
import math

import numpy as np
import scipy.stats
import corner
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

PARAM_NAMES = ['m', 'z', 'Myr', 'v_eq']
PARAM_LABELS = [r'$M$ [M$_\odot$]', r'$Z$', 'age [Myr]', r'$v_{\rm eq}$ [km s$^{-1}$]']


## prior funcs
def uniform_prior(prior_min, prior_max):
    prior = scipy.stats.uniform(loc=prior_min, scale=prior_max-prior_min)
    ## keep the limits as typed. scipy rebuilds them from loc and scale and can miss by one float step
    prior.limits = (prior_min, prior_max)
    return prior


def gaussian_prior(mu, sigma, prior_min, prior_max):
    """Gaussian with mean mu and standard deviation sigma, truncated to [prior_min, prior_max]."""
    prior = scipy.stats.truncnorm((prior_min - mu) / sigma, (prior_max - mu) / sigma, loc=mu, scale=sigma)
    prior.limits = (prior_min, prior_max)
    return prior


def prior_limits(prior):
    ## any frozen scipy distribution works as a prior. ours carry their limits with them
    if hasattr(prior, 'limits'):
        return prior.limits
    return tuple(float(edge) for edge in prior.support())


def paper_priors():
    """The priors of the Pleiades paper, in the order m, z, Myr, v_eq."""
    mass_prior = uniform_prior(1.4, 2.5)
    ## based on the spectroscopic metallicity of the Pleiades from Soderblom et al. (2009)
    Z_prior = gaussian_prior(0.015215574134374013, 0.001886700793196596, 0.001, 0.026)
    age_prior = uniform_prior(50, 200)
    veq_prior = uniform_prior(0, 200)
    return [mass_prior, Z_prior, age_prior, veq_prior]


## mode table funcs
def mode_name(n, ell, m):
    ## p modes are n1ell0m0, n2ell1m-1 and so on. g modes are ng1..., and f modes are stored as ng0...
    return f'n{n}ell{ell}m{m}' if n > 0 else f'ng{abs(n)}ell{ell}m{m}'


def load_modes(star_name):
    """Read stars/<star_name>/<star_name>_modes.csv and keep the identified modes.

    Returns the emulator column names, the observed frequencies and their uncertainties,
    both in cycles per day, in the order of the table.
    """
    path = f'stars/{star_name}/{star_name}_modes.csv'
    mode_cols, freqs, freq_errs = [], [], []

    with open(path, newline='') as fp:
        reader = csv.DictReader(fp)
        missing = {'label', 'f_obs', 'freq_err', 'n_obs', 'l_obs', 'm_obs'} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f'[load_modes] {path} needs the columns {sorted(missing)}.')

        for row in reader:
            try:
                n, ell, m = int(row['n_obs']), int(row['l_obs']), int(row['m_obs'])
                freq, freq_err = float(row['f_obs']), float(row['freq_err'])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(freq) and math.isfinite(freq_err)):
                continue

            ## a row counts as identified only when l_obs >= 0. n_obs cannot decide this,
            ## because n_obs = -1 is also the radial order of a real g mode
            if ell < 0:
                continue
            if abs(m) > ell:
                raise ValueError(f'[load_modes] {path}: mode {row["label"]} has |m_obs| > l_obs.')
            if freq_err <= 0.0:
                raise ValueError(f'[load_modes] {path}: mode {row["label"]} needs a positive freq_err.')

            mode_cols.append(mode_name(n, ell, m))
            freqs.append(freq)
            freq_errs.append(freq_err)

    repeated = sorted({name for name in mode_cols if mode_cols.count(name) > 1})
    if repeated:
        raise ValueError(f'[load_modes] {path}: more than one frequency is assigned to {repeated}.')

    return mode_cols, np.asarray(freqs), np.asarray(freq_errs)


## results funcs
def save_results(results, path):
    """Save the posterior samples of an UltraNest run as plain arrays."""
    weighted = results['weighted_samples']
    np.savez_compressed(
        path,
        samples=np.asarray(results['samples']),
        points=np.asarray(weighted['points']),
        weights=np.asarray(weighted['weights']),
        paramnames=np.asarray(results['paramnames'], dtype=str),
        logz=float(results['logz']),
        logzerr=float(results['logzerr']),
        ess=float(results.get('ess', np.nan)),
        ncall=int(results.get('ncall', -1)),
    )


def load_results(star_name, filename):
    """Load samples written with save_as, in the layout the plotting and cluster-age funcs expect."""
    with np.load(f'stars/{star_name}/{filename}', allow_pickle=False) as saved:
        return {
            'samples': saved['samples'],
            'weighted_samples': {'points': saved['points'], 'weights': saved['weights']},
            'paramnames': [str(name) for name in saved['paramnames']],
            'logz': float(saved['logz']),
            'logzerr': float(saved['logzerr']),
            'ess': float(saved['ess']),
            'ncall': int(saved['ncall']),
        }


def weighted_quantile(values, weights, probabilities):
    order = np.argsort(values)
    x = values[order]
    w = weights[order]
    cumulative = (np.cumsum(w) - 0.5 * w) / float(w.sum())
    return np.interp(np.asarray(probabilities), cumulative, x)


def posterior_summary(results):
    """Weighted mean, standard deviation, median and 68 and 95 per cent limits of each parameter."""
    points = np.asarray(results['weighted_samples']['points'], dtype=np.float64)
    weights = np.asarray(results['weighted_samples']['weights'], dtype=np.float64)
    weights = weights / weights.sum()

    summary = {}
    for j, name in enumerate(results['paramnames']):
        values = points[:, j]
        mean = float(weights @ values)
        variance = float(weights @ ((values - mean) ** 2))
        q025, q16, q50, q84, q975 = weighted_quantile(values, weights, (0.025, 0.16, 0.5, 0.84, 0.975))
        summary[name] = {
            'mean': mean, 'sd': float(np.sqrt(variance)), 'median': float(q50),
            'p16': float(q16), 'p84': float(q84), 'p2_5': float(q025), 'p97_5': float(q975),
        }
    return summary


## plotting funcs
def posterior_plot(results, star_name=None, color='#D33682', include_prior=False, n_prior_samples=10000):
    priors = results.get('priors')
    if include_prior and priors is None:
        print('[posterior_plot] these results carry no priors, so I plot the posterior alone')
        include_prior = False

    if include_prior:
        prior_samples = np.array([prior.rvs(size=n_prior_samples) for prior in priors])
        figure = corner.corner(prior_samples.T, labels=PARAM_LABELS, color='black', hist_kwargs={'density': True}, smooth=True)
        corner.corner(results['samples'], fig=figure, color=color, hist_kwargs={'density': True}, smooth=True, show_titles=True, title_fmt='.4g', title_kwargs={'fontsize': 10})
    else:
        figure = corner.corner(results['samples'], color=color, labels=PARAM_LABELS, hist_kwargs={'density': True}, smooth=True, show_titles=True, title_fmt='.4g', title_kwargs={'fontsize': 10})

    if star_name:
        plt.suptitle(f'posterior samples for {star_name}')

    return figure


## cluster age funcs
def smoothed_log_density(values, weights, age_grid, bandwidth):
    ## weighted histogram on the age grid, smoothed with a Gaussian kernel.
    ## do not tidy the arithmetic: the cluster age is compared with the paper's value
    weights = weights / weights.sum()
    spacing = age_grid[1] - age_grid[0]
    edges = np.concatenate((age_grid - 0.5 * spacing, [age_grid[-1] + 0.5 * spacing]))
    histogram, _ = np.histogram(values, bins=edges, weights=weights)

    ## Silverman's rule with the effective number of samples, times the bandwidth factor
    mean = float(weights @ values)
    standard_deviation = math.sqrt(float(weights @ ((values - mean) ** 2)))
    effective_samples = 1.0 / float(np.sum(weights**2))
    kernel_sd = 1.06 * standard_deviation * effective_samples ** (-0.2) * bandwidth

    half_width = max(1, int(math.ceil(4.0 * kernel_sd / spacing)))
    offsets = (np.arange(2 * half_width + 1) - half_width) * spacing
    kernel = np.exp(-0.5 * (offsets / kernel_sd) ** 2)
    kernel /= kernel.sum()

    ## reflect at the edges so that a posterior leaning on a prior limit keeps its mass
    padded = np.pad(histogram, half_width, mode='reflect')
    smoothed = fftconvolve(padded, kernel, mode='same')[half_width : half_width + len(age_grid)]
    smoothed = np.maximum(smoothed, 0.0)
    norm = float(np.trapezoid(smoothed, age_grid))
    return np.log(np.maximum(smoothed / norm, 1.0e-300))


def cluster_age(results_by_star, bandwidth=0.6, age_min=50.0, age_max=200.0, n_age=601):
    """Multiply the smoothed age posteriors of coeval stars, as the paper does for the Pleiades.

    results_by_star maps a star name to its results. The product treats the stars as independent
    given a shared age. Returns the median, the 68 and 95 per cent limits, and the density on the age grid.
    """
    ages = np.linspace(age_min, age_max, int(n_age))
    log_joint = np.zeros_like(ages)
    for star in sorted(results_by_star):
        results = results_by_star[star]
        age_column = list(results['paramnames']).index('Myr')
        values = np.asarray(results['weighted_samples']['points'][:, age_column], dtype=np.float64)
        weights = np.asarray(results['weighted_samples']['weights'], dtype=np.float64)
        log_joint += smoothed_log_density(values, weights, ages, bandwidth)

    log_joint -= log_joint.max()
    density = np.exp(log_joint)
    density /= float(np.trapezoid(density, ages))

    mean = float(np.trapezoid(density * ages, ages))
    variance = float(np.trapezoid(density * (ages - mean) ** 2, ages))
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(ages)
    cdf = np.concatenate(([0.0], np.cumsum(increments)))
    cdf /= cdf[-1]
    q025, q16, q50, q84, q975 = (float(np.interp(p, cdf, ages)) for p in (0.025, 0.16, 0.5, 0.84, 0.975))

    return {
        'median': q50, 'minus': q50 - q16, 'plus': q84 - q50,
        'p16': q16, 'p84': q84, 'p2_5': q025, 'p97_5': q975,
        'mean': mean, 'sd': math.sqrt(variance),
        'ages': ages, 'density': density,
    }
