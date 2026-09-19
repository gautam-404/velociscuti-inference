import json
import os
import re

import numpy as np
import torch
from torch import nn

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

## frequency columns look like n5ell1m-1 (p modes) or ng1ell2m0 (g modes; f modes are ng0)
MODE_RE = re.compile(r'^(n|ng)\d+ell\d+m-?\d+$')


def yeo_johnson(x, lambdas):
    """Column-wise Yeo-Johnson transform, the same one scikit-learn's PowerTransformer applies."""
    lam = np.asarray(lambdas, dtype=x.dtype)
    out = np.zeros_like(x)
    for j, value in enumerate(lam):
        col = x[..., j]
        positive = col >= 0.0
        if abs(value) < np.spacing(1.0):
            out[..., j][positive] = np.log1p(col[positive])
        else:
            out[..., j][positive] = (
                np.power(col[positive] + 1, value) - 1
            ) / value
        negative = ~positive
        if abs(value - 2.0) > np.spacing(1.0):
            out[..., j][negative] = -(
                np.power(-col[negative] + 1, 2.0 - value) - 1
            ) / (2.0 - value)
        else:
            out[..., j][negative] = -np.log1p(-col[negative])
    return out


def inverse_yeo_johnson(y, lambdas):
    """Undo yeo_johnson, column by column."""
    lam = np.asarray(lambdas, dtype=y.dtype)
    out = np.zeros_like(y)
    for j, value in enumerate(lam):
        col = y[..., j]
        positive = col >= 0.0
        if abs(value) < np.spacing(1.0):
            out[..., j][positive] = np.exp(col[positive]) - 1
        else:
            out[..., j][positive] = np.power(
                col[positive] * value + 1, 1.0 / value
            ) - 1
        negative = ~positive
        if abs(value - 2.0) > np.spacing(1.0):
            out[..., j][negative] = 1.0 - np.power(
                -(2.0 - value) * col[negative] + 1,
                1.0 / (2.0 - value),
            )
        else:
            out[..., j][negative] = 1.0 - np.exp(-col[negative])
    return out


def make_head(n_in, n_out, n_hidden, n_layers):
    layers = []
    current = n_in
    for _ in range(n_layers):
        layers.extend((nn.Linear(current, n_hidden), nn.GELU()))
        current = n_hidden
    layers.append(nn.Linear(current, n_out))
    return nn.Sequential(*layers)


class multihead_network(nn.Module):
    """Shared residual blocks, then one output head each for the 14 non-frequency outputs, the p modes and the g and f modes."""

    def __init__(self, n_inputs, output_cols, hidden_dims, head_hidden_dim, head_n_layers):
        super().__init__()
        self.output_cols = tuple(output_cols)

        ## these attribute names are the keys of the weight file, so they cannot be renamed
        self.blocks = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        previous = n_inputs
        for n_neurons in hidden_dims:
            self.blocks.append(nn.Sequential(nn.Linear(previous, n_neurons), nn.GELU()))
            self.residual_projections.append(
                nn.Linear(previous, n_neurons) if previous != n_neurons else nn.Identity()
            )
            previous = n_neurons

        self.classical_idx = [i for i, name in enumerate(self.output_cols) if MODE_RE.match(name) is None]
        self.p_idx = [i for i, name in enumerate(self.output_cols) if name.startswith('n') and not name.startswith('ng')]
        self.g_idx = [i for i, name in enumerate(self.output_cols) if name.startswith('ng')]

        self.classical_head = make_head(previous, len(self.classical_idx), head_hidden_dim, head_n_layers)
        self.p_head = make_head(previous, len(self.p_idx), head_hidden_dim, head_n_layers)
        self.g_head = make_head(previous, len(self.g_idx), head_hidden_dim, head_n_layers)

    def forward(self, x):
        shared = x
        for block, projection in zip(self.blocks, self.residual_projections):
            residual = projection(shared)
            shared = block(shared) + residual
        result = torch.zeros((x.shape[0], len(self.output_cols)), dtype=shared.dtype, device=shared.device)
        result[:, self.classical_idx] = self.classical_head(shared)
        result[:, self.p_idx] = self.p_head(shared)
        result[:, self.g_idx] = self.g_head(shared)
        return result


class velociscuti_emulator():
    def __init__(self, network, info):
        self.network = network
        self.input_cols = list(info['input_cols'])
        self.output_cols = list(info['output_cols'])

        ## everything stays in float32, the precision of the stored weights
        pre_in = info['preprocessing']['input']
        pre_out = info['preprocessing']['output']
        self.input_min = np.asarray(pre_in['bounds_min'], dtype=np.float32)
        self.input_max = np.asarray(pre_in['bounds_max'], dtype=np.float32)
        self.input_lambdas = np.asarray(pre_in['lambdas'], dtype=np.float32)
        self.input_center = np.asarray(pre_in['center'], dtype=np.float32)
        self.input_scale = np.asarray(pre_in['scale'], dtype=np.float32)
        self.output_power_mask = np.asarray(pre_out['power_mask'], dtype=bool)
        self.output_lambdas = np.asarray(pre_out['lambdas'], dtype=np.float32)
        self.output_center = np.asarray(pre_out['center'], dtype=np.float32)
        self.output_scale = np.asarray(pre_out['scale'], dtype=np.float32)

    def check_range(self, theta):
        """Raise if any row of theta lies outside the range the emulator was trained on."""
        ## a few float32 steps of slack, so that a value rounded onto the edge of the range still counts as inside
        tolerance = 8.0 * np.finfo(np.float32).eps * np.maximum(
            1.0, np.maximum(np.abs(self.input_min), np.abs(self.input_max))
        )
        outside = (theta < self.input_min - tolerance) | (theta > self.input_max + tolerance)
        if np.any(outside):
            cols = [self.input_cols[j] for j in np.flatnonzero(np.any(outside, axis=0))]
            raise ValueError(
                f'[velociscuti_emulator] inputs outside the trained range for {cols}. '
                'The emulator does not extrapolate, see velociscuti/model/velociscuti_info.md for the range.'
            )

    def predict(self, theta):
        """Predict all 232 outputs for rows of (m, z, Myr, v_eq). Returns physical units, frequencies in cycles per day."""
        theta = np.asarray(theta, dtype=np.float32)
        if theta.ndim == 1:
            theta = theta.reshape(1, -1)
        if not np.all(np.isfinite(theta)):
            raise ValueError('[velociscuti_emulator] inputs must be finite.')
        self.check_range(theta)

        ## the pipeline this code came from handed the network a column-ordered array, and the
        ## network's matrix products round differently (in the last float32 digit) for row-ordered
        ## input. keep the column order so that predictions stay identical to the frozen ones
        theta = np.asfortranarray(theta)

        ## the three blocks below reproduce the training pipeline step for step.
        ## do not tidy the arithmetic: fits are compared with frozen values

        # -----------------------
        # scale the inputs
        # -----------------------
        x = np.clip(theta, self.input_min, self.input_max)
        x = yeo_johnson(x, self.input_lambdas)
        x -= self.input_center
        x /= self.input_scale
        x = x.astype(np.float32, copy=False)

        # -----------------------
        # forward pass
        # -----------------------
        with torch.inference_mode():
            y = self.network(torch.from_numpy(x)).numpy()

        # -----------------------
        # undo the output scaling
        # -----------------------
        y = y.copy()
        y *= self.output_scale
        y += self.output_center
        mask = self.output_power_mask
        y[:, mask] = inverse_yeo_johnson(y[:, mask], self.output_lambdas[mask])
        return y.astype(np.float32, copy=False)


def load_emulator(model_dir=None):
    """Load the emulator from velociscuti/model, or from another folder that holds the same two files."""
    if model_dir is None:
        model_dir = MODEL_DIR

    ## one thread on purpose. with more than one thread the output changes slightly with batch
    ## size, which makes a fit unrepeatable
    torch.set_num_threads(1)

    with open(os.path.join(model_dir, 'velociscuti.json'), 'r') as fp:
        info = json.load(fp)

    arch = info['architecture']
    network = multihead_network(
        len(info['input_cols']),
        info['output_cols'],
        arch['hidden_dims'],
        arch['head_hidden_dim'],
        arch['head_n_layers'],
    )

    ## the weights are plain arrays, so nothing is unpickled when the emulator loads
    with np.load(os.path.join(model_dir, 'velociscuti_weights.npz'), allow_pickle=False) as weights:
        state = {name: torch.from_numpy(np.asarray(weights[name]).copy()) for name in weights.files}
    network.load_state_dict(state, strict=True)
    network.eval()

    print(f'[load_emulator] velociscuti loaded: {len(info["input_cols"])} inputs, {len(info["output_cols"])} outputs')
    return velociscuti_emulator(network, info)
