#!/usr/bin/env python3
"""Exact portable tree inference for the eight NHL Item-5 research models.

No sklearn/joblib or sportsbook data required. Trained trees are unchanged;
this module only reads the bundled numeric model artifact.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, poisson


class PortablePoissonModel:
    def __init__(self, raw: dict):
        if raw.get('model_type') != 'HistGradientBoostingRegressorPoisson' or raw.get('loss') != 'poisson':
            raise ValueError('Unsupported research model; refusing approximate inference')
        self.features = tuple(raw['features'])
        self.alpha = float(raw['negative_binomial_alpha'])
        self.baseline = float(raw['baseline_raw'])
        self.trees = []
        for tree in raw['trees']:
            arr = np.asarray(tree, dtype=np.float64)
            if arr.ndim != 2 or arr.shape[1] != 7:
                raise ValueError('Unexpected portable model tree structure')
            # value, feature_index, threshold, missing_go_left, left, right, leaf
            self.trees.append((arr[:, 0], arr[:, 1].astype(np.int32), arr[:, 2],
                               arr[:, 3].astype(bool), arr[:, 4].astype(np.int32),
                               arr[:, 5].astype(np.int32), arr[:, 6].astype(bool)))
        if not self.trees or not math.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('Invalid model trees or dispersion')

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.features):
            raise ValueError(f'Expected {len(self.features)} features, got shape {X.shape}')
        raw = np.full(len(X), self.baseline, dtype=np.float64)
        for values, feats, thresholds, missing_left, left, right, leaf in self.trees:
            nodes = np.zeros(len(X), dtype=np.int32)
            for _ in range(100):
                active = ~leaf[nodes]
                if not active.any():
                    break
                row_ids = np.flatnonzero(active)
                loc = nodes[row_ids]
                xv = X[row_ids, feats[loc]]
                go_left = np.where(np.isnan(xv), missing_left[loc], xv <= thresholds[loc])
                nodes[row_ids] = np.where(go_left, left[loc], right[loc])
            else:
                raise ValueError('Portable tree exceeded safe depth')
            raw += values[nodes]
        return np.exp(raw)


def load_models(path: Path) -> dict[str, PortablePoissonModel]:
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        obj = json.load(fh)
    if obj.get('format') != 'nhl_hist_gbm_poisson_portable_v1':
        raise ValueError('Unsupported research model export')
    models = {name: PortablePoissonModel(payload)
              for name, payload in obj['models'].items()}
    expected = {'shots_on_goal', 'points', 'assists', 'goals', 'blocked_shots',
                'power_play_points', 'saves', 'goals_against'}
    if set(models) != expected:
        raise ValueError(f'Eight-market model set mismatch: {sorted(set(models) ^ expected)}')
    return models


def over_probability(mu: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    """P(count > threshold) for the *same* count family as the trained models."""
    means = np.asarray(mu, dtype=np.float64)
    if np.any(~np.isfinite(means)) or np.any(means < 0):
        raise ValueError('Nonfinite or negative forecast mean')
    means = np.maximum(means, 1e-5)
    k = math.floor(threshold)
    if alpha > 1e-10:
        p = nbinom.sf(k, 1 / alpha, 1 / (1 + alpha * means))
    else:
        p = poisson.sf(k, means)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid probability result')
    return p
