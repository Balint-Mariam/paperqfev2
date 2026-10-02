"""Frozen functions extracted from compute_portfolio_greeks.py."""
from __future__ import annotations
import sys, os, json, time, inspect, hashlib, warnings, logging, itertools
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, spearmanr
import exchange_calendars as xc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io_utils import digest, token, write_json, save


def compute_bs_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sqrt_t = np.sqrt(T)
    vol_sqrt_t = sigma * sqrt_t
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    nd1 = norm.cdf(d1)
    nd2 = norm.cdf(d2)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    disc_r = np.exp(-r * T)

    delta_call = disc_q * nd1
    delta_put = disc_q * (nd1 - 1.0)
    delta = np.where(is_call, delta_call, delta_put)

    gamma = disc_q * pdf_d1 / (S * vol_sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t

    theta_call = (
        -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
        - r * K * disc_r * nd2
        + q * S * disc_q * nd1
    )
    theta_put = (
        -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_t)
        + r * K * disc_r * norm.cdf(-d2)
        - q * S * disc_q * norm.cdf(-d1)
    )
    theta = np.where(is_call, theta_call, theta_put)
    return delta, gamma, vega, theta
