"""Numerical gates; reference values are never used as computed outputs."""
import numpy as np
import pandas as pd


def check(name, observed, reference, tolerance=0):
    a, b = np.asarray(observed), np.asarray(reference)
    same_shape = a.shape == b.shape
    numeric = a.dtype.kind in 'biufc' and b.dtype.kind in 'biufc'
    if same_shape and numeric:
        finite = np.isfinite(a) & np.isfinite(b)
        missing_same = np.array_equal(np.isnan(a), np.isnan(b))
        diff = float(np.max(np.abs(a[finite].astype(float) - b[finite].astype(float)))) if finite.any() else 0.
        passed = missing_same and np.array_equal(np.isinf(a), np.isinf(b)) and diff <= tolerance
    else:
        diff = 0. if same_shape and np.array_equal(a, b) else np.nan
        passed = bool(same_shape and np.array_equal(a, b))
    return dict(check_name=name, status='PASS' if passed else 'FAIL',
                observed=str(observed) if a.size <= 3 else f'shape={a.shape}',
                reference=str(reference) if b.size <= 3 else f'shape={b.shape}',
                max_abs_difference=diff, tolerance=tolerance)


def funnel(stats):
    remaining = stats['input']
    rows = []
    reasons = {'basic':'positive book, volume, OI, spot, strike, mid; ask>bid',
               'bad_dates':'parseable quote/expiry dates', 't_window':'14/365 <= T <= 1',
               'no_rates':'exact rate-date inner join', 'moneyness':'0.8 <= K/S <= 1.2',
               'spread':'relative spread <= 0.2', 'option_type':'C or P',
               'lower_bound':'mid + 0.0001 >= discounted European lower bound'}
    for step, reason in reasons.items():
        removed = stats['dropped_' + step]
        rows.append(dict(step=step, rows_before=remaining, rows_removed=removed,
                         rows_after=remaining-removed, reason=reason))
        remaining -= removed
    if remaining != stats['kept_basic']:
        raise ValueError('Cleaning counters do not conserve rows')
    return pd.DataFrame(rows)
