import unittest
from unittest.mock import patch
from src import risk_controls as m
pd,np=m.pd,m.np

class GreekRiskControlTests(unittest.TestCase):

    def fixture(self):
        return pd.DataFrame(dict(model=['arima'] * 4, entry_date=pd.Timestamp('2024-01-05'), contract_key=['a', 'b', 'c', 'd'], signed_weight=[0.2, 0.1, -0.1, -0.2], entry_Delta=[0.5, 0.2, -0.3, -0.4], entry_Gamma=[0.01] * 4, entry_Vega=[100.0, 200.0, 400.0, 300.0], entry_Theta=[-1.0] * 4))

    def test_hand_calculated_uniform_vega_side_scaling(self):
        d, w = m.entry_controls(self.fixture())
        self.assertEqual(d.LongVega.iloc[0], 40.0)
        self.assertEqual(d.ShortVegaMagnitude.iloc[0], 100.0)
        self.assertEqual(d.a_L.iloc[0], 1.0)
        self.assertEqual(d.a_S.iloc[0], 0.4)
        np.testing.assert_allclose(w.w_vega, [0.2, 0.1, -0.04, -0.08])
        self.assertAlmostEqual((w.w_vega * w.entry_Vega).sum(), 0.0)
        self.assertTrue(w.w_vega.abs().le(w.signed_weight.abs()).all())
        self.assertAlmostEqual(d.hedge_quantity.iloc[0], -0.23)

    def test_invalid_or_one_sided_vega_is_unavailable(self):
        for val in [0.0, -1.0, np.nan, np.inf]:
            f = self.fixture()
            f.loc[0, 'entry_Vega'] = val
            d, w = m.entry_controls(f)
            self.assertFalse(d.vega_balance_available.iloc[0])
            self.assertTrue(w.w_vega.isna().all())
        f = self.fixture()
        f['signed_weight'] = f.signed_weight.abs()
        d, w = m.entry_controls(f)
        self.assertFalse(d.vega_balance_available.iloc[0])

    def test_hiding_all_post_entry_observations_preserves_controls(self):
        f = self.fixture()
        expected = m.entry_controls(f)
        f['exit_spot'] = 1000000000000000.0
        f['exit_iv'] = -100.0
        f['pnl_mid'] = 999.0
        f['exit_bid'] = np.nan
        actual = m.entry_controls(f)
        for a, b in zip(expected, actual):
            pd.testing.assert_frame_equal(a, b)

    def test_static_hedge_cancels_first_order_delta(self):
        f = self.fixture()
        d, w = m.entry_controls(f)
        change = -37.0
        delta_pnl = float((f.signed_weight * f.entry_Delta * change).sum())
        self.assertAlmostEqual(delta_pnl + float(d.hedge_quantity.iloc[0] * change), 0.0)

    def test_greek_units_against_price_finite_differences(self):
        S = 100.0
        K = 105.0
        T = 0.7
        r = 0.03
        q = 0.01
        v = 0.23
        g = m.LEGACY.compute_bs_greeks(np.array([S]), np.array([K]), np.array([T]), np.array([r]), np.array([q]), np.array([v]), np.array([True]))
        price = lambda spot, term, vol: float(m.european_price(spot, K, term, r, q, vol, True))
        h = 0.001
        self.assertAlmostEqual(g[0][0], (price(S + h, T, v) - price(S - h, T, v)) / (2 * h), places=7)
        self.assertAlmostEqual(g[1][0], (price(S + h, T, v) - 2 * price(S, T, v) + price(S - h, T, v)) / h ** 2, places=6)
        self.assertAlmostEqual(g[2][0], (price(S, T, v + 1e-05) - price(S, T, v - 1e-05)) / 2e-05, places=6)
        self.assertAlmostEqual(g[3][0], -(price(S, T + 1e-05, v) - price(S, T - 1e-05, v)) / 2e-05, places=6)

    def test_weekend_theta_clock_and_residual(self):
        f = self.fixture().iloc[:1].copy()
        f['weight_at_entry'] = f.signed_weight
        f['entry_bid'] = 9.0
        f['entry_ask'] = 11.0
        f['exit_bid'] = 10.0
        f['exit_ask'] = 12.0
        f['entry_spot'] = 100.0
        f['exit_spot'] = 101.0
        f['strike'] = 100.0
        f['option_type'] = 'C'
        f['underlying_symbol'] = '^SPX'
        f['entry_T'] = 0.2
        f['entry_r'] = 0.03
        f['entry_q'] = 0.0
        f['entry_implied_vol'] = 0.2
        f['entry_source_observation_count'] = 1
        f['all_entry_greeks_finite'] = True
        f['entry_greeks_quality_valid'] = True
        f['decision_date'] = pd.Timestamp('2024-01-04')
        f['intended_exit_date'] = pd.Timestamp('2024-01-08')
        f['exit_quote_source'] = 'RAW_1545_RECOVERED'
        f = m.S4.contract_pnl(f)
        spots, cross = m.spot_diagnostics(f)
        columns = ['quote_date', 'contract_key', 'implied_vol', 'T', 'r', 'q', 'mid', 'bid', 'ask', 'source_observation_count']
        a = m.attribution(f, spots, pd.DataFrame(columns=columns))
        self.assertAlmostEqual(a.dt_years.iloc[0], 3 / 365)
        self.assertAlmostEqual(a.Theta_component.iloc[0], -0.2 * 3 / 365)
        self.assertAlmostEqual(a[['Delta_component', 'Gamma_component', 'Theta_component', 'DGT_residual']].sum(axis=1).iloc[0], a.pnl_mid.iloc[0])
        self.assertFalse(a.DGVT_available.iloc[0])

    def test_inconsistent_spots_not_averaged(self):
        f = self.fixture()
        f['entry_spot'] = [100.0, 100.0, 100.0, 100.1]
        f['exit_spot'] = 101.0
        f['decision_date'] = pd.Timestamp('2024-01-04')
        f['exit_date'] = pd.Timestamp('2024-01-08')
        f['underlying_symbol'] = '^SPX'
        d, cross = m.spot_diagnostics(f)
        self.assertFalse(d.spot_valid.iloc[0])
        self.assertTrue(np.isnan(d.S_entry.iloc[0]))
        self.assertFalse(cross.valid.all())
        self.assertAlmostEqual(d.entry_spot_dispersion.iloc[0], 0.1)

    def ledger(self):
        c = self.fixture()
        c['weight_at_entry'] = c.signed_weight
        c['entry_bid'] = 10.0
        c['entry_ask'] = 12.0
        c['exit_bid'] = 11.0
        c['exit_ask'] = 13.0
        c['entry_spot'] = 100.0
        c['exit_spot'] = 102.0
        c['underlying_symbol'] = '^SPX'
        c['decision_date'] = pd.Timestamp('2024-01-04')
        c['intended_exit_date'] = pd.Timestamp('2024-01-08')
        c = m.S4.contract_pnl(c)
        b = pd.DataFrame(dict(model=['arima'], entry_date=[pd.Timestamp('2024-01-05')], decision_date=[pd.Timestamp('2024-01-04')], exit_date=[pd.Timestamp('2024-01-08')], PnL_mid=[0.0], PnL_exec=[-1.2], GrossPremium_mid=[6.6], GrossPremium_exec=[6.6], Return_mid=[0.0], Return_exec=[-1.2 / 6.6], number_contracts=[4]))
        return (c, b)

    def test_four_state_accounting_and_denominators(self):
        c, b = self.ledger()
        controls, w = m.entry_controls(c)
        spots, cross = m.spot_diagnostics(c)
        vb, dn, d, v = m.portfolios(c, controls, w, spots, b)
        d = d.set_index('portfolio_state')
        self.assertAlmostEqual(d.loc['DELTA_NEUTRAL', 'PnL_mid'], -0.46)
        self.assertAlmostEqual(d.loc['DELTA_NEUTRAL', 'GrossPremium_mid'], 6.6)
        self.assertAlmostEqual(d.loc['VEGA_BALANCED', 'PnL_mid'], 0.18)
        self.assertAlmostEqual(d.loc['VEGA_BALANCED', 'PnL_exec'], -0.66)
        self.assertAlmostEqual(d.loc['VEGA_BALANCED', 'GrossPremium_mid'], 4.62)
        self.assertAlmostEqual(d.loc['VEGA_BALANCED', 'GrossPremium_exec'], 4.8)
        self.assertAlmostEqual(d.loc['VEGA_BALANCED_DELTA_NEUTRAL', 'PnL_mid'], -0.148)
        self.assertAlmostEqual(d.loc['VEGA_BALANCED_DELTA_NEUTRAL', 'PnL_exec'], -0.988)

    def test_unavailable_vega_day_retained_with_nan_not_zero_pnl(self):
        c, b = self.ledger()
        c.loc[0, 'entry_Vega'] = np.nan
        controls, w = m.entry_controls(c)
        spots, cross = m.spot_diagnostics(c)
        vb, dn, d, v = m.portfolios(c, controls, w, spots, b)
        self.assertEqual(len(d), 4)
        a = d[d.portfolio_state.str.startswith('VEGA_BALANCED')]
        self.assertFalse(a.available.any())
        self.assertTrue(a[['PnL_mid', 'PnL_exec', 'Return_mid', 'Return_exec']].isna().all().all())
