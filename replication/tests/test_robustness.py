import unittest
from unittest.mock import patch
from src import robustness as m
pd,np=m.pd,m.np

class FinalRobustnessTests(unittest.TestCase):

    def test_hand_calculated_holm_and_BH(self):
        a, b = m.corrected([0.01, 0.04, 0.03, 0.2])
        np.testing.assert_allclose(a, [0.04, 0.09, 0.09, 0.2])
        np.testing.assert_allclose(b, [0.04, 0.16 / 3, 0.16 / 3, 0.2])

    def test_corrections_zero_ties_and_random_against_library(self):
        for p in [[0.0, 0.0, 0.05, 0.05, 1.0], np.random.default_rng(42).uniform(size=24)]:
            a, b = m.corrected(p)
            np.testing.assert_allclose(a, m.multipletests(p, method='holm')[1])
            np.testing.assert_allclose(b, m.multipletests(p, method='fdr_bh')[1])
        for p in [[np.nan, 0.1], [-0.1, 0.1], [1.1, 0.1]]:
            with self.assertRaises(ValueError):
                m.corrected(p)

    def cost_fixture(self):
        dates = pd.to_datetime(['2024-01-02', '2024-01-03'])
        d = pd.DataFrame(dict(model=['arima'] * 2, entry_date=dates, portfolio_state=['DELTA_NEUTRAL'] * 2, hedge_quantity=[-1.0, 1.0], PnL_mid=[1.0, 3.0], GrossPremium_mid=[10.0, 30.0], Return_mid=[0.1, 0.1], PnL_exec=[0.5, 2.5], GrossPremium_exec=[11.0, 31.0]))
        spots = pd.DataFrame(dict(model=['arima'] * 2, entry_date=dates, S_entry=[100.0, 100.0]))
        return (d, spots)

    def test_mean_and_aggregate_cost_roots_distinguished(self):
        d, spots = self.cost_fixture()
        c, s, x = m.hedge_costs(d, spots)
        self.assertAlmostEqual(c.break_even_bps_per_side.iloc[0], 75.0)
        self.assertAlmostEqual(c.aggregate_break_even_bps_per_side.iloc[0], 100.0)
        root = c.break_even_decimal_per_side.iloc[0]
        self.assertAlmostEqual(((x.PnL_mid - 2 * root * x.hedge_notional) / x.GrossPremium_mid).mean(), 0.0)
        self.assertAlmostEqual(s.loc[s.bps_per_side.eq(0.5), 'mean_MID_after_hypothetical_cost'].iloc[0], 0.1 - (2 * 5e-05 * 100 / 10 + 2 * 5e-05 * 100 / 30) / 2)

    def test_nonpositive_mean_has_zero_primary_cost_budget(self):
        d, spots = self.cost_fixture()
        d['PnL_mid'] = -d.PnL_mid
        d['Return_mid'] = -d.Return_mid
        c, s, x = m.hedge_costs(d, spots)
        self.assertEqual(c.break_even_bps_per_side.iloc[0], 0.0)
        self.assertFalse(c.positive_mean_friction_budget.iloc[0])

    def test_normalization_changes_no_pnl_and_ratio_of_sums(self):
        dates = pd.to_datetime(['2024-01-02', '2024-01-03'])
        d = pd.DataFrame(dict(model=['arima'] * 2, entry_date=dates, portfolio_state=['VEGA_BALANCED'] * 2, PnL_mid=[2.0, -1.0], PnL_exec=[1.0, -2.0], GrossPremium_mid=[2.0, 10.0], GrossPremium_exec=[3.0, 11.0], Return_mid=[1.0, -0.1], Return_exec=[1 / 3, -2 / 11]))
        b = pd.DataFrame(dict(model=['arima'] * 2, entry_date=dates, GrossPremium_mid=[10.0, 10.0], GrossPremium_exec=[10.0, 10.0]))
        n, x = m.normalization(d, b)
        a = n[n.implementation.eq('MID')].iloc[0]
        self.assertAlmostEqual(a.primary_mean_daily_ratio, 0.45)
        self.assertAlmostEqual(a.mean_Return_common_denom, 0.05)
        self.assertAlmostEqual(a.pooled_premium_return, 1 / 12)
        pd.testing.assert_frame_equal(x[d.columns], d)

    def test_paired_effects_reject_incomplete_date_panel(self):
        dates = pd.bdate_range('2023-01-02', periods=407)
        d = pd.DataFrame([dict(model='arima', entry_date=date, portfolio_state=state, Return_mid=0.0) for date in dates for state in m.STATES])
        with self.assertRaisesRegex(ValueError, '408'):
            m.paired_effects(d)
