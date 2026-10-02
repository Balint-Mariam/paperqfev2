import unittest
from unittest.mock import patch
from src import portfolio as m
pd,np=m.pd,m.np

class EconomicPerformanceTests(unittest.TestCase):

    def fixture(self):
        return pd.DataFrame(dict(weight_at_entry=[2.0, -3.0], entry_bid=[9.0, 19.0], entry_ask=[11.0, 21.0], exit_bid=[12.0, 17.0], exit_ask=[14.0, 19.0], intended_exit_date=pd.to_datetime(['2024-01-04'] * 2)))

    def test_hand_calculated_long_short_ledger(self):
        x = m.contract_pnl(self.fixture())
        self.assertEqual(x.pnl_mid.tolist(), [6.0, 6.0])
        self.assertEqual(x.pnl_exec.tolist(), [2.0, 0.0])
        self.assertEqual(x.gross_entry_premium_mid_component.tolist(), [20.0, 60.0])
        self.assertEqual(x.gross_entry_premium_exec_component.tolist(), [22.0, 57.0])
        self.assertEqual(x.entry_spread_cost.tolist(), [2.0, 3.0])
        self.assertEqual(x.exit_spread_cost.tolist(), [2.0, 3.0])
        self.assertAlmostEqual(x.pnl_mid.sum() / x.gross_entry_premium_mid_component.sum(), 0.15)
        self.assertAlmostEqual(x.pnl_exec.sum() / x.gross_entry_premium_exec_component.sum(), 2 / 79)

    def test_exec_never_exceeds_mid_for_both_directions(self):
        rng = np.random.default_rng(9)
        n = 200
        f = pd.DataFrame(dict(weight_at_entry=rng.normal(size=n), entry_bid=rng.uniform(0.1, 100, n), exit_bid=rng.uniform(0.1, 100, n), intended_exit_date=pd.Timestamp('2024-01-04')))
        f['entry_ask'] = f.entry_bid + rng.uniform(0, 5, n)
        f['exit_ask'] = f.exit_bid + rng.uniform(0, 8, n)
        x = m.contract_pnl(f)
        self.assertTrue((x.pnl_exec <= x.pnl_mid + 1e-12).all())
        np.testing.assert_allclose(x.pnl_mid - x.pnl_exec, x.entry_spread_cost + x.exit_spread_cost, atol=1e-12)

    def test_calendar_poisoning_and_early_close_exclusions(self):
        f = pd.DataFrame(dict(decision_date=pd.to_datetime(['2024-07-02', '2024-07-03', '2024-07-05']), entry_date=pd.to_datetime(['2024-07-03', '2024-07-05', '2024-07-08']), intended_exit_date=pd.to_datetime(['2024-07-05', '2024-07-08', '2024-07-09'])))
        expected = m.economic_calendar(f)
        self.assertEqual(expected.economic_calendar_eligible.tolist(), [False, True, True])
        f['BASE'] = False
        f['exit_bid'] = np.nan
        f['future_pnl'] = -1e+20
        f['weight_at_entry'] = 999
        pd.testing.assert_frame_equal(expected, m.economic_calendar(f))

    def test_hac_matches_independent_bartlett_formula(self):
        x = np.random.default_rng(8).normal(size=60)
        u = x - x.mean()
        n = len(x)
        meat = np.dot(u, u) + 2 * sum(((1 - lag / 6) * np.dot(u[lag:], u[:-lag]) for lag in range(1, 6)))
        expected = np.sqrt(meat / (n * n) * n / (n - 1))
        self.assertAlmostEqual(m.hac(x)['HAC_standard_error'], expected, places=12)

    def test_wealth_guard_and_initial_peak(self):
        w, dd, status = m.wealth([-0.1, 0.05])
        np.testing.assert_allclose(w, [0.9, 0.945])
        self.assertAlmostEqual(dd, -0.1)
        for x in [[-0.1, -1.0], [-1.2, 0.2], [np.nan, 0.1]]:
            w, dd, status = m.wealth(x)
            self.assertIsNone(w)
            self.assertTrue(np.isnan(dd))

    def test_bootstrap_deterministic_and_fixed_settings(self):
        x = np.random.default_rng(7).normal(0.01, 0.1, 30)
        self.assertEqual(m.bootstrap(x, 5), m.bootstrap(x, 5))
        self.assertEqual(m.bootstrap(x, 10)['bootstrap_seed'], 42)
        self.assertEqual(m.bootstrap(x, 5)['bootstrap_repetitions'], 4000)

    def test_pairwise_inference_uses_intersection_of_dates(self):
        dates = pd.bdate_range('2024-01-02', periods=15)
        rows = []
        for k, model in enumerate(m.MODELS):
            for i, date in enumerate(dates):
                if model == 'ridge' and i == 0:
                    continue
                rows.append(dict(model=model, entry_date=date, Return_mid=0.001 * (i + k), Return_exec=0.001 * (i + k) - 0.002))
        result = m.pairwise(pd.DataFrame(rows))
        self.assertEqual(result.loc[result.model_B.eq('ridge'), 'n_inference_dates'].tolist(), [14, 14])
        self.assertEqual(result.loc[result.model_A.eq('ridge'), 'n_inference_dates'].tolist(), [14, 14])
        self.assertTrue(np.allclose(result.mean_difference, [-0.001, -0.002, -0.001] * 2))

    def test_negative_midpoint_value_has_no_loss_fraction(self):
        d = pd.DataFrame(dict(model=['arima'], PnL_mid=[-2.0], PnL_exec=[-3.0], entry_spread_cost=[0.4], exit_spread_cost=[0.6], cost_wedge_pnl=[1.0], cost_wedge_return=[0.1]))
        self.assertTrue(np.isnan(m.wedge_table(d).fraction_midpoint_economic_value_lost.iloc[0]))
