import unittest
from unittest.mock import patch
from src import economic_signals as m
pd,np=m.pd,m.np

class ImplementabilityTests(unittest.TestCase):

    def test_native_two_step_ignores_intermediate_actual(self):
        rng = np.random.default_rng(4)
        series = pd.Series(0.2 + np.cumsum(rng.normal(0, 0.002, 100)), index=pd.bdate_range('2024-01-02', periods=100))
        series.iloc[[12, 72]] = np.nan
        for order in m.S2.ORDERS:
            fitted = m.S2.arima_model(series.iloc[:60].to_numpy(), order).fit(method_kwargs={'maxiter': 200})
            origin = series.index[70]
            actual = m.arima_native(series, order, fitted.params, pd.DatetimeIndex([origin]), 2)[0]
            deleted = series.copy()
            deleted.iloc[71:] = np.nan
            self.assertAlmostEqual(actual, m.arima_native(deleted, order, fitted.params, pd.DatetimeIndex([origin]), 2)[0], places=12)
            native = fitted.append(series.iloc[60:71].to_numpy(), refit=False).forecast(2)[-1]
            self.assertAlmostEqual(actual, native, places=12)

    def fixture(self):
        n = 50
        frame = pd.DataFrame(dict(decision_date=np.repeat(pd.Timestamp('2024-01-02'), n), entry_date=np.repeat(pd.Timestamp('2024-01-03'), n), intended_exit_date=np.repeat(pd.Timestamp('2024-01-04'), n), node=[f'n{i:03d}' for i in range(n)], log_moneyness=np.linspace(-0.2, 0.2, n), T=np.repeat(0.2, n), observed_iv_at_decision=np.repeat(0.2, n), moneyness_region=np.repeat('central', n), maturity_region=np.repeat('short', n)))
        for model in m.MODELS:
            frame[f'forecast_{model}_h1'] = 0.2
            frame[f'forecast_{model}_h2'] = 0.2 + np.linspace(-0.01, 0.01, n)
        return m.candidates(frame)

    def test_rank_universe_ignores_poisoned_future_outcomes(self):
        frame = self.fixture()
        a = m.rank_signals(frame)
        frame['actual_iv_entry'] = np.nan
        frame['actual_iv_exit'] = 1000000000000.0
        frame['future_contract_available'] = False
        b = m.rank_signals(m.candidates(frame))
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue(a.groupby('model').selected.sum().eq(20).all())

    def test_h2_cutoff_purges_training_boundary(self):
        calendar = pd.bdate_range('2020-01-02', '2024-01-02')
        series = pd.Series(0.2, index=calendar)
        y, dates, train, val, final = m.horizon_masks(series, calendar, 2)
        self.assertTrue((series.index[val] >= m.TRAIN_END).all())
        self.assertLessEqual(dates[final].max(), m.FIT_END)
        self.assertLessEqual(dates[train].max(), m.TRAIN_END)

    def test_netting_cancels_opposing_contract_and_preserves_signed_weight(self):
        frame = pd.DataFrame(dict(model=['ridge'] * 4, entry_date=[pd.Timestamp('2024-01-03')] * 4, contract_key=['a', 'a', 'b', 'b'], entry_mapping_status=['MAPPED'] * 4, weight_at_entry=[0.05, -0.05, 0.05, 0.05], signal=[1, -1, 1, 1], node=['n1', 'n2', 'n3', 'n4']))
        net = m.net_contracts(frame).set_index('contract_key')
        self.assertFalse(net.loc['a', 'nonzero_net_position'])
        self.assertTrue(net.loc['a', 'opposing_signals'])
        self.assertAlmostEqual(net.weight_at_entry.sum(), frame.weight_at_entry.sum())
        self.assertAlmostEqual(net.loc['b', 'weight_at_entry'], 0.1)
