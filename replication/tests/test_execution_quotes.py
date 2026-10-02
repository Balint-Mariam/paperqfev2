import unittest
from unittest.mock import patch
from src import execution_quotes as m
pd,np=m.pd,m.np

class ExecutionQuoteTests(unittest.TestCase):

    def fixture(self):
        date = pd.Timestamp('2024-01-04')
        required = pd.DataFrame({'intended_exit_date': [date] * 3, 'contract_key': ['a', 'b', 'c']})
        processed = pd.DataFrame({'quote_date': [date], 'contract_key': ['a'], 'bid': [1.0], 'ask': [2.0], 'trade_volume': [1.0]})
        raw = pd.DataFrame({'quote_date': [date] * 3, 'contract_key': ['a', 'b', 'c'], 'bid_1545': [9.0, 3.0, 4.0], 'ask_1545': [10.0, 3.0, 8.0], 'trade_volume': [0.0, 0.0, 0.0], 'open_interest': [0.0, 0.0, 0.0]})
        return (required, processed, raw)

    def test_processed_priority_zero_volume_and_wide_spread(self):
        req, p, r = self.fixture()
        q = m.choose_quotes(req, p, r, [True] * 3)
        self.assertEqual(q.exit_quote_source.tolist(), ['PROCESSED_1545', 'RAW_1545_RECOVERED', 'RAW_1545_RECOVERED'])
        self.assertEqual(q.exit_bid.iloc[0], 1.0)
        self.assertTrue(q.BASE.all())
        self.assertFalse(q.STRICT_20.iloc[2])
        self.assertFalse(q.POSITIVE_VOLUME.iloc[1])

    def test_invalid_books(self):
        bid = pd.Series([0.0, -1.0, np.nan, np.inf, 1.0, 1.0, 1.0, 1.0])
        ask = pd.Series([2.0, 2.0, 2.0, np.inf, np.nan, np.inf, 0.9, 1.0])
        self.assertEqual(m.valid_two_sided(bid, ask).tolist(), [False] * 7 + [True])

    def test_no_later_date_or_different_contract_or_EOD_fallback(self):
        req, p, r = self.fixture()
        r.loc[1, 'quote_date'] += pd.Timedelta(days=1)
        r.loc[2, 'contract_key'] = 'another_expiry'
        r['bid_eod'] = 100.0
        r['ask_eod'] = 101.0
        q = m.choose_quotes(req, p, r, [True] * 3)
        self.assertEqual(q.BASE.tolist(), [True, False, False])

    def test_drop_liquidity_and_poison_future_fields(self):
        req, p, r = self.fixture()
        core = ['BASE', 'exit_quote_source', 'exit_bid', 'exit_ask', 'exit_mid']
        expected = m.choose_quotes(req, p, r, [True] * 3)[core]
        r = r.drop(columns=['trade_volume', 'open_interest'])
        p = p.drop(columns=['trade_volume'])
        for f in [p, r]:
            f['future_pnl'] = -1e+30
            f['future_return'] = 1e+30
            f['implied_vol'] = np.nan
            f['r'] = np.nan
            f['T'] = 0.0
            f['moneyness'] = 99.0
        pd.testing.assert_frame_equal(expected, m.choose_quotes(req, p, r, [True] * 3)[core])

    def test_inherited_half_day_and_non_session_guard(self):
        dates = pd.Series(pd.to_datetime(['2024-07-03', '2024-07-04', '2024-07-05', '2024-07-06']))
        self.assertEqual(m.calendar_guard(dates).tolist(), [False, False, True, False])
        req, p, r = self.fixture()
        self.assertFalse(m.choose_quotes(req, p, r, [False] * 3).BASE.any())

    def test_recovery_reason_reconciliation(self):
        cases = {'NONPOSITIVE_VOLUME': 'NONPOSITIVE_VOLUME_ONLY', 'SPREAD_ABOVE_DEFAULT_MAX': 'SPREAD_FILTER_ONLY', 'NO_EXACT_RATE_DATE': 'RATE_OR_IV_INPUT_RELATED', 'T_OUTSIDE_DEFAULT_WINDOW': 'MONEYNESS_OR_T_FILTER', 'NONPOSITIVE_VOLUME;NO_EXACT_RATE_DATE': 'MULTIPLE_REASONS', 'NO_OBSERVABLE_DEFAULT_FAILURE_REASON': 'NO_OBSERVABLE_REASON'}
        for value, expected in cases.items():
            self.assertEqual(m.reason_category(value), expected)
