import unittest
from unittest.mock import patch
import pandas as pd
from src import robustness as m


class RegimeTests(unittest.TestCase):
    def test_pretest_boundary_ignores_future_returns_and_IV(self):
        dates=pd.to_datetime(['2023-04-19','2023-04-20','2023-04-21','2023-04-24','2023-04-25'])
        wide=pd.DataFrame(dict(quote_date=dates,iv_a=[.1,.2,.3,.5,.05],iv_b=[.1,.2,.3,.5,.05]))
        decisions=pd.DataFrame(dict(decision_date=dates[-2:],future_return=[1e10,-1e10]))
        with patch.object(m.pd,'read_csv',return_value=wide.copy()),patch.object(m,'digest',return_value='fixture'):
            result,info,levels=m.iv_regimes(decisions)
        self.assertEqual(info['threshold'],.2);self.assertEqual(info['pretest_n_dates'],3)
        self.assertEqual(result.IV_regime.tolist(),['HIGH_IV','LOW_IV'])
        wide.loc[3:,['iv_a','iv_b']]=999.;decisions.future_return=0
        with patch.object(m.pd,'read_csv',return_value=wide),patch.object(m,'digest',return_value='fixture'):
            result2,info2,levels2=m.iv_regimes(decisions)
        self.assertEqual(info['threshold'],info2['threshold'])
