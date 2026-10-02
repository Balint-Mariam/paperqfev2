"""Small deterministic raw/IV/grid/calendar fixtures; no external options needed."""
import unittest
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import ndtr
from src import cleaning as C, implied_volatility as I, iv_surface as S, contract_mapping as M
from src.cache import prepare, complete, valid
from pathlib import Path
import tempfile


class UpstreamTests(unittest.TestCase):
    def test_price_inversion_known_sample(self):
        # Independently tabulated non-dividend European ATM call.
        self.assertAlmostEqual(I.bs_price(100.,100.,1.,.05,0.,.2,True),10.450583572185565,places=12)
        for call in [True,False]:
            price=I.bs_price(100.,105.,.4,.03,.01,.27,call)
            self.assertAlmostEqual(I.implied_vol(price,100.,105.,.4,.03,.01,call),.27,places=6)
        self.assertTrue(np.isnan(I.implied_vol(-1,100,100,1)))

    def test_normal_cdf_is_exact_certified_routine(self):
        x=np.r_[np.random.default_rng(42).normal(size=10000)*10,[-np.inf,np.inf,np.nan]]
        np.testing.assert_array_equal(norm.cdf(x),ndtr(x))

    def test_grid_coordinates_and_identifiers(self):
        x,t,gx,gt,nodes,grid=S.build_grid_and_features(-.22,.18,25,.08,1.,20)
        self.assertEqual(gx.shape,(20,25));self.assertEqual(len(nodes),500)
        self.assertEqual(nodes[:2],['iv_x00_t00','iv_x01_t00']);self.assertEqual(nodes[-1],'iv_x24_t19')
        np.testing.assert_allclose(grid.log_moneyness,np.tile(x,20));np.testing.assert_allclose(grid['T'],np.repeat(t,25))

    def test_surface_median_and_no_nearest_extrapolation(self):
        d=pd.DataFrame(dict(log_moneyness=[-.1,-.1,.1,-.1,.1],T=[.1,.1,.1,.3,.3],implied_vol=[.1,.3,.2,.2,.2]))
        gx,gt=np.meshgrid([0.,.2],[.2])
        z,method=S.interpolate_day_surface(d,gx,gt,'linear',False,0.)
        self.assertEqual(method,'linear');self.assertAlmostEqual(z[0,0],.2);self.assertTrue(np.isnan(z[0,1]))

    def test_next_exchange_session_skips_holiday_and_weekend(self):
        cal=M.xc.get_calendar('XNYS',start='2024-06-28',end='2024-07-09')
        dates=cal.sessions_in_range('2024-07-03','2024-07-08')
        self.assertEqual(dates.strftime('%Y-%m-%d').tolist(),['2024-07-03','2024-07-05','2024-07-08'])

    def test_small_raw_cleaning_sample_funnel(self):
        date=pd.Timestamp('2024-01-02')
        raw=pd.DataFrame(dict(quote_date=[date]*3,expiration=[pd.Timestamp('2024-04-02')]*3,underlying_bid_1545=[99.]*3,underlying_ask_1545=[101.]*3,bid_1545=[5.,0.,5.],ask_1545=[5.5,5.5,5.5],strike=[100.]*3,option_type=['C']*3,trade_volume=[1,1,0],open_interest=[1]*3))
        rates=pd.DataFrame({'Calendar Date':[date],'Dividend Yield (Value-Weighted)':[0.],**{c:[0.] for c in C.RATE_COLS}})
        kept,counts=C.apply_basic_filters(raw,rates,14,1.,.2)
        self.assertEqual(len(kept),1);self.assertEqual(counts['dropped_basic'],2)
        self.assertEqual(kept.mid.iloc[0],5.25);self.assertEqual(kept.r_annual.iloc[0],0.)

    def test_entry_mapping_tie_break_and_wrong_date_rejected(self):
        date=pd.Timestamp('2024-01-03')
        selected=pd.DataFrame(dict(decision_date=[pd.Timestamp('2024-01-02')],entry_date=[date],node=['n'],log_moneyness=[0.],T=[.2],signal=[1]))
        day=pd.DataFrame(dict(quote_date=[date]*2,contract_key=['b','a'],underlying_symbol=['SPX']*2,root=['SPX']*2,expiration=[pd.Timestamp('2024-04-03')]*2,strike=[100.]*2,option_type=['C']*2,bid=[1.]*2,ask=[2.]*2,mid=[1.5]*2,spot=[100.]*2,implied_vol=[.2]*2,T=[.2]*2,r=[0.]*2,q=[0.]*2,trade_volume=[1]*2,open_interest=[1]*2,log_moneyness=[0.]*2,source_observation_count=[1]*2))
        self.assertEqual(M.map_entry(selected,day,True).contract_key.iloc[0],'a')
        day.quote_date+=pd.Timedelta(days=1)
        with self.assertRaises(ValueError):M.map_entry(selected,day,True)

    def test_cache_content_and_signature_must_match(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);prepare(path,'signature');(path/'model.json').write_text('original');complete(path,'signature')
            self.assertTrue(valid(path,'signature'));self.assertFalse(valid(path,'other'))
            (path/'model.json').write_text('tampered');self.assertFalse(valid(path,'signature'))


if __name__=='__main__':unittest.main()
