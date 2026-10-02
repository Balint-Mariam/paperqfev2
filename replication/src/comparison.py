"""Independent frozen-result comparisons and analytical accounting checks."""
import json
import numpy as np
import pandas as pd
from .config import PACKAGE
from .io_utils import digest
from .validation import check
from . import economic_signals as E

TABLE_KEYS={
 'forecast_model_comparison':['model'],
 'rank_ic_summary':['model'],
 'diebold_mariano_results':['loss','model_a','model_b'],
 'model_confidence_set':['loss','model','block_length'],
 'economic_signal_predictive_summary':['model'],
 'stage4_performance_summary':['model','implementation'],
 'stage5_performance_summary':['model','portfolio_state','implementation'],
 'stage5_pnl_attribution_summary':['model','scope','component'],
 'stage5_dgvt_subset_summary':['model','scope','component'],
 'stage6_multiple_testing':['model','portfolio_state','implementation','family'],
 'stage6_risk_control_effects':['model','transformation'],
 'stage6_greek_component_inference':['model','component'],
 'stage6_normalization_sensitivity':['model','portfolio_state','implementation'],
 'stage6_hedge_cost_break_even':['model','portfolio_state'],
 'stage6_hedge_cost_stress':['model','portfolio_state','bps_per_side'],
 'stage6_subperiod_stability':['model','portfolio_state','bucket','implementation'],
 'stage6_iv_regime_stability':['model','portfolio_state','bucket','implementation'],
 'stage6_full_chain_summary':['model'],
 'stage6_model_ranking':['model'],
 'stage6_inference_sensitivity':['model','portfolio_state','implementation','HAC_lag'],
}


def compare_results(config):
    rows=[];tol=config['validation']['results_absolute_tolerance']
    references=PACKAGE/'manifests/reference_results'
    manifest=json.loads((references/'identity.json').read_text())
    for name,keys in TABLE_KEYS.items():
        ref_path=references/(name+'.csv');new_path=config['output_root']/'tables'/(name+'.csv')
        if digest(ref_path)!=manifest['sha256'][name+'.csv']:raise ValueError('Frozen comparison fixture changed: '+name)
        ref=pd.read_csv(ref_path);new=pd.read_csv(new_path)
        ref=ref.sort_values(keys,kind='stable').reset_index(drop=True);new=new.sort_values(keys,kind='stable').reset_index(drop=True)
        key_ok=len(ref)==len(new) and ref[keys].astype(str).equals(new[keys].astype(str))
        rows.append(dict(result_name=name+'.row_keys',model='ALL',portfolio_state='',implementation='',reference_value=len(ref),new_value=len(new),absolute_difference=abs(len(ref)-len(new)),tolerance=0,status='PASS' if key_ok else 'FAIL'))
        if not key_ok:continue
        for col in ref:
            if col in keys or col not in new or not pd.api.types.is_numeric_dtype(ref[col]):continue
            for i in range(len(ref)):
                a,b=new.at[i,col],ref.at[i,col]
                same_missing=pd.isna(a) and pd.isna(b)
                diff=0. if same_missing else abs(float(a)-float(b)) if np.isfinite(a) and np.isfinite(b) else np.inf
                passed=same_missing or diff<=tol
                key={k:str(ref.at[i,k]) for k in keys}
                suffix=';'.join(k+'='+v for k,v in key.items() if k not in ['model','portfolio_state','implementation'])
                rows.append(dict(result_name=name+'.'+col+('['+suffix+']' if suffix else ''),model=key.get('model','ALL'),portfolio_state=key.get('portfolio_state','ORIGINAL' if name.startswith('stage4') else ''),implementation=key.get('implementation',''),reference_value=b,new_value=a,absolute_difference=diff,tolerance=tol,status='PASS' if passed else 'FAIL'))
    # Counts and implementation checks are computed from the actual new artifacts.
    checks=analytic_count_checks(config)
    for name,a,b in checks:
        rows.append(dict(result_name=name,model='ALL',portfolio_state='',implementation='',reference_value=b,new_value=a,absolute_difference=abs(a-b),tolerance=0,status='PASS' if a==b else 'FAIL'))
    return pd.DataFrame(rows)


def analytic_count_checks(config):
    out=config['output_root'];forecast=pd.read_parquet(out/'data/forecast_journal_all.parquet');economic=pd.read_parquet(out/'data/economic_forecasts_3models.parquet');contracts=pd.read_parquet(out/'data/stage4_contract_pnl.parquet');days=pd.read_parquet(out/'data/stage4_daily_performance.parquet')
    targets=json.loads((PACKAGE/'manifests/reference_counts.json').read_text())
    checks=[('forecast_common_rows',int(forecast.common_sample_4models.sum()),targets['forecast_common_rows']),('economic_candidate_rows',int(economic.common_economic_candidate.sum()),targets['economic_candidate_rows']),('trading_dates',days.entry_date.nunique(),targets['trading_dates'])]
    for model,n in targets['contracts'].items():checks.append(('contracts_'+model,int(contracts.model.eq(model).sum()),n))
    return checks


def analytic_checks(config,forecasts,economic,signals,trades,ready,contracts,baseline,attr,exposures,daily,multiple):
    rows=[]
    def add(name,a,b=True,tol=0):rows.append(check(name,a,b,tol))
    common=forecasts[forecasts.common_sample_4models]
    add('common_forecasts_finite',np.isfinite(common[['forecast_'+m for m in ['persistence','arima','ridge','xgboost']]+['actual_iv_at_target']]).all().all())
    poisoned=economic.copy();poisoned[['actual_iv_entry','actual_iv_exit','actual_holding_change']]=1e9
    add('signal_selection_ignores_future_IV',E.rank_signals(economic).equals(E.rank_signals(poisoned)))
    universes=[signals.loc[signals.model.eq(m)&signals.common_economic_candidate,['decision_date','node']].reset_index(drop=True) for m in E.MODELS]
    add('same_candidates_all_three_models',all(u.equals(universes[0]) for u in universes))
    mapped=trades[trades.weight_at_entry.ne(0)]
    sides=mapped.groupby(['model','entry_date','signal']).weight_at_entry.sum()
    add('entry_side_half_budgets',np.allclose(abs(sides),.5,rtol=0,atol=1e-12))
    add('contract_netting_signed_conservation',np.allclose(mapped.groupby(['model','entry_date']).weight_at_entry.sum(),ready.groupby(['model','entry_date']).weight_at_entry.sum(),rtol=0,atol=1e-12))
    add('no_duplicate_active_contracts',not contracts.duplicated(['model','entry_date','contract_key']).any())
    add('BASE_exit_coverage_after_calendar',contracts.BASE.all())
    add('exact_exit_identity',contracts.contract_key.eq(contracts.exit_contract_key).all()&contracts.intended_exit_date.eq(contracts.exit_quote_date).all())
    add('mid_price_identity',contracts.pnl_mid.to_numpy(),(contracts.signed_weight*(contracts.exit_mid-contracts.entry_mid)).to_numpy(),1e-10)
    add('bid_ask_spread_accounting',contracts.cost_wedge_pnl.to_numpy(),(contracts.entry_spread_cost+contracts.exit_spread_cost).to_numpy(),1e-10)
    add('DGT_identity',attr.pnl_mid.to_numpy(),(attr.Delta_component+attr.Gamma_component+attr.Theta_component+attr.DGT_residual).to_numpy(),1e-10)
    add('full_DGT_coverage',attr.DGT_available.all())
    a=attr[attr.DGVT_available]
    add('DGVT_identity',a.pnl_mid.to_numpy(),(a.Delta_component+a.Gamma_component+a.Vega_component+a.Theta_component+a.DGVT_residual).to_numpy(),1e-10)
    add('entry_controls_do_not_leverage',(exposures[['a_L','a_S']].ge(0)&exposures[['a_L','a_S']].le(1+1e-12)).all().all())
    hedged=daily[daily.portfolio_state.str.contains('DELTA_NEUTRAL')]
    add('entry_delta_neutral',hedged.NetDelta.to_numpy(),np.zeros(len(hedged)),1e-10)
    balanced=daily[daily.portfolio_state.str.startswith('VEGA_BALANCED')]
    add('entry_vega_balanced',balanced.NetVega.to_numpy(),np.zeros(len(balanced)),1e-10)
    for suffix in ['mid','exec']:add('normalization_'+suffix,daily['Return_'+suffix].to_numpy(),(daily['PnL_'+suffix]/daily['GrossPremium_'+suffix]).to_numpy(),1e-12)
    for impl in ['MID','BID_ASK']:add('twelve_test_family_'+impl,len(multiple[multiple.family.eq('PRIMARY_'+impl+'_12')]),12)
    add('twelve_BID_ASK_negative',int(multiple[multiple.family.eq('PRIMARY_BID_ASK_12')].mean_daily_return.lt(0).sum()),12)
    for name in ['nominal_1545_timezone_and_fills','frictionless_underlying_hedge_proxy','conditional_DGVT_diagnostic_subset']:
        rows.append(dict(check_name=name,status='WARNING',observed='inherited',reference='frozen limitation',max_abs_difference=None,tolerance=0))
    return pd.DataFrame(rows)
