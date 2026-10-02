"""Corrected chronology from surfaces to inference; no historical audit replay."""
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from . import forecasting as F, economic_signals as E, contract_mapping as M
from . import execution_quotes as Q, portfolio as P, risk_controls as R, robustness as B
from .io_utils import digest, token, write_json, save
from .environment import start_log, end_log, code_identity
from .cache import prepare, complete
from .quote_data import market_partitions, processed_pairs, recover_raw
from .validation import check


def node_worker(task):
    node,series,calendar,origins,root,signature,force=task
    root=Path(root);h1=root/'h1'/node;h2=root/'h2'/node;x2=root/'xgb_h2'/node
    if not prepare(h1,signature,force):F.fit_node((node,series,calendar,origins,str(h1)));complete(h1,signature)
    # The h2 AR/Ridge design uses the observed grid extent, without a terminal label.
    economic_series=series.loc[:origins.max()]
    if not prepare(h2,signature,force):E.fit_h2((node,economic_series,calendar,origins,str(h2)));complete(h2,signature)
    if not prepare(x2,signature,force):
        # Extracted direct-horizon fitting function; fit only h2 here.
        fit_xgb_h2(node,economic_series,calendar,origins,x2)
        complete(x2,signature)
    return node


def fit_xgb_h2(node,series,calendar,origins,folder):
    import xgboost as xgb
    X=M.features(series,calendar,2);labels=series.shift(-2)
    dates=pd.Series(calendar,index=calendar).shift(-2).reindex(series.index)
    train=dates.le(F.TRAIN_END)&labels.notna()
    val=dates.gt(F.TRAIN_END)&dates.le(F.FIT_END)&labels.notna()&(series.index>=F.TRAIN_END)
    final=dates.le(F.FIT_END)&labels.notna()
    info=dict(node=node,horizon=2,n_train=int(train.sum()),n_validation=int(val.sum()),fit_label_max_date=str(dates[final].max()),status='INSUFFICIENT_HISTORICAL_LABELS',candidates=[])
    pred=np.full(len(origins),np.nan)
    if train.sum()>=80 and val.sum()>=20:
        best=None;rmse_best=mae_best=np.inf
        for params in M.GRID:
            model=xgb.XGBRegressor(**M.XGB_BASE,**params).fit(X.loc[train],labels.loc[train])
            error=model.predict(X.loc[val])-labels.loc[val].to_numpy()
            rmse,mae=float(np.sqrt(np.mean(error**2))),float(np.mean(np.abs(error)))
            info['candidates'].append(dict(params=params,RMSE=rmse,MAE=mae))
            if rmse<rmse_best or (np.isclose(rmse,rmse_best,atol=1e-12) and mae<mae_best):best,rmse_best,mae_best=params,rmse,mae
        model=xgb.XGBRegressor(**M.XGB_BASE,**best).fit(X.loc[final],labels.loc[final])
        pred=model.predict(X.reindex(origins)).astype(float);model.get_booster().save_model(folder/(node+'.ubj'))
        info.update(status='OK',params=best)
    save(pd.DataFrame(dict(decision_date=origins,node=node,forecast_xgboost_h2=pred)),folder/(node+'.parquet'));write_json(info,folder/(node+'.json'))


def calendar_data(data):
    wide=pd.read_csv(data/'iv_grid_wide.csv');wide.quote_date=pd.to_datetime(wide.quote_date)
    wide=wide.set_index('quote_date').sort_index()
    grid=pd.read_csv(data/'iv_grid_map.csv').rename(columns={'feature':'node'});wide=wide[grid.node.tolist()]
    cal=M.xc.get_calendar('XNYS',start=wide.index.min()-pd.Timedelta(days=15),end=wide.index.max()+pd.Timedelta(days=20))
    calendar=cal.sessions_in_range(wide.index.min(),wide.index.max()+pd.Timedelta(days=15))
    origins=calendar[(calendar>=F.FIT_END)&(calendar<=wide.index.max())]
    last=pd.Series(calendar,index=calendar).shift(-1).loc[origins[-1]]
    full=wide.reindex(calendar[calendar<=last])
    return wide,grid,cal,calendar,origins,full


def regions(frame,grid):
    for col,labels in [('log_moneyness',['lower','central','upper']),('T',['short','medium','long'])]:
        groups=np.array_split(np.sort(grid[col].unique()),3)
        mapping={v:label for group,label in zip(groups,labels) for v in group}
        frame['moneyness_region' if col=='log_moneyness' else 'maturity_region']=frame[col].map(mapping)
    return frame


def journal_forecasts(config,signature,wide,grid,calendar,origins,full):
    out=config['output_root'];cache=out/'cache'/'models'/signature
    from .certified_cache import adopt
    adopted=adopt(config,cache,signature,full,calendar,origins,grid)
    tasks=[] if adopted else [(node,full[node],calendar,origins,str(cache),signature,config['force']) for node in grid.node]
    print(f'Forecasting {len(grid)} nodes, h1/h2, {len(origins)} origins; {len(tasks)} nodes need fresh fits; cache {signature[:12]}',flush=True)
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        jobs=[pool.submit(node_worker,t) for t in tasks]
        for i,job in enumerate(as_completed(jobs),1):
            job.result()
            if i%25==0:print(f'Forecast checkpoints: {i}/{len(jobs)}',flush=True)
    frame=pd.concat([pd.read_parquet(cache/'h1'/n/(n+'.parquet')) for n in grid.node],ignore_index=True)
    observed=full.stack(future_stack=True).rename('observed_iv_at_origin').reset_index().rename(columns={'quote_date':'forecast_origin_date','level_0':'forecast_origin_date','level_1':'node'})
    frame=frame.merge(observed,on=['forecast_origin_date','node'],how='left',validate='many_to_one')
    actual=observed.rename(columns={'forecast_origin_date':'forecast_target_date','observed_iv_at_origin':'actual_iv_at_target'})
    frame=frame.merge(actual,on=['forecast_target_date','node'],how='left',validate='many_to_one').merge(grid,on='node',validate='many_to_one')
    frame=F.annotate(regions(frame,grid)).sort_values(['forecast_origin_date','node'],kind='stable').reset_index(drop=True)
    save(frame,out/'data/forecast_journal_all.parquet');common=frame[frame.common_sample_4models]
    if common.empty:raise ValueError('No strict four-model forecast sample')
    daily,rank,ic=F.daily_statistics(common)
    save(daily,out/'data/forecast_daily_losses.parquet');save(rank,out/'data/forecast_daily_rank_ic.parquet')
    ic.to_csv(out/'tables/rank_ic_summary.csv',index=False)
    comparison=[]
    for model in F.MODELS:
        metrics=F.metrics(common,model,common.common_sample_qlike)
        comparison.append(dict(model=model,n_predictions_all=int(frame['available_'+model].sum()),n_scored_individual=int(frame['scored_observation_'+model].sum()),n_common=len(common),n_dates_common=common.forecast_target_date.nunique(),n_nodes_common=common.node.nunique(),common_retained_pct=100*len(common)/frame['available_'+model].sum(),**{k:v for k,v in metrics.items() if k!='n'}))
    comparison=pd.DataFrame(comparison)
    for metric in ['RMSE','MAE','QLIKE']:comparison[metric+'_rank']=comparison[metric].rank(method='min')
    intervals=F.bootstrap_intervals(daily)
    for metric in ['RMSE','MAE','QLIKE']:
        ci=intervals[intervals.metric.eq(metric)].set_index('model')
        for bound in ['lower_95','upper_95']:comparison[metric+'_'+bound]=comparison.model.map(ci[bound])
    comparison.to_csv(out/'tables/forecast_model_comparison.csv',index=False);intervals.to_csv(out/'tables/forecast_bootstrap_intervals.csv',index=False)
    dm,mcs=F.inference(daily);dm.to_csv(out/'tables/diebold_mariano_results.csv',index=False);mcs.to_csv(out/'tables/model_confidence_set.csv',index=False)
    regional=[]
    for column in ['moneyness_region','maturity_region']:
        for region,group in common.groupby(column):
            for model in F.MODELS:regional.append(dict(region_type=column,region=region,model=model,**F.metrics(group,model,group.common_sample_qlike)))
    pd.DataFrame(regional).to_csv(out/'tables/forecast_metrics_by_surface_region.csv',index=False)
    F.draw(daily,rank,out/'figures')
    return frame,cache


def economic_forecasts(config,frame,cache,full,grid,calendar):
    out=config['output_root']
    base=frame[['forecast_origin_date','forecast_target_date','node','log_moneyness','T','observed_iv_at_origin','moneyness_region','maturity_region']].rename(columns={'forecast_origin_date':'decision_date','forecast_target_date':'entry_date','observed_iv_at_origin':'observed_iv_at_decision'})
    future=pd.Series(calendar,index=calendar).shift(-2)
    base['intended_exit_date']=base.decision_date.map(future)
    h1=frame[['forecast_origin_date','node']+[f'forecast_{m}' for m in E.MODELS]].rename(columns={'forecast_origin_date':'decision_date',**{f'forecast_{m}':f'forecast_{m}_h1' for m in E.MODELS}})
    h2=pd.concat([pd.read_parquet(cache/'h2'/n/(n+'.parquet')) for n in grid.node],ignore_index=True)
    x2=pd.concat([pd.read_parquet(cache/'xgb_h2'/n/(n+'.parquet')) for n in grid.node],ignore_index=True)
    base=base.merge(h1,on=['decision_date','node'],validate='one_to_one').merge(h2,on=['decision_date','node'],validate='one_to_one').merge(x2,on=['decision_date','node'],validate='one_to_one')
    base=E.candidates(base).sort_values(['decision_date','node'],kind='stable').reset_index(drop=True)
    signals=E.rank_signals(base)
    actual=full.stack(future_stack=True).rename('actual_iv').reset_index().rename(columns={'quote_date':'date','level_0':'date','level_1':'node'})
    for date,col in [('entry_date','actual_iv_entry'),('intended_exit_date','actual_iv_exit')]:base=base.merge(actual.rename(columns={'date':date,'actual_iv':col}),on=[date,'node'],how='left',validate='many_to_one')
    base['actual_holding_change']=base.actual_iv_exit-base.actual_iv_entry
    for name in ['entry','exit','holding_change']:base['actual_'+name+'_available']=np.isfinite(base['actual_iv_'+name] if name!='holding_change' else base.actual_holding_change)
    signals=signals.merge(base[['decision_date','node','actual_iv_entry','actual_iv_exit','actual_holding_change']],on=['decision_date','node'],validate='many_to_one')
    save(base,out/'data/economic_forecasts_3models.parquet');save(signals,out/'data/economic_signals_3models.parquet')
    daily,summary=E.predictive_diagnostics(base,signals)
    save(daily,out/'data/economic_signal_daily_diagnostics.parquet');summary.to_csv(out/'tables/economic_signal_predictive_summary.csv',index=False)
    base.groupby('decision_date').agg(n_common_candidates=('common_economic_candidate','sum')).reset_index().to_csv(out/'tables/stage3_candidate_counts_by_date.csv',index=False)
    return base,signals


def map_trades(signals,market,open_dates):
    rows=[]
    for date,day in signals[signals.selected].groupby('entry_date',sort=True):
        quotes=M.load_day(market,date)
        for model,selected in day.groupby('model',sort=True):
            mapped=M.map_entry(selected,quotes,date in open_dates)
            result=M.lock_weights(selected.merge(mapped,on=['decision_date','entry_date','node'],validate='one_to_one'))
            inputs=result.rename(columns={'entry_spot':'spot','entry_T':'T_contract','entry_implied_vol':'implied_vol','entry_r':'r','entry_q':'q','T':'T_grid'}).rename(columns={'T_contract':'T'})
            for col in ['spot','strike','T','implied_vol','r','q','option_type']:
                if col not in inputs:inputs[col]=np.nan
            greeks=M.bs_greeks(inputs)
            result=pd.concat([result,greeks.rename(columns={c:'entry_'+c for c in greeks})],axis=1)
            result['trade_status']=np.where(result.weight_at_entry.ne(0),'ENTERED','NOT_ENTERED')
            rows.append(result)
    return pd.concat(rows,ignore_index=True)


def execution(config,signature,signals,cal,calendar):
    out=config['output_root'];closes=cal.schedule.loc[calendar,'close'].dt.tz_convert('America/New_York')
    open_dates=set(calendar[(closes.dt.hour.gt(15)|(closes.dt.hour.eq(15)&closes.dt.minute.ge(45))).to_numpy()])
    dates=set(signals.entry_date.dropna())|set(signals.intended_exit_date.dropna())
    market=market_partitions(config,dates,signature)
    trades=map_trades(signals,market,open_dates)
    net=E.net_contracts(trades)
    save(trades,out/'data/trades_3models_preperformance.parquet');save(net,out/'data/trades_3models_contract_net.parquet')
    E.collision_table(trades,net).to_csv(out/'tables/contract_mapping_collisions.csv',index=False)
    required=net[Q.KEYS].drop_duplicates().sort_values(Q.KEYS,kind='stable').reset_index(drop=True)
    processed=processed_pairs(required,market)
    pkeys=pd.MultiIndex.from_arrays([processed.quote_date,processed.contract_key])
    missing=required[~pd.MultiIndex.from_frame(required[Q.KEYS]).isin(pkeys)]
    raw=recover_raw(config,missing,signature)
    quotes=Q.choose_quotes(required,processed,raw,Q.calendar_guard(required.intended_exit_date))
    quotes['recovery_reason_category']=quotes.raw_observable_failure_reasons.map(Q.reason_category)
    ready=Q.join_trades(net,quotes)
    save(required,out/'cache/required_exit_pairs.parquet');save(processed,out/'cache/processed_exit_quotes.parquet');save(raw,out/'cache/raw_exit_quotes.parquet')
    save(quotes,out/'cache/execution_exit_quotes.parquet');save(ready,out/'data/trades_3models_stage3b_ready.parquet')
    Q.coverage_table(ready).to_csv(out/'tables/stage3b_exit_coverage.csv',index=False)
    Q.recovery_reasons(quotes,ready).to_csv(out/'tables/stage3b_raw_recovery_reasons.csv',index=False)
    Q.quality_table(quotes,ready).to_csv(out/'tables/stage3b_quote_quality.csv',index=False)
    return ready,processed,trades


def real_performance(config,ready,signals):
    out=config['output_root']
    selection=signals[signals.selected].groupby(['model','decision_date','entry_date','intended_exit_date']).size().rename('number_selected_nodes').reset_index()
    calendar=pd.concat([selection,P.economic_calendar(selection)],axis=1)
    entered=ready.groupby(P.GROUP).agg(n_mapped_contracts=('contract_key','size'),n_active_contracts=('nonzero_net_position','sum')).reset_index()
    calendar=calendar.merge(entered,on=P.GROUP,how='left',validate='one_to_one');calendar[['n_mapped_contracts','n_active_contracts']]=calendar[['n_mapped_contracts','n_active_contracts']].fillna(0).astype(int)
    net=ready.merge(calendar[P.GROUP+['economic_calendar_eligible']],on=P.GROUP,how='left',validate='many_to_one')
    if net.economic_calendar_eligible.isna().any():raise ValueError('Mapped day absent from selection chronology')
    eligible=net[net.economic_calendar_eligible];active=eligible[eligible.nonzero_net_position]
    if not active.BASE.all():raise ValueError('BASE exit coverage incomplete after calendar exclusions')
    if not Q.valid_two_sided(active.entry_bid,active.entry_ask).all() or not Q.valid_two_sided(active.exit_bid,active.exit_ask).all():raise ValueError('Invalid contract book')
    if active.duplicated(P.KEY).any():raise ValueError('Contract collision not netted')
    contracts=P.contract_pnl(active);daily=P.daily_pnl(contracts,eligible,selection)
    perf,boots,daily=P.performance_tables(daily,contracts)
    joined,relations=P.signal_relation(daily,signals)
    for name,d in [('contract_pnl',contracts),('daily_performance',daily),('economic_calendar',calendar),('signal_pnl_diagnostics',joined)]:save(d,out/'data'/('stage4_'+name+'.parquet'))
    for name,d in [('performance_summary',perf),('bootstrap_sensitivity',boots),('model_return_comparison',P.pairwise(daily)),('transaction_cost_wedge',P.wedge_table(daily)),('quote_source_attribution',P.contribution_table(contracts)),('quote_quality_pnl_sensitivity',P.contribution_table(contracts,True)),('signal_pnl_correlations',relations)]:d.to_csv(out/'tables'/('stage4_'+name+'.csv'),index=False)
    calendar.to_csv(out/'tables/stage4_economic_calendar.csv',index=False);P.figures(daily)
    return contracts,daily,joined,calendar


def greek_analysis(config,contracts,baseline,diag,processed):
    out=config['output_root'];contracts,quality=R.greek_quality(contracts)
    controls,weights=R.entry_controls(contracts);spots,cross=R.spot_diagnostics(contracts)
    if not spots.spot_valid.all() or not cross.valid.all():raise ValueError('Inconsistent timestamp spots')
    attr=R.attribution(contracts,spots,processed)
    exposures=controls.merge(baseline[P.GROUP+['decision_date','exit_date','GrossPremium_mid','GrossPremium_exec']],on=P.GROUP,validate='one_to_one')
    for col in ['Net'+g for g in R.GREEKS]+['GrossAbs'+g for g in R.GREEKS]+['LongVega','ShortVegaMagnitude']:exposures[col+'_per_gross_premium']=exposures[col]/exposures.GrossPremium_mid
    vb,dn,daily,vcontracts=R.portfolios(contracts,controls,weights,spots,baseline)
    perf,boots,daily=R.performance(daily);joined,alignment=R.signal_alignment(daily,diag)
    balance=exposures.groupby('model').agg(n_days=('entry_date','size'),n_available=('vega_balance_available','sum'),mean_a_L=('a_L','mean'),mean_a_S=('a_S','mean'),min_a_L=('a_L','min'),min_a_S=('a_S','min'),original_mean_abs_NetDelta=('NetDelta',lambda x:x.abs().mean()),original_mean_abs_NetVega=('NetVega',lambda x:x.abs().mean())).reset_index()
    balance['unavailable_day_pct']=100*(1-balance.n_available/balance.n_days)
    for name,d in [('contract_attribution',attr),('daily_greek_exposures',exposures),('delta_neutral_portfolios',dn),('vega_balanced_portfolios',vb),('daily_performance',daily),('signal_alignment',joined),('underlying_spot_diagnostics',spots)]:save(d,out/'data'/('stage5_'+name+'.parquet'))
    for name,d in [('greek_quality',quality),('underlying_spot_consistency',cross),('greek_exposure_summary',R.exposure_summary(exposures)),('pnl_attribution_summary',R.attribution_summary(attr)),('dgvt_subset_summary',R.attribution_summary(attr,True)),('performance_summary',perf),('bootstrap_sensitivity',boots),('signal_alignment',alignment),('vega_balance_diagnostics',balance),('exposure_reduction',R.reductions(daily))]:d.to_csv(out/'tables'/('stage5_'+name+'.csv'),index=False)
    R.make_figures(exposures,attr,daily)
    return attr,exposures,daily,perf


def final_robustness(config,baseline,attr,daily,perf):
    out=config['output_root'];multiple=B.multiple_testing(perf);effects,pairedboots=B.paired_effects(daily);components,componentdays=B.component_inference(attr)
    norm,normdaily=B.normalization(daily,baseline)
    spots=attr.groupby(P.GROUP).entry_spot.first().rename('S_entry').reset_index()
    costs,stress,hedgedays=B.hedge_costs(daily,spots)
    periods=daily.copy();periods['decision_year']=periods.decision_date.dt.year
    subperiod=B.stability(periods,'decision_year',[2023,2024])
    regimes,regimeinfo,levels=B.iv_regimes(daily)
    # Confirm the pre-test threshold and use the fixed certified number.
    locked=config['regime']['threshold']
    if abs(regimeinfo['threshold']-locked)>1e-14:raise ValueError('Certified pre-test IV regime threshold not reproduced')
    regimes['IV_regime']=np.where(regimes.market_IV_level.le(locked),'LOW_IV','HIGH_IV');regimeinfo['threshold']=locked
    labelled=daily.merge(regimes,on='decision_date',validate='many_to_one');regime=B.stability(labelled,'IV_regime',['LOW_IV','HIGH_IV'])
    chain,ranks,forecast,signal=B.chain_tables(perf,multiple,costs)
    sensitivity=B.inference_sensitivity(daily)
    for name,d in [('multiple_testing',multiple),('risk_control_effects',effects),('paired_bootstrap_sensitivity',pairedboots),('greek_component_inference',components),('normalization_sensitivity',norm),('hedge_cost_break_even',costs),('hedge_cost_stress',stress),('subperiod_stability',subperiod),('iv_regime_stability',regime),('full_chain_summary',chain),('model_ranking',ranks),('inference_sensitivity',sensitivity)]:d.to_csv(out/'tables'/('stage6_'+name+'.csv'),index=False)
    for name,d in [('daily_greek_components',componentdays),('normalization_daily',normdaily),('hedge_cost_inputs',hedgedays),('decision_IV_regimes',regimes)]:save(d,out/'data'/('stage6_'+name+'.parquet'))
    pd.read_csv(out/'tables/stage5_bootstrap_sensitivity.csv').to_csv(out/'tables/stage6_bootstrap_inference_sensitivity.csv',index=False)
    pd.read_csv(out/'tables/stage4_quote_source_attribution.csv').to_csv(out/'tables/stage6_quote_source_check.csv',index=False)
    write_json(regimeinfo,out/'diagnostics/stage6_iv_regime_definition.json');B.figures(forecast,chain,attr,daily,baseline)
    return multiple,effects,costs,chain


def run(config):
    from .results_export import export_results, delivery_manifest
    from .comparison import compare_results, analytic_checks
    started=time.perf_counter();out=config['output_root'];info=start_log(config,2)
    try:
        stage1=json.loads((out/'diagnostics/stage1_manifest.json').read_text())
        if stage1.get('status')!='REPRODUCED':raise ValueError('Stage 1 is not reproduced; analysis gate closed')
        checks=pd.read_csv(out/'diagnostics/stage1_reproduction_checks.csv')
        if checks.status.eq('FAIL').any():raise ValueError('Upstream numerical FAIL; analysis gate closed')
        if digest(out/'data/iv_grid_wide.csv')!=stage1['grid_sha256'] or digest(out/'data/options_with_iv.parquet')!=stage1['iv_sha256']:raise ValueError('Stage 1 output content changed')
        if digest(config['raw_options_file'])!=stage1['identity']['inputs']['raw_options_file']['sha256']:raise ValueError('Raw quote source changed')
        if digest(config['rates_file'])!=stage1['identity']['inputs']['rates_file']['sha256']:raise ValueError('Rate source changed')
        for module in [F,E,M,Q,P,R,B]:module.HERE=out;module.ROOT=out/'data'
        (out/'reports').mkdir(exist_ok=True)
        signature=token(dict(stage1=stage1['grid_sha256'],processed=stage1['iv_sha256'],raw=stage1['identity']['inputs'],config={k:v for k,v in config.items() if k not in ['force','workers','config_path','reference_root','output_root']},code=code_identity(),versions=info['versions'],OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS')))
        wide,grid,cal,calendar,origins,full=calendar_data(out/'data')
        forecasts,cache=journal_forecasts(config,signature,wide,grid,calendar,origins,full)
        economic,signals=economic_forecasts(config,forecasts,cache,full,grid,calendar)
        ready,processed,trades=execution(config,signature,signals,cal,calendar)
        contracts,baseline,diag,audit=real_performance(config,ready,signals)
        attr,exposures,daily,perf=greek_analysis(config,contracts,baseline,diag,processed)
        multiple,effects,costs,chain=final_robustness(config,baseline,attr,daily,perf)
        own=analytic_checks(config,forecasts,economic,signals,trades,ready,contracts,baseline,attr,exposures,daily,multiple)
        comparisons=compare_results(config)
        comparisons.to_csv(out/'diagnostics/replication_final_comparison.csv',index=False)
        own.to_csv(out/'diagnostics/analysis_checks.csv',index=False)
        failed=own.status.eq('FAIL').any() or comparisons.status.eq('FAIL').any()
        runtime=time.perf_counter()-started
        summary=dict(status='FAILED' if failed else 'REPRODUCED',runtime_seconds=runtime,signature=signature,forecast_common_sample=int(forecasts.common_sample_4models.sum()),economic_candidates=int(economic.common_economic_candidate.sum()),trading_days=baseline.entry_date.nunique(),contracts_per_model=contracts.groupby('model').size().to_dict(),BID_ASK_negative=int(multiple[multiple.family.eq('PRIMARY_BID_ASK_12')].mean_daily_return.lt(0).sum()),check_counts=own.status.value_counts().to_dict(),comparison_counts=comparisons.status.value_counts().to_dict())
        write_json(summary,out/'diagnostics/stage2_manifest.json')
        workbook=export_results(config,info,stage1,summary)
        print('FULL ANALYSIS REPRODUCED' if not failed else 'STOP — ANALYSIS REPRODUCTION FAILURE\nFULL ANALYSIS FAILED')
        print(json.dumps(summary,indent=2,default=str));print(pd.read_csv(out/'tables/forecast_model_comparison.csv')[['model','RMSE','MAE','QLIKE']].to_string(index=False));print(perf[['model','portfolio_state','implementation','mean_daily_return']].to_string(index=False))
        manifest_path=Path(__file__).resolve().parents[1]/'REPLICATION_DELIVERY_MANIFEST.json'
        print(f'Excel workbook: {workbook}\nDelivery manifest: {manifest_path}')
        # Close mirrored logs before their final content hashes are collected.
        end_log()
        delivery_manifest(config)
        return int(failed)
    except Exception as exc:
        write_json(dict(status='FAILED',error=str(exc),runtime_seconds=time.perf_counter()-started),out/'diagnostics/stage2_failure.json')
        print('STOP — ANALYSIS REPRODUCTION FAILURE\nFULL ANALYSIS FAILED\n'+str(exc),flush=True)
        raise
