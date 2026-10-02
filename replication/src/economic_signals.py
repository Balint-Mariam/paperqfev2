"""Frozen functions extracted from paper_qfe/04_economic_signals_and_exit_recovery.py."""
from __future__ import annotations
import sys, os, json, time, inspect, hashlib, warnings, logging, itertools
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm, spearmanr
import exchange_calendars as xc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .io_utils import digest, token, write_json, save
import joblib
from . import contract_mapping as S1, forecasting as S2
MODELS=['arima','ridge','xgboost']; TRAIN_END=S2.TRAIN_END; FIT_END=S2.FIT_END
KEYS=['intended_exit_date','contract_key']; MIN_IC=20


def horizon_features(series,calendar,h):
    out=S2.features(series,calendar)
    out['dow']=pd.Series(calendar,index=calendar).shift(-h).reindex(series.index).dt.weekday.astype(float)
    return out

def horizon_masks(series,calendar,h):
    dates=pd.Series(calendar,index=calendar).shift(-h).reindex(series.index);y=series.shift(-h)
    finite=np.isfinite(y)
    train=dates.le(TRAIN_END)&finite
    val=dates.gt(TRAIN_END)&dates.le(FIT_END)&finite&(series.index>=TRAIN_END)
    final=dates.le(FIT_END)&finite
    return y,dates,train,val,final

def arima_native(series,order,params,origins,horizon):
    """Advance the filtered state from t without observing t+1.

    predicted_state[:,t+1] is conditional on observations through t. The
    time-invariant transition propagates it to t+2. It is NOT the in-sample
    one-step forecast at t+2, which would incorporate the t+1 observation.
    Public native dynamic predictions are checked independently in audits.
    """
    model=S2.arima_model(series.to_numpy(),order);res=model.filter(np.asarray(params))
    if not res.filter_results.time_invariant: raise ValueError('ARIMA representation must be time invariant')
    positions=series.index.get_indexer(origins)
    if (positions<0).any(): raise ValueError('Unknown origin')
    state=res.predicted_state[:,positions+1].copy()
    transition=model.ssm['transition'];intercept=model.ssm['state_intercept']
    for _ in range(horizon-1): state=transition@state+intercept[:,None]
    result=(model.ssm['design']@state+model.ssm['obs_intercept'][:,None])[0]
    return np.asarray(result,dtype=float)

def fit_h2(task):
    node,series,calendar,origins,cache_str=task;cache=Path(cache_str)
    meta=cache/f'{node}.json';archive=cache/f'{node}.parquet'
    if meta.exists() and archive.exists(): return json.loads(meta.read_text(encoding='utf-8')),True
    started=time.perf_counter();X=horizon_features(series,calendar,2);y,dates,train,val,final=horizon_masks(series,calendar,2)
    info=dict(node=node,horizon=2,n_train=int(train.sum()),n_validation=int(val.sum()),n_final=int(final.sum()),
              train_label_max=str(dates[train].max()),validation_label_max=str(dates[val].max()),
              validation_origin_min=str(series.index[val].min()),final_label_max=str(dates[final].max()),models={})
    result=pd.DataFrame(dict(decision_date=origins,node=node))
    prediction=np.full(len(origins),np.nan);details=dict(status='INSUFFICIENT_HISTORICAL_LABELS',candidates=[])
    if train.sum()>=80 and val.sum()>=20:
        best=None;best_loss=(np.inf,np.inf)
        for alpha in S2.ALPHAS:
            pipe=S2.ridge_pipeline(alpha).fit(X.loc[train],y.loc[train]);rmse,mae=S2.score(pipe.predict(X.loc[val]),y.loc[val])
            details['candidates'].append(dict(alpha=alpha,validation_RMSE=rmse,validation_MAE=mae,
                preprocessing=S2.preprocessing_evidence(pipe,X.loc[train])))
            if (rmse,mae)<best_loss: best,best_loss=alpha,(rmse,mae)
        pipe=S2.ridge_pipeline(best).fit(X.loc[final],y.loc[final]);prediction=pipe.predict(X.reindex(origins)).astype(float)
        joblib.dump(pipe,cache/f'{node}_ridge.joblib');details.update(status='OK',alpha=best,
            final_preprocessing=S2.preprocessing_evidence(pipe,X.loc[final]))
    result['forecast_ridge_h2']=prediction;info['models']['ridge']=details
    prediction=np.full(len(origins),np.nan);details=dict(status='INSUFFICIENT_HISTORICAL_LABELS',candidates=[])
    historical=series.loc[:TRAIN_END]
    validation_origins=series.index[(series.index>=TRAIN_END)&(dates>TRAIN_END)&(dates<=FIT_END)&np.isfinite(y)]
    if np.isfinite(historical).sum()>=60 and len(validation_origins)>=20:
        best=None;best_loss=(np.inf,np.inf)
        for order in S2.ORDERS:
            candidate=cache/f'{node}_arima_{order[0]}{order[1]}{order[2]}.json'
            if candidate.exists(): row=json.loads(candidate.read_text(encoding='utf-8'))
            else:
                row=dict(order=order,status='ERROR')
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always');fitted=S2.arima_model(historical.to_numpy(),order).fit(method_kwargs={'maxiter':200})
                    converged=bool(fitted.mle_retvals.get('converged',False))
                    vp=arima_native(series.loc[:FIT_END],order,fitted.params,validation_origins,2)
                    rmse,mae=S2.score(vp,y.reindex(validation_origins))
                    row.update(status='OK' if converged and np.isfinite(rmse) else 'REJECTED_NONCONVERGENCE_OR_NONFINITE',
                        converged=converged,validation_RMSE=rmse,validation_MAE=mae,warnings=sorted(set(str(w.message) for w in caught)))
                except Exception as exc: row['error']=str(exc)
                write_json(row,candidate)
            details['candidates'].append(row)
            if row['status']=='OK' and (row['validation_RMSE'],row['validation_MAE'])<best_loss:
                best,best_loss=order,(row['validation_RMSE'],row['validation_MAE'])
        if best is None: details['status']='NO_CONVERGED_CANDIDATE'
        else:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');fitted=S2.arima_model(series.loc[:FIT_END].to_numpy(),best).fit(method_kwargs={'maxiter':200})
                converged=bool(fitted.mle_retvals.get('converged',False))
                details.update(order=best,params=fitted.params.tolist(),converged=converged,
                               warnings=sorted(set(str(w.message) for w in caught)))
                if converged:
                    prediction=arima_native(series,best,fitted.params,origins,2);details['status']='OK'
                else: details['status']='FINAL_NONCONVERGENCE'
            except Exception as exc: details.update(status='FINAL_ERROR',error=str(exc))
    result['forecast_arima_h2']=prediction;info['models']['arima']=details
    info['fit_seconds']=time.perf_counter()-started;save(result,archive);write_json(info,meta)
    return info,False

def candidates(frame):
    out=frame.copy();current=np.isfinite(out.observed_iv_at_decision)
    out['observed_iv_available']=current
    for m in MODELS:
        for h in [1,2]: out[f'available_{m}_h{h}']=np.isfinite(out[f'forecast_{m}_h{h}'])
        out[f'candidate_{m}']=current&out[f'available_{m}_h1']&out[f'available_{m}_h2']
        out[f'forecast_{m}_holding_change']=out[f'forecast_{m}_h2']-out[f'forecast_{m}_h1']
        out[f'available_{m}_holding_change']=np.isfinite(out[f'forecast_{m}_holding_change'])
    out['common_economic_candidate']=out[[f'candidate_{m}' for m in MODELS]].all(axis=1)
    return out

def rank_signals(frame):
    """Explicit whitelist excludes all future outcomes/quotes/availability."""
    inputs=['decision_date','entry_date','intended_exit_date','node','log_moneyness','T','observed_iv_at_decision',
            'moneyness_region','maturity_region','common_economic_candidate']
    results=[]
    for m in MODELS:
        out=frame[inputs].copy();out['model']=m;out['raw_signal']=frame[f'forecast_{m}_holding_change']
        out['z_score']=np.nan;out['signal']=0;out['selected']=False;out['selection_rank']=pd.Series(pd.NA,index=out.index,dtype='Int64')
        for _,day in out[out.common_economic_candidate].groupby('decision_date',sort=True):
            sd=day.raw_signal.std()
            z=(day.raw_signal-day.raw_signal.mean())/sd if sd>0 else pd.Series(0.,index=day.index)
            out.loc[day.index,'z_score']=z
            for sign in [1,-1]:
                ix=z.index[z.ge(.5) if sign==1 else z.le(-.5)]
                ranked=out.loc[ix].sort_values(['z_score','node'],ascending=[sign==-1,True],kind='stable')
                out.loc[ranked.index,'selection_rank']=np.arange(1,len(ranked)+1)
                chosen=ranked.head(10).index;out.loc[chosen,'signal']=sign;out.loc[chosen,'selected']=True
        results.append(out)
    return pd.concat(results,ignore_index=True)

def predictive_diagnostics(forecasts,signals):
    """Outcomes are joined AFTER selection; incomplete selections remain."""
    rows=[]
    selected_days={(model,date):day for (model,date),day in signals[signals.selected].groupby(['model','decision_date'],sort=True)}
    for date,day in forecasts[forecasts.common_economic_candidate].groupby('decision_date',sort=True):
        scored=day[np.isfinite(day.actual_iv_entry)&np.isfinite(day.actual_iv_exit)]
        for m in MODELS:
            x=scored[f'forecast_{m}_holding_change'];y=scored.actual_holding_change
            valid=len(scored)>=MIN_IC and x.nunique()>1 and y.nunique()>1
            ic=float(spearmanr(x,y).statistic) if valid else np.nan
            selections=selected_days.get((m,date),signals.iloc[:0])
            realized=selections[np.isfinite(selections.actual_holding_change)]
            long=realized[realized.signal.eq(1)];short=realized[realized.signal.eq(-1)]
            full_sides=(len(long)>0 and len(short)>0)
            long_hit=long.actual_holding_change.gt(0).mean() if len(long) else np.nan
            short_hit=short.actual_holding_change.lt(0).mean() if len(short) else np.nan
            rows.append(dict(model=m,decision_date=date,n_common_candidates=len(day),n_scored_nodes=len(scored),rank_ic=ic,
                n_selected_long=int(selections.signal.eq(1).sum()),n_selected_short=int(selections.signal.eq(-1).sum()),
                n_scored_long=len(long),n_scored_short=len(short),long_mean_change=long.actual_holding_change.mean(),
                short_mean_change=short.actual_holding_change.mean(),top_minus_bottom=long.actual_holding_change.mean()-short.actual_holding_change.mean() if full_sides else np.nan,
                long_hit_rate=long_hit,short_hit_rate=short_hit,balanced_hit_rate=(long_hit+short_hit)/2 if full_sides else np.nan))
    daily=pd.DataFrame(rows);summary=[]
    for m in MODELS:
        a=daily[daily.model.eq(m)];ic=a.rank_ic.dropna();spread=a.top_minus_bottom.dropna()
        hi=S2.hac_mean(ic);hs=S2.hac_mean(spread)
        selected=signals[signals.model.eq(m)&signals.selected&np.isfinite(signals.actual_holding_change)]
        long=selected[selected.signal.eq(1)];short=selected[selected.signal.eq(-1)]
        lh=long.actual_holding_change.gt(0).mean();sh=short.actual_holding_change.lt(0).mean()
        summary.append(dict(model=m,mean_IC=ic.mean(),median_IC=ic.median(),std_IC=ic.std(),valid_IC_dates=len(ic),
            IC_percent_positive=100*ic.gt(0).mean(),IC_NW_t=hi['statistic'],IC_p_value=hi['p_value'],
            mean_top_minus_bottom=spread.mean(),median_top_minus_bottom=spread.median(),valid_spread_dates=len(spread),
            spread_NW_t=hs['statistic'],spread_p_value=hs['p_value'],spread_percent_positive=100*spread.gt(0).mean(),
            long_hit_rate=lh,short_hit_rate=sh,balanced_hit_rate=(lh+sh)/2,mean_daily_balanced_hit=a.balanced_hit_rate.mean(),
            n_scored_long=len(long),n_scored_short=len(short),hac_lag=5))
    return daily,pd.DataFrame(summary)

def net_contracts(trades):
    mapped=trades[trades.entry_mapping_status.eq('MAPPED')&trades.weight_at_entry.ne(0)].copy()
    rows=[]
    for (m,date,key),day in mapped.groupby(['model','entry_date','contract_key'],sort=True):
        row=day.iloc[0].to_dict()
        # Node coordinates/signals are not contract-level identity; retain an
        # explicit list and aggregate signed weight without inventing a node.
        for col in ['node','signal','raw_signal','z_score','selection_rank','selected','T','log_moneyness','moneyness_region','maturity_region',
                    'observed_iv_at_decision','actual_iv_entry','actual_iv_exit','actual_holding_change','common_economic_candidate']:
            row.pop(col,None)
        row.update(model=m,entry_date=date,contract_key=key,weight_at_entry=float(day.weight_at_entry.sum()),
                   n_mapped_nodes=len(day),mapped_nodes=';'.join(sorted(day.node)),
                   node_gross_weight=float(day.weight_at_entry.abs().sum()),
                   opposing_signals=bool(day.signal.nunique()>1))
        row['nonzero_net_position']=abs(row['weight_at_entry'])>1e-12
        row['net_signal']=int(np.sign(row['weight_at_entry'])) if row['nonzero_net_position'] else 0
        row['netting_status']='ACTIVE' if row['nonzero_net_position'] else 'FULLY_CANCELLED'
        if 'entry_ask' in row and 'entry_bid' in row:
            row['entry_execution_price']=(row['entry_ask'] if row['net_signal']>0 else row['entry_bid']) if row['nonzero_net_position'] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def collision_table(trades,net):
    rows=[]
    for (m,date),day in trades.groupby(['model','entry_date'],sort=True):
        mapped=day[day.entry_mapping_status.eq('MAPPED')&day.weight_at_entry.ne(0)]
        counts=mapped.groupby('contract_key').size();contracts=net[net.model.eq(m)&net.entry_date.eq(date)]
        rows.append(dict(model=m,entry_date=date,selected_nodes=len(day),entry_mapped_nodes=len(mapped),unique_mapped_contracts=len(counts),
            duplicated_mappings=len(mapped)-len(counts),collision_rate=(len(mapped)-len(counts))/len(mapped) if len(mapped) else np.nan,
            max_nodes_one_contract=int(counts.max()) if len(counts) else 0,
            contracts_receiving_both_signals=int(contracts.opposing_signals.sum()) if len(contracts) else 0,
            gross_weight_before_netting=mapped.weight_at_entry.abs().sum(),net_signed_weight=contracts.weight_at_entry.sum(),
            gross_weight_after_netting=contracts.weight_at_entry.abs().sum(),active_net_contracts=int(contracts.nonzero_net_position.sum()) if len(contracts) else 0))
    return pd.DataFrame(rows)



def quote_valid(frame,bid,ask):
    bids=pd.to_numeric(frame[bid],errors='coerce');asks=pd.to_numeric(frame[ask],errors='coerce')
    return np.isfinite(bids)&np.isfinite(asks)&bids.gt(0)&asks.ge(bids)
