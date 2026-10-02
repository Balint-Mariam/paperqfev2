"""Exact-contract exit books and execution eligibility."""
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
from . import economic_signals as S3
S2=S3.S2; MODELS=S3.MODELS; KEYS=['intended_exit_date','contract_key']
table=S2.table


def calendar_guard(dates):
    """Identical Stage1B/3 rule: XNYS open at the nominal 15:45 snapshot."""
    dates=pd.to_datetime(dates)
    cal=S3.S1.xc.get_calendar('XNYS',start=dates.min()-pd.Timedelta(days=15),end=dates.max()+pd.Timedelta(days=15))
    schedule=cal.schedule.loc[dates.min():dates.max()];close=schedule['close'].dt.tz_convert('America/New_York')
    allowed=schedule.index[(close.dt.hour.gt(15)|(close.dt.hour.eq(15)&close.dt.minute.ge(45))).to_numpy()]
    return dates.isin(allowed)

def valid_two_sided(bid,ask):
    b=pd.to_numeric(bid,errors='coerce');a=pd.to_numeric(ask,errors='coerce')
    return np.isfinite(b)&np.isfinite(a)&b.gt(0)&a.ge(b)

def choose_quotes(required,processed,raw,guard):
    """Eligibility whitelist: exact key/date, calendar, finite positive book.

    No volume, OI, IV, rates, spread, entry price, return or PnL is consulted.
    Diagnostic fields are added after BASE source/eligibility are fixed.
    """
    out=required[KEYS].copy()
    for label,frame,bid,ask in [('processed',processed,'bid','ask'),('raw',raw,'bid_1545','ask_1545')]:
        quotes=frame[['quote_date','contract_key',bid,ask]].copy().rename(columns={
            'quote_date':'intended_exit_date',bid:f'{label}_bid',ask:f'{label}_ask'})
        quotes[f'{label}_available']=True;quotes[f'{label}_quote_date']=quotes.intended_exit_date
        quotes[f'{label}_contract_key']=quotes.contract_key
        out=out.merge(quotes,on=KEYS,how='left',validate='one_to_one')
        out[f'{label}_available']=out[f'{label}_available'].astype('boolean').fillna(False).astype(bool)
    out['calendar_guard_pass']=np.asarray(guard,dtype=bool)
    p=out.processed_available&out.calendar_guard_pass&valid_two_sided(out.processed_bid,out.processed_ask)
    r=out.raw_available&out.calendar_guard_pass&valid_two_sided(out.raw_bid,out.raw_ask)
    out['processed_valid']=p;out['raw_valid']=r
    recovered=~p&r
    out['exit_quote_source']=np.select([p,recovered],['PROCESSED_1545','RAW_1545_RECOVERED'],default='NO_VALID_1545_QUOTE')
    out['BASE']=p|recovered
    out['exit_bid']=np.where(p,out.processed_bid,np.where(recovered,out.raw_bid,np.nan))
    out['exit_ask']=np.where(p,out.processed_ask,np.where(recovered,out.raw_ask,np.nan))
    out['exit_mid']=(out.exit_bid+out.exit_ask)/2
    out['exit_quote_date']=out.intended_exit_date.where(out.BASE);out['exit_contract_key']=out.contract_key.where(out.BASE)
    out['source_file']=np.select([p,recovered],['options_eod_all_with_iv.csv','options_eod_all.csv'],default='')
    out['quote_interpretation']='observed two-sided quote; quote-based liquidation proxy'
    # Raw was scanned only for clean-absent pairs; absence from this compact
    # cache on a processed pair is NOT evidence of absence from the raw archive.
    out['raw_cache_checked']=out.raw_available
    diagnostics=[]
    for label,frame in [('processed',processed),('raw',raw)]:
        cols=['quote_date','contract_key']+[c for c in ['trade_volume','open_interest','observable_failure_reasons','spot','source_observation_count'] if c in frame]
        d=frame[cols].rename(columns={'quote_date':'intended_exit_date',**{c:label+'_'+c for c in cols if c not in ['quote_date','contract_key']}})
        diagnostics.append(d)
    for d in diagnostics: out=out.merge(d,on=KEYS,how='left',validate='one_to_one')
    for c in ['raw_trade_volume','raw_open_interest','processed_trade_volume','processed_open_interest','raw_observable_failure_reasons']:
        if c not in out: out[c]=np.nan
    out['relative_spread']=(out.exit_ask-out.exit_bid)/out.exit_mid
    out['raw_relative_spread']=(out.raw_ask-out.raw_bid)/((out.raw_bid+out.raw_ask)/2)
    out['zero_volume_flag']=out.raw_trade_volume.eq(0);out['zero_open_interest_flag']=out.raw_open_interest.eq(0)
    out['nonpositive_volume_flag']=out.raw_trade_volume.le(0);out['nonpositive_open_interest_flag']=out.raw_open_interest.le(0)
    out['exit_quote_trade_volume']=np.where(p,out.processed_trade_volume,np.where(recovered,out.raw_trade_volume,np.nan))
    out['STRICT_20']=out.BASE&out.relative_spread.le(.2)
    out['STRICT_10']=out.BASE&out.relative_spread.le(.1)
    out['POSITIVE_VOLUME']=out.BASE&out.exit_quote_trade_volume.gt(0)
    return out

def reason_category(value):
    reasons=set(str(value).split(';')) if pd.notna(value) else set()
    reasons.discard('NO_OBSERVABLE_DEFAULT_FAILURE_REASON');reasons.discard('')
    if not reasons: return 'NO_OBSERVABLE_REASON'
    if len(reasons)>1: return 'MULTIPLE_REASONS'
    reason=next(iter(reasons))
    if reason=='NONPOSITIVE_VOLUME': return 'NONPOSITIVE_VOLUME_ONLY'
    if reason=='SPREAD_ABOVE_DEFAULT_MAX': return 'SPREAD_FILTER_ONLY'
    if reason=='NO_EXACT_RATE_DATE' or any(k in reason for k in ['RATE','IV_INPUT','INVALID_SPOT']): return 'RATE_OR_IV_INPUT_RELATED'
    if reason in ['MONEYNESS_OUTSIDE_DEFAULT_WINDOW','T_OUTSIDE_DEFAULT_WINDOW']: return 'MONEYNESS_OR_T_FILTER'
    if reason=='NONPOSITIVE_OPEN_INTEREST': return 'NONPOSITIVE_OPEN_INTEREST_ONLY'
    return 'OTHER_OBSERVABLE_REASON'

def join_trades(original,quotes):
    """Retain all entry/netting fields and rows; change exit fields only."""
    out=original.copy()
    for col in ['preferred_exit_source','exact_exit_available','exit_status','exit_bid','exit_ask','exit_mid','exit_spot','source_file','exit_quote_date','exit_contract_key']:
        if col in out: out['stage3_'+col]=out[col]
    replacement=[c for c in quotes if c not in KEYS]
    out=out.drop(columns=[c for c in replacement if c in out])
    out=out.merge(quotes,on=KEYS,how='left',validate='many_to_one')
    if out.BASE.isna().any(): raise ValueError('Entered contract has no required-pair quote row')
    out['preferred_exit_source']=out.exit_quote_source
    out['exact_exit_available']=out.BASE
    out['exit_status']=np.where(out.BASE,'VALID_EXACT_1545_QUOTE','NO_VALID_1545_QUOTE')
    out['exit_spot']=np.where(out.exit_quote_source.eq('PROCESSED_1545'),out.get('processed_spot',np.nan),
        np.where(out.exit_quote_source.eq('RAW_1545_RECOVERED'),out.get('raw_spot',np.nan),np.nan))
    return out

def coverage_table(trades):
    rows=[]
    for model,day in trades[trades.nonzero_net_position].groupby('model',sort=True):
        n=len(day);p=int(day.exit_quote_source.eq('PROCESSED_1545').sum());r=int(day.exit_quote_source.eq('RAW_1545_RECOVERED').sum())
        row=dict(model=model,entered_active_net_contracts=n,processed_valid_exits=p,processed_only_coverage_pct=100*p/n,
            raw_additional_recovered_exits=r,total_valid_BASE_exits=int(day.BASE.sum()),remaining_missing=int((~day.BASE).sum()),
            BASE_coverage_pct=100*day.BASE.mean(),remaining_missing_pct=100*(~day.BASE).mean())
        for subset in ['STRICT_20','STRICT_10','POSITIVE_VOLUME']:
            row[subset+'_valid_exits']=int(day[subset].sum());row[subset+'_coverage_pct']=100*day[subset].mean()
        rec=day[day.exit_quote_source.eq('RAW_1545_RECOVERED')]
        row['recovered_zero_volume_pct']=100*rec.zero_volume_flag.mean()
        row['recovered_nonpositive_volume_only_pct']=100*rec.recovery_reason_category.eq('NONPOSITIVE_VOLUME_ONLY').mean()
        rows.append(row)
    return pd.DataFrame(rows)

def recovery_reasons(quotes,trades):
    rows=[];categories=['NONPOSITIVE_VOLUME_ONLY','SPREAD_FILTER_ONLY','RATE_OR_IV_INPUT_RELATED','MONEYNESS_OR_T_FILTER',
                       'MULTIPLE_REASONS','NO_OBSERVABLE_REASON','NONPOSITIVE_OPEN_INTEREST_ONLY','OTHER_OBSERVABLE_REASON']
    samples=[('unique_required_pairs','ALL',quotes[quotes.exit_quote_source.eq('RAW_1545_RECOVERED')])]
    samples += [('active_net_contracts',model,day[day.exit_quote_source.eq('RAW_1545_RECOVERED')])
                for model,day in trades[trades.nonzero_net_position].groupby('model')]
    for level,model,day in samples:
        for category in categories:
            n=int(day.recovery_reason_category.eq(category).sum())
            rows.append(dict(level=level,model=model,category=category,count=n,n_recovered=len(day),percent=100*n/len(day) if len(day) else np.nan))
    return pd.DataFrame(rows)

def quality_table(quotes,trades):
    rows=[]
    samples=[('unique_required_pairs','ALL',quotes)] + [('active_net_contracts',m,d) for m,d in trades[trades.nonzero_net_position].groupby('model')]
    for level,model,day in samples:
        for source in ['PROCESSED_1545','RAW_1545_RECOVERED','NO_VALID_1545_QUOTE']:
            a=day[day.exit_quote_source.eq(source)];spread=a.relative_spread.dropna()
            rows.append(dict(level=level,model=model,source=source,n=len(a),mean_relative_spread=spread.mean(),median_relative_spread=spread.median(),
                q90_relative_spread=spread.quantile(.9),max_relative_spread=spread.max(),n_spread_gt_20=int(spread.gt(.2).sum()),
                n_spread_gt_10=int(spread.gt(.1).sum()),n_raw_volume_observed=int(a.raw_trade_volume.notna().sum()),
                n_raw_zero_volume=int(a.zero_volume_flag.sum()),n_raw_zero_OI=int(a.zero_open_interest_flag.sum()),
                n_calendar_rejected=int((~a.calendar_guard_pass).sum())))
    return pd.DataFrame(rows)

def missingness_table(trades):
    frame=trades[trades.nonzero_net_position].copy();grid=pd.read_csv(ROOT/'iv_grid_map.csv')
    frame['side']=np.where(frame.net_signal>0,'LONG','SHORT');frame['year_month']=frame.intended_exit_date.dt.strftime('%Y-%m')
    for col,bucket,labels in [('T','maturity_tercile',['short','medium','long']),('log_moneyness','moneyness_tercile',['lower','central','upper'])]:
        groups=np.array_split(np.sort(grid[col].unique()),3);x=frame['entry_'+col]
        frame[bucket]=np.select([x.le(max(groups[0])),x.le(max(groups[1]))],labels[:2],default=labels[2])
    rows=[]
    for dimension in ['model','side','option_type','maturity_tercile','moneyness_tercile','year_month']:
        group=['model'] if dimension=='model' else ['model',dimension]
        for key,day in frame.groupby(group,sort=True,dropna=False):
            key=key if isinstance(key,tuple) else (key,)
            rows.append(dict(model=key[0],dimension=dimension,bucket=str(key[-1]),n_active=len(day),
                n_processed=int(day.processed_valid.sum()),n_raw_recovered=int(day.exit_quote_source.eq('RAW_1545_RECOVERED').sum()),
                raw_recovery_pct=100*day.exit_quote_source.eq('RAW_1545_RECOVERED').mean(),n_remaining_missing=int((~day.BASE).sum()),
                remaining_missing_pct=100*(~day.BASE).mean(),BASE_coverage_pct=100*day.BASE.mean()))
    return pd.DataFrame(rows)
