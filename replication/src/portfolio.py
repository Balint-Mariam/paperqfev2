"""Contract PnL, premiums, and daily performance."""
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
import statsmodels.api as sm
from . import execution_quotes as B
table=B.table
MODELS=B.MODELS; GROUP=['model','entry_date']; KEY=GROUP+['contract_key']
LAG=5; SEED=42; REPS=4000; BLOCKS=(5,10)
FIGURE_NAMES=['cumulative_mid_performance.png','cumulative_bidask_performance.png','mid_vs_bidask_by_model.png','transaction_cost_wedge_timeseries.png']


def economic_calendar(frame):
    """Whitelist ONLY chronology dates: one decision and two nominal snapshots."""
    dates=frame[['decision_date','entry_date','intended_exit_date']].copy()
    lo=dates.min().min()-pd.Timedelta(days=15);hi=dates.max().max()+pd.Timedelta(days=15)
    cal=B.S3.S1.xc.get_calendar('XNYS',start=lo,end=hi);schedule=cal.schedule
    close=schedule['close'].dt.tz_convert('America/New_York')
    snapshot=schedule.index[(close.dt.hour.gt(15)|(close.dt.hour.eq(15)&close.dt.minute.ge(45))).to_numpy()]
    out=pd.DataFrame(index=frame.index)
    out['decision_session']=dates.decision_date.isin(schedule.index)
    out['entry_snapshot_session']=dates.entry_date.isin(snapshot)
    out['exit_snapshot_session']=dates.intended_exit_date.isin(snapshot)
    out['economic_calendar_eligible']=out.all(axis=1)
    return out

def contract_pnl(frame):
    out=frame.copy();w=out.weight_at_entry
    out['signed_weight']=w;out['exit_date']=out.intended_exit_date
    out['entry_mid']=(out.entry_bid+out.entry_ask)/2;out['exit_mid']=(out.exit_bid+out.exit_ask)/2
    out['entry_execution_price']=np.where(w>0,out.entry_ask,out.entry_bid)
    out['exit_execution_price']=np.where(w>0,out.exit_bid,out.exit_ask)
    out['pnl_mid']=w*(out.exit_mid-out.entry_mid)
    out['pnl_exec']=w*(out.exit_execution_price-out.entry_execution_price)
    out['gross_entry_premium_mid_component']=w.abs()*out.entry_mid
    out['gross_entry_premium_exec_component']=w.abs()*out.entry_execution_price
    out['relative_entry_spread']=(out.entry_ask-out.entry_bid)/out.entry_mid
    out['relative_exit_spread']=(out.exit_ask-out.exit_bid)/out.exit_mid
    out['entry_spread_cost']=w.abs()*(out.entry_ask-out.entry_bid)/2
    out['exit_spread_cost']=w.abs()*(out.exit_ask-out.exit_bid)/2
    out['cost_wedge_pnl']=out.pnl_mid-out.pnl_exec
    return out

def daily_pnl(contracts,eligible_net,selection):
    daily=contracts.groupby(GROUP,sort=True).agg(decision_date=('decision_date','first'),exit_date=('exit_date','first'),
        PnL_mid=('pnl_mid','sum'),PnL_exec=('pnl_exec','sum'),
        GrossPremium_mid=('gross_entry_premium_mid_component','sum'),GrossPremium_exec=('gross_entry_premium_exec_component','sum'),
        number_contracts=('contract_key','size'),number_raw_recovered_exits=('exit_quote_source',lambda x:int(x.eq('RAW_1545_RECOVERED').sum())),
        number_processed_exits=('exit_quote_source',lambda x:int(x.eq('PROCESSED_1545').sum())),
        net_gross_weight=('signed_weight',lambda x:x.abs().sum()),net_signed_weight=('signed_weight','sum'),
        entry_spread_cost=('entry_spread_cost','sum'),exit_spread_cost=('exit_spread_cost','sum')).reset_index()
    pre=eligible_net.groupby(GROUP).agg(number_unique_mapped_contracts=('contract_key','size'),
        number_mapped_nodes=('n_mapped_nodes','sum'),gross_pre_net_signal_weight=('node_gross_weight','sum')).reset_index()
    daily=daily.merge(pre,on=GROUP,validate='one_to_one').merge(selection[['model','entry_date','number_selected_nodes']],on=GROUP,validate='one_to_one')
    for suffix in ['mid','exec']: daily['Return_'+suffix]=daily['PnL_'+suffix]/daily['GrossPremium_'+suffix]
    daily['cost_wedge_pnl']=daily.PnL_mid-daily.PnL_exec;daily['cost_wedge_return']=daily.Return_mid-daily.Return_exec
    daily['cost_wedge']=daily.cost_wedge_pnl;daily['trading_day']=daily.entry_date
    return daily

def hac(values):
    x=np.asarray(values,dtype=float)
    if len(x)<LAG+2 or not np.isfinite(x).all(): raise ValueError('Invalid daily inference sample')
    fit=sm.OLS(x,np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds={'maxlags':LAG,'kernel':'bartlett','use_correction':True},use_t=False)
    return dict(mean_daily_return=float(fit.params[0]),HAC_standard_error=float(fit.bse[0]),HAC_t_stat=float(fit.tvalues[0]),HAC_p_value=float(fit.pvalues[0]),HAC_lag=LAG,n_inference_dates=len(x))

def bootstrap(values,block):
    x=np.asarray(values,dtype=float);n=len(x);rng=np.random.default_rng(SEED)
    # Fixed circular blocks of entire consecutive eligible portfolio dates.
    starts=rng.integers(0,n,size=(REPS,int(np.ceil(n/block))))
    idx=((starts[:,:,None]+np.arange(block))%n).reshape(REPS,-1)[:,:n]
    samples=x[idx];means=samples.mean(axis=1);sd=samples.std(axis=1,ddof=1)
    sharpes=np.divide(np.sqrt(252)*means,sd,out=np.full(REPS,np.nan),where=sd>0)
    a,b=np.quantile(means,[.025,.975]);c,d=np.nanquantile(sharpes,[.025,.975])
    return dict(bootstrap_mean_CI_low=float(a),bootstrap_mean_CI_high=float(b),bootstrap_sharpe_CI_low=float(c),bootstrap_sharpe_CI_high=float(d),
        bootstrap_block_length=block,bootstrap_repetitions=REPS,bootstrap_seed=SEED)

def wealth(values):
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).all() or np.any(x<=-1): return None,np.nan,'INVALID_RETURN_LE_MINUS_ONE_OR_NONFINITE'
    with np.errstate(over='ignore',under='ignore'):
        w=np.exp(np.cumsum(np.log1p(x)))
    if not np.isfinite(w).all() or np.any(w<=0): return None,np.nan,'NUMERICALLY_UNSTABLE_WEALTH'
    peak=np.maximum.accumulate(np.r_[1.,w])[1:]
    return w,float(np.min(w/peak-1)),'MATHEMATICALLY_ADMISSIBLE_PREMIUM_EXPOSURE_INDEX'

def performance_tables(daily,contracts):
    rows=[];sens=[];out=daily.copy()
    for model,d in daily.groupby('model',sort=True):
        d=d.sort_values('entry_date')
        for impl,suffix in [('MID','mid'),('BID_ASK','exec')]:
            x=d['Return_'+suffix].to_numpy();sd=x.std(ddof=1);w,dd,status=wealth(x)
            out.loc[d.index,'cumulative_arithmetic_'+suffix]=np.cumsum(x)
            out.loc[d.index,'wealth_'+suffix]=w if w is not None else np.nan
            row=dict(model=model,implementation=impl,n_days=len(x),n_contracts=int(d.number_contracts.sum()),
                n_distinct_contract_keys=int(contracts.loc[contracts.model.eq(model),'contract_key'].nunique()),
                median_daily_return=float(np.median(x)),daily_vol=float(sd),annualized_mean=float(252*x.mean()),
                annualized_vol=float(np.sqrt(252)*sd),annualized_sharpe=float(np.sqrt(252)*x.mean()/sd) if sd>0 else np.nan,
                cumulative_normalized_return=float(x.sum()),compounded_normalized_return=float(w[-1]-1) if w is not None else np.nan,
                positive_day_pct=float(100*np.mean(x>0)),max_drawdown=dd,worst_daily_return=float(x.min()),best_daily_return=float(x.max()),
                wealth_index_status=status,total_pnl=float(d['PnL_'+suffix].sum()),**hac(x),**bootstrap(x,BLOCKS[0]))
            rows.append(row)
            for block in BLOCKS: sens.append(dict(model=model,implementation=impl,**bootstrap(x,block)))
    return pd.DataFrame(rows),pd.DataFrame(sens),out

def pairwise(daily):
    rows=[]
    for impl,suffix in [('MID','mid'),('BID_ASK','exec')]:
        wide=daily.pivot(index='entry_date',columns='model',values='Return_'+suffix)
        for a,b in [('arima','ridge'),('arima','xgboost'),('ridge','xgboost')]:
            aligned=wide[[a,b]].dropna();h=hac(aligned[a]-aligned[b]);h['mean_difference']=h.pop('mean_daily_return')
            rows.append(dict(model_A=a,model_B=b,implementation=impl,**h))
    return pd.DataFrame(rows)

def wedge_table(daily):
    rows=[]
    for model,d in daily.groupby('model'):
        mid=d.PnL_mid.sum();entry=d.entry_spread_cost.sum();exit=d.exit_spread_cost.sum()
        rows.append(dict(model=model,n_days=len(d),mean_pnl_wedge=d.cost_wedge_pnl.mean(),median_pnl_wedge=d.cost_wedge_pnl.median(),
            cumulative_pnl_wedge=d.cost_wedge_pnl.sum(),mean_return_wedge=d.cost_wedge_return.mean(),median_return_wedge=d.cost_wedge_return.median(),
            cumulative_arithmetic_return_wedge=d.cost_wedge_return.sum(),total_mid_pnl=mid,total_exec_pnl=d.PnL_exec.sum(),
            fraction_midpoint_economic_value_lost=d.cost_wedge_pnl.sum()/mid if mid>0 else np.nan,
            midpoint_loss_fraction_status='POSITIVE_AGGREGATE_MID_PNL' if mid>0 else 'UNDEFINED_NONPOSITIVE_AGGREGATE_MID_PNL',
            entry_spread_contribution=entry,exit_spread_contribution=exit,
            entry_cost_fraction=entry/(entry+exit) if entry+exit>0 else np.nan))
    return pd.DataFrame(rows)

def contribution_table(contracts,quality=False):
    rows=[]
    for model,d in contracts.groupby('model'):
        if quality:
            spread=d.relative_exit_spread
            masks={'spread_le_10':spread.le(.1),'spread_10_to_20':spread.gt(.1)&spread.le(.2),'spread_gt_20':spread.gt(.2),
                'zero_volume_raw_recovered':d.exit_quote_source.eq('RAW_1545_RECOVERED')&d.raw_trade_volume.eq(0),
                'positive_volume_chosen_quote':d.exit_quote_trade_volume.gt(0),
                'STRICT_20':d.STRICT_20,'STRICT_10':d.STRICT_10,'POSITIVE_VOLUME':d.POSITIVE_VOLUME}
        else: masks={s:d.exit_quote_source.eq(s) for s in ['PROCESSED_1545','RAW_1545_RECOVERED']}
        for bucket,mask in masks.items():
            a=d[mask];mid=a.pnl_mid.sum();ex=a.pnl_exec.sum()
            rows.append(dict(model=model,bucket=bucket,n_contracts=len(a),contract_share_pct=100*len(a)/len(d),
                pnl_mid=mid,pnl_exec=ex,pnl_mid_share_of_total=mid/d.pnl_mid.sum() if d.pnl_mid.sum()!=0 else np.nan,
                pnl_exec_share_of_total=ex/d.pnl_exec.sum() if d.pnl_exec.sum()!=0 else np.nan,
                entry_spread_cost=a.entry_spread_cost.sum(),exit_spread_cost=a.exit_spread_cost.sum(),
                interpretation='ex-post quote-quality PnL contribution; unchanged entry weights; not investable subset returns' if quality else 'BASE source PnL contribution; unchanged strategy'))
    return pd.DataFrame(rows)

def signal_relation(daily,signals):
    diag=pd.read_parquet(HERE/'data/economic_signal_daily_diagnostics.parquet')
    disp=signals[signals.common_economic_candidate].groupby(['model','decision_date']).agg(
        predicted_cross_sectional_dispersion=('raw_signal',lambda x:x.std(ddof=1)),predicted_cross_sectional_range=('raw_signal',lambda x:x.max()-x.min())).reset_index()
    joined=daily.merge(diag,on=['model','decision_date'],how='left',validate='one_to_one').merge(disp,on=['model','decision_date'],how='left',validate='one_to_one')
    rows=[]
    for model,d in joined.groupby('model'):
        for a,b in [('top_minus_bottom','Return_mid'),('top_minus_bottom','Return_exec'),('Return_mid','Return_exec'),
                    ('predicted_cross_sectional_dispersion','Return_mid'),('predicted_cross_sectional_dispersion','Return_exec')]:
            x=d[[a,b]].replace([np.inf,-np.inf],np.nan).dropna()
            rows.append(dict(model=model,variable_A=a,variable_B=b,n_dates=len(x),pearson=x[a].corr(x[b]),spearman=x[a].corr(x[b],method='spearman')))
    return joined,pd.DataFrame(rows)

def figures(daily):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'arima':'#333333','ridge':'#0072B2','xgboost':'#D55E00'}
    def finish(fig,name):
        fig.tight_layout();fig.savefig(HERE/'figures'/name,dpi=300,bbox_inches='tight');plt.close(fig)
    for suffix,name in [('mid','cumulative_mid_performance.png'),('exec','cumulative_bidask_performance.png')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for model,d in daily.groupby('model'):
            d=d.sort_values('entry_date');ax.plot(d.exit_date,d['cumulative_arithmetic_'+suffix],label=model.upper(),color=colors[model])
        ax.axhline(0,color='gray',lw=.6);ax.set_ylabel('Cumulative arithmetic return per gross entry premium')
        ax.set_xlabel('Intended exit date');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);finish(fig,name)
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,model in zip(axes,MODELS):
        d=daily[daily.model.eq(model)].sort_values('entry_date')
        for suffix,label,color in [('mid','MID','#333333'),('exec','BID/ASK','#0072B2')]: ax.plot(d.exit_date,d['cumulative_arithmetic_'+suffix],label=label,color=color)
        ax.set_title(model.upper());ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Cumulative arithmetic normalized return');axes[-1].legend(frameon=False);finish(fig,'mid_vs_bidask_by_model.png')
    fig,ax=plt.subplots(figsize=(8,4.5))
    for model,d in daily.groupby('model'):
        d=d.sort_values('entry_date');ax.plot(d.exit_date,d.cost_wedge_return,label=model.upper(),color=colors[model],lw=.8,alpha=.8)
    ax.set_ylabel('Daily MID minus BID/ASK normalized return');ax.set_xlabel('Intended exit date');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    finish(fig,'transaction_cost_wedge_timeseries.png')

def eligible_gross(ready,audit,daily):
    eligible_days=audit[audit.economic_calendar_eligible][GROUP]
    gross=ready.merge(eligible_days,on=GROUP,validate='many_to_one').groupby(GROUP).node_gross_weight.sum()
    return gross.loc[pd.MultiIndex.from_frame(daily[GROUP])].to_numpy()
