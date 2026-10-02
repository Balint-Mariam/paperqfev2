"""Frozen functions extracted from paper_qfe/08_final_robustness.py."""
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
from statsmodels.stats.multitest import multipletests
HERE=Path(__file__).resolve().parents[1]/'outputs'
ROOT=HERE/'data'
from . import risk_controls as S5
S4=S5.S4; MODELS=S4.MODELS; STATES=S5.STATES; GROUP=S4.GROUP
ALPHA=.05; LAGS=(1,5,10); COST_BPS=(.5,1,2,5)
TRANSFORMS=[('DELTA_NEUTRAL','ORIGINAL'),('VEGA_BALANCED','ORIGINAL'),('VEGA_BALANCED_DELTA_NEUTRAL','ORIGINAL'),('VEGA_BALANCED_DELTA_NEUTRAL','DELTA_NEUTRAL')]


def corrected(p):
    """Independent step-down Holm and step-up BH, restored to input order."""
    p=np.asarray(p,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid correction-family p values')
    order=np.argsort(p,kind='stable');x=p[order];n=len(x)
    holm=np.minimum(1,np.maximum.accumulate(x*(n-np.arange(n))))
    bh=np.minimum(1,np.minimum.accumulate((x*n/np.arange(1,n+1))[::-1])[::-1])
    a=np.empty(n);b=np.empty(n);a[order]=holm;b[order]=bh;return a,b

def adjust(frame,pcol='raw_p'):
    out=frame.copy();a,b=corrected(out[pcol]);out['holm_adjusted_p']=a;out['holm_reject_5pct']=a<=ALPHA
    out['BH_adjusted_p']=b;out['BH_reject_5pct']=b<=ALPHA;out['family_size']=len(out);return out

def mean_stats(x,lag=5):
    x=np.asarray(x,float)
    if len(x)<max(7,lag+2) or not np.isfinite(x).all():raise ValueError('Invalid portfolio-date inference sample')
    fit=sm.OLS(x,np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds=dict(maxlags=lag,kernel='bartlett',use_correction=True),use_t=False)
    return dict(mean=float(x.mean()),median=float(np.median(x)),HAC_standard_error=float(fit.bse[0]),HAC_t_stat=float(fit.tvalues[0]),raw_p=float(fit.pvalues[0]),HAC_lag=lag,n_days=len(x))

def multiple_testing(perf):
    rows=[]
    for impl,d in perf.groupby('implementation',sort=True):
        f=d[['model','portfolio_state','implementation','mean_daily_return','HAC_p_value']].rename(columns={'HAC_p_value':'raw_p'})
        primary=adjust(f);primary['family']='PRIMARY_'+impl+'_12';rows.append(primary)
        for model,a in f.groupby('model'):
            second=adjust(a);second['family']='SECONDARY_'+impl+'_'+model+'_4';rows.append(second)
    return pd.concat(rows,ignore_index=True)

def paired_effects(daily):
    rows=[];boots=[]
    for model,d in daily.groupby('model'):
        wide=d.pivot(index='entry_date',columns='portfolio_state',values='Return_mid').sort_index()
        if len(wide)!=408 or wide[STATES].isna().any().any():raise ValueError('Paired transformations require exact same 408 dates')
        for a,b in TRANSFORMS:
            x=(wide[a]-wide[b]).to_numpy();key=dict(model=model,transformation=a+'_MINUS_'+b,target_state=a,reference_state=b)
            rows.append(dict(**key,**mean_stats(x),**S4.bootstrap(x,5)))
            for block in [5,10]:boots.append(dict(**key,**S4.bootstrap(x,block)))
    return adjust(pd.DataFrame(rows)),pd.DataFrame(boots)

def component_inference(attr):
    rows=[];parts=['Delta_component','Gamma_component','Theta_component','DGT_residual']
    if not attr.DGT_available.all():raise ValueError('Full-sample DGT unavailable')
    day=attr.groupby(GROUP)[parts].sum().reset_index()
    for model,d in day.groupby('model'):
        for col in parts:rows.append(dict(model=model,component=col,units='weighted quoted option price',**mean_stats(d[col])))
    return adjust(pd.DataFrame(rows)),day

def normalization(daily,baseline):
    x=daily.merge(baseline[GROUP+['GrossPremium_mid','GrossPremium_exec']].rename(columns={'GrossPremium_mid':'original_gross_premium_mid','GrossPremium_exec':'original_gross_premium_exec'}),on=GROUP,validate='many_to_one')
    rows=[]
    for impl,suffix in [('MID','mid'),('BID_ASK','exec')]:
        # User's common denominator is original MID premium for BOTH implementations.
        x['Return_common_denom_'+suffix]=x['PnL_'+suffix]/x.original_gross_premium_mid
        for (model,state),d in x.groupby(['model','portfolio_state']):
            a=d['Return_'+suffix];b=d['Return_common_denom_'+suffix];pooled=d['PnL_'+suffix].sum()/d['GrossPremium_'+suffix].sum()
            rows.append(dict(model=model,portfolio_state=state,implementation=impl,n_days=len(d),primary_mean_daily_ratio=a.mean(),
                mean_Return_common_denom=b.mean(),pooled_premium_return=pooled,common_sign_matches_primary=np.sign(a.mean())==np.sign(b.mean()),
                pooled_sign_matches_primary=np.sign(a.mean())==np.sign(pooled),total_pnl=d['PnL_'+suffix].sum(),total_state_gross_premium=d['GrossPremium_'+suffix].sum()))
    return pd.DataFrame(rows),x

def hedge_costs(daily,spots):
    x=daily[daily.portfolio_state.isin(['DELTA_NEUTRAL','VEGA_BALANCED_DELTA_NEUTRAL'])].merge(spots[GROUP+['S_entry']],on=GROUP,validate='many_to_one')
    x['hedge_notional']=x.hedge_quantity.abs()*x.S_entry
    rows=[];stress=[]
    for (model,state),d in x.groupby(['model','portfolio_state']):
        mean=d.Return_mid.mean();load=(2*d.hedge_notional/d.GrossPremium_mid).mean();total=d.PnL_mid.sum();notional=d.hedge_notional.sum()
        root=mean/load if load>0 else np.nan;c=max(0,root) if np.isfinite(root) else np.nan
        aggregate=max(0,total/(2*notional)) if notional>0 else np.nan
        rows.append(dict(model=model,portfolio_state=state,n_days=len(d),uncosted_mean_MID=mean,uncosted_total_MID_pnl=total,
            mean_hedge_notional=d.hedge_notional.mean(),mean_normalized_cost_load=load,break_even_decimal_per_side=c,
            break_even_bps_per_side=10000*c,break_even_bps_round_trip=20000*c,aggregate_break_even_decimal_per_side=aggregate,
            aggregate_break_even_bps_per_side=10000*aggregate,positive_mean_friction_budget=mean>0,
            status='POSITIVE_MEAN_BUDGET' if mean>0 else 'NO_POSITIVE_COST_BUDGET_ALREADY_NONPOSITIVE',cost_interpretation='hypothetical constant per-side proportional hedge friction; not observed tradable cost'))
        for bp in COST_BPS:
            cost=2*(bp/10000)*d.hedge_notional;ret=(d.PnL_mid-cost)/d.GrossPremium_mid
            stress.append(dict(model=model,portfolio_state=state,bps_per_side=bp,bps_round_trip=2*bp,n_days=len(d),mean_MID_after_hypothetical_cost=ret.mean(),
                total_MID_pnl_after_hypothetical_cost=(d.PnL_mid-cost).sum(),mean_BID_ASK_proxy_after_hypothetical_cost=((d.PnL_exec-cost)/d.GrossPremium_exec).mean(),interpretation='illustrative stress only; no strategy/cost estimate selected'))
    return pd.DataFrame(rows),pd.DataFrame(stress),x

def stability(daily,variable,buckets,states=STATES,bid_states=('ORIGINAL','VEGA_BALANCED')):
    rows=[]
    for (model,state,bucket),d in daily.groupby(['model','portfolio_state',variable]):
        if state not in states or bucket not in buckets:continue
        for impl,suffix in [('MID','mid'),('BID_ASK','exec')]:
            if impl=='BID_ASK' and state not in bid_states:continue
            x=d['Return_'+suffix];sd=x.std(ddof=1)
            rows.append(dict(model=model,portfolio_state=state,bucket=str(bucket),partition_variable=variable,implementation=impl,annualized_sharpe=np.sqrt(252)*x.mean()/sd,**mean_stats(x)))
    return pd.DataFrame(rows)

def iv_regimes(decisions):
    wide=pd.read_csv(ROOT/'iv_grid_wide.csv');wide['quote_date']=pd.to_datetime(wide.quote_date);nodes=[c for c in wide if c.startswith('iv_')]
    observed=wide[nodes].replace([np.inf,-np.inf],np.nan)
    levels=pd.DataFrame(dict(decision_date=wide.quote_date,market_IV_level=observed.median(axis=1,skipna=True),n_observed_nodes=observed.notna().sum(axis=1)))
    cutoff=S5.S4.B.S3.S2.FIT_END;pre=levels[levels.decision_date.le(cutoff)&levels.market_IV_level.notna()]
    if pre.empty:raise ValueError('No pre-test IV levels')
    threshold=float(pre.market_IV_level.median())
    test=decisions[['decision_date']].drop_duplicates().merge(levels,on='decision_date',how='left',validate='one_to_one')
    if test.market_IV_level.isna().any():raise ValueError('Missing decision-time surface regime label; no imputation')
    test['IV_regime']=np.where(test.market_IV_level.le(threshold),'LOW_IV','HIGH_IV')
    info=dict(threshold=threshold,cutoff=str(cutoff.date()),pretest_n_dates=len(pre),pretest_first_date=str(pre.decision_date.min().date()),pretest_last_date=str(pre.decision_date.max().date()),
        rule='daily cross-sectional median frozen observed IV; threshold pre-test median through validation cutoff; LOW<=threshold, HIGH>threshold',surface_sha256=digest(ROOT/'iv_grid_wide.csv'))
    return test,info,levels

def inference_sensitivity(daily):
    rows=[]
    for (model,state),d in daily.groupby(['model','portfolio_state']):
        for impl,suffix in [('MID','mid'),('BID_ASK','exec')]:
            for lag in LAGS:rows.append(dict(model=model,portfolio_state=state,implementation=impl,**mean_stats(d['Return_'+suffix],lag)))
    out=[]
    for (impl,lag),d in pd.DataFrame(rows).groupby(['implementation','HAC_lag']):out.append(adjust(d))
    return pd.concat(out,ignore_index=True)

def chain_tables(perf,multiple,costs):
    forecast=pd.read_csv(HERE/'tables/forecast_model_comparison.csv');ic=pd.read_csv(HERE/'tables/rank_ic_summary.csv');signal=pd.read_csv(HERE/'tables/economic_signal_predictive_summary.csv')
    forecast=forecast[forecast.model.isin(MODELS)]
    chain=forecast[['model','RMSE','MAE','QLIKE']].merge(ic[['model','mean_daily_IC']],on='model').rename(columns={'mean_daily_IC':'forecast_Rank_IC'})
    chain=chain.merge(signal[['model','mean_IC','mean_top_minus_bottom']],on='model').rename(columns={'mean_IC':'holding_period_Rank_IC','mean_top_minus_bottom':'holding_IV_top_minus_bottom'})
    for state,name in [('ORIGINAL','original'),('DELTA_NEUTRAL','delta_neutral'),('VEGA_BALANCED','vega_balanced'),('VEGA_BALANCED_DELTA_NEUTRAL','joint')]:
        for impl in ['MID','BID_ASK'] if state=='ORIGINAL' else ['MID']:
            d=perf[perf.portfolio_state.eq(state)&perf.implementation.eq(impl)][['model','mean_daily_return']].rename(columns={'mean_daily_return':name+'_'+impl+'_mean'})
            chain=chain.merge(d,on='model')
    joint=multiple[multiple.family.eq('PRIMARY_MID_12')&multiple.portfolio_state.eq('VEGA_BALANCED_DELTA_NEUTRAL')]
    chain=chain.merge(joint[['model','holm_adjusted_p','holm_reject_5pct','BH_adjusted_p','BH_reject_5pct']],on='model')
    for state,name in [('DELTA_NEUTRAL','delta_neutral'),('VEGA_BALANCED_DELTA_NEUTRAL','joint')]:
        chain=chain.merge(costs[costs.portfolio_state.eq(state)][['model','break_even_bps_per_side']].rename(columns={'break_even_bps_per_side':name+'_break_even_bps_per_side'}),on='model')
    rank=chain[['model']].copy()
    for col,asc in [('RMSE',True),('forecast_Rank_IC',False),('holding_period_Rank_IC',False),('original_MID_mean',False),('delta_neutral_MID_mean',False),('joint_MID_mean',False)]:rank[col+'_rank']=chain[col].rank(method='min',ascending=asc).astype(int)
    return chain,rank,forecast,signal

def figures(forecast,chain,attr,daily,baseline):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def finish(fig,name):fig.tight_layout();fig.savefig(HERE/'figures'/name,dpi=300,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(forecast.model,forecast.RMSE,color=['#333333','#0072B2','#D55E00']);ax.set_ylabel('Common-sample one-session IV RMSE');ax.grid(axis='y',alpha=.2);finish(fig,'stage6_main_forecast_performance.png')
    signal_days=pd.read_parquet(HERE/'data/economic_signal_daily_diagnostics.parquet').decision_date.nunique()
    count=[int(signal_days),int(baseline.entry_date.nunique()),int(baseline.entry_date.nunique())]
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].bar(['Signal days','Calendar-eligible','BASE priced'],count,color='#0072B2');axes[0].set_ylabel('Days per model');axes[0].tick_params(axis='x',rotation=15)
    x=np.arange(3);axes[1].bar(x-.18,100*chain.original_MID_mean,.36,label='MID',color='#333333');axes[1].bar(x+.18,100*chain.original_BID_ASK_mean,.36,label='BID/ASK',color='#D55E00');axes[1].set_xticks(x,chain.model);axes[1].set_ylabel('Mean daily return (% gross entry premium)');axes[1].legend(frameon=False);finish(fig,'stage6_main_implementation_funnel.png')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(3)
    for i,col in enumerate(['Delta_component','Gamma_component','Theta_component','DGT_residual']):
        vals=[attr.loc[attr.model.eq(model),col].sum() for model in MODELS];ax.bar(x+(i-1.5)*.18,vals,.18,label=col.replace('_',' '))
    ax.set_xticks(x,MODELS);ax.set_ylabel('Aggregate weighted midpoint PnL');ax.axhline(0,color='gray',lw=.6);ax.legend(frameon=False,ncol=2);finish(fig,'stage6_main_greek_attribution.png')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,model in zip(axes,MODELS):
        for state,color in [('ORIGINAL','#333333'),('DELTA_NEUTRAL','#0072B2')]:
            d=daily[daily.model.eq(model)&daily.portfolio_state.eq(state)].sort_values('entry_date')
            for suffix,style in [('mid','-'),('exec','--')]:ax.plot(d.exit_date,np.cumsum(d['Return_'+suffix]),color=color,linestyle=style,label=state.replace('_',' ')+' '+('MID' if suffix=='mid' else 'BID/ASK'))
        ax.set_title(model.upper());ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Cumulative arithmetic normalized return');axes[-1].legend(frameon=False,fontsize=7);finish(fig,'stage6_main_original_vs_delta_neutral.png')
    text='''# Stage6 figure manifest

Exactly four candidate main-text figures are created, all from frozen data:

1. stage6_main_forecast_performance.png — common-sample RMSE point estimates. Supports the forecast comparison; no dominance claim.
2. stage6_main_implementation_funnel.png — fixed signal days, calendar eligibility and BASE pricing, alongside ORIGINAL MID/BID_ASK means. Shows implementation coverage and friction; exclusions are calendar-only.
3. stage6_main_greek_attribution.png — full-sample DGT aggregate components; residual is not pure Vega PnL.
4. stage6_main_original_vs_delta_neutral.png — all models, MID and BID_ASK, ORIGINAL and static Delta proxy. Dashed curves are option bid/ask; hedged curves include an uncosted underlying proxy. Cumulative values are arithmetic premium-normalized returns, not margin-capital wealth.

Appendix: the six existing Stage5 figures for Vega exposure and all four states, and Stage2 regional/diagnostic figures. Supplement: existing quote/mapping/audit figures. No prior figure is modified. Color/line style is consistent and no decorative plots are added.
'''
    (HERE/'reports/stage6_figure_manifest.md').write_text(text,encoding='utf-8')
