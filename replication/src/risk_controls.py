"""Entry Greeks, attribution, and Delta/Vega portfolio rules."""
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
from . import portfolio as S4, greeks as LEGACY
GROUP=S4.GROUP; KEY=S4.KEY; GREEKS=['Delta','Gamma','Vega','Theta']
STATES=['ORIGINAL','DELTA_NEUTRAL','VEGA_BALANCED','VEGA_BALANCED_DELTA_NEUTRAL']
SPOT_TOL=1e-8; IV_PRICE_TOL=.001


def greek_quality(c):
    out=c.copy();n=len(c)
    for g in GREEKS:out['finite_'+g]=np.isfinite(c['entry_'+g])
    out['all_entry_greeks_finite']=out[['finite_'+g for g in GREEKS]].all(axis=1)
    out['nonpositive_or_invalid_vega']=~out.finite_Vega|c.entry_Vega.le(0)
    dq=np.exp(-c.entry_q*c.entry_T);call=c.option_type.eq('C')
    out['delta_outside_admissible_bounds']=~out.finite_Delta|np.where(call,(c.entry_Delta< -1e-10)|(c.entry_Delta>dq+1e-10),(c.entry_Delta< -dq-1e-10)|(c.entry_Delta>1e-10))
    bound=dq/(c.entry_spot*c.entry_implied_vol*np.sqrt(c.entry_T)*np.sqrt(2*np.pi))
    out['implausible_gamma']=~out.finite_Gamma|c.entry_Gamma.le(0)|c.entry_Gamma.gt(bound*(1+1e-10))
    args=[c[x].to_numpy(float) for x in ['entry_spot','strike','entry_T','entry_r','entry_q','entry_implied_vol']]+[call.to_numpy()]
    reference=LEGACY.compute_bs_greeks(*args)
    for g,x in zip(GREEKS,reference):out['legacy_'+g+'_matches']=np.isclose(c['entry_'+g],x,rtol=1e-11,atol=1e-9,equal_nan=False)
    out['entry_greeks_quality_valid']=out.all_entry_greeks_finite&~out.nonpositive_or_invalid_vega&~out.delta_outside_admissible_bounds&~out.implausible_gamma
    rows=[]
    for model,d in out.groupby('model'):
        row=dict(model=model,n_contracts=len(d),all_finite_pct=100*d.all_entry_greeks_finite.mean(),quality_valid_pct=100*d.entry_greeks_quality_valid.mean(),
            n_invalid_vega=int(d.nonpositive_or_invalid_vega.sum()),n_invalid_delta=int(d.delta_outside_admissible_bounds.sum()),n_implausible_gamma=int(d.implausible_gamma.sum()))
        for g in GREEKS:row[g+'_finite_pct']=100*d['finite_'+g].mean();row[g+'_legacy_max_abs_error']=float(np.max(np.abs(d['entry_'+g]-pd.Series(reference[GREEKS.index(g)],index=out.index).loc[d.index])))
        rows.append(row)
    return out,pd.DataFrame(rows)

def spot_diagnostics(c):
    rows=[]
    for key,d in c.groupby(GROUP,sort=True):
        row=dict(model=key[0],entry_date=key[1],decision_date=d.decision_date.iloc[0],exit_date=d.exit_date.iloc[0])
        for label,col in [('entry','entry_spot'),('exit','exit_spot')]:
            x=d[col];disp=float(x.max()-x.min());valid=np.isfinite(x).all() and x.gt(0).all() and disp<=SPOT_TOL
            row[label+'_spot_min']=x.min();row[label+'_spot_max']=x.max();row[label+'_spot_dispersion']=disp
            row[label+'_spot_valid']=valid;row['S_'+label]=float(x.iloc[0]) if valid else np.nan
        row['spot_valid']=row['entry_spot_valid'] and row['exit_spot_valid'];row['dS']=row['S_exit']-row['S_entry'];rows.append(row)
    local=pd.DataFrame(rows)
    observations=pd.concat([c[['underlying_symbol','entry_date','entry_spot']].rename(columns={'entry_date':'date','entry_spot':'spot'}),
                            c[['underlying_symbol','exit_date','exit_spot']].rename(columns={'exit_date':'date','exit_spot':'spot'})])
    cross=observations.groupby(['underlying_symbol','date']).spot.agg(['min','max','count']);cross['dispersion']=cross['max']-cross['min']
    cross['valid']=np.isfinite(cross[['min','max']]).all(axis=1)&cross['min'].gt(0)&cross.dispersion.le(SPOT_TOL)
    return local,cross.reset_index()

def entry_controls(c):
    """Only signed entry weights and entry Greeks determine hedge/scaling."""
    fields=KEY+['signed_weight']+['entry_'+g for g in GREEKS]
    x=c[fields].copy();rows=[]
    if x.empty:
        daily=x[GROUP].drop_duplicates().copy()
        for col in ['LongVega','ShortVegaMagnitude','B','a_L','a_S','VegaImbalance','hedge_quantity']+['Net'+g for g in GREEKS]+['GrossAbs'+g for g in GREEKS]:daily[col]=pd.Series(dtype=float)
        daily['vega_balance_available']=pd.Series(dtype=bool);daily['vega_balance_status']=pd.Series(dtype=str)
        weights=x.copy()
        for col in ['a_L','a_S','leg_scale','w_vega']:weights[col]=pd.Series(dtype=float)
        weights['vega_balance_available']=pd.Series(dtype=bool)
        return daily,weights
    for key,d in x.groupby(GROUP,sort=True):
        w=d.signed_weight;long=w>0;short=w<0
        valid_vega=np.isfinite(d.entry_Vega).all() and d.entry_Vega.gt(0).all()
        VL=float((w[long]*d.loc[long,'entry_Vega']).sum()) if valid_vega else np.nan
        VS=float(-(w[short]*d.loc[short,'entry_Vega']).sum()) if valid_vega else np.nan
        available=valid_vega and VL>0 and VS>0
        budget=min(VL,VS) if available else np.nan;aL=budget/VL if available else np.nan;aS=budget/VS if available else np.nan
        row=dict(model=key[0],entry_date=key[1],LongVega=VL,ShortVegaMagnitude=VS,B=budget,a_L=aL,a_S=aS,
            vega_balance_available=available,vega_balance_status='AVAILABLE' if available else 'INVALID_OR_ZERO_LEG_VEGA')
        for g in GREEKS:
            v=w*d['entry_'+g];row['Net'+g]=v.sum(min_count=len(v));row['GrossAbs'+g]=v.abs().sum(min_count=len(v))
        row['VegaImbalance']=row['NetVega']/row['GrossAbsVega'] if np.isfinite(row['NetVega']) and np.isfinite(row['GrossAbsVega']) and row['GrossAbsVega']>0 else np.nan
        row['hedge_quantity']=-row['NetDelta']
        rows.append(row)
    daily=pd.DataFrame(rows);weights=x.merge(daily[GROUP+['a_L','a_S','vega_balance_available']],on=GROUP,validate='many_to_one')
    weights['leg_scale']=np.where(weights.signed_weight>0,weights.a_L,weights.a_S)
    weights['w_vega']=weights.signed_weight*weights.leg_scale
    return daily,weights

def attribution(c,spots,processed):
    out=c.merge(spots[GROUP+['S_entry','S_exit','dS','spot_valid']],on=GROUP,validate='many_to_one')
    out['dt_calendar_days']=(out.exit_date-out.entry_date).dt.total_seconds()/86400;out['dt_years']=out.dt_calendar_days/365
    valid=out.spot_valid&out.all_entry_greeks_finite
    out['DGT_available']=valid
    for name,value in [('Delta',out.signed_weight*out.entry_Delta*out.dS),('Gamma',.5*out.signed_weight*out.entry_Gamma*out.dS**2),('Theta',out.signed_weight*out.entry_Theta*out.dt_years)]:out[name+'_component']=value.where(valid)
    out['DGT_residual']=(out.pnl_mid-out.Delta_component-out.Gamma_component-out.Theta_component).where(valid)
    p=processed[['quote_date','contract_key','implied_vol','T','r','q','mid','bid','ask','source_observation_count']].rename(columns={
        'quote_date':'intended_exit_date',**{x:'processed_exit_'+x for x in ['implied_vol','T','r','q','mid','bid','ask','source_observation_count']}})
    for col in p:
        if col.startswith('processed_exit_'):p[col]=pd.to_numeric(p[col],errors='coerce').astype(float)
    out=out.merge(p,on=['intended_exit_date','contract_key'],how='left',validate='many_to_one')
    compatible=out.exit_quote_source.eq('PROCESSED_1545')&out.DGT_available&out.entry_greeks_quality_valid
    compatible &= np.isfinite(out[['entry_implied_vol','processed_exit_implied_vol','processed_exit_r','processed_exit_q','processed_exit_T']]).all(axis=1)
    compatible &= out.entry_implied_vol.gt(0)&out.processed_exit_implied_vol.gt(0)&out.processed_exit_T.gt(0)
    compatible &= np.isclose(out.processed_exit_mid,out.exit_mid,rtol=0,atol=1e-10)&np.isclose(out.processed_exit_bid,out.exit_bid,rtol=0,atol=1e-10)&np.isclose(out.processed_exit_ask,out.exit_ask,rtol=0,atol=1e-10)
    compatible &= np.isclose(out.entry_T-out.processed_exit_T,out.dt_years,rtol=0,atol=1e-10)
    compatible &= out.entry_source_observation_count.eq(1)&out.processed_exit_source_observation_count.eq(1)
    # Reprice observed IVs under their source model; never invert a price for new IV.
    out['entry_IV_repricing_error']=(european_price(out.entry_spot,out.strike,out.entry_T,out.entry_r,out.entry_q,out.entry_implied_vol,out.option_type.eq('C'))-out.entry_mid).abs()
    out['exit_IV_repricing_error']=(european_price(out.exit_spot,out.strike,out.processed_exit_T,out.processed_exit_r,out.processed_exit_q,out.processed_exit_implied_vol,out.option_type.eq('C'))-out.exit_mid).abs()
    compatible &= out.entry_IV_repricing_error.le(IV_PRICE_TOL)&out.exit_IV_repricing_error.le(IV_PRICE_TOL)
    out['DGVT_available']=compatible;out['dIV_exact_contract']=(out.processed_exit_implied_vol-out.entry_implied_vol).where(compatible)
    out['Vega_component']=(out.signed_weight*out.entry_Vega*out.dIV_exact_contract).where(compatible)
    out['DGVT_residual']=(out.DGT_residual-out.Vega_component).where(compatible)
    return out

def european_price(S,K,T,r,q,sigma,is_call):
    """Vectorized source iv.py BS price, solely to verify already stored IVs."""
    d1=(np.log(S/K)+(r-q+.5*sigma**2)*T)/(sigma*np.sqrt(T));d2=d1-sigma*np.sqrt(T)
    call=S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    put=K*np.exp(-r*T)*norm.cdf(-d2)-S*np.exp(-q*T)*norm.cdf(-d1)
    return pd.Series(np.where(is_call,call,put),index=S.index) if isinstance(S,pd.Series) else np.where(is_call,call,put)

def exposure_summary(exposures):
    rows=[]
    cols=['Net'+g for g in GREEKS]+['GrossAbs'+g for g in GREEKS]+['LongVega','ShortVegaMagnitude','VegaImbalance']
    cols += [c+'_per_gross_premium' for c in cols if c!='VegaImbalance']
    for model,d in exposures.groupby('model'):
        for col in cols:
            x=d[col].dropna();rows.append(dict(model=model,exposure=col,n_days=len(x),mean=x.mean(),median=x.median(),std=x.std(ddof=1),p05=x.quantile(.05),p25=x.quantile(.25),p75=x.quantile(.75),p95=x.quantile(.95),mean_absolute=x.abs().mean()))
    return pd.DataFrame(rows)

def attribution_summary(c,subset=False):
    rows=[];components=['Delta_component','Gamma_component','Theta_component','DGT_residual'] if not subset else ['Delta_component','Gamma_component','Vega_component','Theta_component','DGVT_residual']
    for model,d in c.groupby('model'):
        full_n=len(d);a=d[d.DGVT_available if subset else d.DGT_available]
        total=a.pnl_mid.sum();absolute=sum(a[x].sum().__abs__() for x in components)
        for col in components:
            daily=a.groupby('entry_date')[col].sum(min_count=1)
            rows.append(dict(model=model,scope='EXACT_IV_DIAGNOSTIC_SUBSET' if subset else 'PRIMARY_DGT',component=col,n_contracts=len(a),coverage_pct=100*len(a)/full_n,
                n_days=a.entry_date.nunique(),total_mid_pnl=total,total_component=a[col].sum(),mean_daily_component=daily.mean(),median_daily_component=daily.median(),
                signed_share_of_mid_pnl=a[col].sum()/total if abs(total)>1e-12 else np.nan,absolute_aggregate_component_share=abs(a[col].sum())/absolute if absolute>0 else np.nan,
                n_processed=int(a.exit_quote_source.eq('PROCESSED_1545').sum()),n_raw_recovered=int(a.exit_quote_source.eq('RAW_1545_RECOVERED').sum())))
    return pd.DataFrame(rows)

def portfolios(c,controls,weights,spots,baseline):
    vb=c.merge(weights[KEY+['w_vega','a_L','a_S','leg_scale','vega_balance_available']],on=KEY,validate='one_to_one')
    v=vb[vb.vega_balance_available].copy();v['weight_at_entry']=v.w_vega
    v=S4.contract_pnl(v)  # same prices, new entry-only weights; retains w_vega metadata
    vb=vb.merge(v[KEY+['pnl_mid','pnl_exec','gross_entry_premium_mid_component','gross_entry_premium_exec_component']].rename(columns={x:'vb_'+x for x in ['pnl_mid','pnl_exec','gross_entry_premium_mid_component','gross_entry_premium_exec_component']}),on=KEY,how='left',validate='one_to_one')
    base=baseline.copy();base['portfolio_state']='ORIGINAL';base['available']=True;base['hedge_quantity']=0.;base['PnL_underlying']=0.
    base=base.merge(controls[GROUP+['NetDelta','NetGamma','NetVega','NetTheta']],on=GROUP,validate='one_to_one')
    base['option_NetDelta']=base.NetDelta
    dn=baseline.merge(controls,on=GROUP,validate='one_to_one').merge(spots[GROUP+['S_entry','S_exit','dS','spot_valid']],on=GROUP,validate='one_to_one')
    dn['portfolio_state']='DELTA_NEUTRAL';dn['available']=dn.spot_valid&np.isfinite(dn.hedge_quantity)
    dn['PnL_underlying']=(dn.hedge_quantity*dn.dS).where(dn.available)
    dn['option_NetDelta']=dn.NetDelta;dn['NetDelta']=dn.option_NetDelta+dn.hedge_quantity
    for suffix in ['mid','exec']:
        dn['PnL_'+suffix]=(dn['PnL_'+suffix]+dn.PnL_underlying).where(dn.available);dn['Return_'+suffix]=dn['PnL_'+suffix]/dn['GrossPremium_'+suffix]
    vbday=v.groupby(GROUP).agg(PnL_mid=('pnl_mid','sum'),PnL_exec=('pnl_exec','sum'),GrossPremium_mid=('gross_entry_premium_mid_component','sum'),GrossPremium_exec=('gross_entry_premium_exec_component','sum'),number_contracts=('contract_key','size')).reset_index()
    vexp,vweights=entry_controls(v)
    vbd=baseline[['model','entry_date','decision_date','exit_date']].merge(vbday,on=GROUP,how='left',validate='one_to_one').merge(controls[GROUP+['a_L','a_S','vega_balance_available']],on=GROUP,validate='one_to_one').merge(vexp[GROUP+['Net'+g for g in GREEKS]],on=GROUP,how='left',validate='one_to_one')
    vbd['portfolio_state']='VEGA_BALANCED';vbd['available']=vbd.vega_balance_available;vbd['hedge_quantity']=0.;vbd['PnL_underlying']=0.;vbd['option_NetDelta']=vbd.NetDelta
    for suffix in ['mid','exec']:vbd['Return_'+suffix]=vbd['PnL_'+suffix]/vbd['GrossPremium_'+suffix]
    vd=vbd.merge(spots[GROUP+['S_entry','S_exit','dS','spot_valid']],on=GROUP,validate='one_to_one');vd['portfolio_state']='VEGA_BALANCED_DELTA_NEUTRAL'
    vd['available']=vd.available&vd.spot_valid&np.isfinite(vd.NetDelta);vd['hedge_quantity']=-vd.option_NetDelta;vd['PnL_underlying']=(vd.hedge_quantity*vd.dS).where(vd.available)
    vd['NetDelta']=vd.option_NetDelta+vd.hedge_quantity
    for suffix in ['mid','exec']:
        vd['PnL_'+suffix]=(vd['PnL_'+suffix]+vd.PnL_underlying).where(vd.available);vd['Return_'+suffix]=vd['PnL_'+suffix]/vd['GrossPremium_'+suffix]
    # Retain unavailable model-days with NaN results; never turn them into zero PnL.
    common=['model','entry_date','decision_date','exit_date','portfolio_state','available','number_contracts','PnL_mid','PnL_exec','GrossPremium_mid','GrossPremium_exec','Return_mid','Return_exec','PnL_underlying','hedge_quantity','option_NetDelta']+['Net'+g for g in GREEKS]
    allstates=pd.concat([x[common] for x in [base,dn,vbd,vd]],ignore_index=True).sort_values(['model','portfolio_state','entry_date']).reset_index(drop=True)
    return vb,dn,allstates,v

def performance(daily):
    rows=[];boot=[];out=daily.copy()
    for (model,state),d in daily.groupby(['model','portfolio_state']):
        valid=d[d.available].sort_values('entry_date')
        for impl,suffix in [('MID','mid'),('BID_ASK','exec')]:
            x=valid['Return_'+suffix].to_numpy();n=len(x)
            if n<7 or not np.isfinite(x).all():
                rows.append(dict(model=model,portfolio_state=state,implementation=impl,n_days=n,status='UNAVAILABLE_INFERENCE'));continue
            sd=x.std(ddof=1);w,dd,status=S4.wealth(x)
            out.loc[valid.index,'cumulative_arithmetic_'+suffix]=np.cumsum(x);out.loc[valid.index,'wealth_'+suffix]=w if w is not None else np.nan
            proxy='option bid/ask + frictionless underlying hedge proxy' if 'DELTA_NEUTRAL' in state else 'bid/ask option quote-based proxy'
            label='midpoint option + static underlying-hedge proxy' if 'DELTA_NEUTRAL' in state else 'midpoint option valuation'
            rows.append(dict(model=model,portfolio_state=state,implementation=impl,implementation_label=proxy if impl=='BID_ASK' else label,n_days=n,n_contracts=int(valid.number_contracts.sum()),unavailable_day_pct=100*(1-n/len(d)),
                median_daily_return=np.median(x),daily_vol=sd,annualized_sharpe=np.sqrt(252)*x.mean()/sd,positive_day_pct=100*np.mean(x>0),cumulative_arithmetic_normalized_return=x.sum(),max_drawdown=dd,wealth_index_status=status,
                total_pnl=valid['PnL_'+suffix].sum(),**S4.hac(x),**S4.bootstrap(x,5)))
            for block in [5,10]:boot.append(dict(model=model,portfolio_state=state,implementation=impl,**S4.bootstrap(x,block)))
    return pd.DataFrame(rows),pd.DataFrame(boot),out

def reductions(daily):
    rows=[]
    for model,d in daily.groupby('model'):
        original=d[d.portfolio_state.eq('ORIGINAL')].set_index('entry_date')
        vb=d[d.portfolio_state.eq('VEGA_BALANCED')].set_index('entry_date')
        for before,after,greek in [('ORIGINAL','DELTA_NEUTRAL','Delta'),('ORIGINAL','VEGA_BALANCED','Vega'),('VEGA_BALANCED','VEGA_BALANCED_DELTA_NEUTRAL','Delta'),('VEGA_BALANCED','VEGA_BALANCED_DELTA_NEUTRAL','Vega'),('ORIGINAL','VEGA_BALANCED','Gamma'),('ORIGINAL','VEGA_BALANCED','Theta'),('VEGA_BALANCED','VEGA_BALANCED_DELTA_NEUTRAL','Gamma'),('VEGA_BALANCED','VEGA_BALANCED_DELTA_NEUTRAL','Theta')]:
            a=d[d.portfolio_state.eq(before)&d.available].set_index('entry_date');b=d[d.portfolio_state.eq(after)&d.available].set_index('entry_date');dates=a.index.intersection(b.index)
            x=a.loc[dates,'Net'+greek].abs();y=b.loc[dates,'Net'+greek].abs()
            rows.append(dict(model=model,before_state=before,after_state=after,greek=greek,n_common_days=len(dates),before_mean_abs=x.mean(),after_mean_abs=y.mean(),
                mean_abs_reduction_pct=100*(1-y.mean()/x.mean()) if x.mean()>1e-12 else np.nan,before_p95_abs=x.quantile(.95),after_p95_abs=y.quantile(.95),
                p95_abs_reduction_pct=100*(1-y.quantile(.95)/x.quantile(.95)) if x.quantile(.95)>1e-12 else np.nan))
    return pd.DataFrame(rows)

def signal_alignment(daily,stage4diag):
    diag=stage4diag[['model','entry_date','decision_date','top_minus_bottom','predicted_cross_sectional_dispersion']]
    joined=daily.merge(diag,on=['model','entry_date','decision_date'],how='left',validate='many_to_one');rows=[]
    # Common finite date mask across all four states for fair alignment comparison.
    for model,d in joined.groupby('model'):
        for signal in ['top_minus_bottom','predicted_cross_sectional_dispersion']:
            wide=d.pivot(index='entry_date',columns='portfolio_state',values='Return_mid')
            label=diag[diag.model.eq(model)].set_index('entry_date')[signal]
            common=wide[STATES].join(label).replace([np.inf,-np.inf],np.nan).dropna()
            for state in STATES:rows.append(dict(model=model,portfolio_state=state,signal=signal,n_common_dates=len(common),pearson=common[signal].corr(common[state]),spearman=common[signal].corr(common[state],method='spearman')))
    return joined,pd.DataFrame(rows)

def make_figures(exposure,attr,daily):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'ORIGINAL':'#333333','DELTA_NEUTRAL':'#0072B2','VEGA_BALANCED':'#D55E00','VEGA_BALANCED_DELTA_NEUTRAL':'#009E73'}
    def finish(fig,name):fig.tight_layout();fig.savefig(HERE/'figures'/name,dpi=300,bbox_inches='tight');plt.close(fig)
    for greek in ['Delta','Vega']:
        fig,axes=plt.subplots(3,1,figsize=(9,7),sharex=True)
        for ax,model in zip(axes,S4.MODELS):
            for state in STATES:
                d=daily[daily.model.eq(model)&daily.portfolio_state.eq(state)].sort_values('entry_date');ax.plot(d.entry_date,d['Net'+greek],label=state.replace('_',' '),color=colors[state],lw=.8)
            ax.set_ylabel(model.upper());ax.axhline(0,color='gray',lw=.5);ax.grid(axis='y',alpha=.2)
        units='underlying units' if greek=='Delta' else 'quoted price per 1.00 IV'
        axes[0].set_title('Entry net '+greek+' ('+units+'; multiplier 1)');axes[-1].legend(frameon=False,ncol=2,fontsize=8);finish(fig,'stage5_net_'+greek.lower()+'_timeseries.png')
    fig,ax=plt.subplots(figsize=(9,4.5));components=['Delta_component','Gamma_component','Theta_component','DGT_residual'];width=.18;x=np.arange(3)
    for i,col in enumerate(components):
        values=[attr.loc[attr.model.eq(model)&attr.DGT_available,col].sum() for model in S4.MODELS];ax.bar(x+(i-1.5)*width,values,width,label=col.replace('_',' '))
    ax.set_xticks(x,S4.MODELS);ax.set_ylabel('Aggregate weighted midpoint PnL component');ax.axhline(0,color='gray',lw=.7);ax.legend(frameon=False,ncol=2);ax.grid(axis='y',alpha=.2);finish(fig,'stage5_pnl_attribution.png')
    for name,states in [('stage5_original_vs_delta_neutral.png',['ORIGINAL','DELTA_NEUTRAL']),('stage5_original_vs_vega_balanced.png',['ORIGINAL','VEGA_BALANCED']),('stage5_four_portfolio_states.png',STATES)]:
        fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
        for ax,model in zip(axes,S4.MODELS):
            for state in states:
                d=daily[daily.model.eq(model)&daily.portfolio_state.eq(state)].sort_values('entry_date');ax.plot(d.exit_date,d.cumulative_arithmetic_mid,label=state.replace('_',' '),color=colors[state],lw=1)
            ax.set_title(model.upper());ax.tick_params(axis='x',rotation=30);ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Cumulative arithmetic MID return\nper gross option premium');axes[-1].legend(frameon=False,fontsize=7);finish(fig,name)
