"""Fixed-window model selection and one-session forecasting."""
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
import joblib, arch, sklearn, statsmodels, xgboost as xgb
from arch.bootstrap import MCS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
MODELS=['persistence','arima','ridge','xgboost']
TRAIN_END=pd.Timestamp('2021-08-12'); FIT_END=pd.Timestamp('2023-04-21')
ORDERS=[(1,0,0),(2,0,0),(1,1,0),(1,0,1),(2,0,1)]
ALPHAS=[.01,.1,1.,10.,100.]
from .contract_mapping import GRID, XGB_BASE
HAC_LAG, BLOCKS, REPS, MIN_IC=5,[5,10],2000,20
VERSIONS=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,statsmodels=statsmodels.__version__,xgboost=xgb.__version__,arch=arch.__version__,exchange_calendars=xc.__version__)


def features(series,calendar):
    out=pd.DataFrame(index=series.index)
    for lag in (1,2,3,5,10): out[f'lag_{lag}']=series.shift(lag-1)
    out['roll_mean_5']=series.rolling(5,min_periods=2).mean()
    out['roll_std_5']=series.rolling(5,min_periods=2).std()
    out['dow']=pd.Series(calendar,index=calendar).shift(-1).reindex(series.index).dt.weekday.astype(float)
    return out

def masks(series,calendar):
    dates=pd.Series(calendar,index=calendar).shift(-1).reindex(series.index)
    labels=series.shift(-1)
    finite=np.isfinite(labels)
    train=dates.le(TRAIN_END)&finite
    val=dates.gt(TRAIN_END)&dates.le(FIT_END)&finite&(series.index>=TRAIN_END)
    final=dates.le(FIT_END)&finite
    return labels,dates,train,val,final

def ridge_pipeline(alpha):
    return Pipeline([('imputer',SimpleImputer(strategy='median',add_indicator=True)),
                     ('scaler',StandardScaler()),('ridge',Ridge(alpha=alpha))])

def preprocessing_evidence(pipe,X):
    imp=pipe.named_steps['imputer']; scaler=pipe.named_steps['scaler']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(X.to_numpy(),axis=0)
    transformed=imp.transform(X)
    return dict(n_fit=len(X),medians_equal=bool(np.allclose(imp.statistics_,median,equal_nan=True)),
                scaler_mean_equal=bool(np.allclose(scaler.mean_,transformed.mean(axis=0))),
                scaler_n_fit=int(scaler.n_samples_seen_),imputer_statistics=imp.statistics_.tolist(),
                all_missing_features=int(np.isnan(median).sum()))

def score(pred,actual):
    err=np.asarray(pred)-np.asarray(actual)
    if not np.isfinite(err).all(): return np.inf,np.inf
    return float(np.sqrt(np.mean(err**2))),float(np.mean(np.abs(err)))

def arima_model(values,order):
    return ARIMA(np.asarray(values,dtype=float),order=tuple(order),trend='n',
                 enforce_stationarity=False,enforce_invertibility=False)

def filtered_prediction(series,order,params,targets):
    """Fixed-parameter predicted (never smoothed) means at target positions.

    Filtering may process a full vector efficiently; prediction at k uses only
    values at indices < k. Suffix-deletion checks enforce that distinction.
    """
    res=arima_model(series.to_numpy(),order).filter(np.asarray(params))
    positions=series.index.get_indexer(targets)
    if (positions<0).any(): raise ValueError('target outside calendar')
    pred=np.asarray(res.get_prediction(start=0,end=len(series)-1,information_set='predicted').predicted_mean)
    return pred[positions]

def fit_node(task):
    node,series,calendar,origins,cache_str=task
    cache=Path(cache_str); meta_path=cache/f'{node}.json'; archive=cache/f'{node}.parquet'
    if meta_path.exists() and archive.exists():
        return json.loads(meta_path.read_text(encoding='utf-8')),True
    started=time.perf_counter()
    X=features(series,calendar); labels,dates,train,val,final=masks(series,calendar)
    targets=pd.DatetimeIndex(pd.Series(calendar,index=calendar).shift(-1).reindex(origins))
    result=pd.DataFrame({'forecast_origin_date':origins,'forecast_target_date':targets,'node':node})
    result['forecast_persistence']=series.reindex(origins).to_numpy()
    info=dict(node=node,n_train=int(train.sum()),n_validation=int(val.sum()),n_final=int(final.sum()),
              train_label_max=str(dates[train].max()),validation_label_min=str(dates[val].min()),
              validation_label_max=str(dates[val].max()),final_label_max=str(dates[final].max()),
              validation_origin_min=str(series.index[val].min()),models={})
    # All selection calls explicitly receive only training features/labels and
    # validation features/labels. Test targets never enter these interfaces.
    for model in ['ridge','xgboost']:
        pred=np.full(len(origins),np.nan)
        details=dict(status='INSUFFICIENT_HISTORICAL_LABELS',candidates=[])
        if train.sum()>=80 and val.sum()>=20:
            best=None; best_rmse=best_mae=np.inf
            for cfg in (ALPHAS if model=='ridge' else GRID):
                estimator=ridge_pipeline(cfg) if model=='ridge' else xgb.XGBRegressor(**XGB_BASE,**cfg)
                estimator.fit(X.loc[train],labels.loc[train])
                rmse,mae=score(estimator.predict(X.loc[val]),labels.loc[val])
                row=dict(params=cfg,validation_RMSE=rmse,validation_MAE=mae)
                if model=='ridge': row['preprocessing']=preprocessing_evidence(estimator,X.loc[train])
                details['candidates'].append(row)
                # Exact legacy XGBoost tie convention, including np.isclose rtol.
                choose=(rmse<best_rmse or (np.isclose(rmse,best_rmse,atol=1e-12) and mae<best_mae)) if model=='xgboost' else (rmse,mae)<(best_rmse,best_mae)
                if choose: best,best_rmse,best_mae=cfg,rmse,mae
            estimator=ridge_pipeline(best) if model=='ridge' else xgb.XGBRegressor(**XGB_BASE,**best)
            estimator.fit(X.loc[final],labels.loc[final])
            pred=estimator.predict(X.reindex(origins)).astype(float)
            details.update(status='OK',params=best)
            if model=='ridge':
                details['final_preprocessing']=preprocessing_evidence(estimator,X.loc[final])
                joblib.dump(estimator,cache/f'{node}_ridge.joblib')
            else: estimator.get_booster().save_model(cache/f'{node}_xgboost.ubj')
        result[f'forecast_{model}']=pred; info['models'][model]=details
    pred=np.full(len(origins),np.nan)
    details=dict(status='INSUFFICIENT_HISTORICAL_LABELS',candidates=[])
    historical=series.loc[:TRAIN_END]
    val_dates=series.index[(series.index>TRAIN_END)&(series.index<=FIT_END)]
    # Legacy minimum 60 observations; validation needs 20 observed targets.
    if np.isfinite(historical).sum()>=60 and np.isfinite(series.reindex(val_dates)).sum()>=20:
        best=None; best_loss=(np.inf,np.inf)
        for order in ORDERS:
            candidate_path=cache/f'{node}_arima_{order[0]}{order[1]}{order[2]}.json'
            if candidate_path.exists(): row=json.loads(candidate_path.read_text(encoding='utf-8'))
            else:
                row=dict(order=order,status='ERROR')
                try:
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        fitted=arima_model(historical.to_numpy(),order).fit(method_kwargs={'maxiter':200})
                    converged=bool(fitted.mle_retvals.get('converged',False))
                    vp=filtered_prediction(series.loc[:FIT_END],order,fitted.params,val_dates)
                    actual=series.reindex(val_dates).to_numpy(); ok=np.isfinite(actual)
                    rmse,mae=score(vp[ok],actual[ok])
                    row.update(status='OK' if converged and np.isfinite(rmse) else 'REJECTED_NONCONVERGENCE_OR_NONFINITE',
                               converged=converged,validation_RMSE=rmse,validation_MAE=mae,
                               warnings=sorted(set(str(w.message) for w in caught)))
                except Exception as exc: row['error']=str(exc)
                write_json(row,candidate_path)
            details['candidates'].append(row)
            if row['status']=='OK' and (row['validation_RMSE'],row['validation_MAE'])<best_loss:
                best=order; best_loss=(row['validation_RMSE'],row['validation_MAE'])
        if best is not None:
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fitted=arima_model(series.loc[:FIT_END].to_numpy(),best).fit(method_kwargs={'maxiter':200})
                converged=bool(fitted.mle_retvals.get('converged',False))
                details.update(order=best,params=fitted.params.tolist(),converged=converged,
                               warnings=sorted(set(str(w.message) for w in caught)))
                if converged:
                    pred=filtered_prediction(series,best,fitted.params,targets)
                    details['status']='OK'
                else: details['status']='FINAL_NONCONVERGENCE'
            except Exception as exc: details.update(status='FINAL_ERROR',error=str(exc))
        else: details['status']='NO_CONVERGED_CANDIDATE'
    result['forecast_arima']=pred; info['models']['arima']=details
    info['fit_seconds']=time.perf_counter()-started
    save(result,archive); write_json(info,meta_path)
    return info,False

def annotate(frame):
    out=frame.copy()
    out['target_available']=np.isfinite(out.actual_iv_at_target)
    for m in MODELS:
        out[f'available_{m}']=np.isfinite(out[f'forecast_{m}'])
        out[f'prediction_available_{m}']=out[f'available_{m}']
        out[f'scored_observation_{m}']=out.target_available&out[f'available_{m}']
    out['common_sample_4models']=out.target_available&out[[f'available_{m}' for m in MODELS]].all(axis=1)
    out['common_sample_qlike']=out.common_sample_4models&out.actual_iv_at_target.gt(0)&out[[f'forecast_{m}' for m in MODELS]].gt(0).all(axis=1)
    return out

def qlike(actual,pred):
    actual=np.asarray(actual,dtype=float); pred=np.asarray(pred,dtype=float)
    loss=np.full(actual.shape,np.nan)
    ok=np.isfinite(actual)&np.isfinite(pred)&(actual>0)&(pred>0)
    ratio=(actual[ok]**2)/(pred[ok]**2)
    loss[ok]=ratio-np.log(ratio)-1
    return loss

def metrics(frame,model,qlike_mask=None):
    actual=frame.actual_iv_at_target.to_numpy(); pred=frame[f'forecast_{model}'].to_numpy()
    err=pred-actual
    q=qlike(actual,pred)
    if qlike_mask is not None: q[~np.asarray(qlike_mask)]=np.nan
    finite=np.isfinite(q)
    return dict(n=len(frame),RMSE=float(np.sqrt(np.mean(err**2))) if len(frame) else np.nan,
                MAE=float(np.mean(np.abs(err))) if len(frame) else np.nan,
                QLIKE=float(np.mean(q[finite])) if finite.any() else np.nan,QLIKE_n=int(finite.sum()))

def hac_mean(values):
    values=np.asarray(values,dtype=float); values=values[np.isfinite(values)]
    if len(values)<2: return dict(mean=np.nan,statistic=np.nan,p_value=np.nan,n=len(values),hac_lag=HAC_LAG)
    fit=sm.OLS(values,np.ones((len(values),1))).fit(cov_type='HAC',
                cov_kwds={'maxlags':HAC_LAG,'use_correction':True},use_t=True)
    return dict(mean=float(fit.params[0]),statistic=float(fit.tvalues[0]),p_value=float(fit.pvalues[0]),n=len(values),hac_lag=HAC_LAG)

def bootstrap_intervals(daily,reps=REPS,block=5):
    dates=sorted(daily.forecast_target_date.unique()); n=len(dates)
    rng=np.random.default_rng(42)
    starts=rng.integers(0,n,size=(reps,int(np.ceil(n/block))))
    indices=((starts[:,:,None]+np.arange(block))%n).reshape(reps,-1)[:,:n]
    output=[]
    for m in MODELS:
        day=daily[daily.model.eq(m)].set_index('forecast_target_date').reindex(dates)
        for metric,sum_col,count_col in [('RMSE','sum_SE','n_common'),('MAE','sum_AE','n_common'),('QLIKE','sum_QLIKE','n_qlike')]:
            sums=day[sum_col].to_numpy(); counts=day[count_col].to_numpy()
            vals=sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
            if metric=='RMSE': vals=np.sqrt(vals)
            low,high=np.quantile(vals,[.025,.975])
            output.append(dict(model=m,metric=metric,lower_95=low,upper_95=high,block_length=block,reps=reps,seed=42,
                               weighting='node-observation weighted; entire date cross-section'))
    return pd.DataFrame(output)

def daily_statistics(common):
    rows=[]; ic=[]
    for date,day in common.groupby('forecast_target_date',sort=True):
        qm=day.common_sample_qlike.to_numpy()
        actual=day.actual_iv_at_target.to_numpy(); origin=day.observed_iv_at_origin.to_numpy()
        for m in MODELS:
            pred=day[f'forecast_{m}'].to_numpy(); err=pred-actual
            q=qlike(actual[qm],pred[qm])
            rows.append(dict(forecast_target_date=date,model=m,n_common=len(day),n_qlike=int(qm.sum()),
                L_SE=float(np.mean(err**2)),L_AE=float(np.mean(np.abs(err))),L_QLIKE=float(np.mean(q)) if len(q) else np.nan,
                sum_SE=float(np.sum(err**2)),sum_AE=float(np.sum(np.abs(err))),sum_QLIKE=float(np.sum(q))))
            change=pred-origin; realized=actual-origin
            valid=len(day)>=MIN_IC and np.ptp(change)>0 and np.ptp(realized)>0
            value=float(spearmanr(change,realized).statistic) if valid else np.nan
            reason='OK' if valid else ('TOO_FEW_NODES' if len(day)<MIN_IC else 'CONSTANT_CROSS_SECTION')
            ic.append(dict(forecast_target_date=date,model=m,n_common=len(day),rank_ic=value,status=reason))
    daily=pd.DataFrame(rows); rank=pd.DataFrame(ic); summary=[]
    for m in MODELS:
        a=rank.loc[rank.model.eq(m),'rank_ic'].dropna(); h=hac_mean(a)
        summary.append(dict(model=m,mean_daily_IC=a.mean(),median_daily_IC=a.median(),std_daily_IC=a.std(),
                            n_valid_dates=len(a),percent_positive=100*a.gt(0).mean() if len(a) else np.nan,
                            NW_t_statistic=h['statistic'],p_value=h['p_value'],q25=a.quantile(.25),q75=a.quantile(.75),hac_lag=HAC_LAG))
    return daily,rank,pd.DataFrame(summary)

def inference(daily):
    dm=[]; mcs=[]
    for loss in ['SE','AE','QLIKE']:
        matrix=daily.pivot(index='forecast_target_date',columns='model',values=f'L_{loss}')[MODELS].dropna()
        for a,b in itertools.combinations(MODELS,2):
            h=hac_mean(matrix[a]-matrix[b])
            dm.append(dict(loss=loss,model_a=a,model_b=b,mean_loss_differential=h['mean'],
                       DM_t_statistic=h['statistic'],p_value=h['p_value'],n_dates=h['n'],hac_lag=HAC_LAG,
                       lower_loss_model=a if h['mean']<0 else b,significant_5pct=h['p_value']<.05))
        for block in BLOCKS:
            test=MCS(matrix,size=.05,reps=REPS,block_size=block,method='R',bootstrap='circular',seed=42)
            test.compute()
            for m in MODELS:
                mcs.append(dict(loss=loss,model=m,p_value=float(test.pvalues.loc[m,'Pvalue']),included=m in test.included,
                    confidence_level=.95,test_size=.05,n_dates=len(matrix),block_length=block,primary=block==5,
                    reps=REPS,seed=42,method='R',bootstrap='circular',arch_version=arch.__version__))
    return pd.DataFrame(dm),pd.DataFrame(mcs)

def draw(daily,rank,folder):
    plt.rcParams.update({'font.size':10,'figure.figsize':(10,4),'axes.spines.top':False,'axes.spines.right':False})
    matrix=daily.pivot(index='forecast_target_date',columns='model',values='L_SE')[MODELS]
    fig,ax=plt.subplots()
    for m in MODELS: ax.plot(matrix.index,np.sqrt(matrix[m].rolling(20,min_periods=10).mean()),label=m.title(),lw=1)
    ax.set(xlabel='Target session',ylabel='20-date rolling root mean squared error (IV)');ax.legend(ncol=4);fig.tight_layout()
    fig.savefig(folder/'forecast_daily_rmse.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots()
    im=rank.pivot(index='forecast_target_date',columns='model',values='rank_ic')
    for m in MODELS:
        if im[m].notna().any(): ax.plot(im.index,im[m].rolling(20,min_periods=10).mean(),label=m.title(),lw=1)
    ax.axhline(0,color='black',lw=.6);ax.set(xlabel='Target session',ylabel='20-date rolling mean Spearman rank IC')
    ax.legend();fig.tight_layout();fig.savefig(folder/'forecast_rank_ic_timeseries.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,loss in zip(axes,['SE','AE','QLIKE']):
        lm=daily.pivot(index='forecast_target_date',columns='model',values=f'L_{loss}')
        for m in MODELS[1:]: ax.plot(lm.index,(lm[m]-lm.persistence).cumsum(),label=m.title(),lw=1)
        ax.axhline(0,color='black',lw=.6);ax.set(title=loss,xlabel='Target session',ylabel='Cumulative daily loss difference')
        ax.tick_params(axis='x',rotation=30)
    axes[0].legend();fig.tight_layout();fig.savefig(folder/'forecast_loss_relative_to_persistence.png',dpi=180);plt.close(fig)

def table(frame,columns=None):
    frame=frame[columns] if columns else frame
    def cell(value):
        if pd.isna(value): return 'NA'
        if isinstance(value,(float,np.floating)): return f'{value:.6g}'
        return str(value).replace('|','\\|').replace('\n',' ')
    header='| '+' | '.join(map(str,frame.columns))+' |'
    rule='| '+' | '.join(['---']*len(frame.columns))+' |'
    rows=['| '+' | '.join(cell(value) for value in row)+' |' for row in frame.itertuples(index=False,name=None)]
    return '\n'.join([header,rule]+rows)
