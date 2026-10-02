"""Optional certified fitted-state reuse; all predictions are regenerated."""
import inspect
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from . import forecasting as F, economic_signals as E, contract_mapping as M
from .config import PACKAGE
from .io_utils import digest, save, write_json
from .cache import prepare, complete


def validate_source(config,full,grid):
    root=config['reference_root'];cert=json.loads((PACKAGE/'manifests/certified_checkpoint_identity.json').read_text())
    # Absence means a normal from-scratch run, never a requirement for handoff.
    if config['force'] or not all((root/path).is_dir() for path in cert['directories'].values()):return None
    if cert['versions']!=F.VERSIONS or cert['OMP_NUM_THREADS']!=os.environ.get('OMP_NUM_THREADS'):return None
    locked=json.loads((PACKAGE/'manifests/locked_specification.json').read_text())['forecast']
    if (F.ORDERS!=[tuple(o) for o in locked['orders']] or F.ALPHAS!=locked['ridge_alphas']
            or M.GRID!=locked['xgboost_grid'] or str(F.TRAIN_END.date())!=locked['train_end']
            or str(F.FIT_END.date())!=locked['fit_end']):return None
    ref=root/'iv_grid_wide.csv';mapref=root/'iv_grid_map.csv'
    if digest(ref)!=cert['surface_sha256'] or digest(mapref)!=cert['grid_sha256']:return None
    frozen=pd.read_csv(ref);frozen.quote_date=pd.to_datetime(frozen.quote_date);frozen=frozen.set_index('quote_date').sort_index()
    frozen=frozen[grid.node.tolist()].reindex(full.index)
    if not np.array_equal(full.to_numpy(),frozen.to_numpy(),equal_nan=True):
        print('Certified fit cache not reused: reconstructed model inputs are not bit-identical',flush=True);return None
    provenance=json.loads((PACKAGE/'manifests/method_source_provenance.json').read_text())
    checks=[(F,['features','masks','ridge_pipeline','preprocessing_evidence','score','arima_model','filtered_prediction','fit_node']),
            (E,['horizon_features','horizon_masks','arima_native','fit_h2']),(M,['features','fit_forecasts'])]
    import hashlib
    for module,names in checks:
        label=module.__name__.split('.')[-1]
        for name in names:
            expected=next(r['function_sha256'] for r in provenance if r['module']==label and r['function']==name)
            source=inspect.getsource(getattr(module,name)).rstrip('\n')
            if hashlib.sha256(source.encode()).hexdigest()!=expected:return None
    for path,sha in cert['files'].items():
        checkpoint=root/Path(path.replace('\\','/'))
        if not checkpoint.is_file() or digest(checkpoint)!=sha:raise ValueError('Certified fitted-state checkpoint changed: '+path)
    write_json(dict(certified_input_values_bit_identical=True,certified_files=len(cert['files']),certificate_sha256=digest(PACKAGE/'manifests/certified_checkpoint_identity.json'),mode='regenerate predictions from fitted states'),config['output_root']/'diagnostics/certified_checkpoint_reuse.json')
    return root,cert


def estimator_prediction(path,X):
    booster=xgb.Booster(model_file=str(path))
    # XGBRegressor.predict uses inplace_predict for the certified tree predictor.
    return booster.inplace_predict(X).astype(float)


def adopt(config,cache,signature,full,calendar,origins,grid):
    validated=validate_source(config,full,grid)
    if validated is None:return False
    root,cert=validated;dirs={k:root/v for k,v in cert['directories'].items()}
    print('Validated certified fitted states; regenerating all h1/h2 predictions from the new surface',flush=True)
    targets=pd.DatetimeIndex(pd.Series(calendar,index=calendar).shift(-1).reindex(origins))
    for node in grid.node:
        series=full[node];economic=series.loc[:origins.max()]
        for horizon,source,values in [('h1',dirs['h1'],series),('h2',dirs['h2'],economic)]:
            folder=cache/horizon/node
            if prepare(folder,signature):continue
            info=json.loads((source/(node+'.json')).read_text(encoding='utf-8'))
            X=F.features(values,calendar) if horizon=='h1' else E.horizon_features(values,calendar,2)
            frame=pd.DataFrame(dict(forecast_origin_date=origins,forecast_target_date=targets,node=node)) if horizon=='h1' else pd.DataFrame(dict(decision_date=origins,node=node))
            suffix='' if horizon=='h1' else '_h2'
            if horizon=='h1':frame['forecast_persistence']=series.reindex(origins).to_numpy()
            details=info['models']
            frame['forecast_ridge'+suffix]=joblib.load(source/(node+'_ridge.joblib')).predict(X.reindex(origins)).astype(float) if details['ridge']['status']=='OK' else np.nan
            ar=details['arima']
            frame['forecast_arima'+suffix]=(F.filtered_prediction(series,ar['order'],ar['params'],targets) if horizon=='h1' else E.arima_native(economic,ar['order'],ar['params'],origins,2)) if ar['status']=='OK' else np.nan
            if horizon=='h1':frame['forecast_xgboost']=estimator_prediction(source/(node+'_xgboost.ubj'),X.reindex(origins)) if details['xgboost']['status']=='OK' else np.nan
            save(frame,folder/(node+'.parquet'));write_json(info,folder/(node+'.json'));complete(folder,signature)
        folder=cache/'xgb_h2'/node
        if not prepare(folder,signature):
            source=dirs['xgb_h2'];info=json.loads((source/(node+'_h2.json')).read_text(encoding='utf-8'))
            pred=estimator_prediction(source/(node+'_h2.ubj'),M.features(economic,calendar,2).reindex(origins)) if info['status']=='OK' else np.full(len(origins),np.nan)
            save(pd.DataFrame(dict(decision_date=origins,node=node,forecast_xgboost_h2=pred)),folder/(node+'.parquet'));write_json(info,folder/(node+'.json'));complete(folder,signature)
    return True
