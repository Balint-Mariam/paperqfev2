"""Streaming upstream reconstruction with independently verified stage gates."""
import json
import io
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .cleaning import apply_basic_filters, prepare_rates_df
from .implied_volatility import compute_chunk_iv
from . import iv_surface as surface
from .io_utils import digest, token, write_json, save
from .environment import start_log, code_identity
from .validation import check, funnel


def identity(config, info):
    inputs = {key: dict(name=config[key].name, sha256=digest(config[key]))
              for key in ['raw_options_file', 'rates_file']}
    methods = {k:v for k,v in config.items() if k not in ['force','workers','output_root','reference_root','config_path','raw_options_file','rates_file']}
    code={name:digest(Path(__file__).parent/(name+'.py')) for name in ['cleaning','implied_volatility','iv_surface','data_pipeline','validation']}
    return dict(inputs=inputs, config=methods, code=code, versions=info['versions'])


def count_csv(path):
    with Path(path).open('rb') as stream:
        count = sum(chunk.count(b'\n') for chunk in iter(lambda: stream.read(8*1024*1024), b''))
    return max(count-1, 0)


def clean(config):
    data = config['output_root']/'data'
    settings = config['cleaning']
    rates = prepare_rates_df(config['rates_file'])
    stats = {}; writer = None; started = time.perf_counter()
    try:
        for i, chunk in enumerate(pd.read_csv(config['raw_options_file'], chunksize=settings['chunksize'])):
            frame, counts = apply_basic_filters(chunk, rates, settings['t_min_days'], settings['t_max_years'], settings['spread_rel_max'])
            for k, v in counts.items(): stats[k] = stats.get(k, 0) + v
            if len(frame):
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None: writer = pq.ParquetWriter(data/'options_clean.parquet', table.schema, compression='zstd')
                writer.write_table(table)
            if (i+1)%25 == 0:
                print(f'Cleaning: {stats["input"]:,} raw rows; {stats["kept_basic"]:,} retained; {time.perf_counter()-started:.1f}s', flush=True)
    finally:
        if writer is not None: writer.close()
    write_json(stats, config['output_root']/'diagnostics/cleaning_counts.json')
    funnel(stats).to_csv(config['output_root']/'diagnostics/data_cleaning_funnel.csv',index=False)
    return stats


def upstream_clean_checks(config, stats):
    ref = config['reference_root']/'options_eod_all_clean.csv'
    if not ref.is_file():
        return [dict(check_name='reference_clean_available',status='WARNING',observed=False,reference=True,max_abs_difference=None,tolerance=0)]
    n = count_csv(ref)
    checks = [check('clean_row_count', stats['kept_basic'], n)]
    first = pd.read_csv(ref, nrows=200)
    new = next(pq.ParquetFile(config['output_root']/'data/options_clean.parquet').iter_batches(batch_size=200)).to_pandas()
    # Check the first actual source observations before costly millions of inversions.
    for column in first:
        if column not in new:
            checks.append(check('clean_column_'+column,False,True)); continue
        a,b = new[column], first[column]
        if pd.api.types.is_datetime64_any_dtype(a): b=pd.to_datetime(b,format='mixed',dayfirst=True)
        checks.append(check('clean_sample_'+column,a.to_numpy(),b.to_numpy(),1e-8))
    return checks


def compare_option_values(config, stem, reference_name):
    """Compare every stored row/column, preserving historical CSV round trips."""
    path=config['reference_root']/reference_name
    if not path.is_file():return [dict(check_name=stem+'_reference',status='WARNING',observed=False,reference=True,max_abs_difference=None,tolerance=0)]
    parquet=pq.ParquetFile(config['output_root']/'data'/(stem+'.parquet'))
    rows=parquet.metadata.num_rows;checks=[check(stem+'_all_rows',rows,count_csv(path))]
    if checks[0]['status']=='FAIL':return checks
    reference=pd.read_csv(path,chunksize=50000);differences={};ok={};total=0
    for batch,ref in zip(parquet.iter_batches(batch_size=50000),reference):
        new=batch.to_pandas();total+=len(new)
        # Compare numeric values as read by the historical next CSV stage.
        new=pd.read_csv(io.StringIO(new.drop(columns=['iv_status'],errors='ignore').to_csv(index=False)))
        if set(new)!=set(ref):return checks+[check(stem+'_all_columns',sorted(new),sorted(ref))]
        for col in ref:
            a,b=new[col],ref[col]
            if col in ['quote_date','expiration','Calendar Date']:a=pd.to_datetime(a,format='mixed');b=pd.to_datetime(b,format='mixed')
            result=check(col,a.to_numpy(),b.to_numpy(),1e-9 if col=='implied_vol' else 1e-8)
            ok[col]=ok.get(col,True) and result['status']=='PASS'
            differences[col]=max(differences.get(col,0.),result['max_abs_difference'])
        if total%1000000==0:print(f'Full {stem} comparison: {total:,}/{rows:,}',flush=True)
    for col in ok:checks.append(dict(check_name=stem+'_all_values_'+col,status='PASS' if ok[col] else 'FAIL',observed=rows,reference=rows,max_abs_difference=differences[col],tolerance=1e-9 if col=='implied_vol' else 1e-8))
    return checks


def iv_worker(frame):
    # Preserve the historical clean CSV -> pandas numeric parsing boundary.
    frame=pd.read_csv(io.StringIO(frame.to_csv(index=False)))
    before = len(frame)
    out = compute_chunk_iv(frame, None)
    valid = np.isfinite(out.implied_vol)&out.implied_vol.gt(0)
    out['iv_status'] = np.where(valid,'VALID_IV','SOLVER_FAILURE')
    return out, dict(input=before,invalid_input=before-len(out),valid=int(valid.sum()),solver_failure=int((~valid).sum()))


def invert(config):
    data=config['output_root']/'data'; stats={}; writer=None
    reader=pq.ParquetFile(data/'options_clean.parquet')
    batches=(b.to_pandas() for b in reader.iter_batches(batch_size=config['iv']['chunksize']))
    # Submit bounded batches to avoid loading the complete option archive into RAM.
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        while True:
            block=[]
            for _ in range(config['workers']):
                try: block.append(next(batches))
                except StopIteration: break
            if not block: break
            for frame,counts in pool.map(iv_worker,block):
                for k,v in counts.items(): stats[k]=stats.get(k,0)+v
                table=pa.Table.from_pandas(frame,preserve_index=False)
                if writer is None: writer=pq.ParquetWriter(data/'options_with_iv.parquet',table.schema,compression='zstd')
                writer.write_table(table)
            print(f'IV: {stats["input"]:,} rows; {stats["valid"]:,} valid',flush=True)
    if writer is not None: writer.close()
    write_json(stats,config['output_root']/'diagnostics/iv_counts.json')
    return stats


def build_surface(config):
    s=config['surface']; data=config['output_root']/'data'; parts=[]
    # Same validity and coordinate rules as prepare_points, with Parquet input.
    for batch in pq.ParquetFile(data/'options_with_iv.parquet').iter_batches(batch_size=200000):
        # Surface inputs historically pass through IV CSV numeric parsing.
        d=pd.read_csv(io.StringIO(batch.to_pandas().drop(columns=['iv_status']).to_csv(index=False)))
        d['quote_date']=pd.to_datetime(d.quote_date,dayfirst=True,format='mixed')
        k=pd.to_numeric(d.moneyness,errors='coerce')
        d['log_moneyness']=np.log(k); d=d.replace([np.inf,-np.inf],np.nan)
        d=d.dropna(subset=['quote_date','implied_vol','log_moneyness','T'])
        d=d[d.implied_vol.gt(0)&k.gt(0)&d['T'].between(s['t_min'],s['t_max'])&d.log_moneyness.between(s['x_min'],s['x_max'])]
        parts.append(d[['quote_date','log_moneyness','T','implied_vol']])
    points=pd.concat(parts,ignore_index=True)
    _,_,gx,gt,names,grid=surface.build_grid_and_features(s['x_min'],s['x_max'],s['x_points'],s['t_min'],s['t_max'],s['t_points'])
    rows=[]; long=[]; stats=[]
    for date,day in points.groupby('quote_date',sort=True):
        unique=day[['log_moneyness','T']].drop_duplicates().shape[0]
        if unique<s['min_points_day']:
            stats.append(dict(quote_date=date,n_obs_raw=len(day),n_obs_unique=unique,coverage=0.,status='skipped_min_points',interp_used=''));continue
        z,method=surface.interpolate_day_surface(day,gx,gt,s['method'],s['fill_nearest'],s['smooth_sigma']);coverage=np.isfinite(z).mean()
        kept=coverage>=s['min_coverage'];stats.append(dict(quote_date=date,n_obs_raw=len(day),n_obs_unique=unique,coverage=coverage,status='kept' if kept else 'skipped_min_coverage',interp_used=method))
        if kept:
            rows.append(dict(quote_date=date,**dict(zip(names,z.ravel()))))
            long.append(pd.DataFrame(dict(quote_date=date,log_moneyness=gx.ravel(),T=gt.ravel(),iv_grid=z.ravel())))
    wide=pd.DataFrame(rows); long=pd.concat(long,ignore_index=True); stats=pd.DataFrame(stats)
    for name,frame in [('iv_grid_wide',wide),('iv_grid_long',long),('iv_grid_map',grid),('iv_grid_day_stats',stats)]:
        save(frame,data/(name+'.parquet'));frame.to_csv(data/(name+'.csv'),index=False)
    return wide,grid,stats


def run(config):
    started=time.perf_counter();info=start_log(config,1);out=config['output_root'];meta_path=out/'diagnostics/stage1_manifest.json'
    print('Hashing raw options and rates for reproducible cache identity...',flush=True)
    ident=identity(config,info); signature=token(ident);checks=[]
    old=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    cleanpath=out/'data/options_clean.parquet'
    cached=(not config['force'] and old.get('clean_signature')==signature and cleanpath.exists() and old.get('clean_sha256')==digest(cleanpath))
    stats=old['cleaning'] if cached else clean(config)
    checks+=upstream_clean_checks(config,stats)
    checks+=compare_option_values(config,'options_clean','options_eod_all_clean.csv')
    metadata=dict(identity=ident,clean_signature=signature,clean_sha256=digest(cleanpath),cleaning=stats)
    write_json(metadata,meta_path)
    pd.DataFrame(checks).to_csv(out/'diagnostics/stage1_reproduction_checks.csv',index=False)
    if any(c['status']=='FAIL' for c in checks):
        metadata.update(status='FAILED',runtime_seconds=time.perf_counter()-started)
        pd.DataFrame(checks).to_csv(out/'diagnostics/stage1_reproduction_checks.csv',index=False)
        write_json(metadata,meta_path)
        print('STOP — UPSTREAM REPRODUCTION FAILURE\nSTAGE 1 FAILED',flush=True)
        print(json.dumps(dict(raw_rows=stats['input'],clean_rows=stats['kept_basic'],valid_IV_rows=None,grid_dates=None,grid_nodes=500,grid_non_null_rows=None,runtime_seconds=metadata['runtime_seconds'],checks=pd.DataFrame(checks).status.value_counts().to_dict(),output_root=str(out)),indent=2))
        return 1
    ivpath=out/'data/options_with_iv.parquet'
    ivcached=not config['force'] and old.get('iv_signature')==signature and ivpath.exists() and old.get('iv_sha256')==digest(ivpath)
    ivstats=old['iv'] if ivcached else invert(config)
    checks+=compare_option_values(config,'options_with_iv','options_eod_all_with_iv.csv')
    metadata.update(iv_signature=signature,iv_sha256=digest(ivpath),iv=ivstats)
    write_json(metadata,meta_path)
    wide,grid,days=build_surface(config)
    ref=config['reference_root']
    for name,frame in [('iv_grid_wide',wide),('iv_grid_map',grid),('iv_grid_day_stats',days)]:
        path=ref/(name+'.csv')
        if not path.exists():
            checks.append(dict(check_name=name+'_reference',status='WARNING',observed=False,reference=True,max_abs_difference=None,tolerance=0));continue
        frozen=pd.read_csv(path)
        checks.append(check(name+'_columns',list(frame),list(frozen)))
        checks.append(check(name+'_rows',len(frame),len(frozen)))
        for col in frozen:
            a=frame[col];b=frozen[col]
            if pd.api.types.is_datetime64_any_dtype(a): b=pd.to_datetime(b)
            checks.append(check(name+'_'+col,a.to_numpy(),b.to_numpy(),1e-9))
    pd.DataFrame(checks).to_csv(out/'diagnostics/stage1_reproduction_checks.csv',index=False)
    coverage=wide.filter(like='iv_').notna().mean(axis=1)
    summary=dict(raw_rows=stats['input'],clean_rows=stats['kept_basic'],valid_IV_rows=ivstats['valid'],grid_dates=len(wide),grid_nodes=len(grid),potential_grid_observations=len(wide)*len(grid),non_missing_grid_observations=int(wide.filter(like='iv_').notna().sum().sum()),coverage_mean=coverage.mean(),coverage_median=coverage.median(),coverage_min=coverage.min(),coverage_max=coverage.max(),first_date=str(wide.quote_date.min()),last_date=str(wide.quote_date.max()))
    write_json(summary,out/'diagnostics/iv_grid_summary.json')
    metadata.update(status='FAILED' if any(c['status']=='FAIL' for c in checks) else 'REPRODUCED',grid_sha256=digest(out/'data/iv_grid_wide.csv'),summary=summary,runtime_seconds=time.perf_counter()-started)
    write_json(metadata,meta_path)
    print('STAGE 1 REPRODUCED' if metadata['status']=='REPRODUCED' else 'STOP — UPSTREAM REPRODUCTION FAILURE\nSTAGE 1 FAILED')
    print(json.dumps(dict(**summary,runtime_seconds=metadata['runtime_seconds'],checks=pd.DataFrame(checks).status.value_counts().to_dict(),output_root=str(out)),indent=2))
    return int(metadata['status']=='FAILED')
