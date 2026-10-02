"""Plain numerical workbook, sheet mirrors, and a hashed delivery inventory."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from .config import PACKAGE
from .io_utils import digest, save, write_json
from .environment import code_identity


def read(config,name):return pd.read_csv(config['output_root']/'tables'/(name+'.csv'))


def file_manifest(config):
    out=config['output_root'];rows=[]
    excluded={'QFE_Full_Results.xlsx','22_File_Manifest.csv','22_File_Manifest.parquet','file_manifest.csv'}
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p.name in excluded or 'cache' in p.relative_to(out).parts or p.suffix=='.log':continue
        n_rows=n_columns=None
        if p.suffix=='.csv':
            d=pd.read_csv(p);n_rows,n_columns=d.shape
        elif p.suffix=='.parquet':
            import pyarrow.parquet as pq
            metadata=pq.ParquetFile(p).metadata;n_rows,n_columns=metadata.num_rows,metadata.num_columns
        rows.append(dict(filename=p.name,relative_path=str(p.relative_to(out)),sha256=digest(p),rows=n_rows,columns=n_columns))
    return pd.DataFrame(rows)


def implementation_summary(config):
    out=config['output_root'];signals=pd.read_parquet(out/'data/economic_signals_3models.parquet');trades=pd.read_parquet(out/'data/trades_3models_preperformance.parquet');ready=pd.read_parquet(out/'data/trades_3models_stage3b_ready.parquet');contracts=pd.read_parquet(out/'data/stage4_contract_pnl.parquet');calendar=pd.read_parquet(out/'data/stage4_economic_calendar.parquet')
    rows=[]
    for model,d in contracts.groupby('model'):
        s=signals[signals.model.eq(model)];t=trades[trades.model.eq(model)];r=ready[ready.model.eq(model)];c=calendar[calendar.model.eq(model)]
        rows.append(dict(model=model,selected_signals=int(s.selected.sum()),entry_mapped=int(t.entry_mapping_status.eq('MAPPED').sum()),unique_net_contract_rows=len(r),active_net_contract_rows=int(r.nonzero_net_position.sum()),evaluated_contract_rows=len(d),processed_exits=int(d.exit_quote_source.eq('PROCESSED_1545').sum()),raw_recovered_exits=int(d.exit_quote_source.eq('RAW_1545_RECOVERED').sum()),final_exit_coverage=float(d.BASE.mean()),calendar_excluded_decision_days=int((~c.economic_calendar_eligible).sum()),calendar_excluded_entry_days=int((~c.entry_snapshot_session).sum()),calendar_excluded_exit_days=int((~c.exit_snapshot_session).sum()),trading_days=d.entry_date.nunique()))
    return pd.DataFrame(rows)


def all_checks(config):
    out=config['output_root'];frames=[]
    for stage,name in [(1,'stage1_reproduction_checks'),(2,'analysis_checks')]:
        d=pd.read_csv(out/'diagnostics'/(name+'.csv')).rename(columns={'check_name':'check','max_abs_difference':'absolute_difference'})
        d.insert(0,'stage',stage);frames.append(d)
    compare=pd.read_csv(out/'diagnostics/replication_final_comparison.csv').rename(columns={'result_name':'check','new_value':'observed','reference_value':'reference'})
    compare.insert(0,'stage',2);frames.append(compare)
    return pd.concat(frames,ignore_index=True)[['stage','check','status','observed','reference','absolute_difference','tolerance']]


def export_results(config,info,stage1,stage2):
    out=config['output_root'];raw=stage1['identity']['inputs']['raw_options_file']
    runinfo=dict(run_timestamp=info['timestamp'],python_version=info['python'],platform=info['platform'],random_seed=42,raw_input_name=raw['name'],raw_input_hash=raw['sha256'],cleaned_data_hash=stage1['clean_sha256'],iv_grid_hash=stage1['grid_sha256'],code_hash=code_identity(),Stage1_runtime=stage1['runtime_seconds'],Stage2_runtime=stage2['runtime_seconds'],Stage1_status=stage1['status'],Stage2_status=stage2['status'])
    forecast=read(config,'forecast_model_comparison').merge(read(config,'rank_ic_summary')[['model','mean_daily_IC']],on='model',validate='one_to_one')
    signal=read(config,'economic_signal_predictive_summary')
    candidates=pd.read_parquet(out/'data/economic_forecasts_3models.parquet');selections=pd.read_parquet(out/'data/economic_signals_3models.parquet')
    signal['economic_candidate_count']=int(candidates.common_economic_candidate.sum())
    signal['selected_signals']=signal.model.map(selections[selections.selected].groupby('model').size())
    hedge_break=read(config,'stage6_hedge_cost_break_even');hedge_break.insert(0,'record_type','BREAK_EVEN')
    hedge_stress=read(config,'stage6_hedge_cost_stress');hedge_stress.insert(0,'record_type','STRESS')
    sheets={
        '00_Run_Info':pd.DataFrame(runinfo.items(),columns=['key','value']),
        '01_Data_Cleaning':pd.read_csv(out/'diagnostics/data_cleaning_funnel.csv'),
        '02_IV_Grid_Summary':pd.DataFrame(stage1['summary'].items(),columns=['metric','value']),
        '03_Forecast_Performance':forecast,
        '04_DM_Tests':read(config,'diebold_mariano_results'),
        '05_MCS':read(config,'model_confidence_set'),
        '06_Economic_Signal':signal,
        '07_Implementation':implementation_summary(config),
        '08_Original_Performance':read(config,'stage4_performance_summary'),
        '09_Greek_Exposure':read(config,'stage5_greek_exposure_summary'),
        '10_Greek_Attribution':read(config,'stage5_pnl_attribution_summary'),
        '11_DGVT_Subset':read(config,'stage5_dgvt_subset_summary'),
        '12_Risk_Control':read(config,'stage5_performance_summary'),
        '13_Multiple_Testing':read(config,'stage6_multiple_testing'),
        '14_Paired_Control_Effects':read(config,'stage6_risk_control_effects'),
        '15_Normalization':read(config,'stage6_normalization_sensitivity'),
        '16_Hedge_Costs':pd.concat([hedge_break,hedge_stress],ignore_index=True),
        '17_Subperiods':read(config,'stage6_subperiod_stability'),
        '18_IV_Regimes':read(config,'stage6_iv_regime_stability'),
        '19_Full_Chain':read(config,'stage6_full_chain_summary'),
        '20_Model_Ranks':read(config,'stage6_model_ranking'),
        '21_Checks':all_checks(config),
    }
    for name,frame in sheets.items():frame.to_csv(out/'tables'/(name+'.csv'),index=False)
    sheets['22_File_Manifest']=file_manifest(config)
    sheets['22_File_Manifest'].to_csv(out/'tables/22_File_Manifest.csv',index=False)
    sheets['22_File_Manifest'].to_csv(out/'diagnostics/file_manifest.csv',index=False)
    book=Workbook();book.remove(book.active)
    for name,frame in sheets.items():
        sheet=book.create_sheet(name);sheet.append(list(frame.columns));sheet.freeze_panes='A2'
        for values in frame.itertuples(index=False,name=None):
            row=[]
            for value in values:
                if isinstance(value,np.generic):value=value.item()
                if pd.isna(value) or isinstance(value,float) and not np.isfinite(value):value=None
                if isinstance(value,str) and value.startswith('='):value="'"+value
                row.append(value)
            sheet.append(row)
    path=out/'excel/QFE_Full_Results.xlsx';book.save(path)
    return path


def delivery_manifest(config):
    out=config['output_root'];paths=set()
    for folder in [PACKAGE/'src',PACKAGE/'config',PACKAGE/'tests',PACKAGE/'manifests',PACKAGE/'inputs',out]:
        paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update([PACKAGE/'01_build_clean_iv_data.py',PACKAGE/'02_run_full_paper_analysis.py',PACKAGE/'requirements.txt',PACKAGE/'README_REPLICATION.md',PACKAGE/'METHODOLOGY.md'])
    inventory=[]
    for p in sorted(paths):
        try:relative=str(p.relative_to(PACKAGE))
        except ValueError:relative=str(p)
        inventory.append(dict(path=relative,size_bytes=p.stat().st_size,sha256=digest(p)))
    path=PACKAGE/'REPLICATION_DELIVERY_MANIFEST.json'
    statuses={}
    for stage in [1,2]:
        metadata=out/'diagnostics'/f'stage{stage}_manifest.json'
        statuses['stage'+str(stage)]=json.loads(metadata.read_text())['status'] if metadata.exists() else 'NOT_RUN'
    write_json(dict(entry_points=['01_build_clean_iv_data.py','02_run_full_paper_analysis.py'],replication_status=statuses,files=inventory,workbook=str(out/'excel/QFE_Full_Results.xlsx'),self_hash_excluded=True),path)
    return path
