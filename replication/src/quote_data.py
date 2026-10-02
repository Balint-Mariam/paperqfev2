"""Processed entry partitions and one targeted raw recovery scan."""
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from . import contract_mapping as mapping
from .io_utils import digest, token, write_json, save
from .cache import valid, complete, prepare


def market_partitions(config, dates, signature):
    folder=config['output_root']/'cache'/'entry_market'/signature
    marker=folder/'market_identity.json'
    if marker.is_file():
        import json
        identity=json.loads(marker.read_text())
        if identity['signature']==signature and all((folder/k).is_file() and digest(folder/k)==v for k,v in identity['files'].items()):return folder
    folder.mkdir(parents=True,exist_ok=True)
    parts=[];required=set(dates)
    for i,batch in enumerate(pq.ParquetFile(config['output_root']/'data/options_with_iv.parquet').iter_batches(batch_size=200000)):
        frame=batch.to_pandas();frame=frame[frame.quote_date.isin(required)]
        if frame.empty:continue
        use=['quote_date','underlying_symbol','root','expiration','strike','option_type','bid_1545','ask_1545','bid_eod','ask_eod','spot','mid','moneyness','T','implied_vol','r_annual','q_annual','trade_volume','open_interest']
        normalized=mapping.normalize(frame[use])
        for date,day in normalized.groupby('quote_date'):
            path=folder/('date='+date.strftime('%Y-%m-%d'))/f'part_{i:05d}.parquet';save(day,path)
    files={str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*.parquet'))}
    write_json(dict(signature=signature,files=files),marker)
    return folder


def processed_pairs(required, market):
    frames=[]
    for date,day in required.groupby('intended_exit_date',sort=True):
        quotes=mapping.load_day(market,date)
        if len(quotes):frames.append(quotes[quotes.contract_key.isin(day.contract_key)])
    columns=['quote_date','contract_key','bid','ask','mid','spot','implied_vol','T','r','q','trade_volume','open_interest','source_observation_count']
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)


def normalize_raw(frame):
    out=frame.copy()
    for c in ['quote_date','expiration']:out[c]=pd.to_datetime(out[c],format='%Y-%m-%d',errors='coerce')
    for c in ['underlying_symbol','root','option_type']:out[c]=out[c].astype(str).str.strip().str.upper()
    out['option_type']=out.option_type.str[0];out['strike']=pd.to_numeric(out.strike,errors='coerce').round(6)
    out['contract_key']=out.underlying_symbol+'|'+out.root+'|'+out.expiration.dt.strftime('%Y-%m-%d')+'|'+out.strike.map(lambda v:f'{v:.6f}')+'|'+out.option_type
    out['spot']=(out.underlying_bid_1545+out.underlying_ask_1545)/2
    out['mid']=(out.bid_1545+out.ask_1545)/2
    return out


def raw_failure_flags(out, rate_dates):
    T=(out.expiration-out.quote_date).dt.days/365.;mid=out.mid
    conditions={'NONPOSITIVE_BID':~out.bid_1545.gt(0),'ASK_NOT_ABOVE_BID':~out.ask_1545.gt(out.bid_1545),
                'NONPOSITIVE_VOLUME':~out.trade_volume.gt(0),'NONPOSITIVE_OPEN_INTEREST':~out.open_interest.gt(0),
                'INVALID_SPOT_OR_STRIKE':~out.spot.gt(0)|~out.strike.gt(0),'T_OUTSIDE_DEFAULT_WINDOW':~T.between(14/365,1.),
                'MONEYNESS_OUTSIDE_DEFAULT_WINDOW':~(out.strike/out.spot).between(.8,1.2),'SPREAD_ABOVE_DEFAULT_MAX':~((out.ask_1545-out.bid_1545)/mid).le(.2),
                'INVALID_OPTION_TYPE':~out.option_type.isin(['C','P']),'NO_EXACT_RATE_DATE':~out.quote_date.isin(rate_dates)}
    out=out.copy();out['observable_failure_reasons']=[';'.join(name for name,values in conditions.items() if bool(values.loc[i])) or 'NO_OBSERVABLE_DEFAULT_FAILURE_REASON' for i in out.index]
    return out


def recover_raw(config, required, signature):
    sig=token(dict(analysis=signature,required=required.to_dict('records')))
    folder=config['output_root']/'cache'/'raw_recovery'/sig
    if not config['force'] and valid(folder,sig):return pd.read_parquet(folder/'quotes.parquet')
    prepare(folder,sig,config['force']);wanted=pd.MultiIndex.from_frame(required.rename(columns={'intended_exit_date':'quote_date'})[['quote_date','contract_key']])
    strings=set(required.intended_exit_date.dt.strftime('%Y-%m-%d'));parts=[];seen=0;start=time.perf_counter()
    rates=pd.read_csv(config['rates_file']);rate_dates=set(pd.to_datetime(rates['Calendar Date'],dayfirst=True,format='mixed'))
    use=['quote_date','underlying_symbol','root','expiration','strike','option_type','bid_1545','ask_1545','bid_eod','ask_eod','trade_volume','open_interest','underlying_bid_1545','underlying_ask_1545']
    if len(required):
        for raw in pd.read_csv(config['raw_options_file'],usecols=use,chunksize=200000):
            seen+=len(raw);raw=raw[raw.quote_date.isin(strings)]
            if raw.empty:continue
            d=normalize_raw(raw);d=d[pd.MultiIndex.from_frame(d[['quote_date','contract_key']]).isin(wanted)]
            if len(d):parts.append(raw_failure_flags(d,rate_dates))
            if seen%10000000==0:print(f'Exact raw exit scan: {seen:,} rows; {time.perf_counter()-start:.1f}s',flush=True)
    if parts:
        d=pd.concat(parts,ignore_index=True)
        groups=['quote_date','contract_key','underlying_symbol','root','expiration','strike','option_type']
        numeric=['bid_1545','ask_1545','bid_eod','ask_eod','mid','spot','underlying_bid_1545','underlying_ask_1545']
        agg={c:'mean' for c in numeric};agg.update(trade_volume='sum',open_interest='sum',observable_failure_reasons=lambda x:';'.join(sorted(set(';'.join(x).split(';')))))
        quotes=d.groupby(groups,as_index=False,dropna=False).agg(agg)
        quotes=quotes.merge(d.groupby(['quote_date','contract_key']).size().rename('source_observation_count').reset_index(),on=['quote_date','contract_key'],validate='one_to_one')
    else:quotes=pd.DataFrame(columns=['quote_date','contract_key','bid_1545','ask_1545','spot','trade_volume','open_interest'])
    save(quotes,folder/'quotes.parquet');write_json(dict(source_rows=seen,required_pairs=len(required),scan_seconds=time.perf_counter()-start),folder/'scan.json');complete(folder,sig)
    return quotes
