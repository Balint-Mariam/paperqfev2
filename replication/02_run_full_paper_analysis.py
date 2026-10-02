"""Run the corrected frozen paper design after the upstream reproduction gate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
for stream in [sys.stdout,sys.stderr]:
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
from src.config import load_args
from src.analysis_pipeline import run

if __name__ == '__main__':
    try:
        result=run(load_args(2))
    except Exception as error:
        print('STOP — ANALYSIS REPRODUCTION FAILURE\nFULL ANALYSIS FAILED\n'+str(error))
        result=1
    raise SystemExit(result)
