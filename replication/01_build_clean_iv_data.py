"""Reconstruct and verify raw -> clean -> IV -> surface."""
import sys
sys.dont_write_bytecode = True
for stream in [sys.stdout,sys.stderr]:
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
from src.config import load_args
from src.data_pipeline import run

if __name__ == '__main__':
    try:
        result=run(load_args(1))
    except Exception as error:
        print('STOP — UPSTREAM REPRODUCTION FAILURE\nSTAGE 1 FAILED\n'+str(error))
        result=1
    raise SystemExit(result)
