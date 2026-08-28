@echo off
setlocal
cd /d "%~dp0"
set KMP_DUPLICATE_LIB_OK=TRUE
set OMP_NUM_THREADS=1
conda run -n eth_xgaze_prototype python app.py %*
