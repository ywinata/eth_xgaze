@echo off
setlocal
cd /d "%~dp0"
call conda activate eyetrax2_prototype
if not exist "models\face_landmarker.task" python download_model.py
python app.py %*
