@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  if not exist "models\face_landmarker.task" ".venv\Scripts\python.exe" download_model.py
  ".venv\Scripts\python.exe" app.py %*
) else (
  call conda activate eyetrax_prototype
  if not exist "models\face_landmarker.task" python download_model.py
  python app.py %*
)
