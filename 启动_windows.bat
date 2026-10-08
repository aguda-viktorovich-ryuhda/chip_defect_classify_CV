@echo off
rem Chip inspection (solder and wire) - Windows launcher (double-click)
rem ---------------------------------------------------------------
rem Pick an interpreter that REALLY has numpy/opencv/torch.
rem On Windows "where python" often hits the Microsoft Store alias
rem (%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe) which does not
rem actually run -> the server never starts and the page won't open.
rem So every candidate is verified (import test) before being used.
rem ---------------------------------------------------------------
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  Chip inspection (solder and wire) - local web UI
echo ============================================
echo.

rem use UTF-8 for child python I/O (chinese paths / logs)
set "PYTHONUTF8=1"

set "PYEXE="

rem ---- 1) project virtualenv ----
call :try "%~dp0venv\Scripts\python.exe"
if defined PYEXE goto :ready
call :try "%~dp0.venv\Scripts\python.exe"
if defined PYEXE goto :ready

rem ---- 2) explicit override via environment ----
if defined PIPELINE_PYTHON call :try "%PIPELINE_PYTHON%"
if defined PYEXE goto :ready

rem ---- 3) currently activated conda env ----
if defined CONDA_PREFIX call :try "%CONDA_PREFIX%\python.exe"
if defined PYEXE goto :ready

rem ---- 4) all conda envs (from "conda config --show envs_dirs") ----
for /f "tokens=1,*" %%A in ('conda config --show envs_dirs 2^>nul') do (
  if "%%A"=="-" if not defined PYEXE (
    for /d %%D in ("%%B\*") do if not defined PYEXE call :try "%%~D\python.exe"
  )
)
if defined PYEXE goto :ready

rem ---- 5) common conda install locations ----
for %%R in (
  "%USERPROFILE%\miniconda3"
  "%USERPROFILE%\anaconda3"
  "%ProgramData%\miniconda3"
  "%ProgramData%\anaconda3"
  "C:\conda"
  "D:\conda"
) do (
  if not defined PYEXE call :try "%%~R\python.exe"
  if not defined PYEXE if exist "%%~R\envs" (
    for /d %%D in ("%%~R\envs\*") do if not defined PYEXE call :try "%%~D\python.exe"
  )
)
if defined PYEXE goto :ready

rem ---- 6) py launcher (resolve to a real path first) ----
for /f "delims=" %%P in ('py -3 -c "import sys;print(sys.executable)" 2^>nul') do call :try "%%P"
if defined PYEXE goto :ready

rem ---- 7) python on PATH (Store alias is skipped: it fails the import test) ----
for /f "delims=" %%P in ('where python 2^>nul') do (
  if not defined PYEXE call :try "%%P"
)

:ready
if not defined PYEXE (
  echo [ERROR] No usable Python found ^(needs numpy + opencv + torch^).
  echo         Install Python 3.10+ and run:
  echo         python -m pip install numpy opencv-python torch -i https://pypi.tuna.tsinghua.edu.cn/simple
  echo.
  echo         Tip: point to an interpreter explicitly, e.g.
  echo         set PIPELINE_PYTHON=D:\conda\envs\py310\python.exe
  echo.
  pause
  exit /b 1
)

echo Using interpreter: %PYEXE%
"%PYEXE%" -c "import numpy, cv2, torch; print('deps OK:', numpy.__version__, cv2.__version__, torch.__version__)"
if errorlevel 1 echo [WARN] This interpreter is missing numpy/opencv/torch; analysis would fail.

rem S5 (straightness/gap-fill) needs cv2.ximgproc, which only comes with opencv-contrib-python
"%PYEXE%" -c "import cv2; cv2.ximgproc" >nul 2>nul
if errorlevel 1 (
  echo [WARN] opencv-contrib-python not found ^(cv2.ximgproc missing^); S5 would fail.
  echo        Install it with:
  echo        "%PYEXE%" -m pip install opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo.
echo Starting server ... the browser will open http://127.0.0.1:8000
echo Press Ctrl+C in this window to stop.
echo.
"%PYEXE%" ui\app.py --port 8000 --open
pause
exit /b 0

rem ============ helper: is this interpreter usable (deps present)? ============
:try
if not exist "%~1" exit /b 1
"%~1" -c "import numpy, cv2, torch" >nul 2>nul
if errorlevel 1 exit /b 1
set "PYEXE=%~1"
exit /b 0
