@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set SCHOLAR_FROZEN_DEMO=1
echo [ScholarSearchAgent] Starting FROZEN demo on http://127.0.0.1:8002 ...
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8002
) else (
  echo Virtual environment not found. Creating one...
  rem 优先使用 Windows Python Launcher，避免 python 未加入 PATH 时创建失败。
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto failed
  ".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] < (3, 12))"
  if errorlevel 1 goto failed
  if exist "wheelhouse\*.whl" (
    echo Installing runtime dependencies from bundled wheelhouse...
    ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheelhouse -r requirements.txt
  ) else (
    echo Bundled wheelhouse not found. Installing from configured Python package index...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  )
  if errorlevel 1 goto failed
  ".venv\Scripts\python.exe" -m uvicorn backend.main:app --host 127.0.0.1 --port 8002
)
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Startup failed. Python 3.12 or newer is required. Check the error above.
pause
exit /b 1
