@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Virtual environment not found. Creating one...
  rem 优先通过 py 定位本机 Python，创建失败时停止后续安装和测试。
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
  if errorlevel 1 goto failed
)
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] < (3, 12))"
if errorlevel 1 goto failed
if exist "wheelhouse\*.whl" (
  echo Installing test dependencies from bundled wheelhouse...
  ".venv\Scripts\python.exe" -m pip install --no-index --find-links wheelhouse -r requirements-test.txt
) else (
  echo Bundled wheelhouse not found. Installing from configured Python package index...
  ".venv\Scripts\python.exe" -m pip install -r requirements-test.txt
)
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -c "import fastapi,uvicorn,httpx,pydantic,dotenv,numpy,sklearn,pyarrow,pytest; print('Dependency check OK')"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m compileall -q backend
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pytest -q
if errorlevel 1 goto failed
pause
exit /b 0
:failed
echo Validation failed. Python 3.12 or newer is required. Check the error above.
pause
exit /b 1
