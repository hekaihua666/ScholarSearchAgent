@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  where py >nul 2>nul
  if not errorlevel 1 (
    rem 使用已安装的默认 Python 3，不强制要求本机恰好安装 3.12。
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
)
if not exist ".venv\Scripts\python.exe" goto failed
rem 当前 NumPy 依赖要求 Python 3.12 及以上，安装前先检查解释器版本。
".venv\Scripts\python.exe" -c "import sys; sys.exit(sys.version_info[:2] < (3, 12))"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements-selector.txt
if errorlevel 1 goto failed
".venv\Scripts\python.exe" scripts\prepare_local_model.py
if errorlevel 1 goto failed
if not exist ".env" copy ".env.example" ".env" >nul
echo Installation complete. Fill .env, then run the launch script.
pause
exit /b 0
:failed
echo Installation failed. Check Python 3.12 or newer, network, and the error above.
pause
exit /b 1
