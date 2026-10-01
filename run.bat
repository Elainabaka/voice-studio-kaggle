@echo off
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Khong tim thay Python. Cai Python 3.10+ tu https://python.org roi chay lai.
  pause
  exit /b 1
)
python run.py %*
pause
