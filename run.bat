@echo off
REM Launcher cho ung dung nhan dien bien so (License Plate Recognition)
cd /d "%~dp0"
".venv\Scripts\python.exe" main.py
if errorlevel 1 (
    echo.
    echo === Co loi xay ra. Nhan phim bat ky de thoat ===
    pause >nul
)
