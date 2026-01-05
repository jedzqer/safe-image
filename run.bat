@echo off
setlocal enabledelayedexpansion

set "ROOT=%~dp0"
cd /d "%ROOT%"

set "CONDA_EXE=%ROOT%.conda\Scripts\conda.exe"
set "ENV_DIR=%ROOT%.conda_env"

if not exist "%CONDA_EXE%" (
  echo Miniconda not found at "%CONDA_EXE%".
  echo Please run deploy.ps1 first.
  echo Press any key to exit.
  pause >nul
  exit /b 1
)

if not exist "%ENV_DIR%\python.exe" (
  echo Conda env not found at "%ENV_DIR%".
  echo Please run deploy.ps1 first.
  echo Press any key to exit.
  pause >nul
  exit /b 1
)

set "INPUT_DIR=%ROOT%input"
set "SAFE_DIR=%ROOT%SafeNet"
set "IMG_DIR=%ROOT%images"

if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%SAFE_DIR%" mkdir "%SAFE_DIR%"
if not exist "%IMG_DIR%" mkdir "%IMG_DIR%"

set "HAS_INPUT=0"
for /f "delims=" %%F in ('dir /b /a-d "%INPUT_DIR%\\*" 2^>nul') do (
  set "HAS_INPUT=1"
  goto :HAS_INPUT
)
:HAS_INPUT

if "%HAS_INPUT%"=="1" (
  echo Running SafeNet.py...
  "%CONDA_EXE%" run -p "%ENV_DIR%" python "%ROOT%SafeNet.py"
  if errorlevel 1 goto :ERR
  echo Running nsfwDetector.py...
  "%CONDA_EXE%" run -p "%ENV_DIR%" python "%ROOT%nsfwDetector.py"
  if errorlevel 1 goto :ERR
  echo Moving processed files to images...
  move /Y "%SAFE_DIR%\\*" "%IMG_DIR%\\" >nul 2>&1
  echo Running index_img.py...
  "%CONDA_EXE%" run -p "%ENV_DIR%" python "%ROOT%index_img.py"
  if errorlevel 1 goto :ERR
) else (
  echo No files found in input. Skipping SafeNet/nsfw/index.
)

echo Starting app...
"%CONDA_EXE%" run -p "%ENV_DIR%" python "%ROOT%app.py"
if errorlevel 1 goto :ERR

endlocal
exit /b 0

:ERR
echo.
echo An error occurred. Press any key to exit.
pause >nul
endlocal
exit /b 1
