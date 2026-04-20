@echo off
cd /d "%~dp0"

echo ============================================
echo          SAM-EM  Application Launcher
echo ============================================
echo.

REM ── Locate Miniconda / Anaconda ─────────────────────────────────────────────
set "CONDA_ROOT="

if exist "%USERPROFILE%\miniconda3\condabin\conda.bat"  set "CONDA_ROOT=%USERPROFILE%\miniconda3"
if exist "%USERPROFILE%\Miniconda3\condabin\conda.bat"  set "CONDA_ROOT=%USERPROFILE%\Miniconda3"
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat"   set "CONDA_ROOT=%USERPROFILE%\anaconda3"
if exist "%USERPROFILE%\Anaconda3\condabin\conda.bat"   set "CONDA_ROOT=%USERPROFILE%\Anaconda3"
if exist "%ProgramData%\miniconda3\condabin\conda.bat"  set "CONDA_ROOT=%ProgramData%\miniconda3"
if exist "%ProgramData%\Miniconda3\condabin\conda.bat"  set "CONDA_ROOT=%ProgramData%\Miniconda3"

if "%CONDA_ROOT%"=="" (
    echo ERROR: Miniconda or Anaconda was not found.
    echo.
    echo Please install Miniconda from:
    echo   https://docs.conda.io/en/latest/miniconda.html
    echo.
    goto :end
)

echo Found conda at: %CONDA_ROOT%
echo.

REM ── Use conda.exe directly to avoid batch-file shell exit issues ─────────────
set "CONDA_EXE=%CONDA_ROOT%\Scripts\conda.exe"
set "ENV_DIR=%CONDA_ROOT%\envs\SAM-EM-app"
set "PYTHON_EXE=%ENV_DIR%\python.exe"

REM ── First-time setup: create environment if it does not exist ────────────────
if not exist "%ENV_DIR%" (
    echo First-time setup: creating the SAM-EM environment.
    echo This downloads ~2-3 GB and takes 5-15 minutes.
    echo Please wait and do NOT close this window.
    echo.

    "%CONDA_EXE%" env create -f "%~dp0environment_app.yml"
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create the environment.
        echo Check that environment_app.yml is in the same folder as this file.
        goto :end
    )

    echo.
    echo ============================================
    echo   Setup complete! Launching SAM-EM...
    echo ============================================
    echo.
)

REM ── Check Python exists in environment ───────────────────────────────────────
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python not found in the SAM-EM environment.
    echo.
    echo To fix: delete the folder below and run this file again.
    echo   %ENV_DIR%
    goto :end
)

REM ── Launch the application ───────────────────────────────────────────────────
echo Starting SAM-EM application...
echo You can close this window once the app opens.
echo.
pushd "%~dp0application"
"%PYTHON_EXE%" app.py
popd

:end
echo.
pause
