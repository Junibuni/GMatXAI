@echo off
SETLOCAL ENABLEEXTENSIONS
TITLE Dataset Preparation Script
echo ===================================================
echo       Starting Dataset Generation Process
echo ===================================================

REM Step 1: Activate Python virtual environment
echo [1/4] Activating virtual environment...
call venv\Scripts\activate
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to activate virtual environment.
    goto end
)
echo [1/4] Virtual environment activated successfully.
echo ---------------------------------------------------

REM Step 2: Run dataset generation for 'mpjv'
echo ===================================================
echo [2/4] Generating dataset for: mpjv
python .\data\make_dataset.py -D mpjv
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to generate dataset for mpjv.
    goto end
)

REM Step 3: Run dataset generation for 'mp'
echo ===================================================
echo [3/4] Generating dataset for: mp
python .\data\make_dataset.py -D mp
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to generate dataset for mp.
    goto end
)

REM Step 4: Run dataset generation for 'jv'
echo ===================================================
echo [4/4] Generating dataset for: jv
python .\data\make_dataset.py -D jv
IF ERRORLEVEL 1 (
    echo [ERROR] Failed to generate dataset for jv.
    goto end
)

echo ===================================================
echo [SUCCESS] All dataset generation steps completed.
goto done

:end
echo ===================================================
echo [FAILED] Script terminated due to an error.

:done
ENDLOCAL
pause
