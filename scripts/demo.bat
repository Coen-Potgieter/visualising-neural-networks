@echo off
setlocal enabledelayedexpansion

REM Exit the script if there is some error
REM Note: In batch, we handle errors differently from bash

REM Define Some Variables for Re-Use
set "TARGET_SCRIPT=main.py"
set "VENV_DIR=env"

REM Colors (using Windows color codes)
set "GREEN=0A"
set "YELLOW=0E"
set "BLUE=09"
set "RED=0C"
set "NC=07"

REM Ask For python command
set "PYTHON_CMD=python"
color %YELLOW%
echo Would you like to use python3? (y/n)
color %NC%
set /p response=
if /i "%response%"=="y" (
    set "PYTHON_CMD=python3"
)

REM Ask which script to run
color %YELLOW%
echo Which script would you like to run?
echo 1) main.py
echo 2) 3d-demo.py
echo 3) ml-img.py
echo Enter Either 1, 2 or 3: 
color %NC%
set /p response=

if "%response%"=="1" (
    set "TARGET_SCRIPT=main.py"
) else if "%response%"=="2" (
    set "TARGET_SCRIPT=3d_demo.py"
) else if "%response%"=="3" (
    set "TARGET_SCRIPT=ml-img.py"
) else (
    echo Invalid input. Please enter 1, 2, or 3.
    exit /b 1
)

color %BLUE%
echo Starting setup with %PYTHON_CMD%...
color %NC%

color %YELLOW%
echo Creating virtual environment...
color %NC%
%PYTHON_CMD% -m venv "%VENV_DIR%"

color %YELLOW%
echo Activating virtual environment...
color %NC%
call "%VENV_DIR%\Scripts\activate.bat"

color %YELLOW%
echo Upgrading pip...
color %NC%
pip install --upgrade pip

color %YELLOW%
echo Installing dependencies...
color %NC%
pip install -r requirements.txt

color %YELLOW%
echo Running the script: %TARGET_SCRIPT%
color %NC%
%PYTHON_CMD% "%TARGET_SCRIPT%"

color %YELLOW%
echo Deactivating virtual environment...
color %NC%
call deactivate

color %YELLOW%
echo Cleaning...
color %NC%
rmdir /s /q "%VENV_DIR%"

color %GREEN%
echo Demo complete
color %NC%

endlocal
