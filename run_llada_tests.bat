@echo off
setlocal EnableExtensions

rem Run this from the unredact repository directory.
rem Keep the LLaDA server running on the GPU desktop before starting.
rem Configuration can come from environment variables or optional arguments:
rem   run_llada_tests.bat [redactions] [tag] [remasking] [host] [port]
rem LLADA_URL takes precedence over LLADA_HOST/LLADA_PORT.
if defined LLADA_URL goto :url_ready
set "CLI_HOST=%~4"
if defined CLI_HOST set "LLADA_HOST=%CLI_HOST%"
set "CLI_PORT=%~5"
if defined CLI_PORT set "LLADA_PORT=%CLI_PORT%"
if not defined LLADA_HOST set "LLADA_HOST=127.0.0.1"
if not defined LLADA_PORT set "LLADA_PORT=8000"
set "LLADA_URL=http://%LLADA_HOST%:%LLADA_PORT%"
:url_ready

set "REDACTIONS=%~1"
if not defined REDACTIONS set "REDACTIONS=redactions.json"

set "TAG=%~2"
if not defined TAG set "TAG=llada_8cand"

set "REMASKING=%~3"
if not defined REMASKING set "REMASKING=low_confidence"
if /i not "%REMASKING%"=="low_confidence" if /i not "%REMASKING%"=="random" (
    echo ERROR: remasking must be low_confidence or random.
    exit /b 2
)

echo Checking LLaDA server at %LLADA_URL% ...
python -c "import sys,requests; u=r'%LLADA_URL%/health'; r=requests.get(u,timeout=10); print(r.status_code, r.text); sys.exit(0 if r.ok else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: LLaDA health check failed. Is the desktop server running?
    exit /b 1
)

echo.
echo Sending one tiny LLaDA generation smoke request ...
python -c "import sys,requests; u=r'%LLADA_URL%/generate'; p={'before':'The weather remained','after':' throughout the voyage.','min_chars':4,'max_chars':8,'candidates':1,'steps':64,'gen_length':3,'block_length':3,'temperature':0.8,'remasking':r'%REMASKING%'}; r=requests.post(u,json=p,timeout=300); print(r.status_code, r.text); sys.exit(0 if r.ok and isinstance(r.json().get('candidates'),list) else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: LLaDA generation smoke test failed.
    exit /b 1
)

echo.
echo Running unredact against %REDACTIONS% ...
echo Candidates: 8   Steps: 64   Temperature: 0.8   Remasking: %REMASKING%   Server: %LLADA_URL%
echo Note: this makes one serial 64-step request per candidate and may take a while.
python log_run.py "%TAG%" --backend llada --llada-url "%LLADA_URL%" --llada-steps 64 --llada-temperature 0.8 --llada-remasking "%REMASKING%" --candidates 8 --redactions "%REDACTIONS%"
set "RC=%ERRORLEVEL%"

echo.
if "%RC%"=="0" (
    echo LLaDA test run completed successfully.
    echo Review the newest logs/*_%TAG%.txt and logs/*_%TAG%.json files.
) else (
    echo LLaDA test run failed with exit code %RC%.
)
exit /b %RC%
