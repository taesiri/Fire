@echo off
setlocal
node "%~dp0record-fire.mjs" %*
exit /b %ERRORLEVEL%
